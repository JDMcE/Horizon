#pragma once

#include <juce_dsp/juce_dsp.h>
#include "RoomModels.h"
#include "DecayEq.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace hrzn::ir
{

// The early envelope's resolution: 5 ms bins over the first 160 ms.
inline constexpr int numEarlyEnvBins = 32;
inline constexpr double earlyEnvBinSeconds = 0.005;

// Impulse response analysis (feature 6), v0.6.
//
// This half of the IR feature is deliberately *pure*: it takes a block of
// samples and a sample rate and returns numbers. No files, no engine, no
// parameters. Everything downstream — the seed, the fit, the match report —
// works off this struct, which means the hard part can be tested against
// synthetic responses whose answers are known by construction rather than only
// against whatever WAVs happen to be lying around.
//
// It also means the same extraction runs on both sides of the fit: the target
// IR is measured with it, and so is every candidate the engine renders. A fit
// whose two sides are measured by different rulers is fitting the difference
// between the rulers.
//
// ---------------------------------------------------------------------------
// What is measured, and why each one earns its place
// ---------------------------------------------------------------------------
//   - **onset** → Predelay. ISO 3382's convention: the first crossing of 20 dB
//     below the peak. Not the first non-zero sample, which on any recorded IR
//     is the noise floor.
//   - **per-band RT60** → the six Decay Rate EQ bands. A reverb's decay is
//     frequency-dependent by design and a single broadband figure throws away
//     the part that makes a hall sound like a hall rather than a plate.
//   - **early/late energy ratio** → Distance and the ER weighting. How much of
//     the response arrives in the first 80 ms is most of what "how far away
//     does this sound" means.
//   - **echo density buildup** → Space and, through it, the core. When the
//     response stops being discrete echoes and starts being a room is the one
//     measurement that separates an ambience from a cathedral independently of
//     how long either rings.
//   - **band tilt after decay removal** → the Post EQ fit. Band energy on its
//     own says nothing, because a band that rings twice as long holds twice the
//     energy at the same level. Dividing it out leaves the response's actual
//     spectral balance, which is the thing the Post EQ can correct.
struct Features
{
    bool valid = false;

    float onsetSeconds = 0.0f;
    float lengthSeconds = 0.0f;
    float peak = 0.0f;

    // RMS of the response after its onset. Every other measurement here is
    // deliberately scale-invariant — a room does not decay faster because it
    // was recorded hotter — but the A/B needs one that is not, because two
    // reverbs at different levels cannot be compared by ear at all: the louder
    // one just wins.
    float rms = 0.0f;

    // Per-band decay, at the Decay Rate EQ's own band centres.
    std::array<float, numEqBands> bandRt60 {};

    // The mid band's figure, pulled out because it is what Decay Time means.
    float midRt60 = 0.0f;

    // Energy in the first 80 ms after onset over energy in everything after,
    // in dB. Positive is close, negative is far.
    float earlyLateDb = 0.0f;

    // Time for the response to reach the density of Gaussian noise, ms.
    float echoDensityMs = 0.0f;

    // Time by which half the energy of the first 250 ms has arrived, ms.
    //
    // This is the size measurement, and it is a late addition: echo density
    // was the obvious candidate and it turns out to measure something else.
    // Density rises and falls across the room table by design — a plate is
    // denser than a live room twice its size — so it separates *kinds* of room
    // and cannot locate one along the table. Energy centre time rises
    // monotonically from 5.5 ms on the ambience to 139 ms on the largest
    // space, which is what "how big is this room" needs to look like if a
    // search is going to walk it.
    float earlyCentreMs = 0.0f;

    // Time by which 80% of the same energy has arrived, ms (v0.9). The centre
    // alone stopped ranking the halls once their reflections became
    // front-loaded: half the early energy is in before the long taps that say
    // how big the hall is have arrived. The later quantile still hears them.
    float earlyTailMs = 0.0f;

    // The early energy envelope, as QuickStats::earlyEnvDb.
    std::array<float, numEarlyEnvBins> earlyEnvDb {};

    // Band level with each band's own decay divided out, in dB relative to the
    // mean across bands — so a flat response reads zero everywhere whatever
    // its overall level.
    std::array<float, numEqBands> bandTiltDb {};
};

// Band edges: each Decay Rate EQ centre owns the span between the geometric
// midpoints to its neighbours, so the six bands tile the spectrum exactly once
// with no gaps and no overlap in the edge definition. The outer two run to the
// ends, because a shelf's band is everything past its corner.
inline std::pair<float, float> bandEdges (int band, double sampleRate)
{
    const auto& f = decayEqDefaultFreq;
    const int b = juce::jlimit (0, numEqBands - 1, band);

    const float lo = b == 0 ? 20.0f
                            : std::sqrt (f[(size_t) (b - 1)] * f[(size_t) b]);
    const float hi = b == numEqBands - 1
                       ? (float) juce::jmin (20000.0, sampleRate * 0.45)
                       : std::sqrt (f[(size_t) b] * f[(size_t) (b + 1)]);

    return { lo, (float) juce::jmax ((double) lo * 1.05, (double) hi) };
}

namespace detail
{

// Cascaded second-order sections, `pairs` of one high-pass and one low-pass.
//
// Steepness is not a detail here. A band-pass leaks its neighbours' tails in,
// and once the band's own energy has fallen far enough it is the neighbours'
// decay the curve is following — which biases a short band towards its longer
// neighbours and flattens exactly the frequency-dependence this is trying to
// recover. Four pairs is 48 dB/oct.
inline std::vector<juce::dsp::IIR::Coefficients<float>::Ptr>
bandPassCoeffs (float lowHz, float highHz, double sampleRate, int pairs)
{
    using Coefs = juce::dsp::IIR::Coefficients<float>;

    const double nyquist = sampleRate * 0.5;
    const double lo = juce::jlimit (10.0, nyquist * 0.98, (double) lowHz);
    const double hi = juce::jlimit (lo * 1.02, nyquist * 0.98, (double) highHz);

    std::vector<Coefs::Ptr> c;
    c.reserve ((size_t) (2 * pairs));
    for (int i = 0; i < pairs; ++i)
    {
        c.push_back (Coefs::makeHighPass (sampleRate, lo, 0.707f));
        c.push_back (Coefs::makeLowPass (sampleRate, hi, 0.707f));
    }
    return c;
}

inline std::vector<float> bandPass (const std::vector<float>& x, float lowHz, float highHz,
                                    double sampleRate, int pairs = 4)
{
    auto coeffs = bandPassCoeffs (lowHz, highHz, sampleRate, pairs);

    std::vector<juce::dsp::IIR::Filter<float>> stages (coeffs.size());
    for (size_t i = 0; i < coeffs.size(); ++i)
    {
        stages[i].coefficients = coeffs[i];
        stages[i].reset();
    }

    std::vector<float> y (x.size());
    for (size_t i = 0; i < x.size(); ++i)
    {
        float v = x[i];
        for (auto& s : stages)
            v = s.processSample (v);
        y[i] = v;
    }
    return y;
}

// Equivalent noise bandwidth of that cascade: the width of the ideal
// rectangular filter that would pass the same power from white noise.
//
// This is what the band energies have to be divided by, and the nominal edge
// spacing is not it. Four cascaded second-order sections are 12 dB down at
// their own corner, so a band's usable width is well inside its nominal one —
// and by a margin that depends on how many octaves the band spans, which the
// six here do not equally. Dividing by the nominal width made a flat response
// read as a tilt of several dB and would have had the Post EQ fit spending its
// range correcting the ruler.
inline double equivalentNoiseBandwidth (float lowHz, float highHz, double sampleRate,
                                        int pairs = 4)
{
    const auto coeffs = bandPassCoeffs (lowHz, highHz, sampleRate, pairs);

    constexpr int points = 2048;
    const double nyquist = sampleRate * 0.5;
    const double step = nyquist / points;

    double acc = 0.0;
    for (int i = 0; i < points; ++i)
    {
        const double freq = (i + 0.5) * step;
        double mag = 1.0;
        for (const auto& c : coeffs)
            mag *= c->getMagnitudeForFrequency (freq, sampleRate);
        acc += mag * mag;
    }

    return juce::jmax (1.0, acc * step);
}

// RT60 by Schroeder backward integration, fitted over `spanDb` starting 5 dB
// down and extrapolated to a full 60.
//
// T20 rather than T30 inside a band, for the leakage reason above: fitting
// through the crossover where the neighbours take over measures the
// neighbours. ISO 3382 offers T20 for exactly this case.
//
// Returns 0 when the curve never gets far enough down to fit — a truncated
// tail, a response that is all noise floor — which every caller treats as
// "unknown" rather than as "instant".
//
// The truncation guard is not optional and it is the least obvious thing here.
// A backward integral *always* produces a decaying curve, because integrating
// from the tail towards the head necessarily accumulates: a burst of noise cut
// off before it has decayed at all still yields an energy decay curve that
// falls smoothly through -5 and -25 dB, near the very end of the window, and
// fits to a short, confident, entirely fictional RT60. A 6-second tail
// truncated at 120 ms measured as 113 ms — a factor of fifty out, with nothing
// in the curve to suggest anything was wrong.
//
// So the signal itself is asked whether it decayed, before the integral is
// trusted: unless the response has actually fallen by more than the fit span
// between its head and its tail, there is nothing here to measure.
inline double schroederRt60 (const std::vector<float>& h, double sampleRate,
                             double spanDb = 20.0)
{
    if (h.size() < 16)
        return 0.0;

    {
        const size_t edge = juce::jmax<size_t> (16, h.size() / 20);
        double head = 0.0, tail = 0.0;

        for (size_t i = 0; i < edge; ++i)
            head += (double) h[i] * (double) h[i];
        for (size_t i = h.size() - edge; i < h.size(); ++i)
            tail += (double) h[i] * (double) h[i];

        // 5 dB of headroom over the fit span itself, matching the 5 dB the fit
        // discards at the top before it starts.
        const double fell = 10.0 * std::log10 (juce::jmax (head, 1.0e-30)
                                               / juce::jmax (tail, 1.0e-30));
        if (fell < spanDb + 5.0)
            return 0.0;
    }

    std::vector<double> edc (h.size());
    double acc = 0.0;
    for (size_t i = h.size(); i-- > 0;)
    {
        acc += (double) h[i] * (double) h[i];
        edc[i] = acc;
    }

    if (! (edc[0] > 0.0))
        return 0.0;

    const double ref = edc[0];
    auto timeAtDb = [&] (double db) -> double
    {
        for (size_t i = 0; i < edc.size(); ++i)
            if (10.0 * std::log10 (edc[i] / ref) <= db)
                return (double) i / sampleRate;
        return -1.0;
    };

    const double t5 = timeAtDb (-5.0), tEnd = timeAtDb (-5.0 - spanDb);
    if (t5 < 0.0 || tEnd < 0.0 || tEnd <= t5)
        return 0.0;

    return (tEnd - t5) * (60.0 / spanDb);
}

inline double energyOf (const std::vector<float>& h, size_t from, size_t to)
{
    to = juce::jmin (to, h.size());
    double acc = 0.0;
    for (size_t i = from; i < to; ++i)
        acc += (double) h[i] * (double) h[i];
    return acc;
}

// Echo density buildup (Abel & Huang): the fraction of samples in a sliding
// window exceeding the window's own standard deviation, normalised so that
// Gaussian noise reads exactly 1. The same measure the transparency suite has
// used since v0.2, so an imported IR and a rendered candidate are placed on
// the table by the same ruler the table was built with.
inline double echoDensityBuildupMs (const std::vector<float>& h, double sampleRate,
                                    double threshold = 0.9)
{
    constexpr int win = 1024, hop = 128;
    constexpr double gaussianFraction = 0.3173;   // P(|x| > sigma)

    if (h.size() < (size_t) win)
        return 0.0;

    for (size_t s = 0; s + (size_t) win < h.size(); s += (size_t) hop)
    {
        double mean = 0.0;
        for (int i = 0; i < win; ++i)
            mean += (double) h[s + (size_t) i];
        mean /= win;

        double var = 0.0;
        for (int i = 0; i < win; ++i)
        {
            const double d = (double) h[s + (size_t) i] - mean;
            var += d * d;
        }
        const double sd = std::sqrt (var / win);
        if (sd <= 1.0e-12)
            continue;

        int over = 0;
        for (int i = 0; i < win; ++i)
            if (std::abs ((double) h[s + (size_t) i] - mean) > sd)
                ++over;

        if ((double) over / win / gaussianFraction >= threshold)
            return (double) (s + (size_t) (win / 2)) * 1000.0 / sampleRate;
    }

    // Never got there: a response of discrete echoes, or one too short to
    // tell. Reporting its whole length is the honest answer and puts it at the
    // sparse end of the table, which is where such a response belongs.
    return (double) h.size() * 1000.0 / sampleRate;
}

} // namespace detail

// The measurements that need no band filtering, which is all of the ones the
// parameter search actually walks. Split out because filtering six bands over
// a multi-second response is most of what an analysis costs, and a search that
// only wants "how big" and "how close" should not pay for it thirty times.
struct QuickStats
{
    bool valid = false;
    float peak = 0.0f;
    float onsetSeconds = 0.0f;
    float earlyCentreMs = 0.0f;
    float earlyTailMs = 0.0f;
    float echoDensityMs = 0.0f;
    float earlyLateDb = 0.0f;
    size_t onsetSample = 0;

    // Energy per 5 ms bin after onset, in dB against the loudest bin, floored
    // at -40 (v0.9). The scalar measurements summarise the early field; this
    // keeps its *pattern*, which is what still tells a studio from a small hall
    // heard from close up when every summary of the two agrees.
    std::array<float, numEarlyEnvBins> earlyEnvDb {};
};

inline QuickStats quickAnalyse (const std::vector<float>& h, double sampleRate)
{
    QuickStats q;

    if (h.size() < 64 || sampleRate < 8000.0)
        return q;

    for (float v : h)
        q.peak = juce::jmax (q.peak, std::abs (v));

    if (q.peak <= 1.0e-7f)
        return q;

    // Onset. ISO 3382 puts this 20 dB below the peak, which is right for a
    // room response with a direct sound in it and wrong for a reverb, where
    // there is no direct arrival and the response builds. Measured against the
    // engine's own output at zero predelay, the 20 dB rule reported anywhere
    // from 3 to 19 ms of predelay that was not there — so it goes 50 dB down
    // instead, with a floor guard so a recorded response's noise does not get
    // mistaken for the start of the reverb.
    double quietest = 1.0e30;
    {
        const size_t win = juce::jmax<size_t> (64, (size_t) (0.005 * sampleRate));
        for (size_t s = 0; s + win <= h.size(); s += win)
        {
            double acc = 0.0;
            for (size_t i = 0; i < win; ++i)
                acc += (double) h[s + i] * (double) h[s + i];
            quietest = juce::jmin (quietest, std::sqrt (acc / (double) win));
        }
    }

    const float threshold = juce::jmax (q.peak * 0.00316f,          // -50 dB
                                        (float) (8.0 * quietest));

    for (size_t i = 0; i < h.size(); ++i)
        if (std::abs (h[i]) >= threshold)
        {
            q.onsetSample = i;
            break;
        }

    q.onsetSeconds = (float) ((double) q.onsetSample / sampleRate);

    const size_t n = h.size() - q.onsetSample;
    if (n < 64)
        return q;

    // Energy centre of the first 250 ms after onset — the size measurement.
    {
        const size_t win = juce::jmin (n, (size_t) (0.25 * sampleRate));
        double total = 0.0;
        for (size_t i = 0; i < win; ++i)
        {
            const double v = (double) h[q.onsetSample + i];
            total += v * v;
        }

        double acc = 0.0;
        q.earlyCentreMs = (float) ((double) win * 1000.0 / sampleRate);
        q.earlyTailMs = q.earlyCentreMs;
        bool centreFound = false;
        for (size_t i = 0; i < win; ++i)
        {
            const double v = (double) h[q.onsetSample + i];
            acc += v * v;
            if (! centreFound && acc >= 0.5 * total)
            {
                q.earlyCentreMs = (float) ((double) i * 1000.0 / sampleRate);
                centreFound = true;
            }
            if (acc >= 0.8 * total)
            {
                q.earlyTailMs = (float) ((double) i * 1000.0 / sampleRate);
                break;
            }
        }
    }

    {
        const auto bin = juce::jmax<size_t> (1, (size_t) (earlyEnvBinSeconds * sampleRate));
        std::array<double, numEarlyEnvBins> e {};
        double loudest = 0.0;
        for (int b = 0; b < numEarlyEnvBins; ++b)
        {
            const size_t from = q.onsetSample + (size_t) b * bin;
            for (size_t i = from; i < from + bin && i < h.size(); ++i)
                e[(size_t) b] += (double) h[i] * (double) h[i];
            loudest = juce::jmax (loudest, e[(size_t) b]);
        }
        for (int b = 0; b < numEarlyEnvBins; ++b)
            q.earlyEnvDb[(size_t) b] = (float) juce::jmax (
                -40.0, 10.0 * std::log10 (juce::jmax (1.0e-30, e[(size_t) b])
                                          / juce::jmax (1.0e-30, loudest)));
    }

    {
        const std::vector<float> body (h.begin() + (long) q.onsetSample, h.end());
        const auto split = (size_t) (0.080 * sampleRate);
        const double early = detail::energyOf (body, 0, split);
        const double late = detail::energyOf (body, split, body.size());

        q.earlyLateDb = juce::jlimit (-40.0f, 40.0f,
            (float) (10.0 * std::log10 (juce::jmax (early, 1.0e-20)
                                        / juce::jmax (late, 1.0e-20))));

        q.echoDensityMs = (float) detail::echoDensityBuildupMs (body, sampleRate);
    }

    q.valid = true;
    return q;
}

// The whole pipeline. `ir` is a mono response at `sampleRate`; multichannel
// files are summed before they get here, because everything measured above is
// a property of the room rather than of the microphone array.
inline Features analyse (const float* ir, int numSamples, double sampleRate)
{
    Features f;

    if (ir == nullptr || numSamples < 64 || sampleRate < 8000.0)
        return f;

    std::vector<float> h (ir, ir + numSamples);

    // Anything that is not a finite number is treated as absent rather than
    // propagated: one NaN in a file would otherwise poison every measurement
    // below it and land the user on a set of parameters made of nonsense.
    for (auto& v : h)
        if (! std::isfinite (v))
            v = 0.0f;

    const auto q = quickAnalyse (h, sampleRate);

    f.peak = q.peak;
    f.lengthSeconds = (float) (numSamples / sampleRate);

    if (! q.valid)
        return f;   // silence, or too short: the caller keeps its state

    f.onsetSeconds = q.onsetSeconds;
    f.earlyCentreMs = q.earlyCentreMs;
    f.earlyTailMs = q.earlyTailMs;
    f.earlyEnvDb = q.earlyEnvDb;
    f.echoDensityMs = q.echoDensityMs;
    f.earlyLateDb = q.earlyLateDb;

    // Everything past the onset is the response proper. Measuring the decay
    // across the silent run-in would put a flat stretch at the head of every
    // energy decay curve and read as a longer reverb than there is.
    std::vector<float> body (h.begin() + (long) q.onsetSample, h.end());
    if (body.size() < 64)
        return f;

    f.rms = (float) std::sqrt (detail::energyOf (body, 0, body.size())
                               / (double) body.size());

    // Per-band decay and per-band energy, from the same filtered copy.
    std::array<double, numEqBands> bandEnergy {};
    for (int b = 0; b < numEqBands; ++b)
    {
        const auto [lo, hi] = bandEdges (b, sampleRate);
        const auto banded = detail::bandPass (body, lo, hi, sampleRate);

        f.bandRt60[(size_t) b] = (float) detail::schroederRt60 (banded, sampleRate);
        bandEnergy[(size_t) b] = detail::energyOf (banded, 0, banded.size());
    }

    // Bands that could not be fitted inherit the nearest one that could, so a
    // top octave buried in noise does not read as a zero-length decay and pull
    // the whole fit after it.
    {
        float lastGood = 0.0f;
        for (int b = 0; b < numEqBands; ++b)
            if (f.bandRt60[(size_t) b] > 0.0f)
                lastGood = f.bandRt60[(size_t) b];
            else if (lastGood > 0.0f)
                f.bandRt60[(size_t) b] = lastGood;

        for (int b = numEqBands; b-- > 0;)
            if (f.bandRt60[(size_t) b] > 0.0f)
                lastGood = f.bandRt60[(size_t) b];
            else if (lastGood > 0.0f)
                f.bandRt60[(size_t) b] = lastGood;
    }

    // The mid band is band 2 (900 Hz), which is what Decay Time means and what
    // ISO quotes.
    f.midRt60 = f.bandRt60[2];

    // Spectral balance with each band's own decay divided out. Energy in a
    // band is its initial power times its decay time (to within the constant
    // that cancels in the difference), so dividing by the decay recovers the
    // power — which is what the Post EQ actually shapes.
    //
    // Bandwidth is divided out too, which turns the figure into a power
    // *density*. Without it the six bands are not comparable at all: the top
    // one spans 13 kHz and the bottom one 185 Hz, so a genuinely flat response
    // would read as an 18 dB rising tilt and the Post EQ fit would spend its
    // range correcting the band edges.
    {
        std::array<double, numEqBands> levelDb {};
        double mean = 0.0;

        for (int b = 0; b < numEqBands; ++b)
        {
            const auto [lo, hi] = bandEdges (b, sampleRate);
            const double rt = juce::jmax (0.01, (double) f.bandRt60[(size_t) b]);
            const double width = detail::equivalentNoiseBandwidth (lo, hi, sampleRate);
            const double power = juce::jmax (bandEnergy[(size_t) b], 1.0e-20) / rt / width;
            levelDb[(size_t) b] = 10.0 * std::log10 (power);
            mean += levelDb[(size_t) b];
        }

        mean /= numEqBands;

        for (int b = 0; b < numEqBands; ++b)
            f.bandTiltDb[(size_t) b] =
                juce::jlimit (-40.0f, 40.0f, (float) (levelDb[(size_t) b] - mean));
    }

    f.valid = f.midRt60 > 0.0f;
    return f;
}

} // namespace hrzn::ir
