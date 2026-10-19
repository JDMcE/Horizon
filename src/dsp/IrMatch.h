#pragma once

#include "IrAnalysis.h"
#include "ReverbEngine.h"
#include "../Parameters.h"
#include <functional>

namespace hrzn::ir
{

// Fitting an imported IR to an engine state (feature 6), v0.6.
//
// The reference product's headline v2 feature, and the honest description of
// it is *approximation*: a fourteen-model algorithmic reverb cannot become an
// arbitrary recorded room, and anything that claims otherwise is selling
// something. What it can do is land on the state whose measurements match the
// recording's measurements, and say how close it got — which is why the match
// report is part of the feature rather than a nicety bolted on afterwards.
//
// ---------------------------------------------------------------------------
// Why this is a search and not a formula
// ---------------------------------------------------------------------------
// Two of the five quantities invert directly. Predelay is the onset. Decay Time
// is the mid-band RT60. Both are read straight off the analysis and neither
// needs an engine.
//
// The other three do not invert at all. Space moves fourteen geometric
// properties at once and its effect on echo density depends on the room it
// lands in; Distance moves four things together on purpose (see the engine);
// and the Decay Rate EQ's multipliers are defined *against the room's own
// decay curve*, which is not known until Space is. So those are found by
// rendering candidates and measuring them with the same extraction the target
// went through — the one arrangement in which a ruler error cancels instead of
// becoming the answer.
//
// The search is bounded and its cost is fixed: roughly thirty renders, most of
// them short. It runs off the audio thread and reports progress, because a
// second of work with no indication that anything is happening reads as a
// hang.
struct Match
{
    bool valid = false;

    float space01 = 0.55f;
    float decaySeconds = 2.0f;
    float predelayMs = 0.0f;
    float distance01 = 0.35f;

    std::array<float, numEqBands> decayMult {};
    std::array<float, numEqBands> postGainDb {};
    std::array<float, numEqBands> postFreq {};
    std::array<float, numEqBands> postQ {};

    // How close the fit got. `residualDb` is the physical figure — an RMS over
    // the per-band decay and tone residuals, in dB — and `matchPercent` is a
    // monotone reading of it for the panel. The percentage is not a physical
    // quantity and is documented as such below; the dB figure is the one to
    // argue with.
    float residualDb = 0.0f;
    float matchPercent = 0.0f;

    Features target;    // what was measured off the source
    Features fitted;    // ...and off the state this returned
};

namespace detail
{

// A candidate rendered to a mono impulse response, measured the same way the
// target was.
//
// Space is set before `prepare` deliberately: the engine starts *at* its Space
// position rather than gliding to it from wherever it was, and a candidate
// that spent the first second of its own impulse response morphing would be
// measured mid-journey.
inline std::vector<float> render (const ReverbEngine::Params& p, double seconds,
                                  double sampleRate)
{
    // Always fitted against a stereo room, whatever the session is running in.
    // The response being matched is a stereo (or folded) recording, the features
    // are measured off a mono sum of it, and a fit that rendered candidates in
    // 7.1.4 would be fitting a room's *layout* to a measurement that contains no
    // information about one — at fifteen times the cost of the one it should
    // have rendered.
    SpeakerLayout stereo;
    auto engine = std::make_unique<ReverbEngine>();
    engine->setParams (p);
    engine->prepare (sampleRate, stereo);

    const int total = juce::jmax (1024, (int) (seconds * sampleRate));
    constexpr int block = 512;

    std::vector<float> out;
    out.reserve ((size_t) total + block);

    std::vector<float> l ((size_t) block), r ((size_t) block);
    float* chans[2] = { l.data(), r.data() };

    bool first = true;
    while ((int) out.size() < total)
    {
        std::fill (l.begin(), l.end(), 0.0f);
        std::fill (r.begin(), r.end(), 0.0f);

        if (first)
        {
            l[0] = r[0] = 1.0f;
            first = false;
        }

        engine->process (chans, 2, block);

        // Summed to mono, because that is what the target was reduced to and
        // because everything measured is a property of the room rather than of
        // the stereo image.
        for (int i = 0; i < block; ++i)
            out.push_back (0.5f * (l[(size_t) i] + r[(size_t) i]));
    }

    return out;
}

// The engine parameters a candidate carries, with everything the fit does not
// touch left where the panel's own defaults put it. Character and Thickness
// stay at their midpoints: they are the controls that voice a room *away* from
// its model, and letting the fit move them would let it reach the right
// density by the wrong route and leave the user with a state they cannot
// reason about afterwards.
inline ReverbEngine::Params candidateParams (float space01, float decaySeconds,
                                             float distance01)
{
    ReverbEngine::Params p;
    p.space01 = juce::jlimit (0.0f, 1.0f, space01);
    p.decaySeconds = juce::jlimit (minDecaySeconds, maxDecaySeconds, decaySeconds);
    p.distance01 = juce::jlimit (0.0f, 1.0f, distance01);
    p.predelayMs = 0.0f;      // fitted directly; rendering it would only pad the front
    p.mix01 = 1.0f;
    p.width = 1.0f;
    p.brightness01 = 0.5f;
    p.character01 = 0.5f;
    p.thickness01 = 0.5f;

    for (int b = 0; b < numEqBands; ++b)
    {
        auto& d = p.decayBands[(size_t) b];
        d.on = true;
        d.shape = decayEqShapes[(size_t) b];
        d.freq = decayEqDefaultFreq[(size_t) b];
        d.mult = 1.0f;
        d.q = decayEqDefaultQ[(size_t) b];
    }

    return p;
}

// How long a candidate has to be rendered for its decay to be measurable: the
// extraction needs the response to fall well past its fit span, and the
// truncation guard will refuse it otherwise.
inline double renderSecondsFor (float decaySeconds)
{
    return juce::jlimit (0.8, 8.0, (double) decaySeconds * 1.1 + 0.3);
}

} // namespace detail

// Progress is reported 0..1 so an import can show something moving. Called on
// the calling thread; the fit never touches the audio thread or the editor.
using ProgressFn = std::function<void (float)>;

// Fits a mono impulse response. `sampleRate` is the rate the response is at,
// which the caller has already resampled to the session's.
inline Match fitToIr (const float* ir, int numSamples, double sampleRate,
                      const ProgressFn& progress = {})
{
    Match m;
    m.target = analyse (ir, numSamples, sampleRate);

    if (! m.target.valid)
        return m;

    // A response with no tail is not a room. The engine's own floor is 100 ms,
    // and something measuring below half of that is a click, a gate, or a file
    // that is not an impulse response at all — for which the right answer is
    // "no fit" rather than a set of parameters clamped to the bottom of every
    // range. A genuinely short room still fits; it is clamped, not refused.
    if (m.target.midRt60 < minDecaySeconds * 0.5f)
        return m;

    auto report = [&progress] (float p) { if (progress) progress (juce::jlimit (0.0f, 1.0f, p)); };
    report (0.05f);

    // Decay Time inverts directly: it is the mid-band RT60 and nothing else.
    m.decaySeconds = juce::jlimit (minDecaySeconds, maxDecaySeconds, m.target.midRt60);

    // Predelay does not, and the reason is worth stating. The onset of a
    // response is when its *first reflection* arrives, which is the predelay
    // plus however long the room itself takes to produce anything — and that
    // second term is a property of the room, not of the control. Setting
    // Predelay to the measured onset therefore counts the room's own initial
    // delay twice, and the error is not small: measured against the engine at
    // zero predelay it runs from 6 ms on a studio to 13 ms on a cathedral.
    //
    // So it is fitted as a difference, once Space is known: the gap the room
    // does not already supply. Filled in below, after the search.
    m.predelayMs = 0.0f;

    // Space and Distance, alternately ----------------------------------------
    //
    // Space is found from how long the response takes to deliver its early
    // energy — the size measurement — with echo density as a secondary term
    // that decides *what kind* of room of that size it is. Density cannot
    // carry the search on its own: it rises and falls along the table by
    // design, since a plate is denser than a live room twice its size, so on
    // its own it puts a studio and a small hall equally close to the same
    // target and picks between them by rounding. It is a real measurement of a
    // real property, just not of size.
    //
    // The two controls interact — Distance moves the early/late balance, which
    // moves where the early energy centres — so they are alternated rather
    // than solved once each. Two rounds is enough; a third never moved either
    // in testing.
    const double targetCentre = juce::jmax (0.5, (double) m.target.earlyCentreMs);
    const double targetDensity = juce::jmax (1.0, (double) m.target.echoDensityMs);
    const double targetEarlyLate = (double) m.target.earlyLateDb;

    m.space01 = 0.55f;
    m.distance01 = 0.35f;

    auto sizeError = [&] (const QuickStats& q)
    {
        const double centre = std::abs (std::log (juce::jmax (0.5, (double) q.earlyCentreMs)
                                                  / targetCentre));
        const double density = std::abs (std::log (juce::jmax (1.0, (double) q.echoDensityMs)
                                                   / targetDensity));
        return centre + 0.35 * density;
    };

    constexpr int rounds = 2;
    for (int round = 0; round < rounds; ++round)
    {
        const float base = 0.05f + 0.325f * (float) round;

        // Space, over the table. Short renders: everything this stage looks at
        // lives in the first quarter second.
        {
            double bestErr = 1.0e30;
            float bestSpace = m.space01;

            auto trySpace = [&] (float space01)
            {
                const auto p = detail::candidateParams (space01, m.decaySeconds, m.distance01);
                const auto h = detail::render (p, 0.4, sampleRate);
                const auto q = quickAnalyse (h, sampleRate);

                if (! q.valid)
                    return;

                const double err = sizeError (q);
                if (err < bestErr)
                {
                    bestErr = err;
                    bestSpace = space01;
                }
            };

            for (int model = 0; model < numRoomModels; ++model)
            {
                trySpace ((float) model / (float) (numRoomModels - 1));
                report (base + 0.22f * (float) (model + 1) / (float) numRoomModels);
            }

            // The table is stepless from v0.2, so there is no reason to leave
            // the answer parked on a model when the room being matched is
            // between two of them.
            const float half = 0.5f / (float) (numRoomModels - 1);
            trySpace (juce::jlimit (0.0f, 1.0f, bestSpace - half));
            trySpace (juce::jlimit (0.0f, 1.0f, bestSpace + half));

            m.space01 = bestSpace;
        }

        // Distance, from the early/late energy balance.
        {
            double bestErr = 1.0e30;
            float bestDistance = m.distance01;

            auto tryDistance = [&] (float d)
            {
                d = juce::jlimit (0.0f, 1.0f, d);
                const auto p = detail::candidateParams (m.space01, m.decaySeconds, d);
                const auto h = detail::render (p, 1.2, sampleRate);
                const auto q = quickAnalyse (h, sampleRate);

                if (! q.valid)
                    return;

                const double err = std::abs ((double) q.earlyLateDb - targetEarlyLate);
                if (err < bestErr)
                {
                    bestErr = err;
                    bestDistance = d;
                }
            };

            for (int i = 0; i <= 6; ++i)
            {
                tryDistance ((float) i / 6.0f);
                report (base + 0.22f + 0.09f * (float) (i + 1) / 7.0f);
            }

            // Half a step either side, but only on the last round. Refining
            // Distance mid-alternation feeds a finer value back into the next
            // Space search and moves it: the two are partly degenerate — a
            // slightly larger room heard from slightly closer measures much
            // like a smaller one heard from further away — so a small change
            // in one can hand the other a different answer of equal quality.
            // Doing it once, after Space has settled, keeps the refinement
            // without letting it perturb what it is refining against.
            if (round == rounds - 1)
            {
                const float halfStep = 1.0f / 12.0f;
                tryDistance (bestDistance - halfStep);
                tryDistance (bestDistance + halfStep);
            }

            m.distance01 = bestDistance;
        }
    }

    report (0.7f);

    // The Decay Rate EQ, against the room's own curve -------------------------
    //
    // A multiplier here means "this band rings this many times as long as this
    // room would ring at this frequency on its own", so the room's neutral
    // curve has to be in hand before any of them can be set — which is why
    // this could not have happened before Space did.
    auto params = detail::candidateParams (m.space01, m.decaySeconds, m.distance01);

    for (auto& mult : m.decayMult)
        mult = 1.0f;

    for (int pass = 0; pass < 2; ++pass)
    {
        for (int b = 0; b < numEqBands; ++b)
            params.decayBands[(size_t) b].mult = m.decayMult[(size_t) b];

        const auto h = detail::render (params, detail::renderSecondsFor (m.decaySeconds),
                                       sampleRate);
        const auto f = analyse (h.data(), (int) h.size(), sampleRate);

        if (! f.valid)
            break;

        // The room's own time to first reflection, taken off the target's
        // onset to leave the part Predelay actually has to supply. Read from
        // this render because it is the first one carrying the fitted Space
        // and Distance, and it is rendered at zero predelay.
        if (pass == 0)
            m.predelayMs = juce::jlimit (
                minPredelayMs, maxPredelayMs,
                (m.target.onsetSeconds - f.onsetSeconds) * 1000.0f);

        // Multiplicative correction rather than a solve: the bands overlap, so
        // one band's multiplier moves its neighbours' delivered decay too, and
        // a correction that is applied and then re-measured converges where a
        // single algebraic step would overshoot.
        for (int b = 0; b < numEqBands; ++b)
        {
            const float measured = juce::jmax (0.01f, f.bandRt60[(size_t) b]);
            const float wanted = juce::jmax (0.01f, m.target.bandRt60[(size_t) b]);
            m.decayMult[(size_t) b] = juce::jlimit (
                minDecayMult, maxDecayMult,
                m.decayMult[(size_t) b] * wanted / measured);
        }

        report (0.7f + 0.1f * (float) (pass + 1) / 2.0f);
    }

    for (int b = 0; b < numEqBands; ++b)
        params.decayBands[(size_t) b].mult = m.decayMult[(size_t) b];

    // The Post EQ, from what is left over ------------------------------------
    //
    // Fitted last and against the *residual*, because everything above changes
    // the tone as a side effect of changing the decay. Whatever difference in
    // spectral balance survives all of it is exactly what a tone control on
    // the wet output is for. The band frequencies are moved onto the same
    // centres the tilt was measured at, so the correction lands where it was
    // read rather than half an octave away.
    {
        const auto h = detail::render (params, detail::renderSecondsFor (m.decaySeconds),
                                       sampleRate);
        const auto f = analyse (h.data(), (int) h.size(), sampleRate);

        for (int b = 0; b < numEqBands; ++b)
        {
            const auto [lo, hi] = bandEdges (b, sampleRate);
            const float centre = decayEqDefaultFreq[(size_t) b];

            m.postFreq[(size_t) b] = centre;
            m.postQ[(size_t) b] = (b == 0 || b == numEqBands - 1)
                ? 0.71f
                : juce::jlimit (0.4f, 2.0f, centre / juce::jmax (1.0f, hi - lo));

            m.postGainDb[(size_t) b] = f.valid
                ? juce::jlimit (minPostGainDb, maxPostGainDb,
                                m.target.bandTiltDb[(size_t) b] - f.bandTiltDb[(size_t) b])
                : 0.0f;
        }
    }

    report (0.9f);

    // What it came to --------------------------------------------------------
    //
    // Measured on a render carrying the whole fitted state — both EQ sections
    // included — because a report on the state before its own corrections were
    // applied would be a report on a different reverb.
    {
        for (int b = 0; b < numEqBands; ++b)
        {
            auto& pb = params.postBands[(size_t) b];
            pb.on = std::abs (m.postGainDb[(size_t) b]) > 0.05f;
            pb.shape = postEqDefaultShape[(size_t) b];
            pb.channel = ChannelMode::stereo;
            pb.freq = m.postFreq[(size_t) b];
            pb.gainDb = m.postGainDb[(size_t) b];
            pb.q = m.postQ[(size_t) b];
            pb.slopeIndex = 1;
        }

        const auto h = detail::render (params, detail::renderSecondsFor (m.decaySeconds),
                                       sampleRate);
        m.fitted = analyse (h.data(), (int) h.size(), sampleRate);

        double acc = 0.0;
        int terms = 0;

        if (m.fitted.valid)
        {
            for (int b = 0; b < numEqBands; ++b)
            {
                // Decay residual in dB: a band twice as long as asked for is
                // 6 dB out, which puts decay and tone on one scale and stops
                // the report being dominated by whichever happens to be
                // measured in bigger numbers.
                const double ratio = juce::jmax (0.01f, m.fitted.bandRt60[(size_t) b])
                                   / juce::jmax (0.01f, m.target.bandRt60[(size_t) b]);
                const double decayDb = 20.0 * std::log10 (ratio);
                const double toneDb = m.fitted.bandTiltDb[(size_t) b]
                                    - m.target.bandTiltDb[(size_t) b];

                acc += decayDb * decayDb + toneDb * toneDb;
                terms += 2;
            }

            const double earlyLate = m.fitted.earlyLateDb - m.target.earlyLateDb;
            acc += earlyLate * earlyLate;
            ++terms;
        }

        m.residualDb = terms > 0 ? (float) std::sqrt (acc / terms) : 99.0f;

        // A monotone reading of the residual for the panel, and nothing more:
        // 0 dB is 100%, 8 dB RMS is about a third. There is no principled map
        // from "how far out the measurements are" to "how close it sounds",
        // and inventing one that looked principled would be worse than
        // admitting this is a scale.
        m.matchPercent = 100.0f * std::exp (-m.residualDb / 8.0f);
    }

    m.valid = true;
    report (1.0f);
    return m;
}

} // namespace hrzn::ir
