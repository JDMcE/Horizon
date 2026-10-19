#pragma once

#include <juce_core/juce_core.h>
#include <array>
#include <cmath>
#include <complex>

namespace hrzn
{

// Second-order section maths, adapted from Equinox's `src/dsp/FilterCoeffs.h`
// (the family's EQ, where this shape set has been shipping since its v0.2).
// Horizon needs it twice over at v0.3 — the Post EQ on the wet output and the
// Decay Rate EQ inside the feedback path — and the arithmetic is identical in
// both places, so it is worth having exactly one copy of it here.
//
// Trimmed to the shapes a reverb actually needs: the band-pass, all-pass and
// tilt shapes Equinox offers have no use on a reverb's output and none at all
// inside a feedback loop.
//
// **Promotion candidate**, on the same footing as `ui/PrecisionSlider.h`: two
// products now carry the same RBJ cookbook, and the third time it is wanted it
// should move to `jdm_shared` rather than be copied again.

// A band's cascade is at most four sections — a 48 dB/oct cut.
inline constexpr int maxStagesPerBand = 4;

enum class FilterShape
{
    bell = 0,
    lowShelf,
    highShelf,
    lowCut,
    highCut,
    notch
};

inline const juce::StringArray filterShapeNames {
    "Bell", "Low Shelf", "High Shelf", "Low Cut", "High Cut", "Notch"
};

// Cut slopes, as filter orders: 6, 12, 24 and 48 dB per octave.
inline constexpr std::array<int, 4> slopeOrders { { 1, 2, 4, 8 } };

inline const juce::StringArray slopeNames { "6 dB/oct", "12 dB/oct", "24 dB/oct",
                                            "48 dB/oct" };

// Which channels a Post EQ band acts on (features 15 and 14).
//
// The first five are v0.3's, and they generalise rather than being replaced:
// beyond stereo, `left` and `right` are the halves of the room either side of
// the median plane, and `mid` and `side` fold about every mirrored *pair* of
// speakers at once — the front pair, the surrounds, the height pair — with the
// unpaired channels (centre, top centre) treated as what they are, pure mid.
//
// The last four are v0.7's Speakers setting: the speaker groups of a surround or
// immersive layout. They are appended rather than interleaved because the value
// saved in a state is the choice *index*, so inserting one anywhere else would
// silently rewrite every band in every preset ever saved.
//
// In a stereo session the four group entries still resolve — `Front` is the
// pair, and the other three are empty, so a band set to one of them does
// nothing. The UI says so rather than hiding them, because a preset written for
// 7.1.4 has to survive being opened on a stereo track and come back unchanged.
enum class ChannelMode
{
    stereo = 0, left, right, mid, side,
    front, centre, surround, height
};

inline const juce::StringArray channelModeNames { "Stereo", "Left", "Right",
                                                  "Mid", "Side",
                                                  "Front", "Centre", "Surround",
                                                  "Height" };

// The subset the Decay Rate EQ offers. A reverb tank has no stereo image to
// split — the decorrelation happens at the output rows — so left/right/mid/side
// are meaningless there, and what is left is exactly the speaker groups.
inline bool isSpeakerGroupMode (ChannelMode m) noexcept
{
    return m >= ChannelMode::front;
}

// One normalized biquad section (a0 == 1).
struct BiquadCoeffs
{
    double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
};

// Coefficient set for one band: up to maxStagesPerBand cascaded sections.
struct BandCoeffs
{
    std::array<BiquadCoeffs, maxStagesPerBand> stages {};
    int numStages = 0;
};

namespace detail
{
    inline constexpr double pi = 3.141592653589793238;

    inline BiquadCoeffs rbjBell (double w0, double q, double gainDb)
    {
        const double A = std::pow (10.0, gainDb / 40.0);
        const double cs = std::cos (w0), alpha = std::sin (w0) / (2.0 * q);
        const double a0 = 1.0 + alpha / A;
        return { (1.0 + alpha * A) / a0, (-2.0 * cs) / a0, (1.0 - alpha * A) / a0,
                 (-2.0 * cs) / a0, (1.0 - alpha / A) / a0 };
    }

    inline BiquadCoeffs rbjShelf (double w0, double q, double gainDb, bool high)
    {
        const double A = std::pow (10.0, gainDb / 40.0);
        const double cs = std::cos (w0), alpha = std::sin (w0) / (2.0 * q);
        const double sqA2a = 2.0 * std::sqrt (A) * alpha;
        const double sign = high ? -1.0 : 1.0;

        const double b0 =      A * ((A + 1.0) - sign * (A - 1.0) * cs + sqA2a);
        const double b1 = 2.0 * A * ((A - 1.0) - sign * (A + 1.0) * cs) * sign;
        const double b2 =      A * ((A + 1.0) - sign * (A - 1.0) * cs - sqA2a);
        const double a0 =           (A + 1.0) + sign * (A - 1.0) * cs + sqA2a;
        const double a1 = -2.0 *   ((A - 1.0) + sign * (A + 1.0) * cs) * sign;
        const double a2 =           (A + 1.0) + sign * (A - 1.0) * cs - sqA2a;

        return { b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0 };
    }

    inline BiquadCoeffs rbjNotch (double w0, double q)
    {
        const double cs = std::cos (w0), alpha = std::sin (w0) / (2.0 * q);
        const double a0 = 1.0 + alpha;
        return { 1.0 / a0, -2.0 * cs / a0, 1.0 / a0, -2.0 * cs / a0, (1.0 - alpha) / a0 };
    }

    inline BiquadCoeffs rbjCut2 (double w0, double q, bool highCut)
    {
        const double cs = std::cos (w0), alpha = std::sin (w0) / (2.0 * q);
        const double a0 = 1.0 + alpha;

        if (highCut) // low-pass response
            return { (1.0 - cs) * 0.5 / a0, (1.0 - cs) / a0, (1.0 - cs) * 0.5 / a0,
                     -2.0 * cs / a0, (1.0 - alpha) / a0 };

        return { (1.0 + cs) * 0.5 / a0, -(1.0 + cs) / a0, (1.0 + cs) * 0.5 / a0,
                 -2.0 * cs / a0, (1.0 - alpha) / a0 };
    }

    inline BiquadCoeffs cut1 (double freq, double sampleRate, bool highCut)
    {
        // First-order section via bilinear transform, as a degenerate biquad.
        const double w = std::tan (pi * freq / sampleRate);
        const double n = 1.0 / (1.0 + w);

        if (highCut)
            return { w * n, w * n, 0.0, (w - 1.0) * n, 0.0 };

        return { n, -n, 0.0, (w - 1.0) * n, 0.0 };
    }
} // namespace detail

// Computes the biquad cascade for a band. freq is clamped against Nyquist.
inline BandCoeffs computeBandCoeffs (FilterShape shape, double freq, double gainDb,
                                     double q, int slopeIndex, double sampleRate)
{
    using namespace detail;
    BandCoeffs out;

    freq = juce::jlimit (10.0, sampleRate * 0.49, freq);
    q    = juce::jlimit (0.05, 20.0, q);
    const double w0 = 2.0 * pi * freq / sampleRate;

    switch (shape)
    {
        case FilterShape::bell:
            out.stages[(size_t) out.numStages++] = rbjBell (w0, q, gainDb);
            break;

        case FilterShape::lowShelf:
            out.stages[(size_t) out.numStages++] = rbjShelf (w0, q, gainDb, false);
            break;

        case FilterShape::highShelf:
            out.stages[(size_t) out.numStages++] = rbjShelf (w0, q, gainDb, true);
            break;

        case FilterShape::notch:
            out.stages[(size_t) out.numStages++] = rbjNotch (w0, q);
            break;

        case FilterShape::lowCut:
        case FilterShape::highCut:
        {
            const bool highCut = (shape == FilterShape::highCut);
            const int order = slopeOrders[(size_t) juce::jlimit (
                0, (int) slopeOrders.size() - 1, slopeIndex)];
            const int numSecond = order / 2;

            // Butterworth pole placement; the band's Q scales the highest-Q
            // (last) section so it doubles as an adjustable resonance.
            for (int k = 0; k < numSecond; ++k)
            {
                const double psi = (order % 2 == 0) ? pi * (2.0 * k + 1.0) / (2.0 * order)
                                                    : pi * (k + 1.0) / order;
                double sectionQ = 1.0 / (2.0 * std::cos (psi));
                if (k == numSecond - 1)
                    sectionQ *= q / 0.7071067811865476;
                out.stages[(size_t) out.numStages++] = rbjCut2 (w0, sectionQ, highCut);
            }

            if (order % 2 != 0)
                out.stages[(size_t) out.numStages++] = cut1 (freq, sampleRate, highCut);
            break;
        }
    }

    return out;
}

// Magnitude of one section at a given frequency, in dB.
inline double stageMagnitudeDb (const BiquadCoeffs& st, double freq, double sampleRate)
{
    const double w = 2.0 * detail::pi * freq / sampleRate;
    const std::complex<double> z1 = std::polar (1.0, -w);
    const std::complex<double> z2 = z1 * z1;

    const auto num = st.b0 + st.b1 * z1 + st.b2 * z2;
    const auto den = 1.0 + st.a1 * z1 + st.a2 * z2;
    return 20.0 * std::log10 (juce::jmax (1.0e-30, std::abs (num))
                              / juce::jmax (1.0e-30, std::abs (den)));
}

// Magnitude of a whole band cascade at a given frequency, in dB.
inline double bandMagnitudeDb (const BandCoeffs& c, double freq, double sampleRate)
{
    double db = 0.0;
    for (int s = 0; s < c.numStages; ++s)
        db += stageMagnitudeDb (c.stages[(size_t) s], freq, sampleRate);
    return db;
}

// Runtime state for a cascade: transposed direct form II, double state.
//
// Double rather than float deliberately. The Decay Rate EQ's sections sit
// inside the feedback loop and are close to unity gain, where a float TDF-II
// at 20-100 Hz is poorly conditioned — and an error there does not decay away,
// it recirculates.
template <int MaxStages>
struct BiquadCascade
{
    std::array<BiquadCoeffs, (size_t) MaxStages> stages {};
    std::array<double, (size_t) MaxStages> z1 {}, z2 {};
    int numStages = 0;

    void clear() noexcept
    {
        z1.fill (0.0);
        z2.fill (0.0);
    }

    void setStages (const BiquadCoeffs* src, int count) noexcept
    {
        const int n = juce::jlimit (0, MaxStages, count);
        for (int s = 0; s < n; ++s)
            stages[(size_t) s] = src[s];

        // Sections coming into use start from rest; leaving stale state in one
        // that was silent for a minute and is now in the signal path is how a
        // filter bank clicks.
        for (int s = numStages; s < n; ++s)
            z1[(size_t) s] = z2[(size_t) s] = 0.0;

        numStages = n;
    }

    double process (double x) noexcept
    {
        for (int s = 0; s < numStages; ++s)
        {
            const auto& c = stages[(size_t) s];
            const double y = c.b0 * x + z1[(size_t) s];
            z1[(size_t) s] = c.b1 * x - c.a1 * y + z2[(size_t) s];
            z2[(size_t) s] = c.b2 * x - c.a2 * y;
            x = y;
        }
        return x;
    }

    bool isFinite() const noexcept
    {
        for (int s = 0; s < numStages; ++s)
            if (! std::isfinite (z1[(size_t) s]) || ! std::isfinite (z2[(size_t) s]))
                return false;
        return true;
    }
};

} // namespace hrzn
