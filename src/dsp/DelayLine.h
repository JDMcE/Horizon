#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <vector>
#include <cmath>

namespace hrzn
{

// Power-of-two circular delay line with fractional, linearly interpolated
// reads.
//
// Read convention: read(0) returns the sample most recently pushed, read(d)
// the sample d samples before it. Push first, then read — the FDN and the
// allpass stages both rely on that ordering.
//
// Interpolation note (matters for Freeze): linear interpolation is a mild
// lowpass whose loss compounds once per trip around a feedback loop, so an
// infinite-feedback tank read at a fractional delay would still decay. From
// v0.2 the FDN base delays are continuous floats — that is what makes the
// Space morph stepless — so the engine cannot simply keep them integral.
// Instead Freeze ramps the read position to the nearest integer *and* the
// modulation offset to zero, together, over its 50 ms slew: at most half a
// sample of glide, inaudible, after which reads are exact and the tank is
// genuinely lossless again.
class DelayLine
{
public:
    void prepare (int maxDelaySamples)
    {
        int size = 4;
        while (size < maxDelaySamples + 4)
            size <<= 1;

        buffer.assign ((size_t) size, 0.0f);
        mask = size - 1;
        writePos = 0;
    }

    void clear() noexcept
    {
        std::fill (buffer.begin(), buffer.end(), 0.0f);
        writePos = 0;
    }

    void push (float x) noexcept
    {
        buffer[(size_t) writePos] = x;
        writePos = (writePos + 1) & mask;
    }

    float read (float delaySamples) const noexcept
    {
        const float d = juce::jlimit (0.0f, (float) (mask - 1), delaySamples);
        const int i0 = (int) d;
        const float frac = d - (float) i0;

        const int last = (writePos - 1) & mask;
        const int r0 = (last - i0) & mask;
        const int r1 = (r0 - 1) & mask;

        const float a = buffer[(size_t) r0];
        return a + frac * (buffer[(size_t) r1] - a);
    }

    int capacity() const noexcept { return mask; }

private:
    std::vector<float> buffer;
    int mask = 0, writePos = 0;
};

// Schroeder allpass, used in series for input diffusion:
//   v[n] = x[n] + g*v[n-D]
//   y[n] = v[n-D] - g*v[n]
// Unity magnitude response, so stacking them raises echo density without
// colouring the signal.
//
// Delay and gain arrive as arguments rather than as stored state: from v0.2
// the diffuser lengths glide with the Space morph and their gains ride
// Thickness and Character, and all of that has to move per sample.
class Allpass
{
public:
    void prepare (int maxDelaySamples) { line.prepare (maxDelaySamples + 2); }
    void clear() noexcept { line.clear(); }

    float process (float x, float delaySamples, float gain) noexcept
    {
        const float d = juce::jmax (1.0f, delaySamples);
        const float g = juce::jlimit (-0.85f, 0.85f, gain);

        const float vD = line.read (d - 1.0f);
        const float v = x + g * vD;
        line.push (v);
        return vD - g * v;
    }

private:
    DelayLine line;
};

// Two one-pole *shelves* in series, sitting in each FDN feedback path: a high
// shelf for air absorption and a low shelf for the room's low-end character.
//
// ---------------------------------------------------------------------------
// Why shelves, and not the lowpass this used to be
// ---------------------------------------------------------------------------
// Through v0.4 this was a one-pole lowpass plus a one-pole highpass, and both
// of them attenuate *everywhere*. Inside a feedback loop that is not a tonal
// choice, it is a decay-time error: a one-pole at 9.5 kHz still costs about
// 0.05 dB per pass at 1 kHz, and a small room traverses its lines hundreds of
// times a second, so the mid band decayed measurably shorter than the feedback
// gain was set for. v0.1 found the same thing on the highpass and dealt with it
// by pushing the cutoffs subsonic, which bought accuracy by giving up the
// control.
//
// It became untenable at v0.4.1, when Decay Time stopped being a percentage of
// the room and became an absolute number of seconds. "1.8 s" has to mean 1.8 s
// on an ambience and on a cathedral alike, and it could not: the smallest rooms
// delivered a quarter less than they were asked for, entirely because of this
// filter.
//
// A shelf is unity in its passband. Both shelves are flat through the mid band,
// so the feedback gain alone decides the mid-band decay — exactly, on every
// room — and the filter's whole job becomes what it should always have been:
// saying how the *rest* of the spectrum decays relative to that.
//
// ---------------------------------------------------------------------------
// Gains are ratios, not fixed losses
// ---------------------------------------------------------------------------
// The shelf gains are derived per control tick from the current feedback
// attenuation (see ReverbEngine::updateControlRate) so that each model's
// `hfDecayRatio` and `lfDecayRatio` mean "this band rings for this fraction of
// the mid band's time" at *any* decay setting. A fixed per-pass loss would
// instead make long settings progressively darker than short ones, which is not
// what changing a room's size does to it.
//
// The coefficients and gains are passed in per sample: Brightness moves the
// high shelf a long way, and stepping a filter coefficient once per block steps
// the output by (delta * (x - state)), which is exactly the kind of zipper the
// Smart Parameter Interpolation contract forbids.
struct DampingFilter
{
    float hfState = 0.0f, lfState = 0.0f;

    void clear() noexcept { hfState = lfState = 0.0f; }

    // coeff for a one-pole at cutoff Hz, sample rate fs.
    static float coeffFor (float cutoffHz, double sampleRate) noexcept
    {
        const float x = (float) std::exp (-juce::MathConstants<double>::twoPi
                                          * (double) cutoffHz / sampleRate);
        return juce::jlimit (0.0f, 1.0f, 1.0f - x);
    }

    // Magnitude of the combined response at a frequency whose cosine is `cosW`.
    //
    //   high shelf: gh + (1-gh)*La(z),      La = a/(1 - (1-a)z^-1)
    //   low shelf:  1 - (1-gl)*Lb(z),       Lb = b/(1 - (1-b)z^-1)
    //
    // Both collapse to the v0.4 filter at the extremes — gh = 0 is the plain
    // lowpass, gl = 0 the plain highpass — which is what made the change
    // checkable against the old measurements one step at a time.
    //
    // The Decay Rate EQ designs its multipliers against this, so it has to be
    // the exact magnitude rather than an approximation of it.
    static float magnitudeAt (float hfCoeff, float hfGain,
                              float lfCoeff, float lfGain, float cosW) noexcept
    {
        const float a = juce::jlimit (1.0e-6f, 1.0f, hfCoeff);
        const float b = juce::jlimit (0.0f, 1.0f, lfCoeff);
        const float gh = juce::jlimit (0.0f, 1.0f, hfGain);
        const float gl = juce::jlimit (0.0f, 1.0f, lfGain);

        const float pa = 1.0f - a, pb = 1.0f - b;

        const float ch = gh + (1.0f - gh) * a;
        const float ghpa = gh * pa;
        const float hs = std::sqrt (
            juce::jmax (0.0f, ch * ch + ghpa * ghpa - 2.0f * ch * ghpa * cosW)
            / juce::jmax (1.0e-12f, 1.0f + pa * pa - 2.0f * pa * cosW));

        const float cl = 1.0f - (1.0f - gl) * b;
        const float ls = std::sqrt (
            juce::jmax (0.0f, cl * cl + pb * pb - 2.0f * cl * pb * cosW)
            / juce::jmax (1.0e-12f, 1.0f + pb * pb - 2.0f * pb * cosW));

        return juce::jlimit (0.0f, 1.0f, hs * ls);
    }

    float process (float x, float hfCoeff, float hfGain,
                   float lfCoeff, float lfGain) noexcept
    {
        hfState += hfCoeff * (x - hfState);
        const float shelved = hfState + hfGain * (x - hfState);

        lfState += lfCoeff * (shelved - lfState);
        return (shelved - lfState) + lfGain * lfState;
    }
};

} // namespace hrzn
