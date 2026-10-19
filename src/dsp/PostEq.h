#pragma once

#include "FilterCoeffs.h"
#include "SpeakerLayout.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <array>
#include <cmath>

namespace hrzn
{

// The six-band Post EQ (feature 21), on the wet output only.
//
// Its counterpart, `DecayEq.h`, changes how long each part of the spectrum
// rings. This one changes what the tail *sounds* like without touching the
// decay at all: a high cut here darkens the reverb, a high cut there makes it
// die away sooner. Both are wanted, which is why the reference product has
// both and why they are two sections rather than one.
//
// The filter maths is the family EQ's (see `FilterCoeffs.h`), including the
// full shape set — bells, shelves, notches and Butterworth cuts up to 48
// dB/oct — and every band carries a channel mode, so a band can work on the
// stereo pair, one side, or mid/side (feature 15). Widening only the tail's
// side content, or rolling off only its mid, is a large part of what makes a
// reverb sit behind a mix rather than in front of it.
//
// Two departures from Equinox's `EqBand`:
//
//   - it runs per sample rather than per block, because Horizon's engine is
//     one sample loop from input to mix and splitting it in two to run a
//     block-based EQ would mean buffering the wet path for no benefit
//   - enabling or disabling a band crossfades over ~20 ms rather than
//     switching. A band's `on` is an automatable parameter like any other, and
//     a low cut appearing instantaneously in a reverb tail is a click. Below
//     the crossfade the band's state is cleared and it costs nothing.
class PostEq
{
public:
    static constexpr int numBands = 6;

    struct Band
    {
        bool on = false;
        FilterShape shape = FilterShape::bell;
        ChannelMode channel = ChannelMode::stereo;
        float freq = 1000.0f;
        float gainDb = 0.0f;
        float q = 1.0f;
        int slopeIndex = 1;   // 12 dB/oct
    };

    // The speakers are the tank's, not the host's: no LFE (it never reaches the
    // wet path) and a mono session arrives here as the two-channel pair it
    // internally is. `mirror` indices are tank slots by the time they get here.
    void prepare (double newSampleRate, const Speaker* speakers, int count)
    {
        sampleRate = newSampleRate;
        numChannels = juce::jlimit (1, maxSpeakers, count);

        layout.fill ({});
        mirror.fill (-1);

        for (int c = 0; c < numChannels; ++c)
        {
            layout[(size_t) c] = speakers[(size_t) c];
            mirror[(size_t) c] = speakers[(size_t) c].mirror;
        }

        for (int b = 0; b < numBands; ++b)
        {
            smoothed[(size_t) b] = target[(size_t) b];
            smoothing[(size_t) b].snap (target[(size_t) b]);
            wet[(size_t) b] = target[(size_t) b].on ? 1.0f : 0.0f;
            dirty[(size_t) b] = true;
            buildMask (b);
        }

        reset();
        update();
    }

    int getNumChannels() const noexcept { return numChannels; }

    void reset() noexcept
    {
        for (auto& band : cascades)
            for (auto& ch : band)
                ch.clear();
    }

    void setBands (const std::array<Band, numBands>& bands) noexcept { target = bands; }

    // See DecayEq::snapToTargets — a wholesale state change arrives, it does
    // not glide.
    void snapToTargets() noexcept
    {
        for (int b = 0; b < numBands; ++b)
        {
            smoothed[(size_t) b] = target[(size_t) b];
            smoothing[(size_t) b].snap (target[(size_t) b]);
            wet[(size_t) b] = target[(size_t) b].on ? 1.0f : 0.0f;
            dirty[(size_t) b] = true;
            buildMask (b);
        }
        reset();
        update();
    }

    bool isActive() const noexcept { return active; }

    // Control rate: glide the continuous parameters, redesign what moved.
    void update() noexcept
    {
        constexpr float rate = 0.05f;      // ~60 ms at a 1.5 kHz control rate
        constexpr float wetRate = 0.045f;  // ~20 ms crossfade on enable/disable

        bool anyActive = false;

        for (int b = 0; b < numBands; ++b)
        {
            const auto& t = target[(size_t) b];
            auto& s = smoothed[(size_t) b];
            const auto bi = (size_t) b;

            // A shape, slope or channel change is a *different filter*, not a
            // moved one, and the state left in the cascade means nothing in the
            // new configuration — a 48 dB/oct low cut's four sections holding
            // the history of a bell is not an approximation of anything. It
            // also changes the number of sections, so there is nothing to glide
            // between even in principle.
            //
            // So it takes the same route the on/off switch does: the band fades
            // out, the swap happens while it is out of circuit and its state has
            // been cleared, and it fades back in. Forty milliseconds rather than
            // twenty, and no splice.
            //
            // v0.8 found this. It was invisible to every earlier test because a
            // band's shape only matters while the band is *on*, and until the
            // automation suite threw the shapes and the on/off switches in the
            // same block, nothing had ever changed one of a running band.
            const bool swapPending = s.shape != t.shape
                                     || s.channel != t.channel
                                     || s.slopeIndex != t.slopeIndex;

            const float wetGoal = (t.on && ! swapPending) ? 1.0f : 0.0f;
            if (! juce::approximatelyEqual (wet[bi], wetGoal))
            {
                wet[bi] += (wetGoal - wet[bi]) * wetRate;
                if (std::abs (wetGoal - wet[bi]) < 1.0e-4f)
                    wet[bi] = wetGoal;
            }

            if (wet[bi] <= 0.0f)
            {
                // Fully out of circuit: hold the design at the requested shape
                // so that switching back on starts from the right coefficients,
                // but let the state go so it cannot ring stale content back in.
                for (auto& ch : cascades[bi])
                    ch.clear();

                if (swapPending)
                {
                    const bool channelChanged = s.channel != t.channel;

                    s.shape = t.shape;
                    s.channel = t.channel;
                    s.slopeIndex = t.slopeIndex;
                    dirty[bi] = true;

                    if (channelChanged)
                        buildMask (b);
                }
            }
            s.on = t.on;

            // Frequency and Q glide geometrically; all three carry a slew cap
            // as well as a time constant.
            //
            // The cap is what keeps this section honest. An exponential glide
            // covers most of its distance in its first few ticks, and these
            // biquads sit directly on the output carrying full-level signal —
            // unlike the Decay Rate EQ's, which are sixteen parallel copies
            // inside a tank and average each other out. Without the cap,
            // automating gain, frequency and Q together stepped the output
            // three times harder than every other control in the plugin put
            // together, which is exactly what the interpolation regression is
            // there to catch. The limits work out at 120 dB and 13 octaves per
            // second: faster than a hand, slow enough that a coefficient set
            // never jumps. Gain is the one that mattered — Q gets a much
            // looser cap so that dragging it still feels immediate.
            // The log domain is where the smoothed value lives, rather than
            // somewhere it visits — taking the log of the stored frequency and
            // exponentiating it back is not the identity, and a settled band
            // that drifts by an ulp every tick is a band that redesigns its
            // coefficients forever. See the same note in DecayEq.h, where it
            // was expensive enough to show up on the CPU audit.
            auto& sm = smoothing[bi];
            sm.retarget (t);

            if (approach (sm.logFreq, sm.targetLogFreq, rate, 1.0e-5f, 0.006f))
            {
                s.freq = std::exp (sm.logFreq);
                dirty[bi] = true;
            }

            if (approach (s.gainDb, t.gainDb, rate, 1.0e-4f, 0.08f))
                dirty[bi] = true;

            if (approach (sm.logQ, sm.targetLogQ, rate, 1.0e-5f, 0.02f))
            {
                s.q = std::exp (sm.logQ);
                dirty[bi] = true;
            }

            if (dirty[bi])
            {
                const auto c = computeBandCoeffs (s.shape, s.freq, s.gainDb, s.q,
                                                  s.slopeIndex, sampleRate);
                coeffs[bi] = c;
                for (auto& ch : cascades[bi])
                    ch.setStages (c.stages.data(), c.numStages);
                dirty[bi] = false;
            }

            if (wet[bi] > 0.0f && coeffs[bi].numStages > 0)
                anyActive = true;
        }

        active = anyActive;
    }

    // In place across the tank's wet channels.
    //
    // The direct modes are a mask and a filter per channel; mid and side fold
    // about the layout's mirrored pairs. On a stereo pair every branch here
    // reduces to exactly the two-channel arithmetic v0.3 shipped — the mid case
    // takes the left channel's cascade and the side case the right's, which is
    // what makes the routing null tests measure the same thing they always did.
    void processSample (float* x, int count) noexcept
    {
        const int n = juce::jmin (count, numChannels);

        for (int b = 0; b < numBands; ++b)
        {
            const auto bi = (size_t) b;
            const float w = wet[bi];
            if (w <= 0.0f || coeffs[bi].numStages == 0)
                continue;

            const auto mode = smoothed[bi].channel;

            if (mode == ChannelMode::mid || mode == ChannelMode::side)
            {
                for (int c = 0; c < n; ++c)
                {
                    const auto ci = (size_t) c;
                    const int m = mirror[ci];

                    if (m < 0 || m >= n)
                    {
                        // A speaker with no mirror is on the median plane, so it
                        // is all mid and no side. Filtering it in `mid` and
                        // leaving it alone in `side` is the only reading of the
                        // control that stays true when the layout has a centre.
                        if (mode == ChannelMode::mid)
                            x[ci] = blend (x[ci], cascades[bi][ci], w);
                        continue;
                    }

                    if (m <= c)
                        continue;   // the pair is handled from its lower slot

                    const auto mi = (size_t) m;
                    const float mid = (x[ci] + x[mi]) * 0.5f;
                    const float side = (x[ci] - x[mi]) * 0.5f;

                    if (mode == ChannelMode::mid)
                    {
                        const float f = blend (mid, cascades[bi][ci], w);
                        x[ci] = f + side;
                        x[mi] = f - side;
                    }
                    else
                    {
                        const float f = blend (side, cascades[bi][mi], w);
                        x[ci] = mid + f;
                        x[mi] = mid - f;
                    }
                }

                continue;
            }

            for (int c = 0; c < n; ++c)
                if (mask[bi][(size_t) c])
                    x[(size_t) c] = blend (x[(size_t) c], cascades[bi][(size_t) c], w);
        }
    }

    // Analytic magnitude of the bands acting on one output channel, in dB.
    // Mid/side bands are deliberately excluded: their effect on L and R is not
    // a magnitude response at all, and pretending otherwise would make the
    // curve the display draws a lie. They are verified by routing null tests
    // instead.
    double magnitudeDbAt (double hz, int channel) const noexcept
    {
        double db = 0.0;
        for (int b = 0; b < numBands; ++b)
        {
            const auto bi = (size_t) b;
            if (wet[bi] <= 0.0f)
                continue;

            const auto mode = smoothed[bi].channel;
            if (mode == ChannelMode::mid || mode == ChannelMode::side)
                continue;

            if (channel < 0 || channel >= numChannels || ! mask[bi][(size_t) channel])
                continue;

            // Mid-crossfade, a band's contribution is partial. Amplitude
            // interpolation, matching what processSample actually does at DC.
            const double bandDb = bandMagnitudeDb (coeffs[bi], hz, sampleRate);
            const double mag = std::pow (10.0, bandDb / 20.0);
            db += 20.0 * std::log10 (juce::jmax (1.0e-9,
                                                 1.0 + ((double) wet[bi]) * (mag - 1.0)));
        }
        return db;
    }

    bool allFinite() const noexcept
    {
        for (const auto& band : cascades)
            for (const auto& ch : band)
                if (! ch.isFinite())
                    return false;
        return true;
    }

private:
    // See DecayEq::BandSmoothing — the glide lives in the log domain so that a
    // settled band is exactly settled.
    struct BandSmoothing
    {
        float logFreq = 0.0f, logQ = 0.0f;
        float targetLogFreq = 0.0f, targetLogQ = 0.0f;
        float seenFreq = -1.0f, seenQ = -1.0f;

        void retarget (const Band& t) noexcept
        {
            if (! juce::approximatelyEqual (seenFreq, t.freq))
            {
                seenFreq = t.freq;
                targetLogFreq = std::log (juce::jmax (1.0f, t.freq));
            }
            if (! juce::approximatelyEqual (seenQ, t.q))
            {
                seenQ = t.q;
                targetLogQ = std::log (juce::jmax (0.01f, t.q));
            }
        }

        void snap (const Band& t) noexcept
        {
            retarget (t);
            logFreq = targetLogFreq;
            logQ = targetLogQ;
        }
    };

    // See the note on DecayEq::approach: the underflow snap is what stops a
    // glide stalling one ULP short of its goal and reporting a change forever.
    static bool approach (float& v, float goal, float rate, float epsilon,
                          float maxStep = 1.0e9f) noexcept
    {
        const float delta = goal - v;

        if (std::abs (delta) <= epsilon)
        {
            if (juce::approximatelyEqual (v, goal))
                return false;
            v = goal;
            return true;
        }

        const float next = v + juce::jlimit (-maxStep, maxStep, delta * rate);
        v = juce::approximatelyEqual (next, v) ? goal : next;
        return true;
    }

    static float blend (float x, BiquadCascade<maxStagesPerBand>& c, float w) noexcept
    {
        const float y = (float) c.process ((double) x);
        return x + (y - x) * w;
    }

    // Which speakers a band's direct modes act on. Recomputed only when the mode
    // or the layout changes, because it is a question about geometry and the
    // geometry does not move while the plugin is running.
    //
    // `left` and `right` are sides of the room rather than channel indices, so
    // in 7.1.4 a Left band takes the left surround and the two left height
    // speakers with it — which is what somebody reaching for it wants, and is
    // not something a channel-index reading could have given them.
    void buildMask (int band) noexcept
    {
        const auto bi = (size_t) band;
        const auto mode = smoothed[bi].channel;

        for (int c = 0; c < maxSpeakers; ++c)
        {
            const auto ci = (size_t) c;

            if (c >= numChannels)
            {
                mask[bi][ci] = false;
                continue;
            }

            const auto& sp = layout[ci];
            constexpr float centred = 1.0e-4f;

            switch (mode)
            {
                case ChannelMode::stereo:   mask[bi][ci] = true; break;
                case ChannelMode::left:     mask[bi][ci] = sp.azimuth < -centred; break;
                case ChannelMode::right:    mask[bi][ci] = sp.azimuth > centred; break;
                case ChannelMode::front:    mask[bi][ci] = sp.group == SpeakerGroup::front; break;
                case ChannelMode::centre:   mask[bi][ci] = sp.group == SpeakerGroup::centre; break;
                case ChannelMode::surround: mask[bi][ci] = sp.group == SpeakerGroup::surround; break;
                case ChannelMode::height:   mask[bi][ci] = sp.group == SpeakerGroup::height; break;
                case ChannelMode::mid:
                case ChannelMode::side:     mask[bi][ci] = true; break;
            }
        }
    }

    double sampleRate = 48000.0;
    int numChannels = 2;

    std::array<Speaker, maxSpeakers> layout {};
    std::array<int, maxSpeakers> mirror {};

    std::array<Band, numBands> target {}, smoothed {};
    std::array<BandSmoothing, numBands> smoothing {};
    std::array<BandCoeffs, numBands> coeffs {};
    std::array<std::array<BiquadCascade<maxStagesPerBand>, maxSpeakers>, numBands> cascades {};
    std::array<std::array<bool, maxSpeakers>, numBands> mask {};
    std::array<float, numBands> wet {};
    std::array<bool, numBands> dirty {};
    bool active = false;
};

} // namespace hrzn
