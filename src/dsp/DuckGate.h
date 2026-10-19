#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <array>
#include <cmath>

namespace hrzn
{

// Ducking and the Auto Gate (feature 7), v0.5.
//
// Both are the same shape of thing — a level taken off a key signal, turned
// into a gain, applied to the wet path — so they share one detector and one
// file. What they are *for* is opposite, which is why they are two controls
// and not one: ducking pulls the reverb down while the source is playing and
// lets it bloom in the gaps, and the gate cuts the tail off once the source
// stops. Used together they are the gated-reverb sound; used apart they are
// the two commonest reasons a reverb won't sit in a mix.
//
// This produces a *gain*, not audio. The processor computes a block of wet
// gains and the engine multiplies them into the wet path just before the tap
// and the mix (see ReverbEngine::process), which means three things worth
// having: the analyzer trace shows the ducking rather than hiding it, the dry
// path is still untouched bit for bit at Mix 0, and nothing about the reverb's
// own geometry has to know that either control exists.
//
// ---------------------------------------------------------------------------
// Why the timing constants sit on the gain and not on the level
// ---------------------------------------------------------------------------
// The obvious build is one envelope follower whose attack and release are the
// user's controls, with a static curve on the end. It is also the one that
// makes "attack" mean something different at every threshold and amount: the
// gain is a nonlinear function of the level, so smoothing the level and
// smoothing the gain are not the same operation, and the time the user dialled
// in is not the time the reverb takes to get out of the way.
//
// So the detector is fast and fixed — it exists only to answer "how loud is
// the key right now" — and each stage smooths its own *gain*, in dB. Attack is
// then exactly the time the duck takes to reach 63% of its depth, whatever the
// depth is, and release is exactly the time the gate takes to shut. Those are
// the figures on the panel and they are the figures the harness measures.
class DuckGate
{
public:
    struct Params
    {
        bool  duckOn = false;
        float duckThresholdDb = -30.0f;  // key level at which ducking begins
        float duckAmountDb    = 12.0f;   // depth at full drive
        float duckAttackMs    = 10.0f;
        float duckReleaseMs   = 250.0f;

        bool  gateOn = false;
        float gateThresholdDb = -40.0f;
        float gateHoldMs      = 120.0f;
        float gateReleaseMs   = 200.0f;
    };

    // Full ducking depth is reached this far above the threshold. A hard knee
    // would make the control a switch — a reverb that jumps 12 dB the moment a
    // vocal crosses a line is worse than no ducking at all — and a range wide
    // enough to follow a performance means the depth tracks how hard the
    // source is playing, which is the behaviour people actually want from it.
    static constexpr float duckRangeDb = 12.0f;

    // The gate's fade spans this much level, and its release control is the
    // time to cross it. Defining release as "time to shut" rather than as a
    // time constant is the only version of it that can be checked, and the
    // only one where two plugins set to 200 ms do the same thing.
    static constexpr float gateRangeDb = 60.0f;

    // Opening is not a control: a gate that takes a settable time to open is a
    // gate that misses the transient it was opened by. Fast enough not to be
    // heard as a fade, slow enough not to click.
    static constexpr float gateAttackMs = 2.0f;

    // The key has to fall this far back below the threshold before the gate
    // starts counting its hold, so a source sitting exactly on the line does
    // not chatter the tail on and off.
    static constexpr float gateHysteresisDb = 3.0f;

    void prepare (double newSampleRate)
    {
        sampleRate = newSampleRate;

        // The detector is deliberately not a control. 0.1 ms up so a transient
        // is not missed; 5 ms down, which is long enough that a mid-band tone
        // reads as a level rather than as its own waveform and short enough
        // that the gate's hold starts when the source stops rather than a
        // fifth of a second later.
        detectorAttack  = onePole (0.1f);
        detectorRelease = onePole (5.0f);

        setParams (params);
        reset();
    }

    void reset() noexcept
    {
        envelope = 0.0f;
        duckDb = 0.0f;
        gateDb = 0.0f;
        holdCountdown = 0;
        gateArmed = true;
    }

    void setParams (const Params& p) noexcept
    {
        params = p;

        duckAttack  = onePole (p.duckAttackMs);
        duckRelease = onePole (p.duckReleaseMs);

        // Both gate ramps are linear in dB across the fade range, so the step
        // per sample is the range divided by the time. A linear-dB fade is what
        // a fader sounds like, and it is the only shape that can reach zero.
        gateAttackStep  = gateRangeDb / juce::jmax (1.0f, gateAttackMs * 0.001f * (float) sampleRate);
        gateReleaseStep = gateRangeDb / juce::jmax (1.0f, p.gateReleaseMs * 0.001f * (float) sampleRate);
        holdSamples = (int) (p.gateHoldMs * 0.001f * (float) sampleRate);
    }

    // True when neither stage is doing anything and neither is on its way back
    // to doing nothing — the processor skips the whole wet-gain path on this,
    // so a session with both controls off pays for none of it.
    bool idle() const noexcept
    {
        return ! params.duckOn && ! params.gateOn
            && duckDb <= 1.0e-4f && gateDb >= -1.0e-4f;
    }

    // Fills `wetGain` with the per-sample linear gain for the wet path.
    // `key` is the mono trigger — the dry input or the external sidechain,
    // chosen upstream, because which one is keying it is a routing question
    // and not one this needs an opinion about.
    void process (const float* key, float* wetGain, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
        {
            const float rectified = std::abs (key[i]);
            envelope += (rectified > envelope ? detectorAttack : detectorRelease)
                        * (rectified - envelope);

            // -140 dB floor: a silent key must produce a finite level rather
            // than an infinity that poisons every ramp downstream of it.
            const float levelDb = juce::Decibels::gainToDecibels (envelope, -140.0f);

            // Ducking ------------------------------------------------------
            //
            // Switching the control off targets zero depth and lets the
            // release carry it there, so turning ducking off sounds like the
            // source stopping rather than like an edit.
            const float over = levelDb - params.duckThresholdDb;
            const float drive = juce::jlimit (0.0f, 1.0f, over / duckRangeDb);
            const float duckTarget = params.duckOn ? params.duckAmountDb * drive : 0.0f;

            duckDb += (duckTarget > duckDb ? duckAttack : duckRelease) * (duckTarget - duckDb);

            // Auto Gate ----------------------------------------------------
            //
            // Hysteresis is one-sided on purpose: the gate opens the moment the
            // key crosses the threshold, and only re-arms once it has fallen
            // clear of it. Making the opening edge wait for the same margin
            // would cost the transient the gate exists to pass.
            if (levelDb > params.gateThresholdDb)
                gateArmed = false;
            else if (levelDb < params.gateThresholdDb - gateHysteresisDb)
                gateArmed = true;

            const bool open = ! params.gateOn || ! gateArmed;

            if (open)
            {
                holdCountdown = holdSamples;
                gateDb = juce::jmin (0.0f, gateDb + gateAttackStep);
            }
            else if (holdCountdown > 0)
            {
                --holdCountdown;
            }
            else
            {
                gateDb = juce::jmax (-gateRangeDb, gateDb - gateReleaseStep);
            }

            // Shut means shut. Below the fade range the gate is zero rather
            // than 60 dB down, so "keeping long tails out of quiet passages"
            // means what it says.
            const float gateGain = gateDb <= -gateRangeDb
                                 ? 0.0f
                                 : juce::Decibels::decibelsToGain (gateDb);

            wetGain[i] = juce::Decibels::decibelsToGain (-duckDb) * gateGain;
        }
    }

    // Readouts for the display's trigger trace (v0.5). All three are what the
    // last processed sample left behind, which at frame rate is the right
    // answer and costs nothing to keep.
    float keyLevelDb() const noexcept
    {
        return juce::Decibels::gainToDecibels (envelope, -140.0f);
    }

    float duckDepthDb() const noexcept { return duckDb; }

    float gateGain() const noexcept
    {
        return gateDb <= -gateRangeDb ? 0.0f : juce::Decibels::decibelsToGain (gateDb);
    }

private:
    // Coefficient for a one-pole reaching 63.2% of a step in `ms`.
    float onePole (float ms) const noexcept
    {
        const float samples = juce::jmax (1.0f, ms * 0.001f * (float) sampleRate);
        return 1.0f - std::exp (-1.0f / samples);
    }

    double sampleRate = 48000.0;
    Params params;

    float detectorAttack = 0.0f, detectorRelease = 0.0f;
    float duckAttack = 0.0f, duckRelease = 0.0f;
    float gateAttackStep = 0.0f, gateReleaseStep = 0.0f;
    int holdSamples = 0;

    float envelope = 0.0f;     // rectified key level, linear
    float duckDb = 0.0f;       // current ducking depth, dB (positive = pulled down)
    float gateDb = 0.0f;       // current gate fade, dB (0 = open)
    int holdCountdown = 0;
    bool gateArmed = true;     // the key has fallen clear of the threshold
};

} // namespace hrzn
