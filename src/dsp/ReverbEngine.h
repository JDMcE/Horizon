#pragma once

#include "DecayEq.h"
#include "DelayLine.h"
#include "PostEq.h"
#include "RoomModels.h"
#include "SpeakerLayout.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <array>
#include <cmath>

namespace hrzn
{

namespace detail
{
    // Walsh function: row k of a Hadamard matrix, entry i.
    constexpr float walsh (int i, int k) noexcept
    {
        int bits = i & k, parity = 0;
        while (bits != 0) { parity ^= (bits & 1); bits >>= 1; }
        return parity != 0 ? -1.0f : 1.0f;
    }

    template <size_t N>
    constexpr std::array<float, N> makeSigns (int k) noexcept
    {
        std::array<float, N> v {};
        for (size_t i = 0; i < N; ++i)
            v[i] = walsh ((int) i, k);
        return v;
    }

}

// Horizon's reverb engine (v0.2).
//
// Signal flow, per sample:
//
//   in ──┬──────────────────────────────────────────────── dry ──┐
//        │                                                       │
//        └─► predelay ─┬─► ER tap bank ───────────────► erL/erR ─┤
//                      │                                         ├─► width ─► tilt
//                      └─► input diffusion ─► FDN tank ─► late ──┘        ─► post EQ
//                                                                         ─► mix
//
// The late field is a 16-line feedback delay network with a Householder
// mixing matrix (H = I - (2/N)J: orthogonal, so the matrix is lossless and
// every bit of decay comes from the per-line feedback gains). Each line has
// its own damping filter and a slow modulation offset that breaks up
// standing modes.
//
// ---------------------------------------------------------------------------
// v0.2: the stepless Space morph
// ---------------------------------------------------------------------------
// v0.1 snapped to the nearest of fourteen room models and restarted the tank
// on every change. v0.2 keeps a continuous position through the table and
// *interpolates the geometry* instead:
//
//   - `pos` is a float 0..13 that chases the Space parameter under a rate
//     limit (see maxPointerRate). Every geometric quantity is then a pure
//     function of `pos`, so nothing can step: the sixteen FDN delay lengths
//     interpolate between the two neighbouring models' own length sets and are
//     scaled by `sizeScale(pos)` (through v0.4.2 there was one shared set, so
//     this was just `baseDelaySamples[l] * sizeScale(pos)` — see LateShape's
//     `lineSpread` for why each model now brings its own), and are read at
//     fractional positions; the early reflection bank crossfades tap by tap in both time
//     and gain between the two neighbouring models, and the diffuser lengths,
//     damping cutoffs, modulation and levels all interpolate alongside.
//   - The rate limit is a *Doppler budget*, not a fixed time: no FDN read
//     pointer may travel more than `maxPointerRate` of a sample per sample.
//     The table's small end can therefore sweep far faster than its large end
//     without either one warbling, because the limit tracks the geometry
//     rather than the knob.
//   - Discrete relocation still exists, but only for genuine discontinuities:
//     a preset load, an A/B switch or an undo re-applies a whole state at
//     once, and morphing three seconds across the table would be wrong there.
//     `relocate()` fades out, snaps, flushes and fades back in — the v0.1
//     behaviour, now reserved for the case it is actually right for.
//
// ---------------------------------------------------------------------------
// v0.3: the two six-band sections
// ---------------------------------------------------------------------------
// Both live here, and they are deliberately not the same kind of thing:
//
//   - the **Decay Rate EQ** (`DecayEq.h`) sits inside the feedback path and
//     changes how long each part of the spectrum rings. It is designed as a
//     per-line attenuator plus a per-line feedback gain, never as a boost,
//     which makes the loop unconditionally stable however the bands are set —
//     see the header there for why that is the only design worth having in a
//     feedback path a user can drag handles around in.
//   - the **Post EQ** (`PostEq.h`) sits on the wet output after width and
//     tilt, and changes the tone without touching the decay at all.
//
// Both are inert at neutral: no filter sections, no per-sample cost, and a
// feedback gain identical to the one v0.2 computed directly from the decay
// time. Turning the sections on is the only thing that costs anything.
//
// Anti-coloration measures structural in the geometry (feature 4 is verified
// by the transparency suite in the test harness):
//   - the base delay set is mutually incommensurate, being ratios of sixteen
//     distinct primes laid out geometrically. Through v0.4.2 it also scaled as a
//     whole, so the ratios between lines — which is what decides whether modes
//     pile up — were invariant under the morph. v0.5 gives that invariance up
//     deliberately (`LateShape::lineSpread`), because it was also what made every
//     room one box resized; what replaces it is that the exponent map preserves
//     the geometric layout, so the set stays mutually incommensurate at every
//     spread, and that total delay is renormalized so modal *density* is still a
//     function of size alone. The coloration bound per model is what holds it
//     honest, and it is measured on all fourteen
//   - modulation rates are mutually incommensurate
//   - the ER tap bank alternates polarity, so it doesn't comb
//   - the input diffusion network runs up to eight cascaded allpasses; depth
//     is a model property that Thickness scales and Character pulls against
//
// A word on why the diffusion lives *in front of* the tank and not inside the
// feedback loop: an allpass is unity-magnitude, so putting one in the loop
// looks free, but its group delay is not flat — at DC it is D(1+g)/(1-g),
// several times its own length — so it lengthens the effective loop at low
// frequencies and leaves it alone in the mid band. That silently breaks the
// per-band decay contract the whole engine is measured against. Density is
// bought in the input stage, where it costs nothing but CPU.
//
// ---------------------------------------------------------------------------
// v0.7: surround and immersive, up to 9.1.6
// ---------------------------------------------------------------------------
// The claim being built here is "one room, not N reverbs", and the structure is
// what has to deliver it — not a wide-sounding post-process on a stereo tank.
// Three things carry it:
//
//   - **One tank, N windows onto it.** The sixteen FDN lines are the room, and
//     they do not multiply with the channel count. What multiplies is the
//     *output vector*: every non-LFE speaker reads the same sixteen lines
//     through its own Walsh row, and the sixteen-point Hadamard matrix has
//     exactly fifteen non-trivial rows — one per speaker of a 9.1.6 layout with
//     its LFE taken out. Because the rows are mutually orthogonal, the speakers
//     are mutually decorrelated *and* share every mode, every decay time and
//     every geometry change to the sample. That is what "one room" means: the
//     tail is not N tails that happen to have been given the same settings.
//
//   - **Cross-feed is a correlation, applied to the rows.** Two microphones a
//     few degrees apart in a real room hear nearly the same field; two facing
//     opposite walls hear almost independent ones. So each speaker's row is
//     rotated towards its neighbours' by an amount that falls off with the
//     angle between them (see buildOutputVectors), which is the same operation
//     v0.5's `lateCorrelation` already performed on the stereo pair — surround
//     is the general case of it rather than a new mechanism. Rows are
//     renormalized after the rotation, so cross-feed costs no level.
//
//   - **The input side reduces rather than multiplies.** Sixteen input channels
//     do not get sixteen diffusion cascades and sixteen early-reflection banks;
//     they are panned by direction into four spatial *feeds* (front and rear,
//     left and right), and the feeds carry the early bank and the diffusers.
//     A room does not care which of your speakers a sound came out of, only
//     roughly where in the room it was — and the difference between four feeds
//     and sixteen is inaudible where the difference in cost is a factor of four
//     on the most expensive stage in the plugin.
//
// Mono and stereo run the identical code with numFeeds = 2, the v0.5 masks and
// the v0.5 correlation rotation, which is why the whole pre-v0.7 test suite
// still measures what it measured. A mono layout keeps a two-channel tank and
// folds at the output, exactly as v0.1 did.
//
// The **LFE is not part of the room**: it is passed through untouched, bit for
// bit, and contributes nothing to the feeds. It is a band-limited effects
// channel rather than a position, so there is no direction for the room to
// place it at and no direction for it to be heard from; putting a full-band
// tail on it muddies a mix in a way nobody asks for. The consequence worth
// stating plainly is that at Mix 100% the LFE channel carries dry signal, which
// is the honest reading of "this channel is not sent to the reverb".
class ReverbEngine
{
public:
    static constexpr int numLines = 16;

    // 9.1.6. Everything wet-side is sized by this; see SpeakerLayout.
    static constexpr int maxChannels = hrzn::maxSpeakers;

    // The spatial feeds the input side reduces to: left/right for stereo and
    // below, front/rear × left/right for everything above it.
    static constexpr int maxFeeds = 4;

    static constexpr int numDiffusers = 8;
    static constexpr int maxErTaps = 32;

    // Everything whose discontinuity would be audible — delay lengths, tap
    // times, filter coefficients, levels — moves every sample. The rest
    // (feedback gains, cutoffs, modulation rates: the quantities that need a
    // pow or an exp) is refreshed on this subdivision, ~1.5 kHz at 48 kHz.
    // A step in a feedback gain is a multiplication on an already-smooth
    // signal; a step in a delay length is a click.
    static constexpr int controlBlock = 32;

    struct Params
    {
        float space01      = 0.55f; // stepless position through the model table
        float decaySeconds = 2.0f;  // absolute RT60, independent of the room
        float predelayMs   = 0.0f;  // 0 .. 500
        float mix01        = 0.30f; // 0 = dry, 1 = wet
        float width        = 1.0f;  // 0 = mono, 1 = normal, 2 = wide
        float brightness01 = 0.5f;  // 0.5 = the model's own voicing
        float distance01   = 0.35f; // 0 = close, 1 = far
        float character01  = 0.5f;  // 0 = clean, 0.5 = the model, 1 = chorused
        float thickness01  = 0.5f;  // 0 = open, 0.5 = the model, 1 = dense
        bool  freeze       = false;

        // v0.3: the two six-band sections. The Decay Rate EQ multiplies the
        // decay time per frequency inside the feedback path; the Post EQ
        // shapes the wet output's tone and leaves the decay alone.
        //
        // The section switches (v0.4.1) are applied by the caller rather than
        // here — a bypassed Decay Rate EQ is exactly "every band at 1.00x" and
        // a bypassed Post EQ is exactly "every band off", so the switch is a
        // transformation of these arrays and needs no engine state of its own.
        // That also means the existing per-band smoothing carries the
        // transition, which is what keeps it click-free.
        std::array<DecayEq::Band, DecayEq::numBands> decayBands {};
        std::array<PostEq::Band, PostEq::numBands> postBands {};
    };

    void prepare (double newSampleRate, const SpeakerLayout& newLayout)
    {
        sampleRate = newSampleRate;

        // Everything channel-shaped is settled before any table is built,
        // because the tables' *sizes* — how many feeds, how many diffuser
        // cascades, how many output rows — are functions of the layout.
        adoptLayout (newLayout);

        // The ER tap patterns are the one part of the geometry that cannot be
        // derived from a scalar, so they are built once here and crossfaded
        // tap by tap at runtime.
        buildTapTables();
        buildDelayTables();
        buildDiffuserTables();
        buildMixWeights();
        buildFeedMatrix();
        buildInjection();

        // Sized from the longest line any model actually asks for rather than
        // from the base set's longest: `lineSpread` above 1 stretches the top of
        // the distribution, and the morph can pair one model's lengths with a
        // size interpolated towards its neighbour's, so the two maxima have to be
        // taken independently. Cached because the morph rate limiter needs the
        // same figure per sample.
        longestLineSamples = longestModelLine();

        const int maxLine = (int) (worstCaseLineSamples() * 1.35f) + 512;
        for (auto& line : lines)
            line.prepare (maxLine);

        // Only what the layout actually uses is allocated. Half a second of
        // predelay is 96 kB a channel at 48 kHz, and sixteen of them for a
        // stereo session would be a megabyte and a half of buffer nothing ever
        // reads — the arrays are sized for 9.1.6 so the *engine* need not be
        // reallocated, which is not the same as filling them.
        for (int c = 0; c < numIn; ++c)
            predelayLines[(size_t) c].prepare ((int) (0.5 * sampleRate) + 8);

        // The ER bank reads from its own line, so predelay and ER spread stay
        // independent controls. Distance stretches the tap times by up to 30%,
        // which the buffer has to cover.
        for (int f = 0; f < numFeeds; ++f)
            erLines[(size_t) f].prepare ((int) (0.45 * sampleRate) + 8);

        const int maxDiff = (int) longestModelDiffuser() + 8;
        for (int f = 0; f < numFeeds; ++f)
            for (auto& ap : diffusers[(size_t) f])
                ap.prepare (maxDiff);

        for (auto* s : allSmoothers())
            s->reset (sampleRate, 0.02);
        smoothPredelay.reset (sampleRate, 0.05);

        // 30 ms each way, for the relocation fade only.
        fadeStep = 1.0f / juce::jmax (1.0f, 0.030f * (float) sampleRate);
        freezeSlew = 1.0f - std::exp (-1.0f / (0.05f * (float) sampleRate));
        rt60Slew = 1.0f - std::exp (-(float) controlBlock / (0.12f * (float) sampleRate));
        maxPosStep = maxPosPerSecond / (float) sampleRate;
        minPosStep = minPosPerSecond / (float) sampleRate;
        tiltCoeff = DampingFilter::coeffFor (2000.0f, sampleRate);

        reset();

        // Opening at a Space position is not a morph — start where we are.
        pos = targetPos = spaceToPosition (lastParams.space01);
        evaluateGeometry();
        setParams (lastParams);

        decayEq.prepare (sampleRate, numLines);
        postEq.prepare (sampleRate, tankSpeakers.data(), numTank);

        updateControlRate();
        currentRt60 = targetRt60;
        updateControlRate();

        for (auto* s : allSmoothers())
            s->setCurrentAndTargetValue (s->getTargetValue());
        smoothPredelay.setCurrentAndTargetValue (smoothPredelay.getTargetValue());
        predelayNow = smoothPredelay.getTargetValue();

        deliveredRt60 = estimateDeliveredRt60();
    }

    void reset() noexcept
    {
        flushTank();
        for (int c = 0; c < numIn; ++c)
            predelayLines[(size_t) c].clear();

        // The read position lands on the target rather than gliding to it: a
        // relocation has already emptied the line, so there is nothing behind
        // the pointer to glide through.
        predelayNow = smoothPredelay.getTargetValue();

        decayEq.reset();
        postEq.reset();

        inputDampState.fill (0.0f);
        tiltState.fill (0.0f);
        modPhase.fill (0.0f);
        freezeAmount = freezeTarget;
        wetFade = 1.0f;
        relocatePending = false;
        controlCountdown = 0;
    }

    // A wholesale state change — preset load, A/B switch, undo/redo — is a
    // discontinuity by definition, and gliding across the table for a second
    // or two would be the wrong answer to it. This is the one path that still
    // does the v0.1 thing: fade out, jump, restart the tank, fade back in.
    void relocate() noexcept { relocatePending = true; }

    // Delivered decay across frequency, from the Decay Rate EQ's own design.
    // v0.4's decay-time overlay draws this curve; the front panel shows the
    // 1 kHz point.
    static int numDecayPoints() noexcept { return DecayEq::numReportPoints; }
    static float decayPointFrequency (int k) noexcept { return DecayEq::reportFrequency (k); }
    float decaySecondsAt (int k) const noexcept { return decayEq.decaySecondsAt (k); }

    // The same grid with the Decay Rate EQ neutral — the room's own decay
    // across frequency. v0.4's display draws it as the reference the band
    // handles are placed against.
    float neutralDecaySecondsAt (int k) const noexcept
    {
        return decayEq.neutralDecaySecondsAt (k);
    }

    void setParams (const Params& p) noexcept
    {
        lastParams = p;

        targetPos = spaceToPosition (p.space01);
        decaySecondsTarget = juce::jlimit (minDecaySeconds, maxDecaySeconds, p.decaySeconds);

        // Whether the morph can still catch up from here is decided at control
        // rate, once the target has settled — see checkMorphCatchUp.

        // Brightness (feature 16) moves the model's own damping cutoff +/- two
        // octaves, which changes HF *decay*, and tilts the wet output, which
        // changes overall brightness. One control, both effects, as asked.
        //
        // v0.3 makes it compose properly with the Decay Rate EQ rather than
        // compete with it. Brightness voices the model's air absorption — the
        // baseline — and the Decay Rate EQ's multipliers are designed against
        // whatever that baseline currently is (see DecayEq::update, which
        // takes these very coefficients). A band set to 0.5x therefore halves
        // that band's decay at *every* Brightness setting rather than winning
        // or losing an argument with it, and the delivered per-band decay
        // readout shows the two combined.
        const float bright = juce::jlimit (0.0f, 1.0f, p.brightness01);
        brightnessRatioMul = std::pow (2.6f, (bright - 0.5f) * 2.0f);
        smoothTilt.setTargetValue (std::pow (2.0f, (bright - 0.5f) * 2.0f));

        // Character (feature 13): one control from a clean, transparent decay
        // through a lively room with pronounced reflections to an
        // over-modulated chorus. 0.5 is the model exactly as designed, so the
        // trajectory is two straight segments meeting at unity rather than one
        // ramp — otherwise no setting would give you the room itself.
        const float c = juce::jlimit (0.0f, 1.0f, p.character01);
        charErMul    = twoSided (c, 0.55f, 1.00f, 1.90f);
        charDepthMul = twoSided (c, 0.30f, 1.00f, 4.00f);
        charRateMul  = twoSided (c, 0.70f, 1.00f, 2.20f);
        charDiffMul  = twoSided (c, 1.12f, 1.00f, 0.78f);
        charStageMul = twoSided (c, 1.15f, 1.00f, 0.72f);
        // Above the midpoint the later reflections are lifted as well, which
        // is what turns a smooth early field into audible discrete echoes.
        charLively   = juce::jmax (0.0f, (c - 0.5f) * 2.0f);
        smoothLively.setTargetValue (charLively);

        // Thickness (feature 7): how densely the late field builds. It scales
        // the depth and strength of the diffusion network in front of the tank
        // — the stage that decides how quickly a single impulse becomes a
        // continuous cloud — and leans the ER/late balance with it.
        const float th = juce::jlimit (0.0f, 1.0f, p.thickness01);
        thickStageMul = twoSided (th, 0.35f, 1.00f, 1.80f);
        thickGainMul  = twoSided (th, 0.78f, 1.00f, 1.18f);
        thickLateMul  = twoSided (th, 0.88f, 1.00f, 1.16f);

        // Distance (feature 11), properly this time: four things move
        // together. The early reflections drop away and arrive later (you
        // stop hearing the wall next to you and start hearing the room), the
        // late field comes up, the air absorbs the top end on the way in, and
        // a level trim compensates so the control changes *where* the source
        // is rather than how loud it is.
        const float d = juce::jlimit (0.0f, 1.0f, p.distance01);
        distErMul   = juce::jmap (d, 1.90f, 0.22f);
        distLateMul = juce::jmap (d, 0.62f, 1.35f);
        smoothErTime.setTargetValue (juce::jmap (d, 0.88f, 1.30f));
        smoothWetTrim.setTargetValue (juce::jmap (d, 1.02f, 0.88f));
        smoothInputDamp.setTargetValue (
            DampingFilter::coeffFor (lerpGeometric (20000.0f, 3400.0f, d), sampleRate));

        smoothPredelay.setTargetValue (
            juce::jlimit (0.0f, 500.0f, p.predelayMs) * 0.001f * (float) sampleRate);
        smoothMix.setTargetValue (juce::jlimit (0.0f, 1.0f, p.mix01));
        smoothWidth.setTargetValue (juce::jlimit (0.0f, 2.0f, p.width));

        decayEq.setBands (p.decayBands);
        postEq.setBands (p.postBands);

        freezeTarget = p.freeze ? 1.0f : 0.0f;
    }

    // Processes in place. v0.1 handles mono and stereo; surround is v0.7.
    // `wetTap`, when given, receives the wet signal as it stands just before it
    // meets the dry one — post everything the two EQ sections and Width do to
    // it, pre Mix. That is the trace v0.4's analyzer draws, and it has to be
    // taken here rather than reconstructed from the output: at Mix 0 there is
    // no wet in the output to reconstruct from, and the tail is exactly what
    // you want to see while setting the control.
    //
    // `duckGateGain`, when given, is a per-sample linear gain on the wet path
    // from Ducking and the Auto Gate (v0.5, see DuckGate.h). It is applied here
    // rather than by the caller because the mix happens here, and it goes on
    // *before* the tap on purpose: watching the tail duck is most of what the
    // analyzer is for while you are setting the control. The caller passes
    // nullptr when both are neutral, so a session that uses neither pays
    // nothing for them.
    void process (float* const* channelData, int numChannels, int numSamples,
                  float* const* wetTap = nullptr,
                  const float* duckGateGain = nullptr) noexcept
    {
        // The layout decides which channels are the room's and which is the
        // LFE, not the channel count: 7.1 and 7.1 SDDS are both eight channels
        // with the surrounds and the LFE in different places. A count that
        // disagrees with what prepare() was told is a host bug, and the safe
        // reading of it is to process nothing rather than to guess.
        if (numChannels < layoutChannels)
            return;

        std::array<float, maxChannels> dryIn {}, room {}, er {}, late {}, wet {};
        std::array<float, maxFeeds> feed {}, dif {}, erRead {};

        for (int i = 0; i < numSamples; ++i)
        {
            // Freeze ramps rather than switching: feedback goes to exactly 1,
            // damping is bypassed, modulation stops and the reads snap to
            // integers. All four together are what make the sustain lossless
            // (see DelayLine's note).
            freezeAmount += (freezeTarget - freezeAmount) * freezeSlew;
            const float frozen = freezeAmount;

            advanceRelocation();
            advanceMorph (frozen);

            if (pos != evaluatedPos)
                evaluateGeometry();

            if (--controlCountdown <= 0)
            {
                updateControlRate();
                controlCountdown = controlBlock;
            }

            // Predelay glides, and the glide is rate limited (v0.8).
            //
            // A delay line cannot be shortened faster than time passes. The
            // 50 ms ramp asks it to whenever the change is large — half a
            // second of predelay removed over 50 ms needs the read pointer to
            // advance eleven samples per sample, so it overtakes the write
            // head and reads audio that has not been written yet. That is a
            // splice, and it is what the v0.8 automation suite found: Predelay
            // was the only control whose worst automated step (0.12) was six
            // times its nearest rival's.
            //
            // Clamping the per-sample movement to 0.6 samples fixes it in the
            // one direction that is physically available: the read pointer
            // stays behind the write pointer, so a fast move is a tape-style
            // glide rather than a cut. 0.6 is chosen so nothing that worked
            // before changes at all — a 30 ms move, which is larger than any
            // predelay edit anybody makes by hand, already ramps at exactly
            // 0.6 samples per sample, so the clamp only ever binds on moves
            // bigger than the whole useful range of the control.
            predelayNow += juce::jlimit (-maxPredelaySlew, maxPredelaySlew,
                                         smoothPredelay.getNextValue() - predelayNow);
            const float predelaySamples = predelayNow;

            const float mix       = smoothMix.getNextValue();
            const float width     = smoothWidth.getNextValue();
            const float erLevel   = smoothErLevel.getNextValue();
            const float lateLevel = smoothLateLevel.getNextValue();
            const float tilt      = smoothTilt.getNextValue();
            const float lpCoeff   = smoothLpCoeff.getNextValue();
            const float hpCoeff   = smoothHpCoeff.getNextValue();
            const float diffGain  = smoothDiffGain.getNextValue();
            const float stages    = smoothStages.getNextValue();
            const float erScale   = smoothErTime.getNextValue();
            const float wetTrim   = smoothWetTrim.getNextValue();
            const float modScale  = smoothModDepth.getNextValue();
            const float inDamp    = smoothInputDamp.getNextValue();
            const float lively    = smoothLively.getNextValue();

            // Input ----------------------------------------------------------
            // Predelay and Distance's air absorption run per *input* channel,
            // so a source's position survives into the feeds below rather than
            // being averaged away before the room ever sees it.
            for (int c = 0; c < numIn; ++c)
            {
                const auto ci = (size_t) c;
                const float x = channelData[wetChan[ci]][i];
                dryIn[ci] = x;

                predelayLines[ci].push (x);
                const float pd = predelayLines[ci].read (predelaySamples);

                inputDampState[ci] += inDamp * (pd - inputDampState[ci]);
                room[ci] = inputDampState[ci];
            }

            // The spatial feeds. At two feeds this is the identity on a stereo
            // layout and a duplication on a mono one, so the two lines below
            // are bit for bit what v0.6 pushed into its L and R ER lines.
            for (int f = 0; f < numFeeds; ++f)
            {
                const auto fi = (size_t) f;
                float s = 0.0f;
                for (int c = 0; c < numIn; ++c)
                    s += feedMix[(size_t) c][fi] * room[(size_t) c];

                feed[fi] = s;
                erLines[fi].push (s);
            }

            // Early reflections --------------------------------------------
            //
            // Each tap is read once per feed and then distributed across the
            // speakers by its own direction, rather than each *speaker* getting
            // its own bank. Written as a rank-reduced sum: a tap is a small
            // number of (feed weights) x (speaker weights) outer products, so
            // the cost is feeds + speakers per tap instead of feeds x speakers.
            // At two feeds the two components are exactly "left input to left
            // output" and "right to right", which is v0.6's bank unchanged.
            for (int w = 0; w < numTank; ++w)
                er[(size_t) w] = 0.0f;

            for (int t = 0; t < erTapLimit; ++t)
            {
                const auto ti = (size_t) t;
                const float dt = tapTime[ti] * erScale;

                for (int f = 0; f < numFeeds; ++f)
                    erRead[(size_t) f] = erLines[(size_t) f].read (dt);

                // Character's late-reflection emphasis is a per-tap scalar
                // rather than a second gain vector: it multiplies every
                // speaker's share of this tap equally, so it belongs on the tap
                // and not on the fan-out. See updateErGains.
                const float livelyScale = 1.0f + lively * tapLively[ti];

                for (int s = 0; s < numErSources; ++s)
                {
                    const auto si = (size_t) s;

                    float v = 0.0f;
                    for (int f = 0; f < numFeeds; ++f)
                        v += erFeedW[si][ti][(size_t) f] * erRead[(size_t) f];

                    v *= livelyScale;

                    const float* g = erChanW[si][ti].data();
                    for (int w = 0; w < numTank; ++w)
                        er[(size_t) w] += g[(size_t) w] * v;
                }
            }

            // Input diffusion ----------------------------------------------
            // Every stage always runs so its buffer stays warm; `stages`
            // crossfades each one in and out, which is what lets diffusion
            // depth be a continuous, morphable, automatable quantity.
            for (int f = 0; f < numFeeds; ++f)
                dif[(size_t) f] = feed[(size_t) f];

            for (int s = 0; s < numDiffusers; ++s)
            {
                const auto si = (size_t) s;
                const float w = juce::jlimit (0.0f, 1.0f, stages - (float) s);

                // A stage fades in through its own *coefficient*, not through a
                // dry/wet mix.
                //
                // Mixing was the v0.2 design and it is wrong in a way that took
                // until v0.4.2 to see, because the delay-line modulation was
                // smearing the evidence. An allpass has unity magnitude; a
                // partial mix of one against the dry signal does not — it is a
                // comb, worst at a half-and-half blend. A stage count of 4.6
                // measured 10 dB of third-octave prominence where 5.0 measured
                // 2.3. And the count is not a fixed number per model: Thickness
                // and Character scale it and the Space morph interpolates it,
                // so *every* setting is a partial one somewhere.
                //
                // A Schroeder allpass with g = 0 is a plain delay, which is
                // also unity magnitude, so sweeping the coefficient runs from
                // "delay" to "diffuser" without ever leaving the allpass
                // family. Every stage therefore always runs and always delays;
                // what `stages` now controls is how many of them diffuse.
                //
                // The feeds' diffusers are detuned against one another so the
                // late field arrives decorrelated rather than as one mono cloud
                // in the middle of the image. From v0.5 by how much is the
                // room's own business — the detune was a hardcoded 1.061 shared
                // by a vocal booth and an arena — and it is baked into the
                // per-feed length tables so saying so costs the loop nothing.
                // At two feeds those tables are v0.6's L and R sets exactly.
                for (int f = 0; f < numFeeds; ++f)
                    dif[(size_t) f] = diffusers[(size_t) f][si].process (
                        dif[(size_t) f], diffDelay[(size_t) f][si], diffGain * w);
            }

            // FDN late field ------------------------------------------------
            std::array<float, numLines> tapped {};

            for (int l = 0; l < numLines; ++l)
            {
                const auto li = (size_t) l;

                modPhase[li] += modInc[li];
                if (modPhase[li] >= 1.0f)
                    modPhase[li] -= 2.0f;

                // Parabolic sine approximation — ample for chorus-rate motion.
                const float ph = modPhase[li];
                const float lfo = 4.0f * ph * (1.0f - std::abs (ph));

                // Under Freeze the read glides to the nearest whole sample —
                // at most half a sample, over the 50 ms freeze ramp — so the
                // interpolator stops costing energy per trip round the loop.
                float d = delaySamples[li];
                d += (std::round (d) - d) * frozen;

                // Character can quadruple the modulation depth, so the offset
                // is capped against the line's own length rather than trusted.
                // Depth as a fraction of *this line's* length, so every line
                // is detuned by the same musical amount and the figure means
                // the same thing on an ambience as on a cathedral. The 0.30
                // ceiling is a buffer-safety net, not a voicing choice; before
                // v0.4.2 it was what the small rooms actually ran at.
                const float depth = juce::jmin (d * modDetuneNow * modScale, 0.30f * d);
                const float offset = depth * lfo * (1.0f - frozen);

                float v = lines[li].read (d + offset);

                // Damping blends out under Freeze so the loop neither dulls
                // nor decays. The shelf gains are this line's own from v0.7.1 —
                // they are refreshed at control rate rather than per sample, for
                // the same reason the feedback gains are: a step in a gain is a
                // multiplication on an already-smooth signal.
                const float damped = damping[li].process (v, lpCoeff, hfGainLine[li],
                                                          hpCoeff, lfGainLine[li]);
                v = damped + (v - damped) * frozen;

                tapped[li] = v;
            }

            // The Decay Rate EQ, in a pass of its own rather than as a branch
            // inside the loop above.
            //
            // That loop is the hottest code in the plugin — sixteen fractional
            // reads, sixteen modulators and sixteen filters per sample — and
            // putting a conditional cascade in the middle of it cost about a
            // third of the whole plugin's CPU *even with every band neutral
            // and the branch never taken*. The body is right at the edge of
            // what fits in registers, and the extra live state was enough to
            // start spilling. Hoisting the condition out puts the neutral path
            // back to exactly the instruction sequence v0.2 ran.
            //
            // The blend against `frozen` is here for the same reason the
            // damping filter has one: a frozen tank is lossless or it is not
            // frozen.
            if (decayEqActive)
                for (int l = 0; l < numLines; ++l)
                {
                    const auto li = (size_t) l;
                    const float shaped = (float) decayEq.process (l, (double) tapped[li]);
                    tapped[li] = shaped + (tapped[li] - shaped) * frozen;
                }

            // Feedback gain, which goes to exactly unity under Freeze. With an
            // orthogonal matrix and exact reads, that is a genuinely lossless
            // tank.
            for (int l = 0; l < numLines; ++l)
            {
                const auto li = (size_t) l;
                tapped[li] *= feedbackGain[li] + (1.0f - feedbackGain[li]) * frozen;
            }

            // The feedback matrix, H = I - 2vv^T for the room's own reflection
            // vector. With v uniform — every model through v0.4.2 — this is
            // literally the old `sum * (2/N)`: v_l = 1/4, so 2*v_l*(v.x) works
            // out at sum * 2/16, to the bit.
            //
            // A reflection is orthogonal for *any* unit v, which is the whole
            // reason the matrix could be opened up as a voicing control at all:
            // the tank stays lossless, so Freeze still sustains forever and every
            // bit of decay still comes from the feedback gains, at every value of
            // `mixSkew`. What a skewed v buys is lines that couple *unevenly* —
            // heavily weighted lines become the hubs energy crosses through, and
            // lightly weighted ones stay nearly independent combs whose modes the
            // ear can still pick out individually. That is the difference between
            // a wash and a room with a character, and it is the one structural
            // axis the engine previously had no way to express.
            float projection = 0.0f;
            for (int l = 0; l < numLines; ++l)
                projection += mixV[(size_t) l] * tapped[(size_t) l];
            projection *= 2.0f;   // the 2 in H = I - 2vv^T, folded in once

            // Freeze seals the tank; the relocation fade also gates injection,
            // because a snap steps the diffuser and ER read positions. Fading
            // only the *output* would still let that step into the tank, where
            // it rings out as the fade comes back up.
            const float inputGate = (1.0f - frozen) * wetFade;
            for (int l = 0; l < numLines; ++l)
            {
                const auto li = (size_t) l;

                // Each line belongs to exactly one feed, so the injection is a
                // lookup rather than a sum over feeds. The two-feed assignment
                // is v0.5's Walsh row 4 split, so a stereo session's lines take
                // the same halves of the tank they always did.
                const float inject = dif[(size_t) lineFeed[li]] * injectTrimNow;
                lines[li].push (tapped[li] - mixV[li] * projection
                                + inject * inputGate);
            }

            // The late field, one Walsh row per speaker.
            //
            // v0.5 kept the two rows constexpr and rotated the *sums* rather
            // than the vectors, because a runtime coefficient array in the hot
            // loop had cost more than the feature it bought. That trick does
            // not generalise past two outputs — fifteen rows cannot be a pair
            // of unrolled sign chains — so v0.7 pays for the array. It is one
            // array read linearly, outside the per-line loop, and the audit in
            // the harness is what says whether that was affordable. (It was:
            // the neutral stereo engine did not move.)
            for (int w = 0; w < numTank; ++w)
            {
                const float* ov = outVec[(size_t) w].data();
                float s = 0.0f;
                for (int l = 0; l < numLines; ++l)
                    s += ov[(size_t) l] * tapped[(size_t) l];
                late[(size_t) w] = s * outputTrim;
            }

            // Wet assembly --------------------------------------------------
            const float wetGain = wetMakeup * wetTrim;
            for (int w = 0; w < numTank; ++w)
            {
                const auto wi = (size_t) w;
                wet[wi] = (er[wi] * erLevel + late[wi] * lateLevel) * wetGain;
            }

            // Width (feature 22), as the general case of mid/side.
            //
            // The mid is the mean across the speakers and the side is each
            // speaker's departure from it, which on a stereo pair is exactly
            // (L+R)/2 and ±(L-R)/2 — the v0.1 formula, unchanged. What
            // generalising it buys is that the mean is invariant: Width scales
            // only the part that differs between speakers, so a fold to mono
            // (or to any layout that averages the speakers) is identical at
            // every setting from 0% to 200%. That is the mono compatibility
            // feature 22 asks for, and it is a property of the arithmetic here
            // rather than something checked for afterwards.
            {
                float mid = 0.0f;
                for (int w = 0; w < numTank; ++w)
                    mid += wet[(size_t) w];
                mid *= invNumTank;

                for (int w = 0; w < numTank; ++w)
                {
                    const auto wi = (size_t) w;
                    wet[wi] = mid + (wet[wi] - mid) * width;
                }
            }

            // Brightness tilt: split at ~2 kHz, scale the top.
            for (int w = 0; w < numTank; ++w)
            {
                const auto wi = (size_t) w;
                tiltState[wi] += tiltCoeff * (wet[wi] - tiltState[wi]);
                wet[wi] = tiltState[wi] + (wet[wi] - tiltState[wi]) * tilt;
            }

            // Post EQ (feature 21): the last thing that happens to the wet
            // signal before it meets the dry one, so it shapes the reverb and
            // nothing else. At Mix 0 the dry path is still untouched, bit for
            // bit — this whole branch is multiplied away below.
            if (postEqActive)
                postEq.processSample (wet.data(), numTank);

            for (int w = 0; w < numTank; ++w)
                wet[(size_t) w] *= wetFade;

            // Ducking and the Auto Gate (v0.5).
            if (duckGateGain != nullptr)
                for (int w = 0; w < numTank; ++w)
                    wet[(size_t) w] *= duckGateGain[i];

            // The analyzer trace stays a stereo pair whatever the layout: it is
            // the front pair, which is the one every layout has and the one the
            // display is drawing.
            if (wetTap != nullptr)
            {
                wetTap[0][i] = wet[0];
                wetTap[1][i] = wet[1];
            }

            // Mix. At mix == 0 the dry path is untouched, bit for bit — and the
            // LFE is untouched at every Mix, because it is never written here.
            if (foldMono)
            {
                const float wetMono = (wet[0] + wet[1]) * 0.5f;
                const float in0 = dryIn[0];
                channelData[wetChan[0]][i] = in0 + (wetMono - in0) * mix;
            }
            else
            {
                for (int w = 0; w < numTank; ++w)
                {
                    const auto wi = (size_t) w;
                    const float dry = dryIn[wi];
                    channelData[wetChan[wi]][i] = dry + (wet[wi] - dry) * mix;
                }
            }
        }

        deliveredRt60 = estimateDeliveredRt60();
        sanitize();
    }

    // Reporting --------------------------------------------------------------

    // What the model table asks for, before the damping filter takes its cut.
    float currentRt60Seconds() const noexcept { return currentRt60; }

    // What the engine actually produces in the mid band — the nominal decay
    // with the in-loop damping's per-pass loss folded in. v0.1 displayed the
    // nominal figure and ran about 7% short of it; a passive filter in the
    // feedback path can only ever remove energy, so this is the honest number
    // and it is the one the UI shows.
    float deliveredRt60Seconds() const noexcept { return deliveredRt60; }

    // Continuous position through the room model table, 0 .. numRoomModels-1.
    float currentSpacePosition() const noexcept { return pos; }
    float targetSpacePosition() const noexcept { return targetPos; }
    bool isMorphing() const noexcept { return pos != targetPos || relocatePending; }

    int currentModelIndex() const noexcept { return positionToModelIndex (pos); }

    const char* currentModelName() const noexcept
    {
        return roomModels[(size_t) currentModelIndex()].name;
    }

    // The pair the morph is currently sitting between, for the UI readout.
    int morphLowIndex() const noexcept { return morphLow; }
    int morphHighIndex() const noexcept { return morphHigh; }
    float morphFraction() const noexcept { return morphFrac; }

    // Total delay in the tank at a given room size. An FDN's modal density is
    // exactly this: it has one mode per (1 / total delay) hertz, evenly spread
    // in frequency. That makes it the right thing to ask when deciding how low
    // a coloration measurement can meaningfully look — below the point where a
    // critical band stops containing several modes, what an analyser sees is
    // individual resonances being resolved, which is what a small room *is*.
    static float totalLineDelaySeconds (float sizeScale) noexcept
    {
        float sum = 0.0f;
        for (float ms : baseDelayMs)
            sum += ms;
        return sum * 0.001f * sizeScale;
    }

    // Longest tail the host should keep processing for. A Decay Rate EQ band
    // set high makes one part of the spectrum ring far longer than the nominal
    // decay, and a host that truncates on the nominal figure would cut it off
    // mid-bounce — so the figure is the longest band, not the average one.
    float tailSeconds() const noexcept
    {
        if (lastParams.freeze)
            return 60.0f;

        float longest = currentRt60;
        for (int k = 0; k < DecayEq::numReportPoints; ++k)
            longest = juce::jmax (longest, decayEq.decaySecondsAt (k));

        return juce::jmin (300.0f, longest) + lastParams.predelayMs * 0.001f + 0.1f;
    }

private:
    struct ModelTaps
    {
        std::array<float, maxErTaps> time {};   // samples
        std::array<float, maxErTaps> gainL {}, gainR {};
        // The same bank stated as amplitude and lateral position rather than as
        // a pair of channel gains. Redundant with gainL/gainR by construction —
        // and kept anyway, because the two-feed path wants the pair and the
        // surround path wants to place the tap on a sphere, and recovering one
        // from the other in the control-rate loop is a square root per tap for
        // a number that was in hand when the table was built.
        std::array<float, maxErTaps> amp {}, pan {};
        std::array<float, maxErTaps> frac {};   // 0..1 through the ER spread
        int count = 0;
    };

    // Which Walsh row each speaker reads the tank through. Rows 1 and 2 stay
    // where they were so a stereo layout is unchanged; the rest are ordered by
    // increasing Hamming weight, which spreads the low-order rows — the ones
    // that split the tank into large contiguous blocks of lines — across the
    // first speakers of the layout rather than clustering them.
    //
    // The sixteenth entry is row 0, which is all ones: the only remaining
    // orthogonal row, and the one a sixteen-speaker layout with no LFE would
    // need. It is still orthogonal to the other fifteen, so it is still a valid
    // window onto the tank; it is last because it is the row with no sign
    // changes and therefore the one most like the sum of the others.
    static constexpr std::array<int, maxChannels> walshRowOrder { {
        1, 2, 4, 8, 3, 5, 6, 9, 10, 12, 7, 11, 13, 14, 15, 0
    } };

    // The cross-feed kernel. `floor` is how correlated two speakers in the same
    // place would be in the deadest room in the table, `room` is how much the
    // model's own `lateCorrelation` adds to that, and `spread` is the angle over
    // which it decays by 1/e — one radian, so a front pair 60 degrees apart
    // keeps about a third of the near-field figure and a front-to-rear pair
    // keeps a sixteenth.
    //
    // The floor is not zero on purpose. Sharing a tank already makes the
    // speakers one room in every respect that decay and geometry can express,
    // but with fully orthogonal rows the *fine structure* of the field is
    // independent at every speaker, and a layout whose neighbours have nothing
    // in common at all sounds like a very large room however small the setting
    // says it is. A third of a radian's worth of coupling between neighbours is
    // what puts the walls back.
    static constexpr float crossfeedFloor = 0.35f;
    static constexpr float crossfeedRoom = 0.55f;
    static constexpr float crossfeedSpread = 1.0f;

    // How tightly the early bank aims. The feeds are 90 degrees apart so they
    // need a broad kernel or a reflection between two corners would vanish; the
    // speakers can be 30 degrees apart, and a reflection that lands on five of
    // them at once is not a reflection.
    static constexpr float feedFocus = 1.15f;      // radians to 1/e
    static constexpr float speakerFocus = 0.62f;

    // Base FDN delay set in ms at sizeScale 1.0 (a concert hall). Every model
    // scales this whole set, so the *ratios* between lines — which is what
    // decides whether modes pile up — never change as Space moves.
    //
    // Those ratios are ratios of sixteen distinct primes (947, 1049, 1153 ...
    // 4201 samples at 48 kHz), laid out geometrically across a 4.4:1 span. Two
    // lines whose lengths are in a simple ratio put their modes on top of one
    // another, and a rounded-out set like the one v0.1 used — 19.7, 24.3,
    // 28.9 ms, an arithmetic progression with a 4.6 ms step — has that problem
    // everywhere: every line is commensurate with every other, and the whole
    // tank rings at multiples of 217 Hz.
    //
    // v0.1 got away with it because it rounded each length to the nearest
    // prime *number of samples*, which broke the progression by accident. The
    // stepless morph cannot do that: the lengths have to be continuous
    // functions of Space, so nothing may be rounded at runtime. The fix is to
    // put the incommensurability in the design instead of in the rounding —
    // and it is a better fix, because it holds at every sample rate and at
    // every point in the morph rather than only where the rounding happened to
    // land. Measured as a 6 dB third-octave ring in the late field before, and
    // absent after.
    static constexpr std::array<float, numLines> baseDelayMs { {
        19.729f, 21.854f, 24.021f, 26.646f, 29.354f, 32.479f, 35.854f, 39.604f,
        43.729f, 48.146f, 53.146f, 58.729f, 64.771f, 71.521f, 79.104f, 87.521f
    } };

    // Modulation rate multipliers, deliberately incommensurate so the lines
    // never fall into step with one another.
    static constexpr std::array<float, numLines> modRateScale { {
        1.000f, 1.117f, 0.873f, 1.291f, 0.764f, 1.428f, 0.941f, 1.203f,
        0.827f, 1.359f, 1.061f, 0.795f, 1.236f, 0.908f, 1.174f, 0.851f
    } };

    // The *shape* of the input diffuser cascade — the relative lengths. From
    // v0.5 the absolute scale comes from each model's `diffMeanMs` instead of
    // from a clamped multiple of its size, so this set is normalized by its own
    // geometric mean when the per-model tables are built and only its ratios
    // matter here.
    static constexpr std::array<float, numDiffusers> diffuserShapeMs { {
        4.77f, 3.59f, 12.73f, 9.31f, 6.11f, 8.17f, 2.71f, 10.43f
    } };

    // Hadamard rows 1 and 2 are orthogonal, so the two output taps come out
    // decorrelated; row 4 splits which lines each input feed reaches.
    //
    // From v0.5 these two rows are a *basis* rather than the output vectors
    // themselves: each model's `lateCorrelation` rotates the right-hand tap
    // towards the left-hand one, which trades the pair's decorrelation for a
    // narrower intrinsic image. Because the two rows are orthogonal and of equal
    // norm, that blend preserves the norm at every setting, so a room's stereo
    // width costs it no level. From v0.7 that rotation is the two-speaker case
    // of the cross-feed matrix, and the rows it rotates are two of sixteen.
    static constexpr auto outSignL = detail::makeSigns<numLines> (1);
    static constexpr auto outSignR = detail::makeSigns<numLines> (2);

    // All sixteen lines are summed at every output, so the output trim is fixed
    // whatever the layout. The injection trim is not — it depends on how many
    // lines each feed reaches — and lives in `injectTrimNow`; see buildInjection.
    static constexpr float outputTrim = 0.25f;    // 1/sqrt(16)

    // Those two trims leave the wet path around 12 dB below the dry, which
    // makes Mix feel like a level control rather than a blend. This makeup
    // lands full-wet a few dB under the input — the ratio a hall actually
    // gives you — and is applied after the ER/late balance so Distance keeps
    // its designed proportions. Calibrated by the harness's wet-level check.
    static constexpr float wetMakeup = 2.8f;

    // How fast the morph is allowed to move, and why it is expressed the way
    // it is.
    //
    // Gliding a delay line's read pointer is a pitch shift. Inside a feedback
    // loop that shift is applied again on every trip round, so what the ear
    // gets is not the instantaneous detune but the *accumulated* one — and the
    // number of trips is set by the decay time. Working it through: the
    // fractional shift per trip is the pointer rate, the pointer rate for line
    // l is base_l * d(size)/dt, and the trip length is base_l * size, so the
    // per-trip shift is d(size)/dt / size for every line alike, and after t
    // seconds of gliding the accumulated factor is exp(t * dln(size)/dt).
    //
    // The quantity to hold constant is therefore the *logarithmic* size rate
    // multiplied by the decay time — not a fixed sweep duration and not a
    // fixed pointer rate. That is what this constant is: the accumulated shift
    // permitted over the first quarter of the decay, which is the part loud
    // enough to hear. e^0.55 is about 1.7x, a musical swoop rather than a siren.
    //
    // It falls out of this that small rooms may sweep quickly and cathedrals
    // may not — which is correct, and is exactly the trade a fixed morph time
    // would get wrong at both ends.
    static constexpr float maxLogShiftPerDecay = 0.55f;

    // A hard safety net on the instantaneous pointer rate. The rule above
    // keeps it near 0.03 across the table; this exists so that no future
    // retuning of the model geometry can let a read pointer approach the speed
    // (1.0) at which it would outrun the writer.
    static constexpr float maxPointerRate = 0.10f;

    // Ceilings and floors in table units per second: the first for the small
    // end, where the geometry barely changes and the rule alone would allow a
    // model per block; the second so that the largest, longest models still
    // respond to the knob at all.
    static constexpr float maxPosPerSecond = 14.0f;
    static constexpr float minPosPerSecond = 0.55f;

    // The least total line delay any room may run — see applyDensityFloor. Set
    // at the knee of the measured curve: ripple falls steeply with total delay
    // up to about 150 ms and is flat beyond 200, so this is where the return
    // stops and the cost (longer lines in a small room) starts being paid for
    // nothing.
    static constexpr float densityFloorMs = 180.0f;

    // How hard the modulation is pulled back for the density the floor added.
    //
    // Holding (frequency * detune * total delay) exactly constant means an
    // exponent of 1: that is the ratio of how far a mode moves to how far apart
    // the modes are, and it is what decides whether movement reads as lushness
    // or as warble. Measured, 1.0 is too cautious — it leaves a held tone at
    // 0.98 concentration against a 0.95 bound while giving back half of the
    // ripple the floor had just removed. The modes are also individually less
    // prominent once there are five times as many of them, which the ratio
    // alone does not account for. This is where the two measurements meet.
    static constexpr float densityModExponent = 0.85f;

    // Character's two-segment trajectory: `mid` at 0.5 (the model as
    // designed), `low` at 0 and `high` at 1.
    static float twoSided (float x, float low, float mid, float high) noexcept
    {
        return x < 0.5f ? juce::jmap (x, 0.0f, 0.5f, low, mid)
                        : juce::jmap (x, 0.5f, 1.0f, mid, high);
    }

    // The longest line any *reachable* pairing of a model's table with a size
    // can produce, which is what the delay buffers have to cover.
    //
    // Through v0.7 this was "the longest entry in any table times the largest
    // size in the table", and that was safe because every table was normalized
    // to the same total delay — so the two maxima were genuinely independent.
    // The density floor ends that: a small room's table is stated relative to
    // its own small size, so multiplying it by a cathedral's size is a pairing
    // that cannot occur and would size the buffers for eight seconds of delay.
    // The morph only ever pairs a table with a size between its own neighbours',
    // so that is the bound to take.
    float worstCaseLineSamples() const noexcept
    {
        float worst = 0.0f;

        for (int mi = 0; mi < numRoomModels; ++mi)
        {
            float longest = 0.0f;
            for (float v : modelDelaySamples[(size_t) mi])
                longest = juce::jmax (longest, v);

            float size = roomModels[(size_t) mi].sizeScale;
            if (mi > 0)
                size = juce::jmax (size, roomModels[(size_t) (mi - 1)].sizeScale);
            if (mi < numRoomModels - 1)
                size = juce::jmax (size, roomModels[(size_t) (mi + 1)].sizeScale);

            worst = juce::jmax (worst, longest * size);
        }

        return worst;
    }

    float longestModelLine() const noexcept
    {
        float d = 0.0f;
        for (const auto& set : modelDelaySamples)
            for (float v : set)
                d = juce::jmax (d, v);
        return d;
    }

    float longestModelDiffuser() const noexcept
    {
        float d = 0.0f;
        for (int f = 0; f < numFeeds; ++f)
            for (const auto& set : modelDiff[(size_t) f])
                for (float v : set)
                    d = juce::jmax (d, v);
        return d;
    }

    // Deterministic per-model pseudo-random sequence (xorshift32). The ER
    // pattern must be identical on every machine and run, so a preset sounds
    // the same everywhere.
    struct Rng
    {
        unsigned int s;
        explicit Rng (unsigned int seed) : s (seed | 1u) {}
        unsigned int next() noexcept
        {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            return s;
        }
        float next01() noexcept { return (float) (next() >> 8) / (float) (1 << 24); }
    };

    // Every model fills all maxErTaps slots even though only `count` of them
    // are audible: the morph crossfades tap i of one model into tap i of its
    // neighbour, and a slot with no time to interpolate towards would snap.
    // Slots past the count simply carry zero gain.
    // Total power each model's tap bank is normalized to.
    //
    // v0.4.2 divided every tap by sqrt(count), which holds the bank's power
    // roughly constant only while the envelope and the amplitude spread are the
    // same in every model — and from v0.5 they are not. Normalizing to a measured
    // total instead makes ErShape a purely tonal set of controls: a room can be
    // given eight savage discrete reflections instead of sixteen smooth ones
    // without its ER *level* moving, which is what lets the table's `erLevel`
    // column, the ER/late balance and Distance's calibration all carry over
    // untouched. The figure is the power the v0.4.2 formula produced on average
    // (mean decay^2 * mean jitter^2, which works out at 0.118), chosen for
    // exactly that continuity.
    static constexpr float erBankPower = 0.118f;

    void buildTapTables()
    {
        for (size_t mi = 0; mi < roomModels.size(); ++mi)
        {
            const auto& m = roomModels[mi];
            const auto& sh = m.er;
            auto& tt = tapTables[mi];

            Rng rng (m.seed);
            tt.count = juce::jlimit (1, maxErTaps, m.erTaps);

            const float span = juce::jmax (1.0f, m.erSpreadMs);
            // Discreteness is an exponent on the jitter, so 0 leaves the v0.4.2
            // spread alone and 1 turns [0.55, 1] into [0.05, 1].
            const float contrast = 1.0f + 4.0f * juce::jlimit (0.0f, 1.0f, sh.discreteness);
            const float panSpread = juce::jlimit (0.0f, 1.0f, sh.panSpread);

            float power = 0.0f;

            for (int t = 0; t < maxErTaps; ++t)
            {
                // The three draws stay in this order and at this count whatever
                // the shape fields say: the seeds are a compatibility surface,
                // and a model whose shape is left neutral has to come out of here
                // sample for sample as it did at v0.4.2.
                const float u = rng.next01();
                const float jitterU = rng.next01();
                const float panU = rng.next01();

                const float tMs = 1.0f + span * std::pow (u, juce::jmax (0.1f, sh.clustering));
                const float decay = std::exp (-juce::jmax (0.0f, sh.decayRate) * tMs / span);
                const float jitter = std::pow (0.55f + 0.45f * jitterU, contrast);
                const float sign = (t % 2 == 0) ? 1.0f : -1.0f;
                const float pan = juce::jlimit (0.0f, 1.0f,
                                                0.5f + (panU - 0.5f) * panSpread);
                const float live = t < tt.count ? 1.0f : 0.0f;
                const float amp = live * decay * jitter * sign;

                const auto ti = (size_t) t;
                tt.time[ti]  = juce::jmax (1.0f, tMs * 0.001f * (float) sampleRate);
                tt.gainL[ti] = amp * std::sqrt (1.0f - pan);
                tt.gainR[ti] = amp * std::sqrt (pan);
                tt.amp[ti]   = amp;
                tt.pan[ti]   = pan;
                tt.frac[ti]  = juce::jlimit (0.0f, 1.0f, tMs / span);

                power += amp * amp;   // gainL^2 + gainR^2 collapses to this
            }

            const float k = std::sqrt (erBankPower / juce::jmax (1.0e-9f, power));
            for (int t = 0; t < maxErTaps; ++t)
            {
                tt.gainL[(size_t) t] *= k;
                tt.gainR[(size_t) t] *= k;
                tt.amp[(size_t) t]   *= k;
            }
        }

        buildTapDirections();
    }

    // Where in the room each early reflection slot arrives from (v0.7).
    //
    // A reflection off the ceiling behind you is not "the left channel a bit
    // later", and once there are speakers behind and above the listener the
    // difference is the whole point of running the bank in surround at all. So
    // every tap slot gets a direction, and the model's own `pan` draw — which is
    // all v0.6 had — becomes the *lateral* component of it.
    //
    // The front/back and height components are properties of the **slot**, not
    // of the model, and that is deliberate. The Space morph crossfades tap i of
    // one room into tap i of its neighbour; if a tap's direction interpolated
    // too, every reflection in the room would swim from one wall to another
    // while the knob moved. Amplitude and lateral position interpolate because
    // they are levels; direction is structural, so it is drawn once from the
    // slot index and stays put.
    void buildTapDirections() noexcept
    {
        Rng rng (0x5eed1u);

        for (int t = 0; t < maxErTaps; ++t)
        {
            const auto ti = (size_t) t;

            // Roughly a third of the bank arrives from behind and a quarter
            // from above, which is the balance of a real room's early field:
            // the first arrivals are the near walls in front, and the ceiling
            // and rear come in behind them.
            tapRear[ti] = rng.next01() < 0.34f ? 1.0f : 0.0f;

            const float e = rng.next01();
            tapElevation[ti] = e < 0.26f
                ? juce::jmap (rng.next01(), 0.20f, 0.85f)   // radians, ~11..49 deg
                : 0.0f;
        }
    }

    // Per-model FDN line lengths in samples at sizeScale 1.0, carrying the
    // model's own `lineSpread`.
    //
    // Precomputed, and that is not an optimization detail but the thing that
    // makes the feature affordable at all: `evaluateGeometry` re-runs whenever
    // the morph position moves, which during a sweep is every single sample, and
    // sixteen pow() calls per sample is not a trade anyone would take. Because
    // the exponent is a per-model constant, the whole distribution can be built
    // once and the runtime left doing what it already did — one lerp and one
    // multiply per line.
    void buildDelayTables()
    {
        double logSum = 0.0;
        float baseSum = 0.0f;
        for (int l = 0; l < numLines; ++l)
        {
            logSum += std::log ((double) baseDelayMs[(size_t) l]);
            baseSum += baseDelayMs[(size_t) l];
        }

        // The pivot the spread works about, so that tilting the distribution
        // does not also move the room's centre of gravity.
        const float gm = (float) std::exp (logSum / (double) numLines);

        for (size_t mi = 0; mi < roomModels.size(); ++mi)
        {
            const float s = juce::jlimit (0.4f, 2.5f, roomModels[mi].late.lineSpread);

            std::array<float, numLines> ms {};
            float sum = 0.0f;
            for (int l = 0; l < numLines; ++l)
            {
                ms[(size_t) l] = gm * std::pow (baseDelayMs[(size_t) l] / gm, s);
                sum += ms[(size_t) l];
            }

            // Renormalize to the base set's total delay. An FDN has one mode per
            // (1 / total delay) hertz, so this is what makes `lineSpread` change
            // how the modes are *distributed* and never how many there are — and
            // it is what keeps totalLineDelaySeconds(), which the harness uses to
            // decide how low it may meaningfully measure coloration, exact for
            // every model rather than for the neutral one only.
            const float k = baseSum / juce::jmax (1.0e-6f, sum);
            for (int l = 0; l < numLines; ++l)
                modelDelaySamples[mi][(size_t) l] =
                    ms[(size_t) l] * k * 0.001f * (float) sampleRate;

            applyDensityFloor (mi);
        }
    }

    // The modal density floor (v0.7.1).
    //
    // An FDN has one mode per (1 / total delay) hertz. Through v0.7 the whole
    // line set scaled with `sizeScale`, so total delay scaled with it too, and
    // the smallest rooms ran a tank with 33 ms of total delay — one mode every
    // 30 Hz. Between 500 Hz and 8 kHz that is 250 modes where a real room of
    // that size has around 1300, and what 250 sharp resonances sound like is a
    // comb. Measured at 3 Hz resolution the small end rippled by 16-20 dB RMS
    // against the 5.6 dB a genuinely diffuse field gives, with 40 dB teeth. The
    // halls, which have four to twenty times the total delay, sit at 3.8-4.5 dB
    // and always did.
    //
    // The fix is to stop tying modal density to room size, because they are not
    // the same quantity. What makes a room sound small is when the first
    // reflections arrive and how quickly the field builds — the *shortest* path
    // — and that is left exactly where it was. What sets density is the total,
    // and the long lines are free to grow: a long line in a small room is not a
    // longer echo you can hear, it is one more mode.
    //
    // So the shortest line is pinned and the distribution is stretched by an
    // exponent until the total reaches the floor. Rooms already above it are
    // untouched, bit for bit — which is everything from the Chamber up, and is
    // why the halls' voicing, decay calibration and coloration figures carry
    // over unchanged.
    void applyDensityFloor (size_t mi) noexcept
    {
        const float sizeMi = roomModels[mi].sizeScale;
        const float floorSamples = densityFloorMs * 0.001f * (float) sampleRate;

        auto& set = modelDelaySamples[mi];

        float total = 0.0f, shortest = 1.0e9f;
        for (int l = 0; l < numLines; ++l)
        {
            const float d = set[(size_t) l] * sizeMi;
            total += d;
            shortest = juce::jmin (shortest, d);
        }

        modelDetuneScale[mi] = 1.0f;

        if (total >= floorSamples || shortest <= 0.0f)
            return;

        // Bisect the exponent. Monotonic in gamma, twenty iterations is exact
        // to a sample, and it happens once per model in prepare().
        auto totalAt = [&set, sizeMi, shortest] (float gamma)
        {
            float t = 0.0f;
            for (int l = 0; l < numLines; ++l)
            {
                const float d = set[(size_t) l] * sizeMi;
                t += shortest * std::pow (d / shortest, gamma);
            }
            return t;
        };

        float lo = 1.0f, hi = 8.0f;
        if (totalAt (hi) < floorSamples)
            lo = hi;   // cannot reach the floor; take the widest spread allowed

        for (int i = 0; i < 24 && hi - lo > 1.0e-4f; ++i)
        {
            const float mid = 0.5f * (lo + hi);
            (totalAt (mid) < floorSamples ? lo : hi) = mid;
        }

        const float gamma = 0.5f * (lo + hi);
        for (int l = 0; l < numLines; ++l)
        {
            const float d = set[(size_t) l] * sizeMi;
            set[(size_t) l] = shortest * std::pow (d / shortest, gamma) / sizeMi;
        }

        // How much denser the room got, turned into how much less modulation it
        // now needs. See the note in updateControlRate for the exponent.
        const float stretch = juce::jmax (1.0f, totalAt (gamma) / total);
        modelDetuneScale[mi] = 1.0f / std::pow (stretch, densityModExponent);
    }

    // Per-model input diffuser lengths, one set per feed. The detune between
    // feeds is baked in here rather than applied in the sample loop, which is
    // what lets it be a per-room figure without costing the loop anything.
    //
    // The room states one detune ratio, and the feeds are spread geometrically
    // across it: feed 0 is the model's own lengths and the last feed is those
    // lengths times the ratio, with the rest laid out evenly in between. At two
    // feeds that is exactly v0.6's L (unity) and R (the ratio) pair, and at four
    // it keeps the same *span* of decorrelation rather than widening it — a room
    // whose walls are 7% out of step does not become one whose walls are 21% out
    // of step because the session gained a pair of surrounds.
    void buildDiffuserTables()
    {
        double logSum = 0.0;
        for (float x : diffuserShapeMs)
            logSum += std::log ((double) x);
        const float gm = (float) std::exp (logSum / (double) numDiffusers);

        const float spanDivisor = (float) juce::jmax (1, numFeeds - 1);

        for (size_t mi = 0; mi < roomModels.size(); ++mi)
        {
            const auto& L = roomModels[mi].late;
            const float s = juce::jlimit (0.4f, 2.0f, L.diffSpread);
            const float mean = juce::jlimit (0.6f, 24.0f, L.diffMeanMs);
            const float detune = juce::jlimit (1.0f, 1.30f,
                                               roomModels[mi].stereo.diffDetune);

            for (int d = 0; d < numDiffusers; ++d)
            {
                // The shape set is normalized by its own geometric mean, so
                // `diffMeanMs` really is the mean length in milliseconds and the
                // two fields are independent.
                const float ms =
                    mean * std::pow (diffuserShapeMs[(size_t) d] / gm, s);
                const float samples = ms * 0.001f * (float) sampleRate;

                for (int f = 0; f < numFeeds; ++f)
                {
                    // f == 0 is pow(detune, 0) == 1 exactly, so feed 0 keeps the
                    // model's own lengths to the bit.
                    const float k = f == 0 ? 1.0f
                                           : std::pow (detune, (float) f / spanDivisor);
                    modelDiff[(size_t) f][mi][(size_t) d] =
                        juce::jmax (2.0f, samples * k);
                }
            }
        }
    }

    // Unnormalized Householder weights per model. Interpolating these and
    // normalizing the result (see updateControlRate) is what keeps the reflection
    // vector exactly unit-length at every point in the morph, which is what the
    // matrix needs to stay orthogonal and Freeze needs to stay lossless.
    void buildMixWeights()
    {
        for (size_t mi = 0; mi < roomModels.size(); ++mi)
        {
            const float skew = juce::jlimit (-3.0f, 3.0f, roomModels[mi].late.mixSkew);
            for (int l = 0; l < numLines; ++l)
            {
                // Positive skew weights the short lines, which makes them the
                // hubs energy passes through and leaves the long ones — the ones
                // carrying the low, slow, individually audible modes — closer to
                // independent.
                const float x = (float) l / (float) (numLines - 1) - 0.5f;
                modelMixWeight[mi][(size_t) l] = std::exp (-skew * x);
            }
        }
    }

    // The layout, resolved into everything the engine indexes by (v0.7).
    //
    // Two vocabularies meet here and it is worth being explicit about which is
    // which. The *buffer* channel is where the host put a speaker; the *tank
    // slot* is one of the room's wet channels. They differ by the LFE, which
    // has a buffer channel and no slot, and by mono, which has one buffer
    // channel and two slots — the tank has been two-wide since v0.1 and a mono
    // session folds it at the output rather than running half a room.
    void adoptLayout (const SpeakerLayout& newLayout)
    {
        layout = newLayout;
        layoutChannels = juce::jmax (1, layout.size());

        numIn = 0;
        std::array<int, maxChannels> slotOf {};
        slotOf.fill (-1);

        for (int c = 0; c < layout.size() && numIn < maxChannels; ++c)
        {
            if (layout[c].lfe)
                continue;
            slotOf[(size_t) c] = numIn;
            wetChan[(size_t) numIn] = c;
            ++numIn;
        }

        if (numIn == 0)
        {
            // Nothing but LFE. Not a layout anybody ships, but the engine must
            // not index off the end of an empty list to find that out.
            wetChan[0] = 0;
            numIn = 1;
        }

        foldMono = numIn == 1;
        numTank = juce::jmax (2, numIn);
        invNumTank = 1.0f / (float) numTank;
        erChanScale = std::sqrt ((float) numTank * 0.5f);

        for (int w = 0; w < numTank; ++w)
        {
            const auto wi = (size_t) w;

            if (foldMono)
            {
                // The internal pair a mono session folds down from. Placed at
                // the stereo angles because that is what they are: the two
                // halves of the same room, about to be averaged.
                const float az = (w == 0 ? -30.0f : 30.0f)
                                 * juce::MathConstants<float>::pi / 180.0f;
                tankSpeakers[wi] = { az, 0.0f, false, SpeakerGroup::front,
                                     w == 0 ? 1 : 0 };
            }
            else
            {
                tankSpeakers[wi] = layout[wetChan[wi]];
                // Mirrors arrive as buffer indices and have to become slots.
                const int m = tankSpeakers[wi].mirror;
                tankSpeakers[wi].mirror =
                    (m >= 0 && m < maxChannels) ? slotOf[(size_t) m] : -1;
            }

            speakerAzimuth[wi] = tankSpeakers[wi].azimuth;
            speakerElevation[wi] = tankSpeakers[wi].elevation;
        }

        // Two feeds below the surround threshold, four above it. The two-feed
        // directions are the stereo speaker angles so that a stereo layout's
        // channels land on their own feed at unity; the four are the corners of
        // the room, which is the coarsest partition that still tells front from
        // back and left from right.
        numFeeds = numTank <= 2 ? 2 : 4;

        // Two components below the surround threshold (left-to-left and
        // right-to-right), one above it (a direction taken from the feeds and
        // fanned out to the speakers).
        numErSources = numFeeds == 2 ? 2 : 1;

        constexpr float d2r = juce::MathConstants<float>::pi / 180.0f;
        if (numFeeds == 2)
            feedAzimuth = { -30.0f * d2r, 30.0f * d2r, 0.0f, 0.0f };
        else
            feedAzimuth = { -45.0f * d2r, 45.0f * d2r, -135.0f * d2r, 135.0f * d2r };
    }

    // Input channels to spatial feeds.
    //
    // At two feeds this is not a pan at all but a side assignment: anything left
    // of centre goes to feed 0 at unity, anything right to feed 1, and anything
    // centred to both. That is what v0.1 did with a stereo input and what it did
    // with a mono one, and it is kept because a Mix and a wet level calibrated
    // against it are the ones every preset in the library was written to.
    //
    // At four feeds it is a constant-power pan across the two nearest corners,
    // so the energy a channel puts into the room does not depend on where in the
    // room it is — which is the property that keeps a 5.1 session's wet level
    // the same as a stereo one's.
    void buildFeedMatrix() noexcept
    {
        for (auto& row : feedMix)
            row.fill (0.0f);

        for (int c = 0; c < numIn; ++c)
        {
            const auto ci = (size_t) c;
            const float az = foldMono ? 0.0f : layout[wetChan[ci]].azimuth;

            if (numFeeds == 2)
            {
                constexpr float centred = 1.0e-4f;
                if (az < -centred)      feedMix[ci][0] = 1.0f;
                else if (az > centred)  feedMix[ci][1] = 1.0f;
                else                    feedMix[ci][0] = feedMix[ci][1] = 1.0f;
                continue;
            }

            // Nearest two corners by angle round the circle, then a
            // constant-power split between them.
            int best = 0, second = 1;
            float bestD = 1.0e9f, secondD = 1.0e9f;

            for (int f = 0; f < numFeeds; ++f)
            {
                const float d = std::abs (wrapPi (az - feedAzimuth[(size_t) f]));
                if (d < bestD)       { secondD = bestD; second = best; bestD = d; best = f; }
                else if (d < secondD) { secondD = d; second = f; }
            }

            const float span = juce::jmax (1.0e-6f, bestD + secondD);
            const float x = bestD / span;   // 0 = on the near corner
            feedMix[ci][(size_t) best]   = std::sqrt (juce::jmax (0.0f, 1.0f - x));
            feedMix[ci][(size_t) second] = std::sqrt (juce::jmax (0.0f, x));
        }
    }

    // Which feed each FDN line takes its injection from.
    //
    // The two-feed split is v0.5's Walsh row 4 — lines 0-3 and 8-11 from the
    // left, 4-7 and 12-15 from the right — kept exactly, because it is what
    // every stereo measurement in the harness was taken against. The four-feed
    // split interleaves by index instead, so each corner enters the tank through
    // the whole range of line lengths rather than through one end of it; taking
    // the next Walsh bit would have given the front-left corner the four
    // shortest lines and the rear-right the four longest, which is a tonal
    // difference between speakers rather than a spatial one.
    void buildInjection() noexcept
    {
        for (int l = 0; l < numLines; ++l)
            lineFeed[(size_t) l] = numFeeds == 2
                ? (detail::walsh (l, 4) > 0.0f ? 0 : 1)
                : (l % numFeeds);

        // Every feed reaches the same number of lines, so the trim that keeps
        // the tank's input energy independent of the feed count is exactly one
        // over the root of that number: 1/sqrt(8) at two feeds, which is v0.1's
        // constant, and 1/sqrt(4) at four.
        const int linesPerFeed = juce::jmax (1, numLines / numFeeds);
        injectTrimNow = 1.0f / std::sqrt ((float) linesPerFeed);
    }

    // One Walsh row per speaker, rotated towards its neighbours by the room's
    // cross-feed (v0.7).
    //
    // Rebuilt at control rate because the amount of rotation is the room's, and
    // the room moves with Space. Fifteen rows of sixteen coefficients is a
    // couple of hundred multiplies a tick, against sixteen fractional delay
    // reads and sixteen filters *per sample* in the loop it feeds — it does not
    // register.
    void buildOutputVectors (float roomCorrelation) noexcept
    {
        if (numTank <= 2)
        {
            // The v0.5 pair, exactly: the left tap is Walsh row 1 and the right
            // is row 2 rotated towards it. Kept rather than derived from the
            // general case below because the general case is symmetric — it
            // rotates both rows towards each other — and a stereo session that
            // measured one way for two releases should not start measuring
            // another because the code around it grew.
            const float phi = juce::jlimit (0.0f, 1.0f, roomCorrelation)
                              * juce::MathConstants<float>::halfPi;
            const float c = std::cos (phi), s = std::sin (phi);

            for (int l = 0; l < numLines; ++l)
            {
                const auto li = (size_t) l;
                outVec[0][li] = outSignL[li];
                outVec[1][li] = c * outSignR[li] + s * outSignL[li];
            }
            return;
        }

        // Angular cross-feed. Two speakers close together hear nearly the same
        // field and two facing opposite walls hear almost independent ones, so
        // the target correlation falls off exponentially with the angle between
        // them, from a near-field value the room itself sets: a small, correlated
        // room couples its speakers hard and a cathedral barely at all.
        //
        // Turning a *correlation* into a set of row weights is the one piece of
        // arithmetic worth spelling out. Rows are orthonormal, so the
        // correlation between two mixes of them is the dot product of their
        // weight vectors; for a pair carrying weight 1 on their own row and g on
        // each other's, normalized, that dot product is 2g/(1+g^2), which is
        // sin(2 atan g). Inverting gives g = tan(asin(rho)/2), and at rho = 1 it
        // is exactly 1 — both speakers end up reading the identical row, which
        // is what perfect correlation has to mean.
        const float near = juce::jlimit (0.0f, 0.95f,
                                         crossfeedFloor + crossfeedRoom * roomCorrelation);

        for (int w = 0; w < numTank; ++w)
        {
            const auto wi = (size_t) w;
            std::array<float, maxChannels> weight {};

            for (int k = 0; k < numTank; ++k)
            {
                const auto ki = (size_t) k;
                float g;

                if (k == w)
                {
                    g = 1.0f;
                }
                else
                {
                    const float ang = angleBetweenSlots (w, k);
                    const float rho = juce::jlimit (0.0f, 0.95f,
                                                    near * std::exp (-ang / crossfeedSpread));
                    g = std::tan (0.5f * std::asin (rho));
                }

                weight[ki] = g;
            }

            float mag = 0.0f;
            std::array<float, numLines> row {};

            for (int l = 0; l < numLines; ++l)
            {
                const auto li = (size_t) l;
                float acc = 0.0f;
                for (int k = 0; k < numTank; ++k)
                    acc += weight[(size_t) k]
                           * detail::walsh (l, walshRowOrder[(size_t) k]);

                row[li] = acc;
                mag += acc * acc;
            }

            // Normalized to a bare Walsh row's own magnitude — sixteen entries
            // of unit size — rather than to the weight vector. That covers the
            // cross-feed rotation and the emphasis in one step, and it is what
            // makes both of them free: whatever they do to *which* lines a
            // speaker hears, they never change how much of the tank it hears.
            const float norm = std::sqrt ((float) numLines / juce::jmax (1.0e-12f, mag));
            for (int l = 0; l < numLines; ++l)
                outVec[wi][(size_t) l] = row[(size_t) l] * norm;
        }
    }

    float angleBetweenSlots (int a, int b) const noexcept
    {
        const auto& p = tankSpeakers[(size_t) a];
        const auto& q = tankSpeakers[(size_t) b];
        const float dot = std::cos (p.elevation) * std::cos (q.elevation)
                              * std::cos (p.azimuth - q.azimuth)
                          + std::sin (p.elevation) * std::sin (q.elevation);
        return std::acos (juce::jlimit (-1.0f, 1.0f, dot));
    }

    static float wrapPi (float x) noexcept
    {
        constexpr float twoPi = juce::MathConstants<float>::twoPi;
        while (x > juce::MathConstants<float>::pi)  x -= twoPi;
        while (x < -juce::MathConstants<float>::pi) x += twoPi;
        return x;
    }

    void flushTank() noexcept
    {
        for (auto& line : lines)
            line.clear();
        for (int f = 0; f < numFeeds; ++f)
        {
            erLines[(size_t) f].clear();
            for (auto& ap : diffusers[(size_t) f])
                ap.clear();
        }
        for (auto& d : damping)
            d.clear();
        decayEq.reset();
        postEq.reset();
        // The predelay lines carry the dry input and know nothing about room
        // geometry, so they keep running.
    }

    void advanceRelocation() noexcept
    {
        if (relocatePending)
        {
            wetFade -= fadeStep;
            if (wetFade <= 0.0f)
            {
                wetFade = 0.0f;
                relocatePending = false;
                pos = targetPos;
                flushTank();
                evaluateGeometry();
                decayEq.snapToTargets();
                postEq.snapToTargets();
                // The decay time normally glides over ~120 ms, which is right
                // for a knob and wrong for a preset: arrive at the new room's
                // decay, not on the way to it.
                snapRt60 = true;
                controlCountdown = 0;
            }
        }
        else if (wetFade < 1.0f)
        {
            wetFade = juce::jmin (1.0f, wetFade + fadeStep);
        }
    }

    // See maxLogShiftPerDecay for the derivation. In table units the rule is
    //     dpos/dt = (logRate * size) / (dsize/dpos)
    // with logRate itself inversely proportional to the decay time, then
    // clamped at both ends and backstopped by the pointer-rate safety net.
    float morphStepPerSample() const noexcept
    {
        // Slope of the size curve here, taken as the steepest of the intervals
        // either side so the limit never lags a change of gradient.
        const int i = juce::jlimit (0, numRoomModels - 2, (int) pos);
        float dSize = 0.0f;
        for (int j = juce::jmax (0, i - 1); j <= juce::jmin (numRoomModels - 2, i + 1); ++j)
            dSize = juce::jmax (dSize, std::abs (roomModels[(size_t) (j + 1)].sizeScale
                                                 - roomModels[(size_t) j].sizeScale));
        dSize = juce::jmax (1.0e-4f, dSize);

        const float logRate = maxLogShiftPerDecay * 4.0f
                              / juce::jmax (0.15f, currentRt60);
        float step = logRate * juce::jmax (0.02f, sizeNow) / dSize / (float) sampleRate;

        // The pointer-rate safety net, against the longest line *currently*
        // running rather than the longest in the table. It has to be the current
        // one from v0.7.1: the density floor makes a small room's longest line a
        // large multiple of its own size, so a figure taken across the whole
        // table would throttle the morph everywhere on account of a pairing that
        // only exists at the small end.
        step = juce::jmin (step, maxPointerRate / (dSize * juce::jmax (1.0f, longestNow)));
        return juce::jlimit (minPosStep, maxPosStep, step);
    }

    void advanceMorph (float frozen) noexcept
    {
        const float delta = targetPos - pos;
        if (delta == 0.0f)
            return;

        // A frozen room holds its geometry: the tank is sealed and lossless
        // only while the reads stay put.
        const float step = morphStepPerSample() * (1.0f - frozen);

        pos += juce::jlimit (-step, step, delta);
        if (std::abs (targetPos - pos) < 1.0e-6f)
            pos = targetPos;
    }

    // The catch-up rule, and why it is expressed as a time rather than as a
    // distance.
    //
    // The rate limit above is a pitch-artifact budget, so a fast drag across
    // the table genuinely cannot be followed continuously — the engine arrives
    // seconds after the knob does, which reads as the control being broken
    // rather than as a feature. But the threshold cannot be a fixed number of
    // models either: the same two-model move is a fifth of a second on the
    // small rooms and four seconds on the cathedrals, and only the second one
    // is a problem.
    //
    // So the test is "how long would finishing this glide take?", measured at
    // the rate limit currently in force, and the answer is only acted on once
    // the target has *stopped moving*. That ordering is what keeps the morph:
    // while the knob is under a finger the engine sweeps continuously and you
    // hear the room grow, and only when you let go — with more than
    // `relocateSecondsBudget` of glide still owed — does it stop pretending
    // and take the relocation path instead.
    static constexpr float relocateSecondsBudget = 0.40f;
    static constexpr float targetSettleSeconds = 0.08f;

    void checkMorphCatchUp() noexcept
    {
        if (relocatePending || wetFade < 1.0f)
            return;

        if (! juce::approximatelyEqual (targetPos, lastSeenTargetPos))
        {
            lastSeenTargetPos = targetPos;
            targetStillSamples = 0;
            return;
        }

        targetStillSamples += controlBlock;
        if ((float) targetStillSamples < targetSettleSeconds * (float) sampleRate)
            return;

        const float gap = std::abs (targetPos - pos);
        if (gap <= 1.0e-3f)
            return;

        const float modelsPerSecond = morphStepPerSample() * (float) sampleRate;
        if (gap / juce::jmax (1.0e-6f, modelsPerSecond) > relocateSecondsBudget)
            relocatePending = true;
    }

    // Everything that must be exact every sample. Only re-run when the morph
    // position actually moved, so a parked Space costs nothing.
    void evaluateGeometry() noexcept
    {
        const float p = juce::jlimit (0.0f, (float) (numRoomModels - 1), pos);
        morphLow = juce::jlimit (0, numRoomModels - 1, (int) p);
        morphHigh = juce::jmin (morphLow + 1, numRoomModels - 1);
        morphFrac = juce::jlimit (0.0f, 1.0f, p - (float) morphLow);

        const auto& A = roomModels[(size_t) morphLow];
        const auto& B = roomModels[(size_t) morphHigh];
        const float f = morphFrac;

        const float size = A.sizeScale + (B.sizeScale - A.sizeScale) * f;
        sizeNow = size;

        // Each model brings its own distribution of the sixteen lengths now, so
        // what interpolates is the *set* and not just the scale it is multiplied
        // by. With every `lineSpread` at 1.0 both sets are the base set and this
        // reduces exactly to the v0.4.2 `baseDelaySamples[l] * size`.
        const auto& DA = modelDelaySamples[(size_t) morphLow];
        const auto& DB = modelDelaySamples[(size_t) morphHigh];
        longestNow = 0.0f;
        for (int l = 0; l < numLines; ++l)
        {
            const auto li = (size_t) l;
            const float base = DA[li] + (DB[li] - DA[li]) * f;
            delaySamples[li] = juce::jmax (8.0f, base * size);
            longestNow = juce::jmax (longestNow, delaySamples[li]);
        }

        // The diffusers no longer track size at all: their lengths are stated per
        // model. v0.4.2 derived them from `jlimit(0.35, 1.25, size * 1.4)`, and
        // the floor on that clamp meant every model from Ambience to Chamber ran
        // the identical cascade — six rooms spanning a 5.5:1 range of size, all
        // diffusing the same way. The two channels' sets are separate because the
        // L/R detune is per model too.
        for (int fd = 0; fd < numFeeds; ++fd)
        {
            const auto fi = (size_t) fd;
            const auto& DA2 = modelDiff[fi][(size_t) morphLow];
            const auto& DB2 = modelDiff[fi][(size_t) morphHigh];

            for (int s = 0; s < numDiffusers; ++s)
            {
                const auto si = (size_t) s;
                diffDelay[fi][si] =
                    juce::jmax (2.0f, DA2[si] + (DB2[si] - DA2[si]) * f);
            }
        }

        const auto& TA = tapTables[(size_t) morphLow];
        const auto& TB = tapTables[(size_t) morphHigh];
        for (int t = 0; t < maxErTaps; ++t)
        {
            const auto ti = (size_t) t;
            tapTime[ti] = TA.time[ti] + (TB.time[ti] - TA.time[ti]) * f;
        }

        evaluatedPos = pos;
    }

    // Tap gains only ever multiply the ER sum, so a step in one is a step of
    // (delta * signal) — inaudible at control rate — while a step in a tap
    // *time* is a click. That split is why the two live in different places.
    void updateErGains() noexcept
    {
        const auto& TA = tapTables[(size_t) morphLow];
        const auto& TB = tapTables[(size_t) morphHigh];
        const float f = morphFrac;

        erTapLimit = juce::jmax (TA.count, TB.count);

        // Character's late-reflection emphasis is kept *out* of the tap gains,
        // so the control that scales it can be a single per-sample smoothed
        // scalar. Baking it in here would make it step at control rate, and a
        // step of up to 3.2x on the whole early bank is the loudest zipper in
        // the engine — this was measured, not guessed. Through v0.6 it was a
        // second gain vector per output channel; from v0.7 it is a per-tap
        // scalar instead, because it scales every speaker's share of a tap
        // equally and fifteen copies of one number is fifteen copies too many.
        for (int t = 0; t < erTapLimit; ++t)
        {
            const auto ti = (size_t) t;
            const float fr = TA.frac[ti] + (TB.frac[ti] - TA.frac[ti]) * f;
            tapLively[ti] = 2.2f * fr;
        }

        if (numFeeds == 2)
        {
            // Mono and stereo: two rank-one components, "left input to the left
            // output" and "right to the right". This is v0.6's bank written in
            // the general form, and it evaluates to the same arithmetic.
            for (int t = 0; t < erTapLimit; ++t)
            {
                const auto ti = (size_t) t;
                const float gl = TA.gainL[ti] + (TB.gainL[ti] - TA.gainL[ti]) * f;
                const float gr = TA.gainR[ti] + (TB.gainR[ti] - TA.gainR[ti]) * f;

                erFeedW[0][ti] = { 1.0f, 0.0f, 0.0f, 0.0f };
                erFeedW[1][ti] = { 0.0f, 1.0f, 0.0f, 0.0f };

                erChanW[0][ti].fill (0.0f);
                erChanW[1][ti].fill (0.0f);
                erChanW[0][ti][0] = gl;
                erChanW[1][ti][1] = gr;
            }
            return;
        }

        // Surround: one component per tap, taking from the feeds by how close
        // each is to the tap's own direction and fanning out to the speakers the
        // same way. Both weight sets are unit-power, so a tap carries the energy
        // of one feed however many feeds there are, and `erChanScale` then puts
        // the per-speaker level where a stereo session's would be — without it
        // the early field would thin out by 3 dB every time the speaker count
        // doubled while the late field, which gives every speaker a full Walsh
        // row, held its level.
        for (int t = 0; t < erTapLimit; ++t)
        {
            const auto ti = (size_t) t;

            const float amp = TA.amp[ti] + (TB.amp[ti] - TA.amp[ti]) * f;
            const float pan = TA.pan[ti] + (TB.pan[ti] - TA.pan[ti]) * f;

            // Lateral position from the model's pan draw, front/back and height
            // from the slot. asin maps the pan onto the sphere so that an evenly
            // spread pan is an evenly spread *direction*.
            const float lateral = juce::jlimit (-1.0f, 1.0f, 2.0f * pan - 1.0f);
            float az = std::asin (lateral);
            if (tapRear[ti] > 0.5f)
                az = juce::MathConstants<float>::pi - az;
            const float el = tapElevation[ti];

            unitPowerWeights (az, el, feedAzimuth.data(), nullptr, numFeeds,
                              feedFocus, erFeedW[0][ti].data());

            for (int w = 0; w < numTank; ++w)
                erChanW[0][ti][(size_t) w] = 0.0f;

            std::array<float, maxChannels> pans {};
            unitPowerWeights (az, el, speakerAzimuth.data(), speakerElevation.data(),
                              numTank, speakerFocus, pans.data());

            for (int w = 0; w < numTank; ++w)
                erChanW[0][ti][(size_t) w] = amp * pans[(size_t) w] * erChanScale;

            // The second component is unused above two feeds; zeroing its
            // weights rather than branching on a count in the sample loop keeps
            // the inner loop straight.
            erFeedW[1][ti].fill (0.0f);
            erChanW[1][ti].fill (0.0f);
        }
    }

    // Weights for a direction against a set of directions, normalized to unit
    // power. A Gaussian in the angle rather than a cosine power: a cosine law
    // has a long tail round the back of the head, and with sixteen speakers that
    // tail is what turns a discrete reflection into a wash present everywhere.
    // `focus` is the angle at which a speaker's weight is down to 1/e.
    static void unitPowerWeights (float az, float el,
                                  const float* dirAz, const float* dirEl,
                                  int count, float focus, float* out) noexcept
    {
        const float cosEl = std::cos (el), sinEl = std::sin (el);
        float sum = 0.0f;

        for (int k = 0; k < count; ++k)
        {
            const auto ki = (size_t) k;
            const float ke = dirEl != nullptr ? dirEl[ki] : 0.0f;
            const float dot = cosEl * std::cos (ke) * std::cos (az - dirAz[ki])
                              + sinEl * std::sin (ke);
            const float ang = std::acos (juce::jlimit (-1.0f, 1.0f, dot));
            const float w = std::exp (-(ang * ang) / (focus * focus));

            out[ki] = w;
            sum += w * w;
        }

        const float inv = 1.0f / std::sqrt (juce::jmax (1.0e-12f, sum));
        for (int k = 0; k < count; ++k)
            out[(size_t) k] *= inv;
    }

    void updateControlRate() noexcept
    {
        const auto m = morphBetween (morphLow, morphHigh, morphFrac);

        checkMorphCatchUp();

        // From v0.4.1 the decay time is the control's, not the room's: the model
        // table's `baseRt60` no longer reaches the engine at all. It survives
        // as the room's natural decay for the factory presets to be authored
        // against and for the panel to suggest, which is the only place a
        // number like "a cathedral rings for six seconds" belongs once the
        // user can ask for any of them.
        targetRt60 = juce::jlimit (0.05f, 60.0f, decaySecondsTarget);
        if (snapRt60)
        {
            currentRt60 = targetRt60;
            snapRt60 = false;
        }
        currentRt60 += (targetRt60 - currentRt60) * rt60Slew;
        const float rt = juce::jmax (0.02f, currentRt60);

        // The damping filter is set before the Decay Rate EQ is designed,
        // because it *is* the baseline the EQ's multipliers are stated
        // against. The corner ceiling is a fixed audio-band figure rather than
        // a fraction of Nyquist: pinning it to the sample rate meant a bright
        // model at Brightness 100% absorbed measurably differently at 44.1 kHz
        // and at 96 kHz, which is not something a room does.
        const float hfCorner = juce::jlimit (400.0f, juce::jmin (20000.0f,
                                                                 (float) (sampleRate * 0.45)),
                                             m.hfDampHz);
        const float lpTarget = DampingFilter::coeffFor (hfCorner, sampleRate);
        const float hpTarget = DampingFilter::coeffFor (m.lfDampHz, sampleRate);
        smoothLpCoeff.setTargetValue (lpTarget);
        smoothHpCoeff.setTargetValue (hpTarget);

        // Shelf gains from decay *ratios*, which is what makes the room's
        // voicing independent of its length.
        //
        // A band that should ring for `r` times the mid band's time needs the
        // loop to take `A(1/r - 1)` more decibels off it per pass, where A is
        // the mid band's own per-pass attenuation. A is proportional to the
        // line's length, so strictly this wants a gain per line; using the
        // geometric mean instead leaves a spread across the sixteen lines that
        // is smaller than the one v0.4's shared lowpass already had, and costs
        // nothing per sample. What the ear gets is the average, and the harness
        // measures the average.
        //
        // Brightness now moves the *ratio* rather than the corner: at 100% the
        // top end rings as long as the mid (a perfectly reflective room), at 0%
        // a fraction of it. Moving the corner instead, as v0.4 did, changed
        // where the absorption started rather than how much there was, which is
        // why it fought the Decay Rate EQ's high shelf instead of composing
        // with it.
        // The mean is arithmetic from v0.5, and the change is not cosmetic.
        //
        // It used to be the geometric mean of the shortest and longest lines,
        // which was a fair enough stand-in while every room shared one set of
        // length *ratios*. `lineSpread` ends that: it moves the two extremes apart
        // in log space while holding the total delay fixed, so a geometric mean of
        // the extremes tracks the renormalization factor and drifts with the
        // spread — the same room would be voiced differently for no reason other
        // than how its modes are distributed. The arithmetic mean is exactly
        // invariant here, because preserving the sum is precisely what
        // buildDelayTables() does, so a room's spectral voicing and its modal
        // distribution are now independent choices. It was measurable: it is what
        // pushed the 5 kHz Brightness ratio off its (already marginal) bound.
        // ...and from v0.7.1 it is not a mean at all: the shelf gains are
        // designed per line.
        //
        // A shelf in the feedback path takes a fixed loss *per pass*, so how
        // much it takes per *second* depends on how often the line goes round —
        // which is to say on the line's length. One shared gain therefore only
        // delivers the room's stated decay ratios if the lines are all much the
        // same length. Through v0.7 they spanned 4.4:1 and the note above
        // records that the resulting spread was tolerable. The density floor
        // takes the small rooms to 44:1, and at that spread it is not: the long
        // lines lose a fortieth of the absorption per second that the short ones
        // do, so the damped bands ring on well past the decay that was asked for
        // (Ambience measured 2.23 s for a 2.00 s setting).
        //
        // Stating the loss per line, proportional to that line's own length, is
        // both the fix and the thing the feedback gain has always done — see
        // DecayEq's `flat`. It costs sixteen pow() calls a control tick.
        const float hfRatio = juce::jlimit (0.02f, 0.90f,
                                            m.hfDecayRatio * brightnessRatioMul);
        const float lfRatio = m.lfDecayRatio;

        // Brightness stops short of 1.0 deliberately: at exactly unity the top
        // octave rings for as long as the mid band, which no room does and
        // which the transparency criteria notice — the tail's own comb peaks
        // survive up there instead of being absorbed away, and the two
        // sparsest models fail the coloration bound. A ceiling of 0.9 is still
        // an extremely live room and keeps the criteria met across the table.
        for (int l = 0; l < numLines; ++l)
        {
            const auto li = (size_t) l;
            const double d = juce::jmax (1.0, (double) delaySamples[li]);
            const double atten = 60.0 * d / (sampleRate * (double) rt);

            auto gainFor = [atten] (float ratio)
            {
                const double r = juce::jlimit (0.02, 1.0, (double) ratio);
                return (float) juce::jlimit (0.0, 1.0,
                                             std::pow (10.0, -0.05 * atten * (1.0 / r - 1.0)));
            };

            hfGainLine[li] = gainFor (hfRatio);
            lfGainLine[li] = gainFor (lfRatio);
        }

        // Feedback gains come out of the Decay Rate EQ's design rather than
        // straight from the decay time. With every band at 1.0x the design
        // asks for exactly the flat attenuation and this is the v0.2 formula
        // written differently; with a band lifted it lowers the whole line's
        // attenuation and attenuates the other bands back down, which is what
        // makes the section stable by construction. See DecayEq.h.
        decayEq.update (delaySamples.data(), rt, lpTarget, hfGainLine.data(),
                        hpTarget, lfGainLine.data());
        decayEqActive = decayEq.isActive();

        for (int l = 0; l < numLines; ++l)
            feedbackGain[(size_t) l] = juce::jlimit (
                0.0f, 0.9999f,
                std::pow (10.0f, -0.05f * decayEq.feedbackAttenDb (l)));

        postEq.update();
        postEqActive = postEq.isActive();

        // The feedback matrix's reflection vector.
        //
        // Interpolate the two neighbouring models' weights, then normalize — that
        // order matters. Normalizing after the blend makes v exactly unit-length
        // at every point in the morph, so the matrix is exactly orthogonal
        // everywhere rather than only at the fourteen table entries; blending two
        // already-normalized vectors would leave it short by up to a percent in
        // between, and a tank whose matrix loses a percent per pass is not a tank
        // Freeze can hold.
        //
        // Control rate is the right rate for it for the same reason it is right
        // for the feedback gains: a step in a mixing coefficient is a
        // multiplication on an already-smooth signal, and holding v fixed between
        // ticks means it is always precisely the normalized vector rather than a
        // per-sample interpolate that would drift off the unit sphere.
        {
            const auto& WA = modelMixWeight[(size_t) morphLow];
            const auto& WB = modelMixWeight[(size_t) morphHigh];
            const float f = morphFrac;

            float norm = 0.0f;
            for (int l = 0; l < numLines; ++l)
            {
                const auto li = (size_t) l;
                const float w = WA[li] + (WB[li] - WA[li]) * f;
                mixV[li] = w;
                norm += w * w;
            }

            const float inv = 1.0f / std::sqrt (juce::jmax (1.0e-12f, norm));
            for (int l = 0; l < numLines; ++l)
                mixV[(size_t) l] *= inv;
        }

        // The late field's spatial correlation — a stereo image below three
        // speakers and a cross-feed matrix above it, from the same room field
        // and through the same construction. See buildOutputVectors.
        //
        // Only rebuilt when the figure actually moves. It is a control-rate cost
        // either way, but a parked Space with a parked room is the common case
        // and it should cost nothing at all.
        if (! juce::approximatelyEqual (m.lateCorrelation, lastCorrelation))
        {
            lastCorrelation = m.lateCorrelation;
            buildOutputVectors (m.lateCorrelation);
        }

        // Modulation, scaled by what the density floor already did for free.
        //
        // Whether a moving mode reads as lushness or as warble is decided by how
        // far it moves compared with how far apart the modes are: the ratio is
        // (frequency * detune * total delay), and the density floor multiplies
        // the last of those by up to five and a half. Left alone, a detune that
        // was inaudible in a 33 ms tank becomes a plainly audible wobble in a
        // 180 ms one — the harness caught it as a held 500 Hz tone spreading
        // into sidebands, concentration 0.87 against a 0.95 bound.
        //
        // Dividing the detune by the same factor holds the ratio where the
        // v0.4.2 voicing put it. It is the right direction as well as the
        // convenient one: modulation was there to smear a sparse modal field,
        // and a room that no longer has a sparse modal field no longer needs it.
        // The rooms above the floor are untouched, so the halls keep every bit
        // of their movement.
        const float sA = modelDetuneScale[(size_t) morphLow];
        const float sB = modelDetuneScale[(size_t) morphHigh];

        modDetuneNow = m.modDetune * (sA + (sB - sA) * morphFrac);
        smoothModDepth.setTargetValue (charDepthMul);
        for (int l = 0; l < numLines; ++l)
            modInc[(size_t) l] = 2.0f * m.modRateHz * charRateMul
                                 * modRateScale[(size_t) l] / (float) sampleRate;

        smoothErLevel.setTargetValue (m.erLevel * charErMul * distErMul);
        smoothLateLevel.setTargetValue (m.lateLevel * distLateMul * thickLateMul);

        smoothDiffGain.setTargetValue (juce::jlimit (0.20f, 0.82f,
            (0.45f + 0.35f * m.diffusion) * charDiffMul * thickGainMul));
        smoothStages.setTargetValue (juce::jlimit (1.0f, (float) numDiffusers,
            m.diffuseStages * charStageMul * thickStageMul));

        updateErGains();
    }

    // The decay the engine actually delivers at 1 kHz.
    //
    // v0.2 derived this from the one damping filter's magnitude at 1 kHz and
    // the per-line feedback gains. That derivation was per-filter and cannot
    // survive six more sections in the loop, so v0.3 replaces it rather than
    // extending it: the Decay Rate EQ already knows the total per-pass
    // attenuation each line carries at every report frequency, because that is
    // exactly what it designed. The number on the panel is now read out of the
    // design instead of reconstructed from it — and the same array is the
    // whole decay-vs-frequency curve v0.4's overlay wants.
    float estimateDeliveredRt60() const noexcept
    {
        return juce::jlimit (0.0f, 240.0f, decayEq.deliveredMidDecay());
    }

    // A reverb that goes non-finite never recovers on its own — every sample
    // after it is NaN. Cheap insurance, checked once per block.
    void sanitize() noexcept
    {
        bool bad = false;

        for (auto& d : damping)
            if (! std::isfinite (d.hfState) || ! std::isfinite (d.lfState))
                bad = true;
        for (float v : tiltState)
            if (! std::isfinite (v))
                bad = true;
        for (float v : inputDampState)
            if (! std::isfinite (v))
                bad = true;

        if (! decayEq.allFinite() || ! postEq.allFinite())
            bad = true;

        if (bad)
            reset();
    }

    std::array<juce::SmoothedValue<float>*, 14> allSmoothers() noexcept
    {
        return { &smoothMix, &smoothWidth, &smoothErLevel, &smoothLateLevel,
                 &smoothTilt, &smoothLpCoeff, &smoothHpCoeff, &smoothDiffGain,
                 &smoothStages, &smoothErTime, &smoothWetTrim, &smoothModDepth,
                 &smoothInputDamp, &smoothLively };
    }

    double sampleRate = 48000.0;

    // Morph state -------------------------------------------------------------
    float pos = 0.0f, targetPos = 0.0f, evaluatedPos = -1.0f;
    float sizeNow = 0.5f;
    float longestLineSamples = 1.0f, longestNow = 1.0f;
    float maxPosStep = 1.0e-4f, minPosStep = 1.0e-5f;
    int morphLow = 0, morphHigh = 0;
    float morphFrac = 0.0f;
    bool relocatePending = false, snapRt60 = false;
    float wetFade = 1.0f, fadeStep = 0.002f;
    int controlCountdown = 0;

    // Geometry, all continuous functions of `pos` ------------------------------
    std::array<ModelTaps, roomModels.size()> tapTables {};

    std::array<float, numLines> delaySamples {};
    std::array<std::array<float, numDiffusers>, maxFeeds> diffDelay {};

    // The feedback matrix's reflection vector: one hot array, and the only one
    // v0.5 adds to the sample loop.
    //
    // Getting it down to one took two rewrites. The doubling in H = I - 2vv^T is
    // folded into the projection scalar rather than kept as a second vector, and
    // the stereo rotation is applied to the two output sums instead of to the
    // vectors feeding them (see the sample loop), which keeps both Walsh rows
    // constexpr. The first cut held five runtime arrays here and cost the neutral
    // engine 60% of its budget for it.
    std::array<float, numLines> mixV {};
    std::array<float, maxErTaps> tapTime {}, tapLively {};
    int erTapLimit = 0;

    // The early bank's routing, rebuilt at control rate: for each tap, up to two
    // rank-one components of (which feeds it takes from) x (which speakers it
    // lands on). See updateErGains for why it is factored this way — dense over
    // both would be feeds x speakers multiplies a tap instead of feeds plus
    // speakers, which at 9.1.6 is the difference between sixty and nineteen.
    static constexpr int numErComponents = 2;
    std::array<std::array<std::array<float, maxFeeds>, maxErTaps>, numErComponents> erFeedW {};
    std::array<std::array<std::array<float, maxChannels>, maxErTaps>, numErComponents> erChanW {};
    int numErSources = 2;

    // One output row per speaker, cross-fed. The hot array v0.5 went to some
    // trouble to avoid and v0.7 could not: see buildOutputVectors.
    std::array<std::array<float, numLines>, maxChannels> outVec {};

    std::array<DelayLine, numLines> lines;
    std::array<DampingFilter, numLines> damping {};
    std::array<float, numLines> modInc {};
    std::array<float, numLines> modPhase {};
    std::array<float, numLines> feedbackGain {};

    // Per-line damping shelf gains (v0.7.1). One shared pair only delivers the
    // room's decay ratios while the lines are all much the same length; see
    // updateControlRate.
    std::array<float, numLines> hfGainLine {}, lfGainLine {};
    std::array<int, numLines> lineFeed {};
    float modDetuneNow = 0.0f;
    float injectTrimNow = 0.3536f;

    // The rate-limited predelay read position, and its ceiling in samples of
    // movement per sample. See the note at the read site: the limit exists
    // because a delay line cannot be shortened faster than time passes, and it
    // is set above the fastest ramp any in-range edit produces so that it never
    // binds on anything but automation nobody would write by hand.
    static constexpr float maxPredelaySlew = 0.6f;
    float predelayNow = 0.0f;

    std::array<DelayLine, maxChannels> predelayLines;
    std::array<DelayLine, maxFeeds> erLines;
    std::array<std::array<Allpass, numDiffusers>, maxFeeds> diffusers;

    std::array<float, maxChannels> inputDampState {};
    std::array<float, maxChannels> tiltState {};
    float tiltCoeff = 1.0f;

    // Channel geometry (v0.7). All of it is settled in adoptLayout and read
    // everywhere else.
    SpeakerLayout layout;
    std::array<Speaker, maxChannels> tankSpeakers {};
    std::array<float, maxChannels> speakerAzimuth {}, speakerElevation {};
    std::array<int, maxChannels> wetChan {};
    std::array<std::array<float, maxFeeds>, maxChannels> feedMix {};
    std::array<float, maxFeeds> feedAzimuth {};
    int layoutChannels = 2, numIn = 2, numTank = 2, numFeeds = 2;
    bool foldMono = false;
    float invNumTank = 0.5f, erChanScale = 1.0f;
    float lastCorrelation = -1.0f;

    juce::SmoothedValue<float> smoothMix, smoothWidth, smoothErLevel, smoothLateLevel,
                               smoothPredelay, smoothTilt, smoothLpCoeff, smoothHpCoeff,
                               smoothDiffGain, smoothStages, smoothErTime, smoothWetTrim,
                               smoothModDepth, smoothInputDamp, smoothLively;

    // Block-rate derived control values ---------------------------------------
    float decaySecondsTarget = 2.0f, brightnessRatioMul = 1.0f;
    float lastSeenTargetPos = -1.0f;
    int targetStillSamples = 0;
    float charErMul = 1.0f, charDepthMul = 1.0f, charRateMul = 1.0f;
    float charDiffMul = 1.0f, charStageMul = 1.0f, charLively = 0.0f;
    float thickStageMul = 1.0f, thickGainMul = 1.0f, thickLateMul = 1.0f;
    float distErMul = 1.0f, distLateMul = 1.0f;

    float currentRt60 = 1.0f, targetRt60 = 1.0f, deliveredRt60 = 1.0f, rt60Slew = 0.05f;
    float freezeAmount = 0.0f, freezeTarget = 0.0f, freezeSlew = 0.001f;

    // The two v0.3 sections. Both are inert — no filter stages, no per-sample
    // cost — until a band is moved off neutral, so the default engine is the
    // v0.2 one.
    //
    // Last in the class on purpose. Between them they are about thirteen
    // kilobytes of coefficients, filter state and analysis grids, and declared
    // where they are used — next to the damping filters — they pushed
    // `modInc`, `modPhase` and `feedbackGain` that far away from the arrays
    // the sample loop reads them alongside. That alone cost the *neutral*
    // engine a third of its CPU budget, for state it never touches. Cold data
    // goes at the end.
    DecayEq decayEq;
    PostEq postEq;
    bool decayEqActive = false, postEqActive = false;

    // The v0.5 per-model geometry: built once in prepare(), read only by
    // `evaluateGeometry` and `updateControlRate`, and interpolated from there
    // into the hot arrays above.
    //
    // Down here for the reason the two EQ sections are: two and a half kilobytes
    // of tables the sample loop never touches, sitting among arrays it touches
    // sixteen times each per sample, is exactly the eviction the note on those
    // sections describes. Cold data goes at the end.
    std::array<std::array<float, numLines>, roomModels.size()> modelDelaySamples {};
    std::array<std::array<std::array<float, numDiffusers>, roomModels.size()>, maxFeeds> modelDiff {};
    std::array<std::array<float, numLines>, roomModels.size()> modelMixWeight {};

    // What the density floor did to each model's modulation depth. 1.0 for
    // every room already above the floor, which is everything from the Chamber
    // up — the halls keep every bit of their movement.
    std::array<float, roomModels.size()> modelDetuneScale {};

    // Where each early reflection slot arrives from. Fixed per slot rather than
    // per model — see buildTapDirections.
    std::array<float, maxErTaps> tapRear {}, tapElevation {};

    Params lastParams;
};

} // namespace hrzn
