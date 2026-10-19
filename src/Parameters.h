#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "dsp/FilterCoeffs.h"
#include "dsp/RoomModels.h"
#include "dsp/DuckGate.h"
#include "dsp/SpeakerLayout.h"
#include "util/TempoSync.h"

namespace hrzn
{

// Parameter ranges ---------------------------------------------------------

// Decay is an absolute time, not a percentage of the room (v0.4.1).
//
// Through v0.4 this was "25%-400% of whatever the model's own RT60 is", which
// is the reference product's scheme and is wrong for the way this engine is
// actually used: it yokes decay length to Space, so you cannot audition the
// same tail in four different rooms, and every Space move is also a decay
// move. Space now changes what the room *is* — its size, density, early
// pattern, absorption and modulation — and this says how long it rings.
//
// The range covers what the old scheme reached (the smallest room at 25% was
// 70 ms; the largest at 400% was nearly a minute) without needing a room to
// reach it.
constexpr float minDecaySeconds = 0.10f, maxDecaySeconds = 30.0f;
constexpr float defaultDecaySeconds = 2.0f;

// Retained for the v1 -> v2 state migration only, which reads a saved
// percentage and multiplies it by the room's own decay.
constexpr float minDecayRatePc = 25.0f, maxDecayRatePc = 400.0f;

constexpr float minPredelayMs = 0.0f, maxPredelayMs = 500.0f;
constexpr float minWidthPc = 0.0f, maxWidthPc = 200.0f;

// Ducking and the Auto Gate (v0.5). Ranges chosen so the defaults are a usable
// setting rather than a neutral one: switching either control on should do
// something audible and sensible, because a control that has to be dialled in
// before it does anything gets switched off again before it ever gets dialled.
constexpr float minDuckThresholdDb = -60.0f, maxDuckThresholdDb = 0.0f;
constexpr float minDuckAmountDb = 0.0f, maxDuckAmountDb = 30.0f;
constexpr float minDuckAttackMs = 0.5f, maxDuckAttackMs = 200.0f;
constexpr float minDuckReleaseMs = 20.0f, maxDuckReleaseMs = 2000.0f;

constexpr float minGateThresholdDb = -80.0f, maxGateThresholdDb = 0.0f;
constexpr float minGateHoldMs = 0.0f, maxGateHoldMs = 1000.0f;
constexpr float minGateReleaseMs = 5.0f, maxGateReleaseMs = 2000.0f;

// Which signal keys the two of them. One choice drives both: a session where
// the reverb ducks from the mix bus but gates from the snare is not a session
// anybody has, and two selectors would be two things to get wrong.
inline const juce::StringArray triggerSourceNames { "Input", "Sidechain" };
enum class TriggerSource { input = 0, sidechain };

constexpr int numEqBands = 6;

// Which of the two six-band sections a band belongs to. Shared vocabulary
// rather than a display-private enum: the display, the band row under it and
// the value-entry callout all address bands as (section, index).
enum class EqSection { decayEq = 0, postEq };

constexpr float minEqFreq = 20.0f, maxEqFreq = 20000.0f;
constexpr float minDecayMult = 0.2f, maxDecayMult = 6.0f;
constexpr float minDecayQ = 0.3f, maxDecayQ = 6.0f;
constexpr float minPostGainDb = -24.0f, maxPostGainDb = 24.0f;
constexpr float minPostQ = 0.1f, maxPostQ = 10.0f;

// The Decay Rate EQ's shapes are fixed by band: a low shelf, four peaks and a
// high shelf (feature 19). A reverb's decay curve is a broad, smooth thing —
// nobody wants a 48 dB/oct cut on how long 3 kHz rings — so the band shape is
// not a control here, which also keeps six handles permanently on the display
// v0.4 draws rather than a set that appears and disappears.
inline constexpr std::array<FilterShape, numEqBands> decayEqShapes { {
    FilterShape::lowShelf, FilterShape::bell, FilterShape::bell,
    FilterShape::bell, FilterShape::bell, FilterShape::highShelf
} };

inline constexpr std::array<float, numEqBands> decayEqDefaultFreq { {
    120.0f, 350.0f, 900.0f, 2200.0f, 5000.0f, 10000.0f
} };

inline constexpr std::array<float, numEqBands> decayEqDefaultQ { {
    0.71f, 0.9f, 0.9f, 0.9f, 0.9f, 0.71f
} };

inline constexpr std::array<FilterShape, numEqBands> postEqDefaultShape { {
    FilterShape::lowShelf, FilterShape::bell, FilterShape::bell,
    FilterShape::bell, FilterShape::bell, FilterShape::highShelf
} };

inline constexpr std::array<float, numEqBands> postEqDefaultFreq { {
    100.0f, 250.0f, 800.0f, 2500.0f, 6000.0f, 12000.0f
} };

inline constexpr std::array<float, numEqBands> postEqDefaultQ { {
    0.71f, 1.0f, 1.0f, 1.0f, 1.0f, 0.71f
} };

// Parameter IDs ------------------------------------------------------------

namespace id
{
    inline const juce::String space      { "space" };
    inline const juce::String decayTime  { "decay_time" };
    inline const juce::String predelay   { "predelay" };
    inline const juce::String mix        { "mix" };
    inline const juce::String width      { "width" };
    inline const juce::String brightness { "brightness" };
    inline const juce::String distance   { "distance" };
    inline const juce::String character  { "character" };
    inline const juce::String thickness  { "thickness" };
    inline const juce::String freeze     { "freeze" };
    inline const juce::String bypass     { "bypass" };

    // Ducking, the Auto Gate and their shared trigger source (v0.5).
    inline const juce::String duckOn        { "duck_on" };
    inline const juce::String duckThreshold { "duck_thresh" };
    inline const juce::String duckAmount    { "duck_amount" };
    inline const juce::String duckAttack    { "duck_attack" };
    inline const juce::String duckRelease   { "duck_release" };
    inline const juce::String triggerSource { "trigger_src" };

    inline const juce::String gateOn        { "gate_on" };
    inline const juce::String gateThreshold { "gate_thresh" };
    inline const juce::String gateHold      { "gate_hold" };
    inline const juce::String gateRelease   { "gate_release" };

    // Predelay tempo sync (v0.5). The free `predelay` above stays exactly what
    // it was — sync is a second way of arriving at the same delay line, not a
    // second delay line — so a saved state that predates this loads unsynced
    // with its milliseconds intact.
    inline const juce::String predelaySync { "predelay_sync" };
    inline const juce::String predelayDiv  { "predelay_div" };

    // Section bypasses (v0.4.1). Six bands at 1.00x is already "no effect", but
    // getting back to it by hand is six drags and getting back to where you
    // were afterwards is impossible — so both sections carry a switch, which
    // is what makes the sections auditionable at all.
    inline const juce::String decayEqOn  { "deq_on" };
    inline const juce::String postEqOn_  { "peq_on" };

    // The two six-band sections. Built rather than listed because sixty
    // spelled-out identifiers would be sixty chances to typo one.
    inline juce::String decayEqFreq (int band) { return "deq" + juce::String (band + 1) + "_freq"; }
    inline juce::String decayEqMult (int band) { return "deq" + juce::String (band + 1) + "_mult"; }
    inline juce::String decayEqQ    (int band) { return "deq" + juce::String (band + 1) + "_q"; }


    inline juce::String postEqOn    (int band) { return "peq" + juce::String (band + 1) + "_on"; }
    inline juce::String postEqType  (int band) { return "peq" + juce::String (band + 1) + "_type"; }
    inline juce::String postEqFreq  (int band) { return "peq" + juce::String (band + 1) + "_freq"; }
    inline juce::String postEqGain  (int band) { return "peq" + juce::String (band + 1) + "_gain"; }
    inline juce::String postEqQ     (int band) { return "peq" + juce::String (band + 1) + "_q"; }
    inline juce::String postEqSlope (int band) { return "peq" + juce::String (band + 1) + "_slope"; }
    inline juce::String postEqChan  (int band) { return "peq" + juce::String (band + 1) + "_ch"; }
}

// State properties that are not parameters ---------------------------------

namespace prop
{
    // Lock Mix (feature 17), v0.5. Not a parameter: it is not a property of
    // the sound, nothing should automate it, and a host that wrote it into an
    // automation lane would be recording the user's workflow rather than their
    // mix. It rides in the session state — so it survives closing the project
    // — and is listed in PresetProductConfig::sessionProperties, which is what
    // keeps it out of every preset written from here on.
    inline const juce::Identifier lockMix { "lock_mix" };
}

// Shared value formatting for the EQ sections ------------------------------

inline juce::String formatFrequency (float hz, int)
{
    if (hz >= 10000.0f) return juce::String (hz / 1000.0f, 1) + " kHz";
    if (hz >= 1000.0f)  return juce::String (hz / 1000.0f, 2) + " kHz";
    if (hz >= 100.0f)   return juce::String (juce::roundToInt (hz)) + " Hz";
    return juce::String (hz, 1) + " Hz";
}

// Decay multipliers read as "x" figures, not percentages or decibels: the
// control means "this band rings twice as long", and 2.00x says that where
// "+6 dB" would say something about level that is simply not true.
inline juce::String formatDecayMultiplier (float mult, int)
{
    return juce::String (mult, 2) + juce::String::fromUTF8 ("\xc3\x97");
}

// Decay times run from 100 ms to 30 s, so one format does not serve the whole
// range: three significant figures either side of a second.
inline juce::String formatSeconds (float seconds, int)
{
    if (seconds < 1.0f)   return juce::String (juce::roundToInt (seconds * 1000.0f)) + " ms";
    if (seconds < 10.0f)  return juce::String (seconds, 2) + " s";
    return juce::String (seconds, 1) + " s";
}

inline juce::NormalisableRange<float> frequencyRange()
{
    juce::NormalisableRange<float> r (minEqFreq, maxEqFreq, 0.01f);
    r.setSkewForCentre (900.0f);
    return r;
}

// Space reads out as the room it is sitting on — and, once the morph is
// genuinely between two models (v0.2), as the pair it is between. Hosts show
// this string in automation lanes and generic editors, where "62%" would tell
// you nothing about which room you had automated towards.
inline juce::String spaceDisplayName (float space01)
{
    const float p = spaceToPosition (space01);
    const int low = juce::jlimit (0, numRoomModels - 1, (int) p);
    const int high = juce::jmin (low + 1, numRoomModels - 1);
    const float f = p - (float) low;

    if (f <= 0.12f || low == high)
        return roomModels[(size_t) low].name;
    if (f >= 0.88f)
        return roomModels[(size_t) high].name;

    return juce::String (roomModels[(size_t) low].name)
           + juce::String::fromUTF8 (" \xe2\x86\x92 ")
           + roomModels[(size_t) high].name;
}

inline juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
{
    using P = juce::AudioParameterFloat;
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    auto pcAttr = juce::AudioParameterFloatAttributes().withLabel ("%");
    auto msAttr = juce::AudioParameterFloatAttributes().withLabel ("ms");

    // Space is a percentage through the room model table, not a choice —
    // deliberately. v0.1 snapped to the nearest model; from v0.2 the engine
    // sits anywhere between two of them and interpolates the geometry
    // (feature 8). Keeping the type and range fixed from the first release
    // means that change invalidated no saved state.
    params.push_back (std::make_unique<P> (
        juce::ParameterID (id::space, 1), "Space",
        juce::NormalisableRange<float> (0.0f, 100.0f, 0.01f), 55.0f,
        juce::AudioParameterFloatAttributes()
            .withStringFromValueFunction ([] (float v, int)
            {
                return spaceDisplayName (v * 0.01f);
            })));

    // Absolute decay time (feature 9), independent of Space. See the note on
    // minDecaySeconds for why this stopped being a percentage at v0.4.1.
    {
        auto decayRange = juce::NormalisableRange<float> (minDecaySeconds, maxDecaySeconds,
                                                          0.001f);
        decayRange.setSkewForCentre (2.0f);

        params.push_back (std::make_unique<P> (
            juce::ParameterID (id::decayTime, 1), "Decay Time",
            decayRange, defaultDecaySeconds,
            juce::AudioParameterFloatAttributes()
                .withLabel ("s")
                .withStringFromValueFunction (formatSeconds)));
    }

    // Predelay, and from v0.5 the two controls that reinterpret it as a note
    // length (feature 18). The free control keeps its range and its ID: when
    // sync is off this is still exactly the parameter v0.1 shipped.
    params.push_back (std::make_unique<P> (
        juce::ParameterID (id::predelay, 1), "Predelay",
        juce::NormalisableRange<float> (minPredelayMs, maxPredelayMs, 0.1f, 0.5f),
        0.0f, msAttr));

    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID (id::predelaySync, 1), "Predelay Sync", false));

    params.push_back (std::make_unique<juce::AudioParameterChoice> (
        juce::ParameterID (id::predelayDiv, 1), "Predelay Division",
        noteDivisionNames(), defaultDivisionIndex));

    params.push_back (std::make_unique<P> (
        juce::ParameterID (id::mix, 1), "Mix",
        juce::NormalisableRange<float> (0.0f, 100.0f, 0.1f), 30.0f, pcAttr));

    // 0% = mono, 100% = the model's natural image, 200% = beyond. The
    // "intelligent" behaviour of feature 22 lands with surround at v0.7.
    params.push_back (std::make_unique<P> (
        juce::ParameterID (id::width, 1), "Width",
        juce::NormalisableRange<float> (minWidthPc, maxWidthPc, 0.1f), 100.0f, pcAttr));

    // 50% is the model's own voicing; below darkens and shortens HF decay,
    // above brightens and lengthens it (feature 16, perfected at v0.3).
    params.push_back (std::make_unique<P> (
        juce::ParameterID (id::brightness, 1), "Brightness",
        juce::NormalisableRange<float> (0.0f, 100.0f, 0.1f), 50.0f, pcAttr));

    // Proximity to the source inside the room: early/late balance, HF
    // absorption and level moving together (feature 11).
    params.push_back (std::make_unique<P> (
        juce::ParameterID (id::distance, 1), "Distance",
        juce::NormalisableRange<float> (0.0f, 100.0f, 0.1f), 35.0f, pcAttr));

    // Character (feature 13): 0% is a clean, transparent decay, 50% is the
    // room model exactly as designed, 100% is the over-modulated chorus-like
    // extreme. It drives early reflection density and level, late modulation
    // depth and rate, and diffusion depth along one tuned trajectory.
    params.push_back (std::make_unique<P> (
        juce::ParameterID (id::character, 1), "Character",
        juce::NormalisableRange<float> (0.0f, 100.0f, 0.1f), 50.0f, pcAttr));

    // Thickness (feature 7): how densely the late field builds behind the
    // onset. 50% is the model's own density; below opens it out, above packs
    // it. Like Character it is centred rather than one-sided, so the factory
    // models stay reachable at the knob's middle.
    params.push_back (std::make_unique<P> (
        juce::ParameterID (id::thickness, 1), "Thickness",
        juce::NormalisableRange<float> (0.0f, 100.0f, 0.1f), 50.0f, pcAttr));

    // Freeze (feature 10): seals the tank at unity feedback — infinite,
    // lossless sustain with the input held out.
    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID (id::freeze, 1), "Freeze", false));

    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID (id::bypass, 1), "Bypass", false));

    // Ducking and the Auto Gate (feature 7) -------------------------------
    //
    // Both key off the same trigger, and both act on the wet path alone —
    // whatever they do, the dry signal arrives unaltered, which is the
    // property that makes them safe to leave switched on.
    {
        auto dbAttr = juce::AudioParameterFloatAttributes().withLabel ("dB");

        auto msRange = [] (float lo, float hi, float centre)
        {
            juce::NormalisableRange<float> r (lo, hi, 0.01f);
            r.setSkewForCentre (centre);
            return r;
        };

        auto msTimeAttr = juce::AudioParameterFloatAttributes()
                              .withLabel ("ms")
                              .withStringFromValueFunction ([] (float ms, int)
                              {
                                  return ms < 100.0f ? juce::String (ms, 1) + " ms"
                                                     : juce::String (juce::roundToInt (ms)) + " ms";
                              });

        params.push_back (std::make_unique<juce::AudioParameterBool> (
            juce::ParameterID (id::duckOn, 1), "Ducking", false));

        params.push_back (std::make_unique<P> (
            juce::ParameterID (id::duckThreshold, 1), "Duck Threshold",
            juce::NormalisableRange<float> (minDuckThresholdDb, maxDuckThresholdDb, 0.1f),
            -30.0f, dbAttr));

        // Depth in dB rather than a percentage: the control says how far the
        // reverb gets pulled down, and "12 dB" is a figure you can reason
        // about against a fader where "40%" is a figure about nothing.
        params.push_back (std::make_unique<P> (
            juce::ParameterID (id::duckAmount, 1), "Duck Amount",
            juce::NormalisableRange<float> (minDuckAmountDb, maxDuckAmountDb, 0.1f),
            12.0f, dbAttr));

        params.push_back (std::make_unique<P> (
            juce::ParameterID (id::duckAttack, 1), "Duck Attack",
            msRange (minDuckAttackMs, maxDuckAttackMs, 15.0f), 10.0f, msTimeAttr));

        params.push_back (std::make_unique<P> (
            juce::ParameterID (id::duckRelease, 1), "Duck Release",
            msRange (minDuckReleaseMs, maxDuckReleaseMs, 300.0f), 250.0f, msTimeAttr));

        params.push_back (std::make_unique<juce::AudioParameterChoice> (
            juce::ParameterID (id::triggerSource, 1), "Trigger Source",
            triggerSourceNames, (int) TriggerSource::input));

        params.push_back (std::make_unique<juce::AudioParameterBool> (
            juce::ParameterID (id::gateOn, 1), "Auto Gate", false));

        params.push_back (std::make_unique<P> (
            juce::ParameterID (id::gateThreshold, 1), "Gate Threshold",
            juce::NormalisableRange<float> (minGateThresholdDb, maxGateThresholdDb, 0.1f),
            -40.0f, dbAttr));

        params.push_back (std::make_unique<P> (
            juce::ParameterID (id::gateHold, 1), "Gate Hold",
            msRange (minGateHoldMs, maxGateHoldMs, 150.0f), 120.0f, msTimeAttr));

        params.push_back (std::make_unique<P> (
            juce::ParameterID (id::gateRelease, 1), "Gate Release",
            msRange (minGateReleaseMs, maxGateReleaseMs, 250.0f), 200.0f, msTimeAttr));
    }

    // Six-band Decay Rate EQ (feature 19) --------------------------------
    //
    // Frequency-dependent decay *multipliers*, applied inside the feedback
    // path. 1.00x everywhere is the room model exactly as designed and costs
    // nothing to run; the range reaches 6x because a low band held far longer
    // than the rest is one of the most useful things a reverb can do, and
    // because the section is stable by construction rather than by margin
    // (see DecayEq.h) there is no reason to be timid about the top of it.
    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID (id::decayEqOn, 1), "Decay Rate EQ On", true));

    for (int b = 0; b < numEqBands; ++b)
    {
        const auto n = juce::String (b + 1);

        auto multRange = juce::NormalisableRange<float> (minDecayMult, maxDecayMult, 0.001f);
        multRange.setSkewForCentre (1.0f);

        auto qRange = juce::NormalisableRange<float> (minDecayQ, maxDecayQ, 0.001f);
        qRange.setSkewForCentre (1.0f);

        params.push_back (std::make_unique<P> (
            juce::ParameterID (id::decayEqFreq (b), 1), "Decay EQ " + n + " Freq",
            frequencyRange(), decayEqDefaultFreq[(size_t) b],
            juce::AudioParameterFloatAttributes()
                .withStringFromValueFunction (formatFrequency)));

        params.push_back (std::make_unique<P> (
            juce::ParameterID (id::decayEqMult (b), 1), "Decay EQ " + n + " Decay",
            multRange, 1.0f,
            juce::AudioParameterFloatAttributes()
                .withStringFromValueFunction (formatDecayMultiplier)));

        params.push_back (std::make_unique<P> (
            juce::ParameterID (id::decayEqQ (b), 1), "Decay EQ " + n + " Q",
            qRange, decayEqDefaultQ[(size_t) b]));
    }

    // Six-band Post EQ (feature 21) ---------------------------------------
    //
    // On the wet output, with the family EQ's full shape set and a channel
    // mode per band (feature 15). Bands start switched off: an EQ band nobody
    // asked for is a band that costs CPU and, sooner or later, surprises
    // somebody.
    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID (id::postEqOn_, 1), "Post EQ On", true));

    for (int b = 0; b < numEqBands; ++b)
    {
        const auto n = juce::String (b + 1);

        auto qRange = juce::NormalisableRange<float> (minPostQ, maxPostQ, 0.001f);
        qRange.setSkewForCentre (1.0f);

        params.push_back (std::make_unique<juce::AudioParameterBool> (
            juce::ParameterID (id::postEqOn (b), 1), "Post EQ " + n + " On", false));

        params.push_back (std::make_unique<juce::AudioParameterChoice> (
            juce::ParameterID (id::postEqType (b), 1), "Post EQ " + n + " Shape",
            filterShapeNames, (int) postEqDefaultShape[(size_t) b]));

        params.push_back (std::make_unique<P> (
            juce::ParameterID (id::postEqFreq (b), 1), "Post EQ " + n + " Freq",
            frequencyRange(), postEqDefaultFreq[(size_t) b],
            juce::AudioParameterFloatAttributes()
                .withStringFromValueFunction (formatFrequency)));

        params.push_back (std::make_unique<P> (
            juce::ParameterID (id::postEqGain (b), 1), "Post EQ " + n + " Gain",
            juce::NormalisableRange<float> (minPostGainDb, maxPostGainDb, 0.01f), 0.0f,
            juce::AudioParameterFloatAttributes().withLabel ("dB")));

        params.push_back (std::make_unique<P> (
            juce::ParameterID (id::postEqQ (b), 1), "Post EQ " + n + " Q",
            qRange, postEqDefaultQ[(size_t) b]));

        params.push_back (std::make_unique<juce::AudioParameterChoice> (
            juce::ParameterID (id::postEqSlope (b), 1), "Post EQ " + n + " Slope",
            slopeNames, 1));

        params.push_back (std::make_unique<juce::AudioParameterChoice> (
            juce::ParameterID (id::postEqChan (b), 1), "Post EQ " + n + " Channel",
            channelModeNames, (int) ChannelMode::stereo));
    }

    return { params.begin(), params.end() };
}

} // namespace hrzn
