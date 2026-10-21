#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include "Parameters.h"
#include "StateManager.h"
#include "presets/PresetManager.h"
#include "dsp/AnalyzerFifo.h"
#include "dsp/ReverbEngine.h"
#include "dsp/DuckGate.h"
#include "dsp/IrFile.h"
#include "dsp/IrMatch.h"
#include "util/TempoSync.h"

class HorizonAudioProcessor : public juce::AudioProcessor
{
public:
    HorizonAudioProcessor();
    ~HorizonAudioProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Horizon"; }
    // True from v0.8: MIDI Learn needs controller messages, and a plugin that
    // declares no MIDI input is a plugin several hosts will not send any to —
    // regardless of what the CMake target says. It still produces none.
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }

    // A reverb must keep running after the input stops, or hosts truncate the
    // tail on bounce. Freeze reports a long-but-finite tail.
    double getTailLengthSeconds() const override { return (double) engine.tailSeconds(); }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    // Saved-state schema version, embedded in every state/preset we write.
    // States from a newer schema are ignored rather than half-applied.
    //
    //   v1: session root carries both A/B slots (the family scheme).
    //   v2: Decay Rate (% of the room) became Decay Time (seconds). A v1
    //       state is migrated on load rather than rejected — see
    //       migrateDecayRateToTime.
    static constexpr int stateVersion = 2;
    bool applyStateXml (const juce::XmlElement&);

    // Rewrites any v1 `decay_rate` percentage into the equivalent absolute
    // `decay_time`, using the room the same state was sitting on. Public
    // because the harness checks the conversion directly.
    static void migrateDecayRateToTime (juce::XmlElement& xml);

    // MIDI Learn (feature 26, v0.8) -----------------------------------------
    //
    // Zenith's system, ported unchanged in structure because the structure is
    // the part that is hard: arm on the message thread, capture on the audio
    // thread, persist on the message thread. The audio thread only ever
    // exchanges atomics — it never touches the ValueTree, which is where every
    // MIDI Learn implementation that has a data race puts one.
    void armMidiLearn (const juce::String& paramId);
    void cancelMidiLearn() { learnTarget.store (nullptr); }
    bool isMidiLearnArmed() const { return learnTarget.load() != nullptr; }
    int getMappedCc (const juce::String& paramId) const;   // -1 if unmapped
    void clearMidiMapping (const juce::String& paramId);
    void commitPendingMidiLearn();

    // The tree child the mappings live in. Named here rather than in the .cpp
    // because the preset config lists it as a tree presets inherit, and the
    // harness reads it directly.
    static const juce::Identifier midiMapType;

    juce::AudioProcessorValueTreeState apvts;

    // Snapshot-based undo/redo with independent history per A/B slot, plus
    // the preset ecosystem (both from jdm_shared; preset loads and name
    // changes route through the StateManager so they are undoable).
    hrzn::StateManager stateManager { *this, apvts, juce::Identifier ("HorizonSession") };
    hrzn::PresetManager presets { apvts, hrzn::makePresetConfig(),
        [this] (const juce::ValueTree& t) { applyPresetState (t); },
        [this] { return stateManager.getCurrentPresetName(); },
        [this] (const juce::String& n) { stateManager.setCurrentPresetName (n); } };

    // Lock Mix (feature 17). The flag is a member rather than a lookup into
    // the state tree, and that is what makes it survive: every A/B switch,
    // undo and preset load replaces the tree wholesale, and a setting about
    // how loading should behave cannot live in the thing being loaded. The
    // tree carries a copy so the session remembers it — see prop::lockMix.
    bool isMixLocked() const noexcept { return mixLocked.load(); }
    void setMixLocked (bool shouldLock);

    // Wet gain currently imposed by Ducking and the Auto Gate, plus the level
    // of the trigger that produced it: the trace the display draws in the
    // sidechain teal (v0.5).
    std::atomic<float> triggerLevelDb { -140.0f };
    std::atomic<float> duckDepthDb { 0.0f };
    std::atomic<float> gateGain { 1.0f };

    // Host tempo, and the predelay actually delivered once sync and the 500 ms
    // ceiling have had their say. The readout shows the latter, so a division
    // that has run into the ceiling is visible rather than silently ignored.
    std::atomic<float> hostBpm { 120.0f };
    std::atomic<float> displayPredelayMs { 0.0f };

    // Impulse response import (feature 6, v0.6) ---------------------------
    //
    // What an import produced, kept so the panel can report it and so the A/B
    // has something to compare against. The conversion is an approximation and
    // the report is how the user finds out how good an approximation — so the
    // numbers are part of the result, not diagnostics.
    struct IrImport
    {
        bool valid = false;
        juce::String error;
        juce::String name;

        hrzn::ir::Match match;

        double sourceSampleRate = 0.0;
        int sourceChannels = 0;
        double sourceSeconds = 0.0;

        juce::AudioBuffer<float> audio;   // resampled, for the A/B convolution
    };

    // Loads and fits a file. Synchronous and slow — several seconds of
    // rendering — so it belongs on a background thread, which is why it
    // reports progress and why it touches no parameters. Applying the result
    // is a separate call that must happen on the message thread.
    IrImport analyseImpulseResponse (const juce::File&,
                                     const hrzn::ir::ProgressFn& = {});

    // Applies a fitted state as one undoable change, and hands the source
    // response to the convolver so it can be auditioned against.
    void applyImpulseResponse (const IrImport&);

    // Both, in order, on the calling thread. The harness uses this.
    bool importImpulseResponse (const juce::File&, const hrzn::ir::ProgressFn& = {});

    // What the editor uses: the fit runs on a worker and the result is applied
    // back on the message thread. The pool lives here rather than on the
    // editor so that closing the window mid-import neither blocks on a fit
    // that has several seconds left to run nor throws the result away.
    void beginImpulseResponseImport (const juce::File&);

    std::atomic<bool> irBusy { false };
    std::atomic<float> irProgress { 0.0f };

    // The last import's report, for the panel. Guarded because it is written
    // on the message thread and read there too, but the audio thread asks
    // whether an IR is loaded at all.
    IrImport lastImport;
    std::atomic<bool> irLoaded { false };

    // A/B: the wet path comes from convolution with the source response
    // instead of from the engine. Not a parameter — it is an audition switch,
    // it belongs to the session rather than to the sound, and a host that
    // recorded it into an automation lane would be recording the user looking
    // at something.
    std::atomic<bool> irCompare { false };
    void setIrCompare (bool shouldCompare);

    // Level match for the A/B: the ratio between what the fitted engine state
    // produces and what the source response produces, so the two sides of the
    // comparison arrive at the same loudness. Without it the convolution came
    // in 17 dB hot and the A/B measured which side was louder.
    std::atomic<float> convolutionGain { 1.0f };

    // Metering for the GUI: peaks held since the last exchange (0).
    std::atomic<float> peakIn { 0.0f }, peakOut { 0.0f };

    // Smoothed processing load as a fraction of realtime (0..1+).
    std::atomic<float> cpuLoad { 0.0f };

    // Set when Space should jump rather than morph: a whole state re-applied
    // at once (preset load, A/B switch, undo/redo), or the editor's jump-to-
    // room menu. Dragging the knob and automating it still glide — a state
    // change and a named choice are discontinuities and should sound like one
    // rather than like a slide across the room table. Read and cleared on the
    // audio thread, which is the only place the engine may be touched.
    std::atomic<bool> relocateRequested { false };

    // What the engine is currently sitting on, for the header readout. The
    // morph position is continuous from v0.2, so the readout needs the pair
    // it is between as well as the nearest name — and the decay it reports is
    // the one actually delivered, not the model's nominal figure.
    std::atomic<int> displayModelIndex { 0 };
    std::atomic<int> displayMorphLow { 0 }, displayMorphHigh { 0 };
    std::atomic<float> displayMorphFraction { 0.0f };
    std::atomic<float> displaySpacePosition { 0.0f };
    std::atomic<float> displayRt60 { 1.0f };

    // Delivered decay across frequency, straight out of the Decay Rate EQ's
    // design (see DecayEq.h). The header readout shows the 1 kHz point; the
    // whole curve is the decay-time overlay the display draws (feature 24).
    std::array<std::atomic<float>, hrzn::DecayEq::numReportPoints> displayDecayCurve {};

    // The same grid with the section neutral: the room's own decay across
    // frequency, which is the reference the display places the six Decay Rate
    // EQ handles against. A handle reading 2.00x sits at twice this curve, and
    // any gap left between the two curves is the part of a request the
    // feedback loop could not meet.
    std::array<std::atomic<float>, hrzn::DecayEq::numReportPoints> displayNeutralCurve {};

    // The layout the host has us in (v0.7). Read by the editor so the panel can
    // name it and so feature 14's Speakers setting can grey out the groups this
    // layout does not have. Written in prepareToPlay and read on the message
    // thread; a layout change is a prepare, and a prepare cannot race a paint.
    const hrzn::SpeakerLayout& speakerLayout() const noexcept { return speakers; }

    // Analyzer feeds (feature 24), drained by the display at frame rate: the
    // signal arriving, and the reverb on its own before it meets the dry path.
    // The wet trace is taken inside the engine rather than derived from the
    // output, because at Mix 0 there is nothing in the output to derive it
    // from and the tail is exactly what you want to see while setting Mix.
    hrzn::AnalyzerFifo inputFifo, wetFifo;

private:
    static constexpr int maxChans = hrzn::ReverbEngine::maxChannels;

    // Applies a preset's state, honouring Lock Mix. Wrapping the apply rather
    // than restoring Mix afterwards keeps it one undoable transaction: undoing
    // a preset load should not need two undos, and should not leave the Mix
    // somewhere neither the preset nor the user put it.
    void applyPresetState (const juce::ValueTree& incoming);

    // First element in the tree carrying the Lock Mix attribute, or null.
    static const juce::XmlElement* findLockMixAttribute (const juce::XmlElement&);

    // MIDI Learn plumbing. `ccTargets` is the audio thread's lookup — 128
    // slots, one per controller number — and the MIDIMAP child of the state
    // tree is the persistent source of truth. The two are kept in step by
    // rebuildMidiCcMap(), which runs on the message thread and, crucially,
    // also runs on every wholesale tree reapply: undo, redo, an A/B switch and
    // a preset load all replace the tree under the cache, and a cache that
    // survived one of those would be pointing at the mappings of a state the
    // session is no longer in. See the StateManager's onStateReapplied hook.
    std::array<std::atomic<juce::RangedAudioParameter*>, 128> ccTargets {};
    std::atomic<juce::RangedAudioParameter*> learnTarget { nullptr };
    std::atomic<juce::RangedAudioParameter*> pendingLearnParam { nullptr };
    std::atomic<int> pendingLearnCc { -1 };
    void rebuildMidiCcMap();   // message thread
    void handleMidi (juce::MidiBuffer&);

    hrzn::SpeakerLayout speakers;
    hrzn::ReverbEngine engine;
    hrzn::DuckGate duckGate;

    juce::SmoothedValue<float> bypassMix;
    juce::AudioBuffer<float> dryBuffer, wetTapBuffer;

    // Mono trigger for Ducking and the Auto Gate, and the block of wet gains
    // they produce.
    juce::AudioBuffer<float> keyBuffer, wetGainBuffer;

    std::atomic<bool> mixLocked { false };

    // The A/B convolver (v0.6). Zero latency by construction: an audition
    // switch that shifted the timing every time it was thrown would make the
    // two sides incomparable, which is the one thing it exists to avoid.
    juce::dsp::Convolution convolver { juce::dsp::Convolution::Latency { 0 } };
    juce::AudioBuffer<float> convolutionBuffer;
    juce::SmoothedValue<float> compareMix, compareDryWet;

    // The convolver's warm-up before the A/B crossfades over to it (v0.9);
    // see processBlock. Audio thread only.
    static constexpr double compareWarmupSeconds = 0.12;
    int compareWarmupLeft = 0;
    bool convolverWarm = false;

    // One worker, because two simultaneous imports would race to apply two
    // states and the second would win for no reason the user could predict.
    juce::ThreadPool irPool { 1 };

    // Liveness token for the message-thread callback that applies a finished
    // import. The callback holds a weak reference to it, so a result that
    // arrives after this processor is gone does nothing instead of writing
    // into freed memory.
    std::shared_ptr<int> lifetime { std::make_shared<int> (0) };

    std::atomic<float>* spaceParam = nullptr;
    std::atomic<float>* decayTimeParam = nullptr;
    std::atomic<float>* decayEqOnParam = nullptr;
    std::atomic<float>* postEqOnParam = nullptr;
    std::atomic<float>* predelayParam = nullptr;
    std::atomic<float>* mixParam = nullptr;
    std::atomic<float>* widthParam = nullptr;
    std::atomic<float>* brightnessParam = nullptr;
    std::atomic<float>* distanceParam = nullptr;
    std::atomic<float>* characterParam = nullptr;
    std::atomic<float>* thicknessParam = nullptr;
    std::atomic<float>* freezeParam = nullptr;
    std::atomic<float>* bypassParam = nullptr;

    std::atomic<float>* predelaySyncParam = nullptr;
    std::atomic<float>* predelayDivParam = nullptr;

    std::atomic<float>* duckOnParam = nullptr;
    std::atomic<float>* duckThresholdParam = nullptr;
    std::atomic<float>* duckAmountParam = nullptr;
    std::atomic<float>* duckAttackParam = nullptr;
    std::atomic<float>* duckReleaseParam = nullptr;
    std::atomic<float>* triggerSourceParam = nullptr;
    std::atomic<float>* gateOnParam = nullptr;
    std::atomic<float>* gateThresholdParam = nullptr;
    std::atomic<float>* gateHoldParam = nullptr;
    std::atomic<float>* gateReleaseParam = nullptr;

    // The two six-band sections, cached the same way. Sixty getRawParameterValue
    // lookups per block would be sixty string hashes on the audio thread.
    struct DecayBandParams
    {
        std::atomic<float>* freq = nullptr;
        std::atomic<float>* mult = nullptr;
        std::atomic<float>* q = nullptr;
    };

    struct PostBandParams
    {
        std::atomic<float>* on = nullptr;
        std::atomic<float>* type = nullptr;
        std::atomic<float>* freq = nullptr;
        std::atomic<float>* gain = nullptr;
        std::atomic<float>* q = nullptr;
        std::atomic<float>* slope = nullptr;
        std::atomic<float>* channel = nullptr;
    };

    std::array<DecayBandParams, hrzn::numEqBands> decayBandParams {};
    std::array<PostBandParams, hrzn::numEqBands> postBandParams {};

    double hostSampleRate = 48000.0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (HorizonAudioProcessor)
};
