#include "PluginProcessor.h"
#include "PluginEditor.h"

HorizonAudioProcessor::HorizonAudioProcessor()
    // The sidechain arrives disabled (v0.5): a reverb that appears in the host
    // demanding a second input it does not need is a reverb people route
    // around. Enabling it is the host's answer to the Trigger control being
    // set to Sidechain.
    : AudioProcessor (BusesProperties()
                          .withInput ("Input", juce::AudioChannelSet::stereo(), true)
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                          .withInput ("Sidechain", juce::AudioChannelSet::stereo(), false)),
      apvts (*this, nullptr, "HorizonState", hrzn::createParameterLayout())
{
    spaceParam      = apvts.getRawParameterValue (hrzn::id::space);
    decayTimeParam  = apvts.getRawParameterValue (hrzn::id::decayTime);
    decayEqOnParam  = apvts.getRawParameterValue (hrzn::id::decayEqOn);
    postEqOnParam   = apvts.getRawParameterValue (hrzn::id::postEqOn_);
    predelayParam   = apvts.getRawParameterValue (hrzn::id::predelay);
    mixParam        = apvts.getRawParameterValue (hrzn::id::mix);
    widthParam      = apvts.getRawParameterValue (hrzn::id::width);
    brightnessParam = apvts.getRawParameterValue (hrzn::id::brightness);
    distanceParam   = apvts.getRawParameterValue (hrzn::id::distance);
    characterParam  = apvts.getRawParameterValue (hrzn::id::character);
    thicknessParam  = apvts.getRawParameterValue (hrzn::id::thickness);
    freezeParam     = apvts.getRawParameterValue (hrzn::id::freeze);
    bypassParam     = apvts.getRawParameterValue (hrzn::id::bypass);

    predelaySyncParam = apvts.getRawParameterValue (hrzn::id::predelaySync);
    predelayDivParam  = apvts.getRawParameterValue (hrzn::id::predelayDiv);

    duckOnParam        = apvts.getRawParameterValue (hrzn::id::duckOn);
    duckThresholdParam = apvts.getRawParameterValue (hrzn::id::duckThreshold);
    duckAmountParam    = apvts.getRawParameterValue (hrzn::id::duckAmount);
    duckAttackParam    = apvts.getRawParameterValue (hrzn::id::duckAttack);
    duckReleaseParam   = apvts.getRawParameterValue (hrzn::id::duckRelease);
    triggerSourceParam = apvts.getRawParameterValue (hrzn::id::triggerSource);
    gateOnParam        = apvts.getRawParameterValue (hrzn::id::gateOn);
    gateThresholdParam = apvts.getRawParameterValue (hrzn::id::gateThreshold);
    gateHoldParam      = apvts.getRawParameterValue (hrzn::id::gateHold);
    gateReleaseParam   = apvts.getRawParameterValue (hrzn::id::gateRelease);

    for (int b = 0; b < hrzn::numEqBands; ++b)
    {
        auto& d = decayBandParams[(size_t) b];
        d.freq = apvts.getRawParameterValue (hrzn::id::decayEqFreq (b));
        d.mult = apvts.getRawParameterValue (hrzn::id::decayEqMult (b));
        d.q    = apvts.getRawParameterValue (hrzn::id::decayEqQ (b));

        auto& q = postBandParams[(size_t) b];
        q.on      = apvts.getRawParameterValue (hrzn::id::postEqOn (b));
        q.type    = apvts.getRawParameterValue (hrzn::id::postEqType (b));
        q.freq    = apvts.getRawParameterValue (hrzn::id::postEqFreq (b));
        q.gain    = apvts.getRawParameterValue (hrzn::id::postEqGain (b));
        q.q       = apvts.getRawParameterValue (hrzn::id::postEqQ (b));
        q.slope   = apvts.getRawParameterValue (hrzn::id::postEqSlope (b));
        q.channel = apvts.getRawParameterValue (hrzn::id::postEqChan (b));
    }

    // A snapshot applied wholesale is not a knob move: tell the engine to
    // relocate rather than morph. The flag is read and cleared on the audio
    // thread, which is the only place the engine may be touched.
    //
    // It is also where the MIDI CC cache is rebuilt (feature 27, v0.8). Undo,
    // redo, an A/B switch and a preset load all replace the whole tree, and
    // the cache is built *from* the tree — so a cache that survived one of
    // them would be routing controllers according to a state the session had
    // left. Both jobs belong to the same hook because they are the same event:
    // "everything you derived from the tree is now stale".
    stateManager.onStateReapplied = [this]
    {
        relocateRequested.store (true);
        rebuildMidiCcMap();
    };
}

HorizonAudioProcessor::~HorizonAudioProcessor()
{
    // Dropped first, so a fit that finishes during the wait below finds an
    // expired token and does nothing rather than reaching into a half-torn-down
    // processor.
    lifetime.reset();
    irPool.removeAllJobs (true, 8000);
}

bool HorizonAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto& in = layouts.getMainInputChannelSet();
    const auto& out = layouts.getMainOutputChannelSet();

    // v0.7: mono, stereo, 5.1, 7.1, 7.1.4 and 9.1.6, in == out.
    //
    // In == out is not a limitation being conceded, it is what a reverb is: the
    // wet field is generated *at* the output speakers, so a host asking for a
    // stereo-in / 7.1-out instance is asking Horizon to be an upmixer, and an
    // upmixer that guessed would be wrong in a way nobody could correct. Feed it
    // seven point one and pan into it.
    if (in != out || in.isDisabled())
        return false;

    if (! hrzn::isSupportedLayout (in))
        return false;

    // The sidechain (v0.5) may be absent, mono or stereo whatever the main bus
    // is doing — it is summed to a mono trigger before anything looks at it,
    // so its width never has to match.
    if (layouts.inputBuses.size() > 1)
    {
        const auto& sc = layouts.getChannelSet (true, 1);
        if (! sc.isDisabled()
            && sc != juce::AudioChannelSet::mono()
            && sc != juce::AudioChannelSet::stereo())
            return false;
    }

    return true;
}

void HorizonAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    hostSampleRate = sampleRate;

    // The layout, not the channel count. Two hosts can both say "eight
    // channels" and mean 7.1 and 7.1 SDDS, which put the surrounds and the LFE
    // in different places — and where the LFE is decides which channel gets no
    // reverb at all. See SpeakerLayout.
    speakers.set (getChannelLayoutOfBus (false, 0));
    engine.prepare (sampleRate, speakers);

    bypassMix.reset (sampleRate, 0.01);
    bypassMix.setCurrentAndTargetValue (bypassParam->load() > 0.5f ? 1.0f : 0.0f);

    dryBuffer.setSize (maxChans, samplesPerBlock);
    dryBuffer.clear();

    // The wet tap is always allocated for the engine's two tank channels, even
    // on a mono layout: the tank itself is stereo and the analyzer wants what
    // the tank produced, not what survived the fold-down.
    wetTapBuffer.setSize (2, samplesPerBlock);
    wetTapBuffer.clear();

    keyBuffer.setSize (1, samplesPerBlock);
    keyBuffer.clear();
    wetGainBuffer.setSize (1, samplesPerBlock);
    wetGainBuffer.clear();

    duckGate.prepare (sampleRate);

    convolutionBuffer.setSize (2, samplesPerBlock);
    convolutionBuffer.clear();

    convolver.prepare ({ sampleRate, (juce::uint32) samplesPerBlock, 2 });

    compareMix.reset (sampleRate, 0.03);
    compareMix.setCurrentAndTargetValue (irCompare.load() ? 1.0f : 0.0f);
    compareDryWet.reset (sampleRate, 0.02);
    compareDryWet.setCurrentAndTargetValue (mixParam->load() * 0.01f);

    inputFifo.clear();
    wetFifo.clear();

    // Predelay is part of the effect, not a processing delay: the dry path
    // is never delayed, so the plugin reports no latency.
    setLatencySamples (0);
}

void HorizonAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                          juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals noDenormals;

    // MIDI Learn capture and mapped CC -> parameter (v0.8). Ahead of the early
    // returns below: a controller move has to reach its parameter on a block
    // this plugin decides not to process, or a fader would stop working
    // whenever the transport ran into silence.
    handleMidi (midi);

    const auto blockStartTicks = juce::Time::getHighResolutionTicks();

    auto mainBuf = getBusBuffer (buffer, false, 0);
    const int numSamples = buffer.getNumSamples();
    const int numChannels = juce::jmin (mainBuf.getNumChannels(), maxChans);

    if (numSamples == 0 || numChannels == 0)
        return;

    // Keep the raw input for the click-free bypass crossfade.
    dryBuffer.setSize (numChannels, numSamples, false, false, true);
    for (int ch = 0; ch < numChannels; ++ch)
        dryBuffer.copyFrom (ch, 0, mainBuf, ch, 0, numSamples);

    float inPk = 0.0f;
    for (int ch = 0; ch < numChannels; ++ch)
        inPk = juce::jmax (inPk, mainBuf.getMagnitude (ch, 0, numSamples));

    // Predelay tempo sync (feature 18, v0.5). The host's tempo is cached
    // rather than read fresh each block, because a host that stops reporting
    // one — offline bounce, a transport that isn't rolling, a standalone build
    // with no clock at all — should leave the predelay where it was rather
    // than snap it to a default nobody chose.
    if (auto* host = getPlayHead())
        if (const auto position = host->getPosition())
            if (const auto bpm = position->getBpm())
                if (*bpm > 0.0)
                    hostBpm.store ((float) *bpm);

    const bool predelaySynced = predelaySyncParam->load() > 0.5f;
    const float predelayMs = predelaySynced
        ? juce::jlimit (hrzn::minPredelayMs, hrzn::maxPredelayMs,
                        hrzn::divisionMilliseconds ((int) predelayDivParam->load(),
                                                    (double) hostBpm.load()))
        : predelayParam->load();

    displayPredelayMs.store (predelayMs);

    hrzn::ReverbEngine::Params p;
    p.space01      = spaceParam->load() * 0.01f;
    p.decaySeconds = decayTimeParam->load();
    p.predelayMs   = predelayMs;
    p.mix01        = mixParam->load() * 0.01f;
    p.width        = widthParam->load() * 0.01f;
    p.brightness01 = brightnessParam->load() * 0.01f;
    p.distance01   = distanceParam->load() * 0.01f;
    p.character01  = characterParam->load() * 0.01f;
    p.thickness01  = thicknessParam->load() * 0.01f;
    p.freeze       = freezeParam->load() > 0.5f;

    // Section bypasses (v0.4.1). Neither needs engine support: "off" for the
    // Decay Rate EQ is every multiplier at 1.00x, which its designer already
    // resolves to no filter at all, and "off" for the Post EQ is every band
    // switched off. Expressing them as parameter transformations here means
    // the per-band smoothing that already exists carries the transition, so
    // toggling a section is as click-free as moving one of its bands.
    const bool decayEqOn = decayEqOnParam->load() > 0.5f;
    const bool postEqOn = postEqOnParam->load() > 0.5f;

    for (int b = 0; b < hrzn::numEqBands; ++b)
    {
        const auto bi = (size_t) b;

        auto& dst = p.decayBands[bi];
        const auto& src = decayBandParams[bi];
        dst.on = true;               // decay bands are always present; 1.00x is "off"
        dst.shape = hrzn::decayEqShapes[bi];
        dst.freq = src.freq->load();
        dst.mult = decayEqOn ? src.mult->load() : 1.0f;
        dst.q    = src.q->load();

        auto& pdst = p.postBands[bi];
        const auto& psrc = postBandParams[bi];
        pdst.on = postEqOn && psrc.on->load() > 0.5f;
        pdst.shape = (hrzn::FilterShape) juce::jlimit (
            0, hrzn::filterShapeNames.size() - 1, (int) psrc.type->load());
        pdst.channel = (hrzn::ChannelMode) juce::jlimit (
            0, hrzn::channelModeNames.size() - 1, (int) psrc.channel->load());
        pdst.freq = psrc.freq->load();
        pdst.gainDb = psrc.gain->load();
        pdst.q = psrc.q->load();
        pdst.slopeIndex = juce::jlimit (0, (int) hrzn::slopeOrders.size() - 1,
                                        (int) psrc.slope->load());
    }

    // A/B against the source response (feature 6, v0.6). Both sides run at
    // once: switching has to be instant, and an engine restarted from silence
    // on every switch would answer with a tail that is not the tail being
    // compared.
    //
    // The blend is applied below as a *correction* to what the engine already
    // produced, rather than by taking the mix over from it. The first version
    // did take it over — the engine was told Mix 0 and the processor did the
    // blend — and it clicked audibly on the way back, because the engine's own
    // mix smoother then had to ramp from 0 to wherever Mix was while the
    // processor had already stopped covering for it. Written as a correction
    // the transition is continuous by construction: at zero blend the term is
    // exactly zero and the output is the engine's, bit for bit.
    const bool comparing = irCompare.load() && irLoaded.load();
    compareMix.setTargetValue (comparing ? 1.0f : 0.0f);

    const bool compareActive = comparing || compareMix.isSmoothing()
                            || compareMix.getCurrentValue() > 0.0f;

    compareDryWet.setTargetValue (p.mix01);

    engine.setParams (p);

    if (relocateRequested.exchange (false))
        engine.relocate();

    float* chans[maxChans] = {};
    for (int ch = 0; ch < numChannels; ++ch)
        chans[ch] = mainBuf.getWritePointer (ch);

    wetTapBuffer.setSize (2, numSamples, false, false, true);
    float* wetTap[2] = { wetTapBuffer.getWritePointer (0),
                         wetTapBuffer.getWritePointer (1) };

    // Ducking and the Auto Gate (feature 7, v0.5) ------------------------
    //
    // The trigger is summed to mono before the detector sees it. A ducker
    // whose depth depends on where the source is panned is a ducker that
    // behaves differently on every track it is put on.
    hrzn::DuckGate::Params dg;
    dg.duckOn          = duckOnParam->load() > 0.5f;
    dg.duckThresholdDb = duckThresholdParam->load();
    dg.duckAmountDb    = duckAmountParam->load();
    dg.duckAttackMs    = duckAttackParam->load();
    dg.duckReleaseMs   = duckReleaseParam->load();
    dg.gateOn          = gateOnParam->load() > 0.5f;
    dg.gateThresholdDb = gateThresholdParam->load();
    dg.gateHoldMs      = gateHoldParam->load();
    dg.gateReleaseMs   = gateReleaseParam->load();
    duckGate.setParams (dg);

    // Asked after the parameters are in, so switching a control off does not
    // cut the ramp back to unity short: idle means both off *and* both gains
    // home, and until then the block still gets computed.
    const float* wetGain = nullptr;

    if (! duckGate.idle())
    {
        keyBuffer.setSize (1, numSamples, false, false, true);
        wetGainBuffer.setSize (1, numSamples, false, false, true);

        float* key = keyBuffer.getWritePointer (0);
        const bool useSidechain =
            (hrzn::TriggerSource) (int) triggerSourceParam->load() == hrzn::TriggerSource::sidechain;

        // A Sidechain setting with nothing patched into it leaves the key
        // silent, which reads as "the source has stopped": ducking opens all
        // the way up and the gate shuts. That is the truthful answer — the
        // trigger really is silent — and the trace shows it, which is how you
        // find out the send was never connected.
        const auto* scBus = useSidechain ? getBus (true, 1) : nullptr;

        if (scBus != nullptr && scBus->isEnabled())
        {
            const auto sc = getBusBuffer (buffer, true, 1);
            const int scChans = sc.getNumChannels();

            if (scChans > 0)
            {
                const float scale = 1.0f / (float) scChans;
                for (int i = 0; i < numSamples; ++i)
                {
                    float sum = 0.0f;
                    for (int ch = 0; ch < scChans; ++ch)
                        sum += sc.getSample (ch, i);
                    key[i] = sum * scale;
                }
            }
            else
            {
                keyBuffer.clear (0, 0, numSamples);
            }
        }
        else if (useSidechain)
        {
            keyBuffer.clear (0, 0, numSamples);
        }
        else
        {
            // The dry copy, not the main buffer: the engine works in place, so
            // by the time this is read the main buffer is already reverb.
            //
            // The LFE is left out of the sum. It is not sent to the reverb, so
            // it is not part of what the reverb is ducking under — and it is the
            // one channel a mix routinely runs ten decibels hot, which on a
            // shared trigger would mean a ducker that behaves differently the
            // moment a session gains a subwoofer.
            int keyChans = 0;
            for (int ch = 0; ch < numChannels; ++ch)
                if (ch >= speakers.size() || ! speakers[ch].lfe)
                    ++keyChans;

            const float scale = 1.0f / (float) juce::jmax (1, keyChans);
            for (int i = 0; i < numSamples; ++i)
            {
                float sum = 0.0f;
                for (int ch = 0; ch < numChannels; ++ch)
                    if (ch >= speakers.size() || ! speakers[ch].lfe)
                        sum += dryBuffer.getSample (ch, i);
                key[i] = sum * scale;
            }
        }

        float* gains = wetGainBuffer.getWritePointer (0);
        duckGate.process (key, gains, numSamples);
        wetGain = gains;

        triggerLevelDb.store (duckGate.keyLevelDb());
        duckDepthDb.store (duckGate.duckDepthDb());
        gateGain.store (duckGate.gateGain());
    }
    else
    {
        triggerLevelDb.store (-140.0f);
        duckDepthDb.store (0.0f);
        gateGain.store (1.0f);
    }

    engine.process (chans, numChannels, numSamples, wetTap, wetGain);

    if (compareActive)
    {
        // The convolution is fed the dry signal, not the engine's output.
        convolutionBuffer.setSize (2, numSamples, false, false, true);
        for (int ch = 0; ch < 2; ++ch)
            convolutionBuffer.copyFrom (ch, 0, dryBuffer,
                                        numChannels > 1 ? ch : 0, 0, numSamples);

        juce::dsp::AudioBlock<float> block (convolutionBuffer);
        juce::dsp::ProcessContextReplacing<float> context (block);
        convolver.process (context);

        const float* convL = convolutionBuffer.getReadPointer (0);
        const float* convR = convolutionBuffer.getReadPointer (1);
        const float convGain = convolutionGain.load();

        for (int i = 0; i < numSamples; ++i)
        {
            const float c = compareMix.getNextValue();
            const float mix = compareDryWet.getNextValue();

            // Ducking and the Auto Gate act on whichever wet is playing: they
            // are part of how the reverb sits in the mix, not part of the
            // reverb, so switching source must not switch them off.
            const float gain = (wetGain != nullptr ? wetGain[i] : 1.0f) * convGain;

            const float convWetL = convL[i] * gain;
            const float convWetR = convR[i] * gain;

            // How far the wet has moved from the engine's towards the
            // convolution's, scaled by the same Mix the engine applied. The
            // engine's output already carries its own wet at that Mix, so
            // adding the difference lands exactly on the blended one.
            const float deltaL = (convWetL - wetTap[0][i]) * c;
            const float deltaR = (convWetR - wetTap[1][i]) * c;

            if (numChannels > 1)
            {
                chans[0][i] += deltaL * mix;
                chans[1][i] += deltaR * mix;
            }
            else
            {
                chans[0][i] += 0.5f * (deltaL + deltaR) * mix;
            }

            // The analyzer follows whichever wet is playing.
            wetTap[0][i] += deltaL;
            wetTap[1][i] += deltaR;
        }
    }

    // Analyzer feeds. Pushed before the bypass crossfade so the display keeps
    // showing what the reverb is doing while the plugin is bypassed — which is
    // the one time you are looking at it rather than listening to it.
    inputFifo.push (dryBuffer.getReadPointer (0),
                    dryBuffer.getReadPointer (numChannels > 1 ? 1 : 0), numSamples);
    wetFifo.push (wetTap[0], wetTap[1], numSamples);

    // Bypass crossfade back to the raw dry signal.
    bypassMix.setTargetValue (bypassParam->load() > 0.5f ? 1.0f : 0.0f);
    if (bypassMix.isSmoothing() || bypassMix.getCurrentValue() > 0.0f)
    {
        for (int i = 0; i < numSamples; ++i)
        {
            const float dry = bypassMix.getNextValue();
            const float wet = 1.0f - dry;
            for (int ch = 0; ch < numChannels; ++ch)
                chans[ch][i] = chans[ch][i] * wet + dryBuffer.getSample (ch, i) * dry;
        }
    }

    float outPk = 0.0f;
    for (int ch = 0; ch < numChannels; ++ch)
        outPk = juce::jmax (outPk, mainBuf.getMagnitude (ch, 0, numSamples));

    // Peak-hold for the GUI (released when the GUI reads with exchange (0)).
    float prev = peakIn.load();
    while (inPk > prev && ! peakIn.compare_exchange_weak (prev, inPk)) {}
    prev = peakOut.load();
    while (outPk > prev && ! peakOut.compare_exchange_weak (prev, outPk)) {}

    displayModelIndex.store (engine.currentModelIndex());
    displayMorphLow.store (engine.morphLowIndex());
    displayMorphHigh.store (engine.morphHighIndex());
    displayMorphFraction.store (engine.morphFraction());
    displaySpacePosition.store (engine.currentSpacePosition());
    displayRt60.store (engine.deliveredRt60Seconds());

    for (int k = 0; k < hrzn::DecayEq::numReportPoints; ++k)
    {
        displayDecayCurve[(size_t) k].store (engine.decaySecondsAt (k));
        displayNeutralCurve[(size_t) k].store (engine.neutralDecaySecondsAt (k));
    }

    // Smoothed fraction-of-realtime CPU estimate for the GUI.
    const double elapsed = juce::Time::highResolutionTicksToSeconds (
        juce::Time::getHighResolutionTicks() - blockStartTicks);
    const double budget = (double) numSamples / hostSampleRate;
    const float load = (float) (elapsed / juce::jmax (budget, 1.0e-9));
    cpuLoad.store (0.9f * cpuLoad.load() + 0.1f * load);
}

// Impulse response import (feature 6, v0.6) ---------------------------------

HorizonAudioProcessor::IrImport
HorizonAudioProcessor::analyseImpulseResponse (const juce::File& file,
                                               const hrzn::ir::ProgressFn& progress)
{
    IrImport result;
    result.name = file.getFileNameWithoutExtension();

    const double rate = hostSampleRate > 0.0 ? hostSampleRate : 48000.0;
    auto loaded = hrzn::ir::loadImpulseResponse (file, rate);

    result.sourceSampleRate = loaded.sourceSampleRate;
    result.sourceChannels = loaded.sourceChannels;
    result.sourceSeconds = loaded.sourceSeconds;

    if (! loaded.valid)
    {
        result.error = loaded.error;
        return result;
    }

    result.match = hrzn::ir::fitToIr (loaded.mono.data(), (int) loaded.mono.size(),
                                      rate, progress);

    if (! result.match.valid)
    {
        // The loader was happy and the fit was not, which means the file is
        // audio but not a room: too short a tail, or no tail at all.
        result.error = "No reverb tail to match — is this an impulse response?";
        return result;
    }

    result.audio = std::move (loaded.audio);
    result.valid = true;
    return result;
}

void HorizonAudioProcessor::applyImpulseResponse (const IrImport& import)
{
    if (! import.valid)
        return;

    const auto& m = import.match;

    // Applied as one transaction so the whole conversion is one undo. Anyone
    // who imports an IR and dislikes the result wants the state they had back,
    // not thirty parameter changes to walk backwards through.
    auto state = apvts.copyState();

    auto set = [&state] (const juce::String& paramId, float value)
    {
        auto node = state.getChildWithProperty ("id", paramId);
        if (node.isValid())
            node.setProperty ("value", (double) value, nullptr);
    };

    set (hrzn::id::space, m.space01 * 100.0f);
    set (hrzn::id::decayTime, m.decaySeconds);
    set (hrzn::id::predelay, m.predelayMs);
    set (hrzn::id::predelaySync, 0.0f);   // a fitted time is a time, not a note
    set (hrzn::id::distance, m.distance01 * 100.0f);

    // The fit voices the room through the two EQ sections and leaves the
    // shaping controls at their midpoints, so the result is a state a user can
    // read: whatever the imported room does differently is visible on the
    // curves rather than baked into Character or Thickness.
    set (hrzn::id::brightness, 50.0f);
    set (hrzn::id::character, 50.0f);
    set (hrzn::id::thickness, 50.0f);

    set (hrzn::id::decayEqOn, 1.0f);
    set (hrzn::id::postEqOn_, 1.0f);

    for (int b = 0; b < hrzn::numEqBands; ++b)
    {
        set (hrzn::id::decayEqFreq (b), hrzn::decayEqDefaultFreq[(size_t) b]);
        set (hrzn::id::decayEqMult (b), m.decayMult[(size_t) b]);
        set (hrzn::id::decayEqQ (b), hrzn::decayEqDefaultQ[(size_t) b]);

        const bool used = std::abs (m.postGainDb[(size_t) b]) > 0.05f;
        set (hrzn::id::postEqOn (b), used ? 1.0f : 0.0f);
        set (hrzn::id::postEqType (b), (float) (int) hrzn::postEqDefaultShape[(size_t) b]);
        set (hrzn::id::postEqFreq (b), m.postFreq[(size_t) b]);
        set (hrzn::id::postEqGain (b), m.postGainDb[(size_t) b]);
        set (hrzn::id::postEqQ (b), m.postQ[(size_t) b]);
    }

    // Mix is left exactly where it was, whether or not Lock Mix is on. An
    // import is a statement about the room, not about how much of it you want.
    stateManager.applyState (state);
    stateManager.setCurrentPresetName (import.name);

    lastImport = import;
    irLoaded.store (true);

    // Loaded *un*-normalised, and level-matched by hand instead. JUCE's
    // normalisation is a sensible default for a convolution reverb and the
    // wrong one here: it scales the response to a fixed energy that has
    // nothing to do with what the engine happens to produce, which left the
    // two sides of the A/B 17 dB apart. The ratio of the two measured RMS
    // figures is the gain that puts them on the same footing, and both were
    // measured by the same extraction on the same kind of signal.
    const float sourceRms = m.target.rms;
    const float fittedRms = m.fitted.valid ? m.fitted.rms : m.target.rms;

    convolutionGain.store (sourceRms > 1.0e-9f
                               ? juce::jlimit (0.02f, 50.0f, fittedRms / sourceRms)
                               : 1.0f);

    if (import.audio.getNumSamples() > 0)
    {
        auto copy = import.audio;
        convolver.loadImpulseResponse (std::move (copy),
                                       hostSampleRate > 0.0 ? hostSampleRate : 48000.0,
                                       juce::dsp::Convolution::Stereo::yes,
                                       juce::dsp::Convolution::Trim::no,
                                       juce::dsp::Convolution::Normalise::no);
    }
}

void HorizonAudioProcessor::beginImpulseResponseImport (const juce::File& file)
{
    if (irBusy.exchange (true))
        return;   // one at a time

    irProgress.store (0.0f);

    irPool.addJob ([this, file]
    {
        auto result = analyseImpulseResponse (file, [this] (float p)
        {
            irProgress.store (p);
        });

        // Parameters may only be touched on the message thread, so the fit
        // hands its result back rather than applying it where it ran.
        std::weak_ptr<int> token = lifetime;
        juce::MessageManager::callAsync ([this, token, result = std::move (result)]
        {
            if (token.expired())
                return;

            if (result.valid)
                applyImpulseResponse (result);
            else
                lastImport = result;   // keeps the error for the panel

            irProgress.store (1.0f);
            irBusy.store (false);
        });
    });
}

bool HorizonAudioProcessor::importImpulseResponse (const juce::File& file,
                                                   const hrzn::ir::ProgressFn& progress)
{
    auto result = analyseImpulseResponse (file, progress);

    if (! result.valid)
    {
        lastImport = result;   // keeps the error for the panel to show
        return false;
    }

    applyImpulseResponse (result);
    return true;
}

void HorizonAudioProcessor::setIrCompare (bool shouldCompare)
{
    irCompare.store (shouldCompare && irLoaded.load());
}

// Lock Mix (feature 17) -----------------------------------------------------

void HorizonAudioProcessor::setMixLocked (bool shouldLock)
{
    mixLocked.store (shouldLock);

    // Mirrored into the tree so the session remembers it. Not read back except
    // when a session is loaded — the member is the authority for as long as
    // the plugin is open, which is what stops an A/B switch or an undo from
    // carrying somebody else's idea of the setting into the present.
    apvts.state.setProperty (hrzn::prop::lockMix, shouldLock, nullptr);
    stateManager.noteNonParameterChange();
}

// A preset's Mix is dropped on the floor while the lock is on, before the
// state is applied rather than after. Restoring it afterwards would work and
// would also put a second entry in the undo history and a visible jump on the
// control, which is two ways of telling the user something happened that they
// specifically asked not to have happen.
void HorizonAudioProcessor::applyPresetState (const juce::ValueTree& incoming)
{
    if (! mixLocked.load())
    {
        stateManager.applyState (incoming);
        return;
    }

    auto state = incoming.createCopy();
    auto mixNode = state.getChildWithProperty ("id", hrzn::id::mix);

    if (mixNode.isValid())
        mixNode.setProperty ("value", (double) mixParam->load(), nullptr);

    stateManager.applyState (state);
}

// MIDI Learn (feature 26) -----------------------------------------------------
//
// Three threads meet here and each is given exactly one job:
//
//   message thread   arms a target, and owns the ValueTree
//   audio thread     captures the CC and applies mapped controllers
//   message thread   persists what the audio thread captured
//
// The audio thread never allocates, never locks and never reads the tree. What
// it writes is two atomics — the parameter and the controller number — which
// commitPendingMidiLearn() picks up on the next editor tick, or on the next
// save if there is no editor open.

const juce::Identifier HorizonAudioProcessor::midiMapType { "MIDIMAP" };

static const juce::Identifier midiMapEntryType ("MAP");
static const juce::Identifier midiMapCcProp ("cc");
static const juce::Identifier midiMapParamProp ("param");

void HorizonAudioProcessor::handleMidi (juce::MidiBuffer& midi)
{
    for (const auto meta : midi)
    {
        const auto msg = meta.getMessage();
        if (! msg.isController())
            continue;

        const int cc = msg.getControllerNumber();
        if (cc < 0 || cc >= (int) ccTargets.size())
            continue;

        // An armed learn takes the next controller that moves, and the mapping
        // is live from that instant — the knob follows the fader on the very
        // message that taught it, rather than on the next one.
        if (auto* target = learnTarget.exchange (nullptr))
        {
            ccTargets[(size_t) cc].store (target);
            pendingLearnParam.store (target, std::memory_order_relaxed);
            pendingLearnCc.store (cc, std::memory_order_release);
        }

        if (auto* mapped = ccTargets[(size_t) cc].load())
            mapped->setValueNotifyingHost ((float) msg.getControllerValue() / 127.0f);
    }
}

void HorizonAudioProcessor::armMidiLearn (const juce::String& paramId)
{
    learnTarget.store (apvts.getParameter (paramId));
}

int HorizonAudioProcessor::getMappedCc (const juce::String& paramId) const
{
    auto map = apvts.state.getChildWithName (midiMapType);
    for (int i = 0; i < map.getNumChildren(); ++i)
        if (map.getChild (i)[midiMapParamProp].toString() == paramId)
            return map.getChild (i)[midiMapCcProp];
    return -1;
}

void HorizonAudioProcessor::clearMidiMapping (const juce::String& paramId)
{
    auto map = apvts.state.getChildWithName (midiMapType);
    for (int i = map.getNumChildren(); --i >= 0;)
        if (map.getChild (i)[midiMapParamProp].toString() == paramId)
            map.removeChild (i, nullptr);

    rebuildMidiCcMap();
    stateManager.noteNonParameterChange();
}

void HorizonAudioProcessor::commitPendingMidiLearn()
{
    const int cc = pendingLearnCc.exchange (-1, std::memory_order_acquire);
    if (cc < 0)
        return;

    auto* param = pendingLearnParam.load (std::memory_order_relaxed);
    if (param == nullptr)
        return;

    auto map = apvts.state.getOrCreateChildWithName (midiMapType, nullptr);

    // One CC per parameter and one parameter per CC. Both directions matter:
    // without the first, relearning a control leaves the old controller still
    // driving it; without the second, one fader ends up moving two knobs and
    // there is no screen that shows you why.
    for (int i = map.getNumChildren(); --i >= 0;)
    {
        auto entry = map.getChild (i);
        if ((int) entry[midiMapCcProp] == cc
            || entry[midiMapParamProp].toString() == param->paramID)
            map.removeChild (i, nullptr);
    }

    juce::ValueTree entry (midiMapEntryType);
    entry.setProperty (midiMapCcProp, cc, nullptr);
    entry.setProperty (midiMapParamProp, param->paramID, nullptr);
    map.appendChild (entry, nullptr);

    rebuildMidiCcMap();

    // The mapping changed the state tree without touching a parameter, so the
    // history has no way of knowing on its own.
    stateManager.noteNonParameterChange();
}

void HorizonAudioProcessor::rebuildMidiCcMap()
{
    for (auto& t : ccTargets)
        t.store (nullptr);

    auto map = apvts.state.getChildWithName (midiMapType);
    for (int i = 0; i < map.getNumChildren(); ++i)
    {
        auto entry = map.getChild (i);
        const int cc = entry[midiMapCcProp];
        if (cc >= 0 && cc < (int) ccTargets.size())
            ccTargets[(size_t) cc].store (
                apvts.getParameter (entry[midiMapParamProp].toString()));
    }
}

// State ---------------------------------------------------------------------

void HorizonAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // A learn captured since the last editor tick is still only in the
    // atomics. Flushing here is what makes a headless session save — a host
    // closing a project with no plugin window open — keep the mapping.
    commitPendingMidiLearn();

    apvts.state.setProperty (hrzn::prop::lockMix, mixLocked.load(), nullptr);

    if (auto xml = stateManager.saveSession().createXml())
    {
        xml->setAttribute ("stateVersion", stateVersion);
        copyXmlToBinary (*xml, destData);
    }
}

// v1 -> v2: Decay Rate was a percentage of whatever room Space was sitting on,
// and is now an absolute time. The conversion needs both values, so it is done
// per state subtree — a session carries two of them, one per A/B slot — and it
// reads the room from the same subtree's own `space`, which is exactly what the
// engine would have done with that pair.
//
// A state that already carries `decay_time` is left alone, so this is safe to
// run over anything.
void HorizonAudioProcessor::migrateDecayRateToTime (juce::XmlElement& xml)
{
    juce::XmlElement* rate = nullptr;
    juce::XmlElement* space = nullptr;
    bool hasTime = false;

    for (auto* child : xml.getChildWithTagNameIterator ("PARAM"))
    {
        const auto paramId = child->getStringAttribute ("id");
        if (paramId == hrzn::id::decayTime) hasTime = true;
        if (paramId == hrzn::id::space)     space = child;
        if (paramId == "decay_rate")        rate = child;
    }

    if (rate != nullptr && ! hasTime)
    {
        const float pc = juce::jlimit (hrzn::minDecayRatePc, hrzn::maxDecayRatePc,
                                       (float) rate->getDoubleAttribute ("value", 100.0));
        const float space01 = space != nullptr
                            ? (float) space->getDoubleAttribute ("value", 55.0) * 0.01f
                            : 0.55f;

        const float seconds = juce::jlimit (
            hrzn::minDecaySeconds, hrzn::maxDecaySeconds,
            hrzn::nominalRt60At (hrzn::spaceToPosition (space01)) * pc * 0.01f);

        auto* migrated = xml.createNewChildElement ("PARAM");
        migrated->setAttribute ("id", hrzn::id::decayTime);
        migrated->setAttribute ("value", (double) seconds);
    }

    for (auto* child : xml.getChildIterator())
        if (! child->hasTagName ("PARAM"))
            migrateDecayRateToTime (*child);
}

bool HorizonAudioProcessor::applyStateXml (const juce::XmlElement& xmlIn)
{
    if (xmlIn.getIntAttribute ("stateVersion", 0) > stateVersion)
        return false; // written by a newer schema than this build understands

    juce::XmlElement xml (xmlIn);
    migrateDecayRateToTime (xml);

    // Lock Mix is read from the file rather than from the tree once it has
    // been applied, and only when the file actually carries it. A state
    // written before v0.5 — or a bare preset, which never carries it at all —
    // therefore leaves the current setting alone instead of clearing it.
    if (const auto* lockAttr = findLockMixAttribute (xml))
        mixLocked.store (lockAttr->getBoolAttribute (hrzn::prop::lockMix.toString()));

    // Session: both A/B slots + which is active.
    if (xml.hasTagName (stateManager.sessionType.toString()))
    {
        const bool ok = stateManager.loadSession (juce::ValueTree::fromXml (xml));
        apvts.state.setProperty (hrzn::prop::lockMix, mixLocked.load(), nullptr);
        return ok;
    }

    // Bare preset state: one state, seeded into both slots.
    if (xml.hasTagName (apvts.state.getType()))
    {
        stateManager.loadLegacyState (juce::ValueTree::fromXml (xml));
        apvts.state.setProperty (hrzn::prop::lockMix, mixLocked.load(), nullptr);
        return true;
    }

    return false;
}

// The property sits on whichever state tree carries it — the session root has
// one per A/B slot — so the first one found wins. It is a session-wide switch
// and the slots cannot disagree about it in any way worth honouring.
const juce::XmlElement* HorizonAudioProcessor::findLockMixAttribute (const juce::XmlElement& xml)
{
    if (xml.hasAttribute (hrzn::prop::lockMix.toString()))
        return &xml;

    for (auto* child : xml.getChildIterator())
        if (const auto* found = findLockMixAttribute (*child))
            return found;

    return nullptr;
}

void HorizonAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary (data, sizeInBytes))
        applyStateXml (*xml);
}

juce::AudioProcessorEditor* HorizonAudioProcessor::createEditor()
{
    return new HorizonAudioProcessorEditor (*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new HorizonAudioProcessor();
}
