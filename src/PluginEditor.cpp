#include "PluginEditor.h"

using namespace hrzn;

namespace
{
    // What the hints bar says when the pointer is not over anything that has
    // help. It has to say how to turn the bar off, because a bar that appears
    // and cannot explain itself is worse than no bar.
    juce::String hintBarIdleText()
    {
        return "Hover any control for help. H hides this bar, F1 lists the "
               "keyboard shortcuts.";
    }
}

HorizonAudioProcessorEditor::HorizonAudioProcessorEditor (HorizonAudioProcessor& p)
    : AudioProcessorEditor (p), processor (p)
{
    setLookAndFeel (&lookAndFeel);
    setWantsKeyboardFocus (true);

    content.onPaint = [this] (juce::Graphics& g) { paintContent (g); };
    addAndMakeVisible (content);

    {
        juce::PropertiesFile::Options o;
        o.applicationName = "Horizon";
        o.folderName = "Equinox Audio";
        o.filenameSuffix = "settings";
        o.osxLibrarySubFolder = "Application Support";
        settings = std::make_unique<juce::PropertiesFile> (o);
    }

    // Header ---------------------------------------------------------------
    undoButton.onClick = [this] { processor.stateManager.undo(); };
    redoButton.onClick = [this] { processor.stateManager.redo(); };

    slotAButton.onClick = [this] { processor.stateManager.setActiveSlot (0); };
    slotBButton.onClick = [this] { processor.stateManager.setActiveSlot (1); };
    copySlotButton.onClick = [this] { processor.stateManager.copyActiveToOther(); };

    prevPresetButton.onClick = [this] { processor.presets.loadAdjacent (false); };
    nextPresetButton.onClick = [this] { processor.presets.loadAdjacent (true); };
    presetNameButton.onClick = [this] { presetBrowser.openBrowser(); };

    compactButton.setClickingTogglesState (true);
    compactButton.onClick = [this] { setCompact (compactButton.getToggleState(), true); };

    fullScreenButton.setClickingTogglesState (true);
    fullScreenButton.onClick = [this] { setFullScreen (fullScreenButton.getToggleState()); };

    bypassButton.setClickingTogglesState (true);
    bypassAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (
        processor.apvts, id::bypass, bypassButton);
    registerMidiLearn (bypassButton, id::bypass);

    helpButton.onClick = [this] { showHelpMenu(); };

    for (auto* b : { &undoButton, &redoButton, &slotAButton, &slotBButton, &copySlotButton,
                     &prevPresetButton, &nextPresetButton, &presetNameButton,
                     &compactButton, &fullScreenButton, &helpButton, &bypassButton })
        content.addAndMakeVisible (*b);

    registerHelp (undoButton,       "Undo",              help::undo);
    registerHelp (redoButton,       "Redo",              help::redo);
    registerHelp (slotAButton,      "A/B state A",       help::slotA);
    registerHelp (slotBButton,      "A/B state B",       help::slotB);
    registerHelp (copySlotButton,   "Copy to other A/B slot", help::copyAb);
    registerHelp (prevPresetButton, "Previous preset",   help::presetPrev);
    registerHelp (nextPresetButton, "Next preset",       help::presetNext);
    registerHelp (presetNameButton, "Preset browser",    help::presetName);
    registerHelp (compactButton,    "Compact layout",    help::compact);
    registerHelp (fullScreenButton, "Full screen",       help::fullScreen);
    registerHelp (helpButton,       "Help",              help::helpButton);
    registerHelp (bypassButton,     "Bypass",            help::bypass);

    cpuLabel.setFont (juce::Font (juce::FontOptions (11.0f)));
    cpuLabel.setColour (juce::Label::textColourId, colors::textDim);
    cpuLabel.setJustificationType (juce::Justification::centredRight);
    content.addAndMakeVisible (cpuLabel);

    // Display ---------------------------------------------------------------
    content.addAndMakeVisible (display);
    content.addAndMakeVisible (bandRow);

    display.onSelectionChanged = [this] (ReverbDisplay::Section s, int band)
    {
        bandRow.setBand (s, band);
        updateSectionTabs();
    };
    display.onPrefsChanged = [this] { savePrefs(); };

    for (auto* tab : { &decayEqTab, &postEqTab })
    {
        tab->setClickingTogglesState (true);
        tab->setRadioGroupId (0x4571);
        content.addAndMakeVisible (*tab);
    }

    decayEqTab.onClick = [this] { display.setSection (EqSection::decayEq); };
    postEqTab.onClick  = [this] { display.setSection (EqSection::postEq); };

    registerHelp (decayEqTab, "Decay Rate EQ", help::decayEqTab);
    registerHelp (postEqTab,  "Post EQ",       help::postEqTab);
    registerHelp (display,    "Reverb display", help::displayText);

    decayCurveLabel.setFont (juce::Font (juce::FontOptions (11.0f)));
    decayCurveLabel.setColour (juce::Label::textColourId, colors::textDim);
    decayCurveLabel.setJustificationType (juce::Justification::centredRight);
    content.addAndMakeVisible (decayCurveLabel);

    // Knobs ----------------------------------------------------------------
    initKnob (spaceKnob, id::space, "SPACE");
    initKnob (decayKnob, id::decayTime, "DECAY TIME");
    initKnob (mixKnob, id::mix, "MIX");
    initKnob (predelayKnob, id::predelay, "PREDELAY");
    initKnob (distanceKnob, id::distance, "DISTANCE");
    initKnob (characterKnob, id::character, "CHARACTER");
    initKnob (thicknessKnob, id::thickness, "THICKNESS");
    initKnob (widthKnob, id::width, "WIDTH");
    initKnob (brightnessKnob, id::brightness, "BRIGHTNESS");

    initKnob (duckThresholdKnob, id::duckThreshold, "THRESHOLD");
    initKnob (duckAmountKnob, id::duckAmount, "AMOUNT");
    initKnob (duckAttackKnob, id::duckAttack, "ATTACK");
    initKnob (duckReleaseKnob, id::duckRelease, "RELEASE");
    initKnob (gateThresholdKnob, id::gateThreshold, "THRESHOLD");
    initKnob (gateHoldKnob, id::gateHold, "HOLD");
    initKnob (gateReleaseKnob, id::gateRelease, "RELEASE");

    // Space is the one control where the exact position matters and the knob
    // alone cannot show it: the parameter reads out as a room name to the host
    // (which is what you want in an automation lane), so on the knob itself it
    // reads as a number instead. The room names live in the readout below.
    spaceKnob.slider.textFromValueFunction = [] (double v) { return juce::String (v, 2); };
    spaceKnob.slider.updateText();
    spaceKnob.slider.onPopupMenu = [this] { showSpaceMenu(); };
    decayKnob.slider.onPopupMenu = [this] { showDecayMenu(); };

    registerHelp (spaceKnob.slider,      "Space",       help::space);
    registerHelp (decayKnob.slider,      "Decay Time",  help::decayTime);
    registerHelp (mixKnob.slider,        "Mix",         help::mix);
    registerHelp (predelayKnob.slider,   "Predelay",    help::predelay);
    registerHelp (distanceKnob.slider,   "Distance",    help::distance);
    registerHelp (characterKnob.slider,  "Character",   help::character);
    registerHelp (thicknessKnob.slider,  "Thickness",   help::thickness);
    registerHelp (widthKnob.slider,      "Width",       help::width);
    registerHelp (brightnessKnob.slider, "Brightness",  help::brightness);

    registerHelp (duckThresholdKnob.slider, "Duck Threshold", help::duckThreshold);
    registerHelp (duckAmountKnob.slider,    "Duck Amount",    help::duckAmount);
    registerHelp (duckAttackKnob.slider,    "Duck Attack",    help::duckAttack);
    registerHelp (duckReleaseKnob.slider,   "Duck Release",   help::duckRelease);
    registerHelp (gateThresholdKnob.slider, "Gate Threshold", help::gateThreshold);
    registerHelp (gateHoldKnob.slider,      "Gate Hold",      help::gateHold);
    registerHelp (gateReleaseKnob.slider,   "Gate Release",   help::gateRelease);

    // Freeze ---------------------------------------------------------------
    freezeButton.setClickingTogglesState (true);
    freezeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (
        processor.apvts, id::freeze, freezeButton);
    registerHelp (freezeButton, "Freeze", help::freeze);
    registerMidiLearn (freezeButton, id::freeze);
    content.addAndMakeVisible (freezeButton);

    // Lock Mix (v0.5) --------------------------------------------------------
    lockMixButton.setClickingTogglesState (true);
    lockMixButton.setToggleState (processor.isMixLocked(), juce::dontSendNotification);
    lockMixButton.onClick = [this]
    {
        processor.setMixLocked (lockMixButton.getToggleState());
    };
    registerHelp (lockMixButton, "Lock Mix", help::lockMix);
    content.addAndMakeVisible (lockMixButton);

    // Predelay tempo sync (v0.5) ---------------------------------------------
    predelaySyncButton.setClickingTogglesState (true);
    predelaySyncAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (
        processor.apvts, id::predelaySync, predelaySyncButton);
    predelaySyncButton.onStateChange = [this] { updatePredelayMode(); };
    registerHelp (predelaySyncButton, "Predelay tempo sync", help::predelaySync);
    registerMidiLearn (predelaySyncButton, id::predelaySync);
    content.addAndMakeVisible (predelaySyncButton);

    predelayDivBox.addItemList (noteDivisionNames(), 1);
    predelayDivAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (
        processor.apvts, id::predelayDiv, predelayDivBox);
    registerHelp (predelayDivBox, "Predelay division", help::predelayDiv);
    registerMidiLearn (predelayDivBox, id::predelayDiv);
    content.addChildComponent (predelayDivBox);

    predelaySyncedLabel.setFont (juce::Font (juce::FontOptions (11.0f)));
    predelaySyncedLabel.setColour (juce::Label::textColourId, colors::textDim);
    predelaySyncedLabel.setJustificationType (juce::Justification::centred);
    content.addChildComponent (predelaySyncedLabel);

    // Ducking and the Auto Gate (v0.5) ---------------------------------------
    duckOnButton.setClickingTogglesState (true);
    duckOnAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (
        processor.apvts, id::duckOn, duckOnButton);
    registerHelp (duckOnButton, "Ducking", help::duckOn);
    duckOnButton.onStateChange = [this] { updateDuckGateVisibility(); };
    registerMidiLearn (duckOnButton, id::duckOn);
    content.addAndMakeVisible (duckOnButton);

    gateOnButton.setClickingTogglesState (true);
    gateOnAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (
        processor.apvts, id::gateOn, gateOnButton);
    registerHelp (gateOnButton, "Auto Gate", help::gateOn);
    gateOnButton.onStateChange = [this] { updateDuckGateVisibility(); };
    registerMidiLearn (gateOnButton, id::gateOn);
    content.addAndMakeVisible (gateOnButton);

    triggerSourceBox.addItemList (triggerSourceNames, 1);
    triggerSourceAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (
        processor.apvts, id::triggerSource, triggerSourceBox);
    registerHelp (triggerSourceBox, "Trigger source", help::triggerSource);
    registerMidiLearn (triggerSourceBox, id::triggerSource);
    content.addAndMakeVisible (triggerSourceBox);

    // Impulse response import (v0.6) -----------------------------------------
    importIrButton.onClick = [this] { showIrImportChooser(); };
    registerHelp (importIrButton, "Import impulse response", help::importIr);
    content.addAndMakeVisible (importIrButton);

    irCompareButton.setClickingTogglesState (true);
    irCompareButton.setEnabled (false);
    irCompareButton.onClick = [this]
    {
        processor.setIrCompare (irCompareButton.getToggleState());
    };
    registerHelp (irCompareButton, "Compare with the imported response", help::irCompare);
    content.addAndMakeVisible (irCompareButton);

    irLabel.setFont (juce::Font (juce::FontOptions (11.0f)));
    irLabel.setColour (juce::Label::textColourId, colors::textDim);
    irLabel.setJustificationType (juce::Justification::centredLeft);
    content.addAndMakeVisible (irLabel);

    auto initReadout = [this] (juce::Label& l, juce::Justification just)
    {
        l.setFont (juce::Font (juce::FontOptions (12.0f, juce::Font::bold)));
        l.setColour (juce::Label::textColourId, colors::accent);
        l.setJustificationType (just);
        content.addAndMakeVisible (l);
    };
    initReadout (roomLabel, juce::Justification::centredLeft);
    initReadout (decayLabel, juce::Justification::centredRight);
    decayLabel.setColour (juce::Label::textColourId, colors::textDim);

    // Meter strip ----------------------------------------------------------
    content.addAndMakeVisible (inMeter);
    content.addAndMakeVisible (outMeter);
    inMeter.setScale (-60.0f);
    outMeter.setScale (-60.0f);

    auto initMeterLabel = [this] (juce::Label& l, const char* text)
    {
        l.setText (text, juce::dontSendNotification);
        l.setFont (juce::Font (juce::FontOptions (11.0f, juce::Font::bold)));
        l.setColour (juce::Label::textColourId, colors::textDim);
        l.setJustificationType (juce::Justification::centred);
        content.addAndMakeVisible (l);
    };
    initMeterLabel (inLabel, "IN");
    initMeterLabel (outLabel, "OUT");

    registerHelp (inMeter,     "Input level",  help::meterStrip);
    registerHelp (outMeter,    "Output level", help::meterStrip);
    registerHelp (roomLabel,   "Current room", help::roomReadout);
    registerHelp (decayLabel,  "Delivered decay", help::decayReadout);
    registerHelp (bandRow,     "Selected band", help::bandLevel);

    // The help layer (v0.8) --------------------------------------------------
    hintBar.setFont (juce::Font (juce::FontOptions (12.0f)));
    hintBar.setJustificationType (juce::Justification::centredLeft);
    hintBar.setColour (juce::Label::textColourId, colors::text);
    hintBar.setColour (juce::Label::backgroundColourId, colors::panel.withAlpha (0.6f));
    hintBar.setInterceptsMouseClicks (false, false);
    content.addChildComponent (hintBar);

    // One listener for the whole tree, rather than one per control. The hints
    // bar has to react to the pointer crossing anything at all — including the
    // gaps between controls, where the right answer is the idle text — so a
    // per-control listener would be both more code and less correct.
    addMouseListener (this, true);

    // Overlays -------------------------------------------------------------
    content.addChildComponent (presetBrowser);
    content.addChildComponent (shortcutsOverlay);

    // Resizable shell -------------------------------------------------------
    loadPrefs();
    updateSectionTabs();
    updatePredelayMode();
    bandRow.setBand (display.getSection(), display.getSelectedBand());

    // Seeded before the first sizing rather than left to the switches to
    // discover: a session that was saved with the ducker on has to open at the
    // height that shows it, not open short and then jump.
    duckGateExpanded = duckOnButton.getToggleState() || gateOnButton.getToggleState();
    updateDuckGateVisibility();

    setResizable (true, true);

    // resized() writes ui_scale back on the very first layout, so a settings
    // file written before the default changed holds the *old* default rather
    // than a size anybody chose — and would go on opening the window at the
    // size this changes. Re-seed once and mark the file, so a resize made
    // after this point is never overruled.
    if (! settings->getBoolValue ("ui_scale_rebased", false))
    {
        settings->setValue ("ui_scale", defaultUiScale);
        settings->setValue ("ui_scale_rebased", true);
        settings->saveIfNeeded();
    }

    applyLogicalSize (juce::jlimit (0.5, 2.0,
                                    settings->getDoubleValue ("ui_scale", defaultUiScale)));

    // GPU-accelerated compositing; repaints stay timer-driven.
    glContext.setContinuousRepainting (false);
    glContext.attachTo (*this);

    startTimerHz (30);
}

HorizonAudioProcessorEditor::~HorizonAudioProcessorEditor()
{
    glContext.detach();
    settings->saveIfNeeded();
    setLookAndFeel (nullptr);
}

// Preferences ----------------------------------------------------------------

void HorizonAudioProcessorEditor::loadPrefs()
{
    compact = settings->getBoolValue ("ui_compact", false);
    compactButton.setToggleState (compact, juce::dontSendNotification);

    // Accessibility and help are per-user, not per-session: somebody who needs
    // a high-contrast palette needs it in every project they open, and it has
    // nothing to do with how the reverb is set.
    colors::setHighContrast (settings->getBoolValue ("high_contrast", false));
    lookAndFeel.applyPalette();
    lookAndFeel.setTextScale ((float) settings->getDoubleValue ("text_scale", 1.0));

    helpHintsOn = settings->getBoolValue ("help_hints", false);
    hintBar.setVisible (helpHintsOn);
    if (helpHintsOn)
        hintBar.setText (hintBarIdleText(), juce::dontSendNotification);

    ReverbDisplay::ViewPrefs p;
    p.showInput = settings->getBoolValue ("analyzer_input", p.showInput);
    p.showWet = settings->getBoolValue ("analyzer_wet", p.showWet);
    p.pianoScale = settings->getBoolValue ("piano_scale", p.pianoScale);
    p.speedIndex = settings->getIntValue ("analyzer_speed", p.speedIndex);
    display.setPrefs (p);
}

void HorizonAudioProcessorEditor::savePrefs()
{
    const auto p = display.getPrefs();
    settings->setValue ("analyzer_input", p.showInput);
    settings->setValue ("analyzer_wet", p.showWet);
    settings->setValue ("piano_scale", p.pianoScale);
    settings->setValue ("analyzer_speed", p.speedIndex);
    settings->saveIfNeeded();
}

// Scaling shell --------------------------------------------------------------

double HorizonAudioProcessorEditor::currentScale() const
{
    return getWidth() > 0 ? (double) getWidth() / (double) logicalWidth() : 1.0;
}

void HorizonAudioProcessorEditor::applyLogicalSize (double scale)
{
    // The 2x ceiling is a taste limit, not a technical one, so full screen is
    // allowed to go past it — on a 4K display an 0.5x-2x range would leave the
    // flagship view sitting in a quarter of the screen.
    const double maxScale = fullScreen ? juce::jmax (2.0, scale) : 2.0;
    const double clamped = juce::jlimit (0.5, maxScale, scale);

    // The limits are rounded the same way the size is, not truncated: rounding
    // one and truncating the other puts the window a pixel outside its own
    // constrainer at the top of the range, and with a fixed aspect ratio the
    // correction for that is a visible snap.
    if (auto* c = getConstrainer())
    {
        c->setFixedAspectRatio ((double) logicalWidth() / (double) logicalHeight());
        c->setSizeLimits (juce::roundToInt (logicalWidth() * 0.5),
                          juce::roundToInt (logicalHeight() * 0.5),
                          juce::roundToInt (logicalWidth() * maxScale),
                          juce::roundToInt (logicalHeight() * maxScale));
    }

    setSize (juce::roundToInt (logicalWidth() * clamped),
             juce::roundToInt (logicalHeight() * clamped));
}

void HorizonAudioProcessorEditor::setCompact (bool shouldBeCompact, bool save)
{
    if (compact == shouldBeCompact)
        return;

    // The aspect ratio is pinned, so a layout change has to re-derive the
    // window size from the *scale* rather than keep the width: keeping the
    // width would silently rescale everything by the height ratio.
    const double scale = currentScale();

    compact = shouldBeCompact;
    compactButton.setToggleState (compact, juce::dontSendNotification);
    display.setVisible (! compact);
    bandRow.setVisible (! compact);
    decayEqTab.setVisible (! compact);
    postEqTab.setVisible (! compact);
    decayCurveLabel.setVisible (! compact);

    duckOnButton.setVisible (! compact);
    gateOnButton.setVisible (! compact);
    importIrButton.setVisible (! compact);
    irCompareButton.setVisible (! compact);
    irLabel.setVisible (! compact);

    // The trigger source and the seven knobs are the block's own business, and
    // whether they show depends on more than Compact.
    updateDuckGateVisibility();

    applyLogicalSize (scale);

    if (save)
    {
        settings->setValue ("ui_compact", compact);
        settings->saveIfNeeded();
    }
}

void HorizonAudioProcessorEditor::setFullScreen (bool shouldBeFullScreen)
{
    if (fullScreen == shouldBeFullScreen)
        return;

    fullScreenButton.setToggleState (shouldBeFullScreen, juce::dontSendNotification);

    if (! shouldBeFullScreen)
    {
        fullScreen = false;
        const double scale = sizeBeforeFullScreen > 0
                           ? (double) sizeBeforeFullScreen / (double) logicalWidth()
                           : defaultUiScale;
        applyLogicalSize (scale);
        settings->setValue ("ui_scale", currentScale());
        settings->saveIfNeeded();
        return;
    }

    sizeBeforeFullScreen = getWidth();
    fullScreen = true;

    // Fit the logical aspect into whatever the display offers. A plugin editor
    // cannot take the screen away from its host, so "full screen" here means
    // "as large as this display will show", which is the useful reading of it
    // in a DAW anyway.
    double scale = 2.0;
    if (auto* d = juce::Desktop::getInstance().getDisplays()
                      .getDisplayForRect (getScreenBounds()))
    {
        const auto area = d->userBounds;
        scale = juce::jmin ((double) area.getWidth() / (double) logicalWidth(),
                            (double) area.getHeight() / (double) logicalHeight());
    }

    applyLogicalSize (juce::jmax (0.5, scale));
}

// A switched-off ducker has nothing to say with four knobs, and a switched-off
// gate nothing with three: the numbers on them describe a process that is not
// running. Each half carries its knobs only while it is on, and with neither on
// the block stands down to its two switches and the window gives the height
// back — which is the state almost every preset is in.
//
// The two switches themselves never leave: what collapses has to be one click
// from coming back, or the feature is hidden rather than tidied away.
void HorizonAudioProcessorEditor::updateDuckGateVisibility()
{
    const bool duck = duckOnButton.getToggleState();
    const bool gate = gateOnButton.getToggleState();

    for (auto* k : { &duckThresholdKnob, &duckAmountKnob, &duckAttackKnob, &duckReleaseKnob })
    {
        k->slider.setVisible (! compact && duck);
        k->caption.setVisible (! compact && duck);
    }

    for (auto* k : { &gateThresholdKnob, &gateHoldKnob, &gateReleaseKnob })
    {
        k->slider.setVisible (! compact && gate);
        k->caption.setVisible (! compact && gate);
    }

    // One trigger feeds both halves, so it belongs to the block rather than to
    // either one, and it goes when the block does.
    triggerSourceBox.setVisible (! compact && (duck || gate));

    const bool expanded = duck || gate;
    if (expanded == duckGateExpanded)
        return;

    // logicalHeight has just changed underneath us, so the window has to be
    // re-derived from the scale it was at — the same reasoning setCompact uses,
    // and for the same pinned aspect ratio.
    const double scale = currentScale();
    duckGateExpanded = expanded;
    applyLogicalSize (scale);

    // Compact is one height whatever the block is doing, and at the ends of the
    // resize range the clamp can land on the size we are already at. Either way
    // setSize would not have fired resized, and the block still has to be laid
    // out at its new height.
    layoutContent();
}

void HorizonAudioProcessorEditor::updateSectionTabs()
{
    const bool decay = (display.getSection() == EqSection::decayEq);
    decayEqTab.setToggleState (decay, juce::dontSendNotification);
    postEqTab.setToggleState (! decay, juce::dontSendNotification);
}

// Knobs ----------------------------------------------------------------------

void HorizonAudioProcessorEditor::initKnob (Knob& k, const juce::String& paramId,
                                            const juce::String& caption)
{
    k.slider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    k.slider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 88, 16);
    k.slider.setTitle (caption);            // screen readers
    k.slider.setWantsKeyboardFocus (true);  // tab traversal + arrow keys
    k.attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (
        processor.apvts, paramId, k.slider);

    // Scroll-wheel moves are routed through the parameter rather than the
    // slider so they arrive at the host wrapped in a change gesture; the
    // attachment carries the value back to the knob.
    k.slider.onWheelValue = [this, paramId] (double newValue)
    {
        if (auto* param = processor.apvts.getParameter (paramId))
        {
            param->beginChangeGesture();
            param->setValueNotifyingHost (param->convertTo0to1 ((float) newValue));
            param->endChangeGesture();
        }
    };

    // Every knob is learnable, and gets the menu that says so. Space and Decay
    // Time replace this handler with one that carries their own section as
    // well — the MIDI part is the same either way.
    k.slider.onPopupMenu = [this, slider = &k.slider, paramId]
    {
        showParameterMenu (*slider, paramId);
    };
    registerMidiLearn (k.slider, paramId);

    content.addAndMakeVisible (k.slider);

    k.caption.setText (caption, juce::dontSendNotification);
    k.caption.setFont (juce::Font (juce::FontOptions (11.0f, juce::Font::bold)));
    k.caption.setColour (juce::Label::textColourId, colors::textDim);
    k.caption.setJustificationType (juce::Justification::centred);
    content.addAndMakeVisible (k.caption);
}

void HorizonAudioProcessorEditor::layoutKnob (Knob& k, juce::Rectangle<int> area)
{
    k.caption.setBounds (area.removeFromTop (16));
    k.slider.setBounds (area);
}

void HorizonAudioProcessorEditor::layoutKnobWithSwitch (Knob& k, juce::Rectangle<int> area,
                                                        juce::Component& sw, int switchWidth)
{
    auto captionRow = area.removeFromTop (16);
    sw.setBounds (captionRow.removeFromRight (switchWidth));
    captionRow.removeFromRight (4);

    // Reserving the switch's width on the far side as well is what keeps the
    // caption centred over its knob instead of nudged off it — and a caption
    // that does not sit over the middle of a knob reads as a knob that is not
    // where it should be. It costs twice the switch, so the narrow Predelay
    // slot cannot afford it and keeps the arrangement it has had since v0.5:
    // a name shortened to stay centred is the worse trade of the two.
    const auto centred = captionRow.withTrimmedLeft (switchWidth + 4);
    k.caption.setBounds (centred.getWidth() >= 84 ? centred : captionRow);

    k.slider.setBounds (area);
}

// The Predelay slot holds either the free millisecond knob or the note
// division, never both. Swapping the control rather than greying one out is
// the honest presentation: a knob that is showing 28 ms while the delay line
// is running 250 ms is a lie, and one that is greyed out still shows the
// number it is not delivering.
void HorizonAudioProcessorEditor::updatePredelayMode()
{
    const bool synced = predelaySyncButton.getToggleState();

    predelayKnob.slider.setVisible (! synced);
    predelayDivBox.setVisible (synced);
    predelaySyncedLabel.setVisible (synced);
}

void HorizonAudioProcessorEditor::layoutPredelay (juce::Rectangle<int> area)
{
    layoutKnobWithSwitch (predelayKnob, area, predelaySyncButton, 40);
    area.removeFromTop (16);

    // The division sits where the knob's own value would, and the time it
    // comes to at the current tempo sits where the knob's text box would —
    // so the slot reads the same way in both modes.
    auto synced = area.reduced (6, 0);
    synced.removeFromTop (juce::jmax (0, (synced.getHeight() - 46) / 2));
    predelayDivBox.setBounds (synced.removeFromTop (24));
    synced.removeFromTop (4);
    predelaySyncedLabel.setBounds (synced.removeFromTop (16));
}

// The help layer (feature 30, v0.8) ------------------------------------------
//
// One registration per control, three uses. The tooltip is what a mouse user
// gets, the hints bar is what somebody who does not want to hover and wait
// gets, and the accessibility description is what a screen reader reads — and
// all three are the same sentence, because a control whose tooltip and whose
// spoken description disagree is a control with two manuals.

void HorizonAudioProcessorEditor::registerHelp (juce::Component& c,
                                                const juce::String& title,
                                                const char* text)
{
    const auto help = juce::String::fromUTF8 (text);

    helpTexts[&c] = help;
    c.setTitle (title);
    c.setHelpText (help);

    if (auto* client = dynamic_cast<juce::SettableTooltipClient*> (&c))
        client->setTooltip (help);
}

// The help for whatever the pointer is over, walking outwards: a knob's text
// box and a combo box's arrow are children of the control they belong to, and
// the answer for both is the control's own help.
juce::String HorizonAudioProcessorEditor::helpTextAt (const juce::Component* c) const
{
    for (; c != nullptr && c != this; c = c->getParentComponent())
    {
        const auto it = helpTexts.find (c);
        if (it != helpTexts.end())
            return it->second;
    }
    return {};
}

void HorizonAudioProcessorEditor::mouseMove (const juce::MouseEvent& e)
{
    if (! helpHintsOn)
        return;

    const auto text = helpTextAt (e.eventComponent);
    hintBar.setText (text.isNotEmpty() ? text : hintBarIdleText(),
                     juce::dontSendNotification);
}

void HorizonAudioProcessorEditor::mouseExit (const juce::MouseEvent&)
{
    if (helpHintsOn)
        hintBar.setText (hintBarIdleText(), juce::dontSendNotification);
}

void HorizonAudioProcessorEditor::setHelpHints (bool on)
{
    if (helpHintsOn == on)
        return;

    helpHintsOn = on;
    hintBar.setVisible (on);
    hintBar.setText (on ? hintBarIdleText() : juce::String(), juce::dontSendNotification);

    if (settings != nullptr)
        settings->setValue ("help_hints", on);

    // The bar changes logicalHeight, so the shell has to re-derive its size at
    // the scale it is currently at rather than keep its pixel height.
    applyLogicalSize (currentScale());
}

void HorizonAudioProcessorEditor::setHighContrast (bool on)
{
    colors::setHighContrast (on);
    lookAndFeel.applyPalette();

    if (settings != nullptr)
        settings->setValue ("high_contrast", on);

    sendLookAndFeelChange();
    repaint();
}

void HorizonAudioProcessorEditor::setTextScale (float scale)
{
    lookAndFeel.setTextScale (scale);

    if (settings != nullptr)
        settings->setValue ("text_scale", scale);

    sendLookAndFeelChange();
    repaint();
}

void HorizonAudioProcessorEditor::showShortcutsOverlay (bool shouldShow)
{
    shortcutsOverlay.setVisible (shouldShow);
    if (shouldShow)
        shortcutsOverlay.toFront (true);
}

void HorizonAudioProcessorEditor::showHelpMenu()
{
    const float scale = lookAndFeel.getTextScale();

    juce::PopupMenu sizes;
    sizes.setLookAndFeel (&lookAndFeel);
    sizes.addItem (10, "100%", true, juce::approximatelyEqual (scale, 1.0f));
    sizes.addItem (11, "115%", true, juce::approximatelyEqual (scale, 1.15f));
    sizes.addItem (12, "130%", true, juce::approximatelyEqual (scale, 1.3f));

    juce::PopupMenu menu;
    menu.setLookAndFeel (&lookAndFeel);
    menu.addItem (1, "Open Manual...");
    menu.addItem (2, "Keyboard Shortcuts (F1)");
    menu.addSeparator();
    menu.addItem (3, "Help Hints Bar (H)", true, helpHintsOn);
    menu.addItem (4, "High Contrast", true, colors::highContrast);
    menu.addSubMenu ("Text Size", sizes);

    juce::Component::SafePointer<HorizonAudioProcessorEditor> safeThis (this);

    menu.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (&helpButton),
                        [safeThis] (int result)
    {
        if (safeThis == nullptr)
            return;

        switch (result)
        {
            case 1:  safeThis->openManual(); break;
            case 2:  safeThis->showShortcutsOverlay (true); break;
            case 3:  safeThis->setHelpHints (! safeThis->helpHintsOn); break;
            case 4:  safeThis->setHighContrast (! colors::highContrast); break;
            case 10: safeThis->setTextScale (1.0f); break;
            case 11: safeThis->setTextScale (1.15f); break;
            case 12: safeThis->setTextScale (1.3f); break;
            default: break;
        }
    });
}

void HorizonAudioProcessorEditor::openManual()
{
    // The manual lives in the binary — there is no installer until v0.9 — so
    // it is written out beside the settings file and handed to the browser.
    auto file = juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
                    .getChildFile ("Equinox Audio").getChildFile ("Horizon Manual.html");

    file.getParentDirectory().createDirectory();

    if (file.replaceWithText (help::manualHtml()))
        juce::URL (file).launchInDefaultBrowser();
}

void HorizonAudioProcessorEditor::ShortcutsOverlay::paint (juce::Graphics& g)
{
    g.fillAll (colors::background.withAlpha (0.82f));

    constexpr int rowH = 21;
    const int numRows = (int) std::size (help::shortcuts);

    auto panel = getLocalBounds()
                     .withSizeKeepingCentre (460, numRows * rowH + 58)
                     .toFloat();

    g.setColour (colors::panel);
    g.fillRoundedRectangle (panel, 6.0f);
    g.setColour (colors::outline);
    g.drawRoundedRectangle (panel.reduced (0.5f), 6.0f, 1.0f);

    auto inner = panel.reduced (16.0f, 12.0f);
    g.setColour (colors::text);
    g.setFont (juce::Font (juce::FontOptions (14.0f, juce::Font::bold)));
    g.drawText ("KEYBOARD SHORTCUTS", inner.removeFromTop (26.0f),
                juce::Justification::centredLeft);

    for (const auto& s : help::shortcuts)
    {
        auto row = inner.removeFromTop ((float) rowH);
        g.setColour (colors::accent);
        g.setFont (juce::Font (juce::FontOptions (12.5f, juce::Font::bold)));
        g.drawText (s.keys, row.removeFromLeft (162.0f), juce::Justification::centredLeft);
        g.setColour (colors::text);
        g.setFont (juce::Font (juce::FontOptions (12.5f)));
        g.drawText (s.action, row, juce::Justification::centredLeft);
    }

    g.setColour (colors::textDim);
    g.setFont (juce::Font (juce::FontOptions (11.0f)));
    g.drawText ("click anywhere or press any key to close",
                panel.reduced (16.0f, 6.0f).removeFromBottom (14.0f),
                juce::Justification::centredRight);
}

// MIDI Learn (feature 26, v0.8) ----------------------------------------------
//
// The menu is the whole user interface for it: right-click a control, choose
// MIDI Learn, move a fader. Item IDs sit far above anything a product-specific
// section uses, so a menu can carry both — Space's fourteen rooms *and* its
// MIDI section — without either having to know the other's numbering.
namespace
{
    constexpr int midiLearnItemId = 9001, midiClearItemId = 9002, midiCancelItemId = 9003;
}

void HorizonAudioProcessorEditor::registerMidiLearn (juce::Component& c,
                                                     const juce::String& paramId)
{
    midiLearnTargets[&c] = paramId;
}

void HorizonAudioProcessorEditor::addMidiLearnSection (juce::PopupMenu& menu,
                                                       const juce::String& paramId)
{
    const int mapped = processor.getMappedCc (paramId);

    menu.addSectionHeader (mapped >= 0 ? "MIDI CC " + juce::String (mapped)
                                       : juce::String ("MIDI"));

    menu.addItem (midiLearnItemId,
                  processor.isMidiLearnArmed()
                      ? juce::String::fromUTF8 ("Learning\xe2\x80\xa6 move a controller")
                      : juce::String ("MIDI Learn"));

    if (mapped >= 0)
        menu.addItem (midiClearItemId, "Remove MIDI CC " + juce::String (mapped));

    if (processor.isMidiLearnArmed())
        menu.addItem (midiCancelItemId, "Cancel MIDI Learn");
}

bool HorizonAudioProcessorEditor::handleMidiLearnChoice (int result,
                                                          const juce::String& paramId)
{
    if (result == midiLearnItemId)  { processor.armMidiLearn (paramId);    return true; }
    if (result == midiClearItemId)  { processor.clearMidiMapping (paramId); return true; }
    if (result == midiCancelItemId) { processor.cancelMidiLearn();          return true; }
    return false;
}

void HorizonAudioProcessorEditor::showParameterMenu (juce::Component& target,
                                                      const juce::String& paramId)
{
    juce::PopupMenu menu;
    menu.setLookAndFeel (&lookAndFeel);
    addMidiLearnSection (menu, paramId);

    juce::Component::SafePointer<HorizonAudioProcessorEditor> safeThis (this);

    menu.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (&target),
                        [safeThis, paramId] (int result)
    {
        if (safeThis != nullptr)
            safeThis->handleMidiLearnChoice (result, paramId);
    });
}

void HorizonAudioProcessorEditor::mouseDown (const juce::MouseEvent& e)
{
    if (! e.mods.isPopupMenu() || e.originalComponent == nullptr)
        return;

    // A slider has already shown its own menu through
    // PrecisionSlider::onPopupMenu — which it has to, because it also has to
    // suppress the drag the right-click would otherwise start. Letting the
    // editor's blanket listener act on it as well would show two.
    if (dynamic_cast<juce::Slider*> (e.originalComponent) != nullptr)
        return;

    const auto it = midiLearnTargets.find (e.originalComponent);
    if (it != midiLearnTargets.end())
        showParameterMenu (*e.originalComponent, it->second);
}

void HorizonAudioProcessorEditor::showSpaceMenu()
{
    auto* param = processor.apvts.getParameter (id::space);
    if (param == nullptr)
        return;

    const float current = processor.apvts.getRawParameterValue (id::space)->load();
    const int nearest = spaceToModelIndex (current * 0.01f);

    juce::PopupMenu menu;
    menu.setLookAndFeel (&lookAndFeel);
    menu.addSectionHeader ("Jump to room");

    for (int i = 0; i < numRoomModels; ++i)
        menu.addItem (i + 1, roomModels[(size_t) i].name, true, i == nearest);

    addMidiLearnSection (menu, id::space);

    juce::Component::SafePointer<HorizonAudioProcessorEditor> safeThis (this);

    menu.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (spaceKnob.slider),
                        [safeThis, param] (int result)
    {
        if (safeThis == nullptr || result <= 0)
            return;

        if (safeThis->handleMidiLearnChoice (result, id::space))
            return;

        const float value = 100.0f * (float) (result - 1) / (float) (numRoomModels - 1);
        param->beginChangeGesture();
        param->setValueNotifyingHost (param->convertTo0to1 (value));
        param->endChangeGesture();

        // Choosing a room by name is a discrete act, so the engine lands on it
        // rather than gliding across the table for a second or two — the same
        // relocation a preset load takes. Dragging the knob still morphs.
        safeThis->processor.relocateRequested.store (true);
    });
}

// Decay Time is deliberately independent of Space from v0.4.1, which loses one
// genuinely useful thing the old percentage control gave for free: the room's
// own decay, the figure a real hall that size would ring for. It comes back
// here rather than as a mode, because wanting it is an occasional act and not
// a state the plugin should be in.
void HorizonAudioProcessorEditor::showDecayMenu()
{
    auto* param = processor.apvts.getParameter (id::decayTime);
    if (param == nullptr)
        return;

    const float natural = juce::jlimit (
        minDecaySeconds, maxDecaySeconds,
        nominalRt60At (processor.displaySpacePosition.load()));

    juce::PopupMenu menu;
    menu.setLookAndFeel (&lookAndFeel);
    menu.addSectionHeader ("Decay Time");
    menu.addItem (1, "Use this room's natural decay (" + formatSeconds (natural, 0) + ")");

    addMidiLearnSection (menu, id::decayTime);

    juce::Component::SafePointer<HorizonAudioProcessorEditor> safeThis (this);

    menu.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (decayKnob.slider),
                        [safeThis, param, natural] (int result)
    {
        if (safeThis == nullptr || result <= 0)
            return;

        if (safeThis->handleMidiLearnChoice (result, id::decayTime))
            return;

        if (result != 1)
            return;

        param->beginChangeGesture();
        param->setValueNotifyingHost (param->convertTo0to1 (natural));
        param->endChangeGesture();
    });
}

// Impulse response import ----------------------------------------------------

void HorizonAudioProcessorEditor::showIrImportChooser()
{
    if (processor.irBusy.load())
        return;

    irChooser = std::make_unique<juce::FileChooser> (
        "Import an impulse response", juce::File(), "*.wav;*.aif;*.aiff;*.flac");

    juce::Component::SafePointer<HorizonAudioProcessorEditor> safeThis (this);

    irChooser->launchAsync (juce::FileBrowserComponent::openMode
                                | juce::FileBrowserComponent::canSelectFiles,
                            [safeThis] (const juce::FileChooser& chooser)
    {
        if (safeThis == nullptr)
            return;

        const auto file = chooser.getResult();
        if (file.existsAsFile())
            safeThis->processor.beginImpulseResponseImport (file);
    });
}

// Driven from the editor's timer. The import runs on a worker for several
// seconds, so the panel has to say so — a button that does nothing visible for
// that long reads as a plugin that has hung.
void HorizonAudioProcessorEditor::updateIrPanel()
{
    const bool busy = processor.irBusy.load();

    importIrButton.setEnabled (! busy);

    if (busy)
    {
        irLabel.setColour (juce::Label::textColourId, colors::textDim);
        irLabel.setText ("Analysing... "
                             + juce::String (juce::roundToInt (processor.irProgress.load() * 100.0f))
                             + "%",
                         juce::dontSendNotification);
        return;
    }

    const auto& import = processor.lastImport;

    if (import.error.isNotEmpty())
    {
        irLabel.setColour (juce::Label::textColourId, colors::warn);
        irLabel.setText (import.error, juce::dontSendNotification);
        irCompareButton.setEnabled (false);
        irCompareButton.setToggleState (false, juce::dontSendNotification);
        return;
    }

    if (! import.valid)
    {
        irLabel.setColour (juce::Label::textColourId, colors::textDim);
        irLabel.setText ("No impulse response loaded.", juce::dontSendNotification);
        irCompareButton.setEnabled (false);
        return;
    }

    irCompareButton.setEnabled (true);
    irCompareButton.setToggleState (processor.irCompare.load(), juce::dontSendNotification);

    // The match figure and the residual it came from, together. The percentage
    // is the readable one and the dB is the honest one, and showing only the
    // first would be claiming a precision the fit does not have.
    const auto dot = juce::String::fromUTF8 ("  \xc2\xb7  ");
    irLabel.setColour (juce::Label::textColourId, colors::text);
    irLabel.setText (import.name
                         + dot + juce::String (juce::roundToInt (import.match.matchPercent))
                         + "% match (" + juce::String (import.match.residualDb, 2) + " dB out)"
                         + dot + juce::String (juce::roundToInt (import.sourceSampleRate / 1000.0))
                         + " kHz " + juce::String (import.sourceChannels) + " ch"
                         + dot + juce::String (import.sourceSeconds, 2) + " s",
                     juce::dontSendNotification);
}

// Layout ---------------------------------------------------------------------

void HorizonAudioProcessorEditor::resized()
{
    content.setBounds (0, 0, logicalWidth(), logicalHeight());
    content.setTransform (juce::AffineTransform::scale ((float) currentScale()));

    if (! fullScreen && getWidth() > 0)
    {
        settings->setValue ("ui_scale", currentScale());
        // Written to disk by saveIfNeeded in the destructor: a drag of a window
        // edge fires this on every frame.
    }

    layoutContent();
}

void HorizonAudioProcessorEditor::layoutContent()
{
    auto area = content.getLocalBounds();

    presetBrowser.setBounds (area);
    shortcutsOverlay.setBounds (area);

    // The hints bar takes its strip off the bottom before anything else is
    // placed, and logicalHeight has already made room for it — so the rest of
    // the layout is bit-for-bit what it is with help off.
    if (helpHintsOn)
        hintBar.setBounds (area.removeFromBottom (hintBarHeight));

    // Header.
    auto header = area.removeFromTop (44).reduced (8, 9);
    header.removeFromLeft (150); // mark + title, painted

    undoButton.setBounds (header.removeFromLeft (52));
    header.removeFromLeft (4);
    redoButton.setBounds (header.removeFromLeft (52));
    header.removeFromLeft (14);
    slotAButton.setBounds (header.removeFromLeft (28));
    header.removeFromLeft (4);
    slotBButton.setBounds (header.removeFromLeft (28));
    header.removeFromLeft (4);
    copySlotButton.setBounds (header.removeFromLeft (52));

    bypassButton.setBounds (header.removeFromRight (70));
    header.removeFromRight (6);
    helpButton.setBounds (header.removeFromRight (46));
    header.removeFromRight (4);
    fullScreenButton.setBounds (header.removeFromRight (54));
    header.removeFromRight (4);
    compactButton.setBounds (header.removeFromRight (74));
    header.removeFromRight (10);
    nextPresetButton.setBounds (header.removeFromRight (24));
    presetNameButton.setBounds (header.removeFromRight (150));
    prevPresetButton.setBounds (header.removeFromRight (24));
    header.removeFromRight (8);
    // Wider from v0.7: the layout name shares this readout with the CPU figure.
    cpuLabel.setBounds (header.removeFromRight (112));

    // Display block.
    if (! compact)
    {
        area.removeFromTop (6);
        display.setBounds (area.removeFromTop (268).reduced (8, 0));

        area.removeFromTop (4);
        auto tabs = area.removeFromTop (24).reduced (8, 0);
        decayEqTab.setBounds (tabs.removeFromLeft (132));
        tabs.removeFromLeft (6);
        postEqTab.setBounds (tabs.removeFromLeft (96));
        decayCurveLabel.setBounds (tabs);

        area.removeFromTop (6);
        bandRow.setBounds (area.removeFromTop (54).reduced (8, 0));
    }

    area.removeFromTop (8);

    // Impulse response footer (v0.6): the import, the A/B against the source,
    // and the match report — kept together, because the report belongs next to
    // the button that produced it rather than in a dialog that has to be
    // dismissed before the result can be heard, and kept at the foot of the
    // window, because loading a file and reading what came of it is
    // housekeeping and not one of the reverb's controls.
    //
    // Taken off the bottom before the meters come off the right, so it runs the
    // full width and the meter strip stops level with the panel above it.
    if (! compact)
    {
        area.removeFromBottom (12);
        footerBounds = area.removeFromBottom (24).reduced (16, 0);

        auto strip = footerBounds;
        importIrButton.setBounds (strip.removeFromLeft (104));
        strip.removeFromLeft (8);
        irCompareButton.setBounds (strip.removeFromLeft (92));
        strip.removeFromLeft (12);
        irLabel.setBounds (strip);

        area.removeFromBottom (10);
    }
    else
    {
        footerBounds = {};
    }

    // Meter strip, down the right of the knob block.
    meterPanelBounds = area.removeFromRight (118).withTrimmedRight (8)
                           .withTrimmedBottom (compact ? 10 : 0);
    auto meters = meterPanelBounds.reduced (10, 8);
    auto meterLabels = meters.removeFromBottom (16);
    const int meterW = meters.getWidth() / 2;
    inLabel.setBounds (meterLabels.removeFromLeft (meterW));
    outLabel.setBounds (meterLabels);
    inMeter.setBounds (meters.removeFromLeft (meterW).reduced (8, 0));
    outMeter.setBounds (meters.reduced (8, 0));

    // Knobs.
    auto controls = area.reduced (16, 0);

    auto bigRow = controls.removeFromTop (146);
    const int bigW = bigRow.getWidth() / 3;
    layoutKnob (spaceKnob, bigRow.removeFromLeft (bigW).reduced (10, 0));
    layoutKnobWithSwitch (decayKnob, bigRow.removeFromLeft (bigW).reduced (10, 0),
                          freezeButton, 58);
    layoutKnobWithSwitch (mixKnob, bigRow.reduced (10, 0), lockMixButton, 46);

    controls.removeFromTop (6);

    auto smallRow = controls.removeFromTop (96);
    const int smallW = smallRow.getWidth() / 6;
    layoutPredelay (smallRow.removeFromLeft (smallW).reduced (4, 0));
    layoutKnob (distanceKnob, smallRow.removeFromLeft (smallW).reduced (4, 0));
    layoutKnob (characterKnob, smallRow.removeFromLeft (smallW).reduced (4, 0));
    layoutKnob (thicknessKnob, smallRow.removeFromLeft (smallW).reduced (4, 0));
    layoutKnob (widthKnob, smallRow.removeFromLeft (smallW).reduced (4, 0));
    layoutKnob (brightnessKnob, smallRow.reduced (4, 0));

    controls.removeFromTop (6);

    // The readout row is now only the readout: Freeze and Lock Mix have gone to
    // the knobs they act on, which leaves the room name and its decay the full
    // width to sit in.
    auto readoutRow = controls.removeFromTop (26);
    roomLabel.setBounds (readoutRow.removeFromLeft (readoutRow.getWidth() / 2));
    decayLabel.setBounds (readoutRow);

    if (compact)
    {
        duckGateBounds = {};
        return;
    }

    // Ducking / Auto Gate block (v0.5). Seven columns: four for the ducker,
    // three for the gate, with each section's switch sitting over its own
    // group so the two THRESHOLD captions can never be read as one control.
    // The columns are the same seven whether one half is showing or both, so
    // switching the gate on never shuffles the ducker's knobs sideways.
    controls.removeFromTop (12);
    duckGateBounds = controls.removeFromTop (duckGateExpanded ? 114 : 38);

    auto block = duckGateBounds.reduced (10, 8);
    const int colW = block.getWidth() / 7;
    duckGateDividerX = block.getX() + colW * 4;

    auto strip = block.removeFromTop (22);
    auto duckStrip = strip.removeFromLeft (colW * 4);
    duckOnButton.setBounds (duckStrip.removeFromLeft (92));
    duckStrip.removeFromLeft (10);
    triggerSourceBox.setBounds (duckStrip.removeFromLeft (116).reduced (0, 1));
    gateOnButton.setBounds (strip.removeFromLeft (100));

    block.removeFromTop (6);

    auto knobs = block.removeFromTop (78);
    layoutKnob (duckThresholdKnob, knobs.removeFromLeft (colW).reduced (4, 0));
    layoutKnob (duckAmountKnob, knobs.removeFromLeft (colW).reduced (4, 0));
    layoutKnob (duckAttackKnob, knobs.removeFromLeft (colW).reduced (4, 0));
    layoutKnob (duckReleaseKnob, knobs.removeFromLeft (colW).reduced (4, 0));
    layoutKnob (gateThresholdKnob, knobs.removeFromLeft (colW).reduced (4, 0));
    layoutKnob (gateHoldKnob, knobs.removeFromLeft (colW).reduced (4, 0));
    layoutKnob (gateReleaseKnob, knobs.removeFromLeft (colW).reduced (4, 0));
}

void HorizonAudioProcessorEditor::paintContent (juce::Graphics& g)
{
    g.fillAll (colors::background);

    // Header bar.
    auto header = juce::Rectangle<int> (0, 0, content.getWidth(), 44);
    g.setColour (colors::panel);
    g.fillRect (header);
    g.setColour (colors::outline);
    g.fillRect (header.removeFromBottom (1));

    // Product mark: the family disc with a horizon line across it.
    const float cx = 24.0f, cy = 22.0f, r = 9.0f;
    g.setColour (colors::accent);
    g.drawEllipse (cx - r, cy - r, r * 2.0f, r * 2.0f, 1.6f);
    g.fillRect (cx - r - 3.0f, cy - 0.9f, r * 2.0f + 6.0f, 1.8f);

    g.setColour (colors::text);
    g.setFont (juce::Font (juce::FontOptions (17.0f, juce::Font::bold)));
    g.drawText ("HORIZON", 42, 0, 120, 44, juce::Justification::centredLeft);

    // The display sits in its own recess, with the active section's hue along
    // the top edge — the same pairing the two curves use.
    if (! compact)
    {
        auto frame = display.getBounds().expanded (1);
        g.setColour (colors::outline);
        g.drawRoundedRectangle (frame.toFloat(), 4.0f, 1.0f);

        g.setColour (display.getSection() == EqSection::decayEq ? colors::decayEq
                                                                : colors::postEq);
        g.fillRect (frame.getX() + 6, frame.getY() - 3, frame.getWidth() - 12, 2);
    }

    // Ducking / Auto Gate block: its own recess, with a rule between the two
    // halves. The teal is the trigger's colour everywhere it appears — here on
    // the section that reads it, and on the display's meter that shows it.
    if (! compact && ! duckGateBounds.isEmpty())
    {
        g.setColour (colors::panel);
        g.fillRoundedRectangle (duckGateBounds.toFloat(), 5.0f);

        g.setColour (colors::outline);
        g.drawRoundedRectangle (duckGateBounds.toFloat(), 5.0f, 1.0f);
        g.fillRect (duckGateDividerX, duckGateBounds.getY() + 10,
                    1, duckGateBounds.getHeight() - 20);

        g.setColour (colors::sidechain.withAlpha (0.75f));
        g.fillRect (duckGateBounds.getX() + 6, duckGateBounds.getY() - 3,
                    duckGateBounds.getWidth() - 12, 2);
    }

    // Meter strip panel.
    if (! meterPanelBounds.isEmpty())
    {
        g.setColour (colors::panel);
        g.fillRoundedRectangle (meterPanelBounds.toFloat(), 5.0f);
    }

    // Impulse response footer: a rule above it and nothing else. A second
    // panel down there would give the housekeeping the same weight as the
    // reverb, which is the opposite of why it was moved.
    if (! footerBounds.isEmpty())
    {
        g.setColour (colors::outline);
        g.fillRect (footerBounds.getX(), footerBounds.getY() - 6,
                    footerBounds.getWidth(), 1);
    }
}

// Keyboard -------------------------------------------------------------------

bool HorizonAudioProcessorEditor::keyPressed (const juce::KeyPress& key)
{
    const auto mods = key.getModifiers();

    // Letter key codes arrive in platform-dependent case; normalize.
    int code = key.getKeyCode();
    if (code < 256)
        code = juce::CharacterFunctions::toUpperCase ((juce::juce_wchar) code);

    // F1 is checked before the modifier gates below, because a shortcuts
    // overlay you cannot summon while holding a key is a shortcuts overlay
    // somebody will conclude is broken.
    if (key == juce::KeyPress::F1Key)
    {
        showShortcutsOverlay (! shortcutsOverlay.isVisible());
        return true;
    }

    if (mods.isCommandDown())
    {
        if (code == 'Z' && mods.isShiftDown()) { processor.stateManager.redo(); return true; }
        if (code == 'Z')                       { processor.stateManager.undo(); return true; }
        if (code == 'Y')                       { processor.stateManager.redo(); return true; }
        return false;
    }

    if (code == juce::KeyPress::escapeKey)
    {
        if (shortcutsOverlay.isVisible()) { showShortcutsOverlay (false); return true; }
        if (presetBrowser.isVisible()) { presetBrowser.closeBrowser(); return true; }
        if (fullScreen)                { setFullScreen (false); return true; }
    }

    if (mods.isAnyModifierKeyDown())
        return false;

    switch (code)
    {
        case 'A': processor.stateManager.setActiveSlot (0); return true;
        case 'B': processor.stateManager.setActiveSlot (1); return true;
        case 'C': processor.stateManager.copyActiveToOther(); return true;
        case 'H': setHelpHints (! helpHintsOn); return true;
        case 'K': setCompact (! compact, true); return true;
        case 'F': setFullScreen (! fullScreen); return true;
        case 'E':
            display.setSection (display.getSection() == EqSection::decayEq
                                    ? EqSection::postEq : EqSection::decayEq);
            return true;
        case 'P':
            if (presetBrowser.isVisible()) presetBrowser.closeBrowser();
            else                           presetBrowser.openBrowser();
            return true;
        case '[': processor.presets.loadAdjacent (false); return true;
        case ']': processor.presets.loadAdjacent (true); return true;
        default: break;
    }
    return false;
}

void HorizonAudioProcessorEditor::timerCallback()
{
    // A MIDI Learn captured on the audio thread is written into the tree here,
    // on the message thread that owns it. Cheap when there is nothing pending:
    // one atomic exchange.
    processor.commitPendingMidiLearn();

    inMeter.push (processor.peakIn.exchange (0.0f));
    outMeter.push (processor.peakOut.exchange (0.0f));

    // Space is stepless from v0.2, so the readout names the pair the engine is
    // morphing between and how far through it is — "CONCERT HALL" alone would
    // be a lie for two thirds of the knob's travel.
    const int low = juce::jlimit (0, numRoomModels - 1, processor.displayMorphLow.load());
    const int high = juce::jlimit (0, numRoomModels - 1, processor.displayMorphHigh.load());
    const float frac = processor.displayMorphFraction.load();

    // The blend sits *between* the two names rather than after them, so it
    // cannot be read as anything but "this far from here to there" — the knob
    // now carries the absolute Space value and two loose percentages on one
    // panel would be one too many.
    juce::String room (roomModels[(size_t) low].name);
    if (high != low && frac > 0.12f && frac < 0.88f)
    {
        const auto arrow = juce::String::fromUTF8 (" \xe2\x86\x92 ");
        room += arrow + juce::String (juce::roundToInt (frac * 100.0f)) + "%"
                + arrow + roomModels[(size_t) high].name;
    }
    else if (frac >= 0.88f)
        room = roomModels[(size_t) high].name;

    roomLabel.setText (room.toUpperCase(), juce::dontSendNotification);

    const float rt = processor.displayRt60.load();
    decayLabel.setText (rt >= 10.0f ? juce::String (rt, 1) + " s DECAY"
                                    : juce::String (rt, 2) + " s DECAY",
                        juce::dontSendNotification);

    // Three points off the same curve the display draws, for the times you are
    // reading numbers rather than looking at the shape — and the only decay
    // readout left in Compact, where the display is hidden.
    if (! compact)
    {
        auto seconds = [] (float s)
        {
            return s >= 10.0f ? juce::String (s, 1) : juce::String (s, 2);
        };

        const auto dot = juce::String::fromUTF8 ("  \xc2\xb7  ");
        decayCurveLabel.setText ("125 Hz " + seconds (display.deliveredSecondsAt (125.0f))
                                     + " s" + dot
                                     + "1 kHz " + seconds (display.deliveredSecondsAt (1000.0f))
                                     + " s" + dot
                                     + "8 kHz " + seconds (display.deliveredSecondsAt (8000.0f))
                                     + " s",
                                 juce::dontSendNotification);
    }

    // The delivered predelay, which is what the note division came to at the
    // host's current tempo — and, at slow tempos on a long division, the 500 ms
    // ceiling it ran into. Showing the figure is what makes the clamp visible.
    if (predelaySyncedLabel.isVisible())
    {
        const float ms = processor.displayPredelayMs.load();
        const bool clamped = ms >= maxPredelayMs - 0.05f;
        predelaySyncedLabel.setText (
            (ms < 100.0f ? juce::String (ms, 1) : juce::String (juce::roundToInt (ms)))
                + (clamped ? " ms MAX" : " ms"),
            juce::dontSendNotification);
        predelaySyncedLabel.setColour (juce::Label::textColourId,
                                       clamped ? colors::warn : colors::textDim);
    }

    // Lock Mix is not a parameter, so nothing pushes it back to the button —
    // and a session that opened with it set has to arrive showing that.
    lockMixButton.setToggleState (processor.isMixLocked(), juce::dontSendNotification);

    if (! compact)
        updateIrPanel();

    undoButton.setEnabled (processor.stateManager.canUndo());
    redoButton.setEnabled (processor.stateManager.canRedo());

    const int active = processor.stateManager.getActiveSlot();
    slotAButton.setToggleState (active == 0, juce::dontSendNotification);
    slotBButton.setToggleState (active == 1, juce::dontSendNotification);

    const auto preset = processor.stateManager.getCurrentPresetName();
    presetNameButton.setButtonText (preset.isEmpty() ? "PRESETS" : preset);

    // The layout sits next to the CPU figure rather than anywhere prominent, and
    // that is the right weight for it: it is not a control and there is nothing
    // to do about it here — the host decides — but when a Speakers setting is
    // greyed out or a band scoped to Height is doing nothing, this is the line
    // that says why.
    cpuLabel.setText (hrzn::layoutShortName (processor.speakerLayout())
                          + "  CPU " + juce::String (processor.cpuLoad.load() * 100.0f, 1)
                          + "%",
                      juce::dontSendNotification);
}
