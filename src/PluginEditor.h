#pragma once

#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_opengl/juce_opengl.h>
#include "PluginProcessor.h"
#include "ui/LookAndFeel.h"
#include "ui/Meters.h"
#include "ui/PrecisionSlider.h"
#include "ui/PresetBrowser.h"
#include "ui/ReverbDisplay.h"
#include "ui/BandRow.h"
#include "ui/HelpTexts.h"
#include <map>

// v0.4 editor: the interactive display on top with the two EQ curves and the
// analyzer, the selected band's values under it, then the knob rows and the
// meter strip — all inside a content component at a fixed logical size that
// the editor shell scales with an AffineTransform. Dragging any edge scales
// the whole UI continuously (0.5x-2x, aspect pinned), which leaves
// setScaleFactor free for the host's own DPI handling, and a full-screen mode
// goes past the 2x cap to whatever the display allows.
//
// The scale, the Compact layout, and the display's analyzer/range/piano-scale
// options persist per-user in Horizon.settings — deliberately not in plugin
// state, because how large somebody's screen is has nothing to do with how the
// reverb is set. An OpenGLContext composites all painting on the GPU; repaints
// stay timer-driven at 30 Hz rather than continuous.
//
// v0.3's six-column band strip is gone, replaced by the display plus a row
// following its selection. Everything it edited was an ordinary parameter, so
// nothing was lost in the swap.
class HorizonAudioProcessorEditor : public juce::AudioProcessorEditor,
                                    private juce::Timer
{
public:
    explicit HorizonAudioProcessorEditor (HorizonAudioProcessor&);
    ~HorizonAudioProcessorEditor() override;

    void paint (juce::Graphics& g) override { g.fillAll (hrzn::colors::background); }
    void resized() override;
    bool keyPressed (const juce::KeyPress&) override;

    // Right-click on a registered control. The knobs route their own popup
    // through PrecisionSlider::onPopupMenu (which also has to suppress the
    // drag); the buttons and combo boxes have no such hook, so the editor
    // listens to them directly.
    void mouseDown (const juce::MouseEvent&) override;

    // The hints bar follows the pointer, so the editor has to see it move
    // across every child.
    void mouseMove (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;

    // Exposed for the harness, which has no message loop and no menus.
    void setHelpHints (bool on);
    bool helpHintsShown() const noexcept { return helpHintsOn; }
    void setHighContrast (bool on);
    void setTextScale (float scale);
    float textScale() const { return lookAndFeel.getTextScale(); }
    void showShortcutsOverlay (bool shouldShow);
    bool shortcutsOverlayShown() const noexcept { return shortcutsOverlay.isVisible(); }
    juce::String helpTextAt (const juce::Component*) const;

    // Exposed for the harness, which drives them the way the buttons do.
    void setCompact (bool shouldBeCompact, bool save);
    void setFullScreen (bool shouldBeFullScreen);
    bool isCompact() const noexcept { return compact; }
    bool isFullScreen() const noexcept { return fullScreen; }
    hrzn::ReverbDisplay& getDisplay() noexcept { return display; }

    // Also exposed for the harness: the timer drives this, and the harness has
    // no message loop to run one.
    void updateIrPanel();

    int logicalWidth() const noexcept { return 940; }

    // The layout is drawn at 940 x 790, which is taller than the rest of the
    // range — Zenith opens at 880 x 580, Meridian at 920 x 570, Equinox at
    // 1080 x 680. Opening at 1.0 would make Horizon the odd window on the
    // desktop and, on a scaled display, the one that does not fit. The default
    // scale brings the opening height (632) in among the others; the aspect
    // ratio is pinned, so the width follows and lands a little narrower.
    //
    // This is only where the window *starts*: the first resize writes ui_scale
    // and every later session opens where the last one was left.
    static constexpr double defaultUiScale = 0.8;

    // Compact hides the editing surfaces — the display, the two EQ sections,
    // the Ducking / Auto Gate block and the impulse response footer — and keeps
    // the controls you reach for while something is playing.
    //
    // The full layout has two heights of its own: the Ducking / Auto Gate block
    // carries seven knobs only while one of the two is switched on, and stands
    // at a single strip of switches otherwise. Eleven controls that do nothing
    // are eleven controls to read past, and the block is off in almost every
    // preset — so the window gives the space back rather than reserving it.
    // The hints bar is a strip along the bottom, and the window grows by its
    // height rather than the layout shrinking to make room: turning help on
    // must not move every control you were about to read about.
    static constexpr int hintBarHeight = 22;

    int logicalHeight() const noexcept
    {
        const int hints = helpHintsOn ? hintBarHeight : 0;

        if (compact)
            return 342 + hints;

        return (duckGateExpanded ? 866 : 790) + hints;
    }

private:
    void timerCallback() override;
    void paintContent (juce::Graphics&);
    void layoutContent();
    void applyLogicalSize (double scale);
    double currentScale() const;
    void loadPrefs();
    void savePrefs();
    void updateSectionTabs();

    // Right-click on Space: the fourteen rooms by name, so a specific one can
    // be reached without hunting for it along a stepless control.
    void showSpaceMenu();

    // Right-click on Decay Time: the room's own natural decay, which the
    // absolute control otherwise has no way to offer.
    void showDecayMenu();

    // MIDI Learn (feature 26, v0.8) ------------------------------------------
    //
    // Every control that maps to one parameter is registered here, which is
    // also the definition of what can be learned: the band row is deliberately
    // left out, because its three sliders re-attach to a different band every
    // time the selection moves, so a mapping made on it would follow the row
    // rather than the parameter and there is no honest label for that.
    void registerMidiLearn (juce::Component&, const juce::String& paramId);
    void addMidiLearnSection (juce::PopupMenu&, const juce::String& paramId);
    bool handleMidiLearnChoice (int result, const juce::String& paramId);
    void showParameterMenu (juce::Component& target, const juce::String& paramId);

    std::map<const juce::Component*, juce::String> midiLearnTargets;

    // The help layer (feature 30, v0.8) --------------------------------------
    //
    // One string per control from `ui/HelpTexts.h`, registered here and then
    // used three ways: as the hover tooltip, as the text the hints bar shows
    // while the pointer is over the control, and as the description a screen
    // reader announces. Registering rather than setting each of the three at
    // each control is what stops them drifting: there is one call site per
    // control and it cannot half-happen.
    void registerHelp (juce::Component&, const juce::String& title, const char* text);
    void showHelpMenu();
    void openManual();

    std::map<const juce::Component*, juce::String> helpTexts;
    juce::Label hintBar;
    bool helpHintsOn = false;

    juce::TextButton helpButton { "HELP" };

    // F1. Painted rather than laid out, because it is a list of pairs and a
    // list of pairs is two drawText calls and a loop.
    struct ShortcutsOverlay : juce::Component
    {
        ShortcutsOverlay() { setWantsKeyboardFocus (true); }
        void paint (juce::Graphics&) override;
        void mouseDown (const juce::MouseEvent&) override { setVisible (false); }
        bool keyPressed (const juce::KeyPress&) override { setVisible (false); return true; }
    };
    ShortcutsOverlay shortcutsOverlay;

    struct Knob
    {
        hrzn::PrecisionSlider slider;
        juce::Label caption;
        std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attachment;
    };

    void initKnob (Knob& k, const juce::String& paramId, const juce::String& caption);
    static void layoutKnob (Knob& k, juce::Rectangle<int> area);

    // A knob whose caption row also carries a small switch belonging to it —
    // Predelay's Sync, Decay Time's Freeze, Mix's Lock. Putting the switch in
    // the caption is what says which knob it acts on; the same three buttons in
    // a row of their own said only that they were buttons.
    static void layoutKnobWithSwitch (Knob& k, juce::Rectangle<int> area,
                                      juce::Component& sw, int switchWidth);

    // Predelay is two controls in one slot: a free time and a note division.
    // Which one is showing follows the Sync switch.
    void layoutPredelay (juce::Rectangle<int> area);
    void updatePredelayMode();

    // The Ducking / Auto Gate knobs are only present while their half is on,
    // and the block collapses to its switches when neither is. Changing that
    // changes logicalHeight, so this re-sizes the window as well.
    void updateDuckGateVisibility();

    // Everything the shell scales lives in here; the editor itself only ever
    // holds the transform and the background colour.
    struct ContentComponent : juce::Component
    {
        std::function<void (juce::Graphics&)> onPaint;
        void paint (juce::Graphics& g) override { if (onPaint) onPaint (g); }
    };

    HorizonAudioProcessor& processor;
    hrzn::LookAndFeel lookAndFeel;
    ContentComponent content;
    juce::OpenGLContext glContext;
    std::unique_ptr<juce::PropertiesFile> settings;

    // Header.
    juce::TextButton undoButton { "UNDO" }, redoButton { "REDO" };
    juce::TextButton slotAButton { "A" }, slotBButton { "B" }, copySlotButton { "COPY" };
    juce::TextButton prevPresetButton { juce::String::fromUTF8 ("\xe2\x80\xb9") },
                     nextPresetButton { juce::String::fromUTF8 ("\xe2\x80\xba") };
    juce::TextButton presetNameButton { "Init" };
    juce::TextButton compactButton { "COMPACT" }, fullScreenButton { "FULL" };
    juce::TextButton bypassButton { "BYPASS" };
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> bypassAttachment;
    juce::Label cpuLabel;

    // The display and the two things that frame it.
    hrzn::ReverbDisplay display { processor };
    hrzn::BandRow bandRow { processor };
    // The tabs select which curve the display is editing, and nothing else:
    // the section bypasses that used to sit beside them now ride on the band
    // row, which is the bar that belongs to the section they switch.
    juce::TextButton decayEqTab { "DECAY RATE EQ" }, postEqTab { "POST EQ" };

    juce::Label decayCurveLabel;

    // Controls.
    Knob spaceKnob, decayKnob, mixKnob;
    Knob predelayKnob, distanceKnob, characterKnob, thicknessKnob,
         widthKnob, brightnessKnob;

    // Freeze rides in the Decay Time caption: it is that control taken to its
    // limit, and it belongs to the knob rather than to a strip of odds and ends
    // under the panel.
    juce::TextButton freezeButton { "FREEZE" };
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> freezeAttachment;

    // Lock Mix (v0.5), likewise in the Mix caption — it is the one control it
    // acts on. Not a parameter, so no attachment: it is a workflow switch and
    // the button owns it outright.
    juce::TextButton lockMixButton { "LOCK" };

    // Predelay tempo sync (v0.5).
    juce::TextButton predelaySyncButton { "SYNC" };
    juce::ComboBox predelayDivBox;
    juce::Label predelaySyncedLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> predelaySyncAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> predelayDivAttachment;

    // Ducking and the Auto Gate (v0.5).
    juce::TextButton duckOnButton { "DUCKING" }, gateOnButton { "AUTO GATE" };
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment>
        duckOnAttachment, gateOnAttachment;

    juce::ComboBox triggerSourceBox;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> triggerSourceAttachment;

    Knob duckThresholdKnob, duckAmountKnob, duckAttackKnob, duckReleaseKnob;
    Knob gateThresholdKnob, gateHoldKnob, gateReleaseKnob;

    // Where the block sits, so paintContent can frame it and rule a line
    // between the two halves without deriving the layout twice.
    juce::Rectangle<int> duckGateBounds;
    int duckGateDividerX = 0;
    bool duckGateExpanded = false;

    // Impulse response import (v0.6), in the footer: it loads a file and
    // reports on it, which is housekeeping rather than a sound control, and it
    // was reading as one more row of the reverb wedged between the knobs and
    // the ducker.
    juce::TextButton importIrButton { "IMPORT IR" };
    juce::TextButton irCompareButton { "COMPARE" };
    juce::Label irLabel;
    std::unique_ptr<juce::FileChooser> irChooser;

    void showIrImportChooser();

    // Readout of what Space is currently sitting on and how long it rings.
    juce::Label roomLabel, decayLabel;

    // Output meter strip, and the footer rule — both painted from the bounds
    // the layout arrived at rather than from the same arithmetic run twice.
    hrzn::LevelMeter inMeter, outMeter;
    juce::Label inLabel, outLabel;
    juce::Rectangle<int> meterPanelBounds, footerBounds;

    bool compact = false, fullScreen = false;
    int sizeBeforeFullScreen = 0;

    // Hover tooltips have carried the help since v0.1; from v0.8 the string
    // they show is the same one the hints bar and the screen reader use (see
    // registerHelp).
    juce::TooltipWindow tooltipWindow { this, 700 };

    hrzn::PresetBrowser presetBrowser { processor.presets };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (HorizonAudioProcessorEditor)
};
