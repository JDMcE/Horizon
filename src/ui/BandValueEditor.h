#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "../PluginProcessor.h"
#include "../util/NoteNames.h"
#include "LookAndFeel.h"

namespace hrzn
{

// Numeric entry for one band, shown in a CallOutBox when a handle on the
// display is double-clicked. Enter applies and closes; Esc closes.
//
// Adapted from Equinox's `ui/BandValueEditor.h`. The one change worth noting is
// the middle field: on a Decay Rate EQ band it is a *multiplier*, not a gain,
// so it is labelled and parsed as one. Typing "2" there means "rings twice as
// long", which is the only reading of that control that makes sense.
class BandValueEditor : public juce::Component
{
public:
    BandValueEditor (HorizonAudioProcessor& p, EqSection s, int bandIndex)
        : proc (p), section (s), band (bandIndex)
    {
        const bool decay = (section == EqSection::decayEq);

        auto initRow = [this] (juce::Label& label, const juce::String& name,
                               juce::TextEditor& editor, const juce::String& text)
        {
            label.setText (name, juce::dontSendNotification);
            label.setFont (juce::Font (juce::FontOptions (12.0f)));
            label.setColour (juce::Label::textColourId, colors::textDim);
            addAndMakeVisible (label);

            editor.setText (text, juce::dontSendNotification);
            editor.setJustification (juce::Justification::centredLeft);
            editor.setSelectAllWhenFocused (true);
            editor.onReturnKey = [this] { applyAndClose(); };
            editor.onEscapeKey = [this] { close(); };
            addAndMakeVisible (editor);
        };

        const float freq = get (freqId());
        const float level = get (levelId());
        const float q = get (qId());

        initRow (freqLabel, "Freq", freqEditor, juce::String (freq, freq < 100.0f ? 1 : 0));
        initRow (levelLabel, decay ? "Decay" : "Gain", levelEditor,
                 juce::String (level, 2));
        initRow (qLabel, "Q", qEditor, juce::String (q, 2));

        noteLabel.setText (freqToNoteName (freq), juce::dontSendNotification);
        noteLabel.setFont (juce::Font (juce::FontOptions (11.0f)));
        noteLabel.setColour (juce::Label::textColourId, colors::textDim.withAlpha (0.8f));
        addAndMakeVisible (noteLabel);

        unitLabel.setText (decay ? juce::String::fromUTF8 ("\xc3\x97") : juce::String ("dB"),
                           juce::dontSendNotification);
        unitLabel.setFont (juce::Font (juce::FontOptions (11.0f)));
        unitLabel.setColour (juce::Label::textColourId, colors::textDim.withAlpha (0.8f));
        addAndMakeVisible (unitLabel);

        setSize (216, 96);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced (6, 4);
        const int rowH = area.getHeight() / 3;

        auto layoutRow = [&] (juce::Label& l, juce::TextEditor& e, juce::Label* extra)
        {
            auto row = area.removeFromTop (rowH).reduced (0, 3);
            l.setBounds (row.removeFromLeft (44));
            if (extra != nullptr)
                extra->setBounds (row.removeFromRight (62));
            e.setBounds (row);
        };

        layoutRow (freqLabel, freqEditor, &noteLabel);
        layoutRow (levelLabel, levelEditor, &unitLabel);
        layoutRow (qLabel, qEditor, nullptr);
    }

    void visibilityChanged() override
    {
        if (isVisible())
            freqEditor.grabKeyboardFocus();
    }

private:
    juce::String freqId() const
    {
        return section == EqSection::decayEq ? id::decayEqFreq (band) : id::postEqFreq (band);
    }

    juce::String levelId() const
    {
        return section == EqSection::decayEq ? id::decayEqMult (band) : id::postEqGain (band);
    }

    juce::String qId() const
    {
        return section == EqSection::decayEq ? id::decayEqQ (band) : id::postEqQ (band);
    }

    float get (const juce::String& paramId) const
    {
        return proc.apvts.getRawParameterValue (paramId)->load();
    }

    void apply (const juce::String& paramId, float plainValue) const
    {
        if (auto* p = proc.apvts.getParameter (paramId))
        {
            p->beginChangeGesture();
            p->setValueNotifyingHost (p->convertTo0to1 (plainValue));
            p->endChangeGesture();
        }
    }

    void applyAndClose()
    {
        const bool decay = (section == EqSection::decayEq);

        const double freq = parseFreqText (freqEditor.getText());
        if (freq > 0.0)
            apply (freqId(), juce::jlimit (minEqFreq, maxEqFreq, (float) freq));

        const auto levelText = levelEditor.getText().trim()
                                   .removeCharacters (juce::String::fromUTF8 ("\xc3\x97"))
                                   .upToLastOccurrenceOf ("x", false, true);
        if (levelText.isNotEmpty())
        {
            const float v = (float) levelText.getDoubleValue();
            apply (levelId(), decay ? juce::jlimit (minDecayMult, maxDecayMult, v)
                                    : juce::jlimit (minPostGainDb, maxPostGainDb, v));
        }

        const double q = qEditor.getText().getDoubleValue();
        if (q > 0.0)
            apply (qId(), decay ? juce::jlimit (minDecayQ, maxDecayQ, (float) q)
                                : juce::jlimit (minPostQ, maxPostQ, (float) q));

        close();
    }

    void close()
    {
        if (auto* box = findParentComponentOfClass<juce::CallOutBox>())
            box->dismiss();
    }

    HorizonAudioProcessor& proc;
    EqSection section;
    int band;

    juce::Label freqLabel, levelLabel, qLabel, noteLabel, unitLabel;
    juce::TextEditor freqEditor, levelEditor, qEditor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (BandValueEditor)
};

} // namespace hrzn
