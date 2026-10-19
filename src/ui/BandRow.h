#pragma once

#include <juce_audio_utils/juce_audio_utils.h>
#include "../PluginProcessor.h"
#include "LookAndFeel.h"
#include "PrecisionSlider.h"

namespace hrzn
{

// The strip under the display: the selected section's own bypass, then the
// selected band's values.
//
// The display can edit every one of these by dragging, and for the shapes and
// channel modes it is the only place they are reachable by mouse — so why keep
// a row of controls at all? Three reasons, and the first is the one that
// matters: a handle on a canvas is not keyboard-reachable and reports nothing
// to a screen reader, and the family's accessibility bar (v0.8) is a promise
// already made to three shipped products. The second is that typing a
// frequency should not require finding a handle first. The third is that these
// are ordinary attachments to ordinary parameters, so hosts, MIDI Learn (v0.8)
// and automation all see the same controls they see everywhere else.
//
// It follows the display's selection rather than showing all twelve bands at
// once: v0.3's six-column strip was thirty-six controls fighting for the same
// space, which is exactly what the display replaced.
class BandRow : public juce::Component,
                private juce::Timer
{
public:
    explicit BandRow (HorizonAudioProcessor& p) : proc (p)
    {
        // The section's own bypass rides on the section's own bar rather than
        // on the tab strip above it: a tab names a view, this switches audio,
        // and the two sitting in one row of buttons made the tabs read as if
        // they were the thing being bypassed. The band's switch keeps its place
        // among the band's controls and says BAND, so the two scopes on this
        // row can never be read as the same control.
        sectionOnButton.setClickingTogglesState (true);
        addAndMakeVisible (sectionOnButton);

        title.setFont (juce::Font (juce::FontOptions (12.0f, juce::Font::bold)));
        title.setJustificationType (juce::Justification::centredLeft);
        addAndMakeVisible (title);

        onButton.setClickingTogglesState (true);
        onButton.setTitle ("Band on");
        onButton.setTooltip ("Bypasses this band on its own. The switch at the left of "
                             "the row is the whole section.");
        addAndMakeVisible (onButton);

        auto initCombo = [this] (juce::ComboBox& box, const juce::StringArray& items,
                                 const juce::String& name)
        {
            box.addItemList (items, 1);
            box.setTitle (name);
            addAndMakeVisible (box);
        };

        initCombo (shapeBox, filterShapeNames, "Shape");
        initCombo (slopeBox, slopeNames, "Slope");
        initCombo (channelBox, channelModeNames, "Channel");

        channelBox.setTooltip ("Which channels this band works on. Beyond stereo, Left "
                               "and Right are the two halves of the room, Mid/Side folds "
                               "about every mirrored speaker pair, and the last four are "
                               "the layout's speaker groups.");

        initSlider (freqSlider, "Frequency");
        initSlider (levelSlider, "Level");
        initSlider (qSlider, "Q");

        for (auto* l : { &freqCaption, &levelCaption, &qCaption })
        {
            l->setFont (juce::Font (juce::FontOptions (10.0f, juce::Font::bold)));
            l->setColour (juce::Label::textColourId, colors::textDim);
            l->setJustificationType (juce::Justification::centredLeft);
            addAndMakeVisible (*l);
        }

        freqCaption.setText ("FREQ", juce::dontSendNotification);
        qCaption.setText ("Q", juce::dontSendNotification);

        setBand (EqSection::decayEq, 0);
        startTimerHz (8);
    }

    void setBand (EqSection s, int bandIndex)
    {
        section = s;
        band = juce::jlimit (0, numEqBands - 1, bandIndex);

        const bool decay = (section == EqSection::decayEq);
        const auto n = juce::String (band + 1);

        // Attachments bind to one parameter id, so following a selection means
        // rebuilding them. Released first, deliberately: two live attachments
        // on one slider would fight over its value for as long as both exist.
        sectionOnAtt.reset();
        onAtt.reset();
        shapeAtt.reset();
        slopeAtt.reset();
        channelAtt.reset();
        freqAtt.reset();
        levelAtt.reset();
        qAtt.reset();

        title.setText ((decay ? "DECAY RATE EQ " : "POST EQ ") + juce::String ("BAND ") + n,
                       juce::dontSendNotification);
        title.setColour (juce::Label::textColourId, decay ? colors::decayEq : colors::postEq);

        // Lit in the section's own hue, which is what ties it to the title
        // beside it and to the curve it switches off rather than to the band
        // controls further along the row.
        sectionOnAtt = std::make_unique<ButtonAttachment> (
            proc.apvts, decay ? id::decayEqOn : id::postEqOn_, sectionOnButton);
        sectionOnButton.setColour (juce::TextButton::buttonOnColourId,
                                   (decay ? colors::decayEq : colors::postEq)
                                       .withMultipliedBrightness (0.57f));
        sectionOnButton.setTitle (decay ? "Decay Rate EQ on" : "Post EQ on");
        sectionOnButton.setTooltip (decay
            ? "Bypasses the whole Decay Rate EQ, so the room decays the way the model "
              "does on its own. The band settings are kept."
            : "Bypasses the whole Post EQ. The band settings are kept.");

        levelCaption.setText (decay ? "DECAY" : "GAIN", juce::dontSendNotification);

        freqAtt = std::make_unique<SliderAttachment> (
            proc.apvts, decay ? id::decayEqFreq (band) : id::postEqFreq (band), freqSlider);
        levelAtt = std::make_unique<SliderAttachment> (
            proc.apvts, decay ? id::decayEqMult (band) : id::postEqGain (band), levelSlider);
        qAtt = std::make_unique<SliderAttachment> (
            proc.apvts, decay ? id::decayEqQ (band) : id::postEqQ (band), qSlider);

        freqSlider.setTitle ((decay ? "Decay EQ " : "Post EQ ") + n + " frequency");
        levelSlider.setTitle ((decay ? "Decay EQ " + n + " decay multiplier"
                                     : "Post EQ " + n + " gain"));
        qSlider.setTitle ((decay ? "Decay EQ " : "Post EQ ") + n + " Q");

        if (! decay)
        {
            onAtt = std::make_unique<ButtonAttachment> (proc.apvts, id::postEqOn (band),
                                                        onButton);
            shapeAtt = std::make_unique<ComboAttachment> (proc.apvts, id::postEqType (band),
                                                          shapeBox);
            slopeAtt = std::make_unique<ComboAttachment> (proc.apvts, id::postEqSlope (band),
                                                          slopeBox);
            channelAtt = std::make_unique<ComboAttachment> (proc.apvts, id::postEqChan (band),
                                                            channelBox);
        }

        // The Decay Rate EQ's shapes are fixed by band (a low shelf, four bells
        // and a high shelf) and its bands are always live, so the controls that
        // would edit those simply are not there for it.
        onButton.setVisible (! decay);
        shapeBox.setVisible (! decay);
        slopeBox.setVisible (! decay);
        channelBox.setVisible (! decay);

        updateSlopeEnablement();
        updateLayoutEnablement();
        resized();
        repaint();
    }

    void paint (juce::Graphics& g) override
    {
        g.setColour (colors::panel);
        g.fillRoundedRectangle (getLocalBounds().toFloat(), 5.0f);

        g.setColour (section == EqSection::decayEq ? colors::decayEq : colors::postEq);
        g.fillRect (4, 1, getWidth() - 8, 2);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced (10, 6);
        area.removeFromTop (2);

        sectionOnButton.setBounds (area.removeFromLeft (40).reduced (0, 2));
        area.removeFromLeft (8);

        title.setBounds (area.removeFromLeft (166));
        area.removeFromLeft (6);

        if (section == EqSection::postEq)
        {
            onButton.setBounds (area.removeFromLeft (44).reduced (0, 2));
            area.removeFromLeft (6);
            shapeBox.setBounds (area.removeFromLeft (94).reduced (0, 2));
            area.removeFromLeft (6);
        }

        // Capped rather than divided across the whole strip: a value bar three
        // hundred pixels wide reads as a progress meter, not as a number, and
        // the row is the width of the display whether it needs to be or not.
        const int fields = (section == EqSection::postEq) ? 5 : 3;
        const int w = juce::jlimit (60, 168, area.getWidth() / fields);

        auto field = [&area, w] (juce::Label& caption, juce::Component& control)
        {
            auto col = area.removeFromLeft (w).reduced (3, 0);
            caption.setBounds (col.removeFromTop (12));
            control.setBounds (col.reduced (0, 1));
        };

        field (freqCaption, freqSlider);
        field (levelCaption, levelSlider);
        field (qCaption, qSlider);

        auto placeBox = [&area, w] (juce::ComboBox& box)
        {
            auto col = area.removeFromLeft (juce::jmin (w, 110)).reduced (3, 0);
            col.removeFromTop (12);
            box.setBounds (col.reduced (0, 1));
        };

        if (section == EqSection::postEq)
        {
            placeBox (slopeBox);
            placeBox (channelBox);
        }
    }

private:
    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ComboAttachment  = juce::AudioProcessorValueTreeState::ComboBoxAttachment;
    using ButtonAttachment = juce::AudioProcessorValueTreeState::ButtonAttachment;

    void initSlider (PrecisionSlider& s, const juce::String& name)
    {
        s.setSliderStyle (juce::Slider::LinearBar);
        s.setTextBoxStyle (juce::Slider::TextBoxLeft, false, 10, 10);
        s.setTitle (name);
        s.setWantsKeyboardFocus (true);
        addAndMakeVisible (s);
    }

    // A slope only means something on a cut shape, and a live combo that does
    // nothing is worse than one that says so.
    void updateSlopeEnablement()
    {
        if (section != EqSection::postEq)
            return;

        const auto shape = (FilterShape) (int)
            proc.apvts.getRawParameterValue (id::postEqType (band))->load();
        slopeBox.setEnabled (shape == FilterShape::lowCut || shape == FilterShape::highCut);
    }

    // A speaker group this layout does not have is not a setting that would be
    // wrong, it is a setting that cannot do anything — so the item stays in the
    // list, keeps its index, and greys out.
    //
    // Greyed rather than removed on purpose. A preset written for 7.1.4 and
    // opened on a stereo track still holds "Height" in that slot, and it has to
    // survive the visit: rebuilding the list would leave the attachment pointing
    // at a different choice and the preset would come back changed. The rule for
    // this whole feature is that a layout the session is not in costs the
    // session nothing and changes nothing.
    void updateLayoutEnablement()
    {
        const auto& layout = proc.speakerLayout();

        auto groupAvailable = [&layout] (SpeakerGroup g)
        {
            return g == SpeakerGroup::all || layout.hasGroup (g);
        };

        for (int i = 0; i < channelModeNames.size(); ++i)
        {
            const auto mode = (ChannelMode) i;
            channelBox.setItemEnabled (
                i + 1, ! isSpeakerGroupMode (mode)
                           || groupAvailable ((SpeakerGroup) (i - (int) ChannelMode::front + 1)));
        }
    }

    void timerCallback() override
    {
        updateSlopeEnablement();
        updateLayoutEnablement();
    }

    HorizonAudioProcessor& proc;
    EqSection section = EqSection::decayEq;
    int band = 0;

    juce::Label title;
    juce::Label freqCaption, levelCaption, qCaption;
    juce::TextButton sectionOnButton { "ON" }, onButton { "BAND" };
    juce::ComboBox shapeBox, slopeBox, channelBox;
    PrecisionSlider freqSlider, levelSlider, qSlider;

    std::unique_ptr<ButtonAttachment> sectionOnAtt, onAtt;
    std::unique_ptr<ComboAttachment> shapeAtt, slopeAtt, channelAtt;
    std::unique_ptr<SliderAttachment> freqAtt, levelAtt, qAtt;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (BandRow)
};

} // namespace hrzn
