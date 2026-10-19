#pragma once

#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_dsp/juce_dsp.h>
#include "../PluginProcessor.h"

namespace hrzn
{

// The large interactive display (features 5 and 24) — the interface the
// product is sold on.
//
// ---------------------------------------------------------------------------
// Two curves, two axes, one canvas
// ---------------------------------------------------------------------------
// Horizon's two six-band sections are not measured in the same unit. The Decay
// Rate EQ multiplies how long a band *rings* (0.2x to 6x); the Post EQ is an
// ordinary output EQ in decibels. Drawing both on a shared dB axis would be a
// lie about the first one, and giving each its own arbitrary axis would leave
// neither readable, so the vertical axis carries both scales at once:
//
//   - decay multiplier, logarithmic, +/- `decayOctaves` octaves around 1.00x.
//     Labelled x8 / x4 / x2 / 1.00x / /2 / /4 / /8 down the left.
//   - Post EQ gain, linear, +/- `dbRange`. Labelled in dB down the right.
//
// They are pinned together at the centre line, which therefore means both "the
// room's own decay" and "0 dB", and one horizontal grid serves both: one
// octave of decay per 8 dB of gain.
//
// v0.4 plotted the decay side in *absolute seconds*, and it was wrong for a
// reason worth recording. The room's own decay across frequency already spans
// more than three octaves before any multiplier is applied — a mid-size hall
// runs 2.2 s at 250 Hz and 0.22 s at 16 kHz, because air absorption is real —
// so an axis anchored on the mid band and wide enough to be draggable could
// not contain the curve. The two top bands fell off the bottom of the plot:
// the high shelf's handle sat below the canvas at 1.00x, which made raising
// the top end look impossible when the engine does it perfectly well. Widening
// the axis to fit would have needed about nine octaves, at which point a whole
// octave of decay is thirty pixels and nothing is draggable either.
//
// Plotting the ratio fixes it by construction: every handle is on screen at
// every setting, the travel is symmetric up and down, and flat means the room
// untouched — the mental model every EQ already teaches. Absolute seconds have
// not gone away; they are on the cursor readout, on the selected band, and in
// the three-frequency strip under the display.
//
// ---------------------------------------------------------------------------
// Where a decay handle sits, and what the curve under it means
// ---------------------------------------------------------------------------
// A handle sits at its own multiplier. The purple curve is what the engine
// *delivers* divided by what the room would give with the section neutral —
// so it passes through the handles wherever the request can be met, and falls
// short of them where it cannot.
//
// That shortfall is the point. Making a band ring longer means lifting inside
// a feedback loop, and the headroom for that is finite (see DecayEq.h); a
// display that drew the handles on the delivered curve would be circular, with
// every band sitting on the curve by construction and a request that could not
// be granted looking as though it had been. The gap between handle and curve
// is the v0.3 headroom story made visible.
//
// ---------------------------------------------------------------------------
// Interaction vocabulary
// ---------------------------------------------------------------------------
// Adapted from Equinox's `EqDisplay`, which has carried this since that
// plugin's v0.2:
//
//  - drag a handle: frequency + decay multiplier (or gain). Shift for fine.
//  - wheel over a handle (or the selected band): Q
//  - double-click a handle: numeric entry for all three values
//  - double-click empty space (Post EQ): switch on the next unused band there
//  - right-click a handle: shape / slope / channel, copy / paste, snap to
//    note, reset, delete
//  - right-click empty space: analyzer sources and speed, display range,
//    piano scale, reset the section
//  - click a dimmed handle belonging to the other section: switch to it
class ReverbDisplay : public juce::Component,
                      public juce::SettableTooltipClient,
                      private juce::Timer
{
public:
    explicit ReverbDisplay (HorizonAudioProcessor&);

    using Section = EqSection;

    void setSection (Section s);
    Section getSection() const noexcept { return section; }

    int getSelectedBand() const noexcept { return selectedBand; }
    void selectBand (int band);

    // Fires on a section switch or a selection change, so the editor can move
    // the band row underneath the display onto the same band.
    std::function<void (Section, int)> onSelectionChanged;

    // Everything here is a per-user *view* preference: it belongs in
    // Horizon.settings, never in plugin state or a preset. The editor owns the
    // file and calls setPrefs() once at startup, then saves on every change.
    struct ViewPrefs
    {
        bool showInput = true;
        bool showWet = true;
        bool pianoScale = false;
        int speedIndex = 1;        // analyzer fall rate: slow / medium / fast
    };

    ViewPrefs getPrefs() const noexcept { return prefs; }
    void setPrefs (const ViewPrefs&);
    std::function<void()> onPrefsChanged;

    // Both axes are fixed, and both span exactly their parameter's range: the
    // multiplier reaches 0.2x-6x, which fits inside +/-3 octaves, and the Post
    // EQ gain is +/-24 dB exactly. There is therefore nothing to zoom, and the
    // display-range option v0.4 carried is gone with the seconds axis that
    // needed it. A fixed axis is also the only way to promise what v0.4 could
    // not: every handle is reachable at every setting, always.
    static constexpr float decayOctaves = 3.0f;
    static constexpr float dbRange = 8.0f * decayOctaves;   // 24 dB

    // Coordinate mapping. Public because the harness checks the round-trips —
    // a display whose inverse mapping is wrong turns every drag into a jump,
    // and that is not something a DSP suite would otherwise catch.
    float freqToX (float hz) const noexcept;
    float xToFreq (float x) const noexcept;
    float multToY (float multiplier) const noexcept;
    float yToMult (float y) const noexcept;
    float dbToY (float db) const noexcept;
    float yToDb (float y) const noexcept;

    // The room's own decay at a frequency, interpolated from the engine's
    // half-octave report grid. This is what a multiplier is stated against,
    // and it is the denominator of the curve the display draws.
    float neutralSecondsAt (float hz) const noexcept;
    float deliveredSecondsAt (float hz) const noexcept;

    // Where a band's handle sits. Public for the same reason the mappings are:
    // "a decay handle sits at multiplier x the room's neutral decay at that
    // band's control frequency" is the display's central claim, and it is
    // worth an assertion rather than a comment.
    juce::Point<float> handlePosition (Section s, int band) const;


    void paint (juce::Graphics&) override;
    void resized() override;

    void mouseMove (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseDrag (const juce::MouseEvent&) override;
    void mouseUp (const juce::MouseEvent&) override;
    void mouseDoubleClick (const juce::MouseEvent&) override;
    void mouseWheelMove (const juce::MouseEvent&, const juce::MouseWheelDetails&) override;

private:
    static constexpr int fftOrder = 12;
    static constexpr int fftSize = 1 << fftOrder;
    static constexpr float displayMinFreq = 16.0f, displayMaxFreq = 24000.0f;
    static constexpr float analyzerMinDb = -96.0f, analyzerMaxDb = 6.0f;
    static constexpr float handleRadius = 8.0f;
    static constexpr int pianoStripHeight = 15;
    static constexpr int freqLabelHeight = 15;

    struct Spectrum
    {
        std::array<float, fftSize> ring {};
        std::array<float, fftSize * 2> fftData {};
        std::array<float, fftSize / 2> levelsDb;
        bool hasData = false;

        Spectrum() { levelsDb.fill (-140.0f); }
    };

    void timerCallback() override;
    void drainFifo (AnalyzerFifo& fifo, Spectrum& s);
    void runFFT (Spectrum& s);

    float spectrumBinY (const Spectrum& s, int bin) const;
    void buildAnalyzerPath (const Spectrum& s, juce::Path& path) const;

    // Parameter plumbing. A band is addressed as (section, index); every value
    // on the display is an ordinary APVTS parameter, which is what makes all of
    // it automatable and undoable for free.
    juce::RangedAudioParameter* param (Section s, int band, int which) const;
    float value (Section s, int band, int which) const;
    void setValue (Section s, int band, int which, float plainValue) const;

    enum Which { pFreq = 0, pLevel, pQ, pOn, pShape, pSlope, pChannel };

    bool postBandOn (int band) const;
    float bandControlFreq (Section s, int band) const;
    int hitTest (juce::Point<float> pos, Section s) const;
    int firstUnusedPostBand() const;

    void paintGrid (juce::Graphics&) const;
    void paintAxisLabels (juce::Graphics&) const;
    void paintPiano (juce::Graphics&) const;
    void paintDecayCurves (juce::Graphics&) const;
    void paintPostCurve (juce::Graphics&) const;
    void paintHandles (juce::Graphics&) const;
    void paintReadout (juce::Graphics&) const;
    void paintTriggerMeter (juce::Graphics&) const;
    void paintCursorReadout (juce::Graphics&) const;

    void showBandMenu (Section s, int band);
    void showDisplayMenu();
    void showValueEditor (Section s, int band);

    void copyBandToClipboard (Section s, int band) const;
    std::unique_ptr<juce::XmlElement> getClipboardBand (Section s) const;
    void applyBandXml (Section s, int band, const juce::XmlElement&) const;
    void resetSection (Section s) const;

    void beginGesture (Section s, int band);
    void endGesture();
    void prefsChanged();

    HorizonAudioProcessor& processor;

    juce::dsp::FFT fft { fftOrder };
    juce::dsp::WindowingFunction<float> window { fftSize,
        juce::dsp::WindowingFunction<float>::hann };

    Spectrum inputSpec, wetSpec;
    std::vector<float> popBuffer;

    ViewPrefs prefs;
    bool frozen = false;   // deliberately not persisted: a held analyzer is a
                           // momentary tool, not a preference

    juce::Rectangle<float> plot;   // the curve area, inside the axis furniture

    Section section = Section::decayEq;
    int selectedBand = 0, hoveredBand = -1, draggedBand = -1;
    Section hoveredSection = Section::decayEq;
    bool dragging = false;

    // Fine drag (Shift) is re-anchored whenever the modifier changes mid-drag,
    // the same correction PrecisionSlider needed: without it, pressing Shift
    // halfway through a gesture makes the handle jump.
    juce::Point<float> dragAnchorMouse, dragAnchorHandle;
    bool dragFine = false;

    // Where the pointer is, for the absolute decay readout the ratio axis
    // cannot show on its own.
    juce::Point<float> cursor;
    bool cursorInside = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ReverbDisplay)
};

} // namespace hrzn
