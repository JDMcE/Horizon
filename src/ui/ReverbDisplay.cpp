#include "ReverbDisplay.h"
#include "BandValueEditor.h"
#include "LookAndFeel.h"
#include "../dsp/FilterCoeffs.h"
#include "../util/NoteNames.h"

namespace hrzn
{

namespace
{
    // Const references onto the shared family palette, so a runtime
    // high-contrast swap is picked up without touching the paint code.
    const juce::Colour& gridCol       = colors::panelRaised;
    const juce::Colour& gridStrongCol = colors::outline;
    const juce::Colour& textCol       = colors::textDim;

    constexpr int labelFreqs[] = { 20, 30, 50, 100, 200, 300, 500,
                                   1000, 2000, 3000, 5000, 10000, 20000 };

    constexpr const char* decayClipTag = "HorizonDecayBand";
    constexpr const char* postClipTag  = "HorizonPostBand";

    enum MenuIds
    {
        menuCopyBand = 1, menuPasteBand, menuSnapToNote, menuResetBand, menuDeleteBand,
        menuBandOn,
        menuShowInput = 20, menuShowWet, menuFreeze, menuPiano, menuResetSection,
        menuSpeedSlow = 30, menuSpeedMedium, menuSpeedFast,
        menuShapeBase = 100, menuSlopeBase = 200, menuChanBase = 300
    };

    float analyzerFallFor (int speedIndex) noexcept
    {
        return speedIndex <= 0 ? 0.6f : speedIndex == 1 ? 1.5f : 3.5f;
    }

    juce::String secondsText (float s)
    {
        if (s < 1.0f)   return juce::String (juce::roundToInt (s * 1000.0f)) + " ms";
        if (s < 10.0f)  return juce::String (s, 2) + " s";
        return juce::String (s, 1) + " s";
    }
} // namespace

ReverbDisplay::ReverbDisplay (HorizonAudioProcessor& p) : processor (p)
{
    popBuffer.resize (fftSize);
    setOpaque (true);
    setWantsKeyboardFocus (false);

    startTimerHz (30);
}

void ReverbDisplay::setPrefs (const ViewPrefs& p)
{
    prefs = p;
    prefs.speedIndex = juce::jlimit (0, 2, prefs.speedIndex);
    resized();
    repaint();
}

void ReverbDisplay::prefsChanged()
{
    if (onPrefsChanged)
        onPrefsChanged();
    resized();
    repaint();
}

void ReverbDisplay::setSection (Section s)
{
    if (section == s)
        return;

    section = s;
    hoveredBand = -1;

    if (onSelectionChanged)
        onSelectionChanged (section, selectedBand);
    repaint();
}

void ReverbDisplay::selectBand (int band)
{
    const int b = juce::jlimit (0, numEqBands - 1, band);
    if (selectedBand == b)
        return;

    selectedBand = b;
    if (onSelectionChanged)
        onSelectionChanged (section, selectedBand);
    repaint();
}

void ReverbDisplay::resized()
{
    auto area = getLocalBounds().toFloat();
    area.removeFromBottom ((float) freqLabelHeight
                           + (prefs.pianoScale ? (float) pianoStripHeight : 0.0f));
    plot = area.reduced (2.0f, 6.0f);
}

// Coordinate mapping ---------------------------------------------------------

float ReverbDisplay::freqToX (float hz) const noexcept
{
    const float norm = std::log (juce::jmax (1.0f, hz) / displayMinFreq)
                       / std::log (displayMaxFreq / displayMinFreq);
    return plot.getX() + norm * plot.getWidth();
}

float ReverbDisplay::xToFreq (float x) const noexcept
{
    const float norm = plot.getWidth() > 0.0f ? (x - plot.getX()) / plot.getWidth() : 0.0f;
    return displayMinFreq * std::pow (displayMaxFreq / displayMinFreq, norm);
}

float ReverbDisplay::multToY (float multiplier) const noexcept
{
    const float octaves = std::log2 (juce::jmax (1.0e-4f, multiplier));
    const float span = decayOctaves;
    return plot.getCentreY() - octaves / span * plot.getHeight() * 0.5f;
}

float ReverbDisplay::yToMult (float y) const noexcept
{
    const float span = decayOctaves;
    const float octaves = plot.getHeight() > 0.0f
                        ? (plot.getCentreY() - y) / (plot.getHeight() * 0.5f) * span
                        : 0.0f;
    return std::pow (2.0f, octaves);
}

float ReverbDisplay::dbToY (float db) const noexcept
{
    const float range = dbRange;
    return plot.getCentreY() - db / range * plot.getHeight() * 0.5f;
}

float ReverbDisplay::yToDb (float y) const noexcept
{
    const float range = dbRange;
    return plot.getHeight() > 0.0f
         ? (plot.getCentreY() - y) / (plot.getHeight() * 0.5f) * range
         : 0.0f;
}

// The engine reports on a half-octave grid from 31.25 Hz. Interpolating in the
// log of the value as well as in log frequency matters: decay times run over
// two orders of magnitude across the grid, and a linear blend between two of
// them puts a visible kink at every grid point.
namespace
{
    template <typename Array>
    float sampleReportCurve (const Array& curve, float hz) noexcept
    {
        constexpr int n = DecayEq::numReportPoints;
        const float p = juce::jlimit (0.0f, (float) (n - 1),
                                      std::log2 (juce::jmax (1.0f, hz) / 31.25f) * 2.0f);
        const int i = juce::jlimit (0, n - 1, (int) p);
        const int j = juce::jmin (i + 1, n - 1);
        const float f = p - (float) i;

        const float a = juce::jmax (1.0e-4f, curve[(size_t) i].load());
        const float b = juce::jmax (1.0e-4f, curve[(size_t) j].load());
        return std::exp (std::log (a) + f * (std::log (b) - std::log (a)));
    }
}

float ReverbDisplay::neutralSecondsAt (float hz) const noexcept
{
    return sampleReportCurve (processor.displayNeutralCurve, hz);
}

float ReverbDisplay::deliveredSecondsAt (float hz) const noexcept
{
    return sampleReportCurve (processor.displayDecayCurve, hz);
}

// Parameters -----------------------------------------------------------------

juce::RangedAudioParameter* ReverbDisplay::param (Section s, int band, int which) const
{
    const int b = juce::jlimit (0, numEqBands - 1, band);

    if (s == Section::decayEq)
    {
        switch (which)
        {
            case pFreq:  return processor.apvts.getParameter (id::decayEqFreq (b));
            case pLevel: return processor.apvts.getParameter (id::decayEqMult (b));
            case pQ:     return processor.apvts.getParameter (id::decayEqQ (b));
            default:     return nullptr;
        }
    }

    switch (which)
    {
        case pFreq:    return processor.apvts.getParameter (id::postEqFreq (b));
        case pLevel:   return processor.apvts.getParameter (id::postEqGain (b));
        case pQ:       return processor.apvts.getParameter (id::postEqQ (b));
        case pOn:      return processor.apvts.getParameter (id::postEqOn (b));
        case pShape:   return processor.apvts.getParameter (id::postEqType (b));
        case pSlope:   return processor.apvts.getParameter (id::postEqSlope (b));
        case pChannel: return processor.apvts.getParameter (id::postEqChan (b));
        default:       return nullptr;
    }
}

float ReverbDisplay::value (Section s, int band, int which) const
{
    if (auto* p = param (s, band, which))
        return p->convertFrom0to1 (p->getValue());
    return 0.0f;
}

void ReverbDisplay::setValue (Section s, int band, int which, float plainValue) const
{
    if (auto* p = param (s, band, which))
        p->setValueNotifyingHost (p->convertTo0to1 (plainValue));
}

bool ReverbDisplay::postBandOn (int band) const
{
    return value (Section::postEq, band, pOn) >= 0.5f;
}

// The frequency a Decay Rate EQ band states its multiplier at — the band
// frequency for a bell, out into the shelf for a shelf. The handle is drawn at
// the band frequency (that is the parameter, and where a shelf's corner is),
// but its *height* means the multiplier at the control point, which is what
// DecayEq designs against.
float ReverbDisplay::bandControlFreq (Section s, int band) const
{
    const float f = value (s, band, pFreq);
    if (s != Section::decayEq)
        return f;

    DecayEq::Band b;
    b.shape = decayEqShapes[(size_t) juce::jlimit (0, numEqBands - 1, band)];
    b.freq = f;
    return DecayEq::controlFrequencyFor (b, processor.getSampleRate() > 0.0
                                                ? processor.getSampleRate() : 48000.0);
}

juce::Point<float> ReverbDisplay::handlePosition (Section s, int band) const
{
    const float x = freqToX (value (s, band, pFreq));

    if (s == Section::decayEq)
        return { x, multToY (value (s, band, pLevel)) };

    const auto shape = (FilterShape) (int) value (s, band, pShape);
    const bool hasGain = (shape == FilterShape::bell || shape == FilterShape::lowShelf
                          || shape == FilterShape::highShelf);
    return { x, dbToY (hasGain ? value (s, band, pLevel) : 0.0f) };
}

int ReverbDisplay::hitTest (juce::Point<float> pos, Section s) const
{
    int best = -1;
    float bestDist = handleRadius + 4.0f;

    for (int b = 0; b < numEqBands; ++b)
    {
        const float d = pos.getDistanceFrom (handlePosition (s, b));
        if (d < bestDist)
        {
            bestDist = d;
            best = b;
        }
    }
    return best;
}

int ReverbDisplay::firstUnusedPostBand() const
{
    for (int b = 0; b < numEqBands; ++b)
        if (! postBandOn (b))
            return b;
    return -1;
}

// Analyzer -------------------------------------------------------------------

void ReverbDisplay::timerCallback()
{
    drainFifo (processor.inputFifo, inputSpec);
    drainFifo (processor.wetFifo, wetSpec);
    repaint();
}

void ReverbDisplay::drainFifo (AnalyzerFifo& fifo, Spectrum& s)
{
    bool gotNew = false;

    while (fifo.getNumReady() > 0)
    {
        const int n = fifo.pop (popBuffer.data(), (int) popBuffer.size());
        if (n <= 0)
            break;

        if (frozen)
            continue; // keep draining so the FIFO stays fresh, but hold the display

        if (n >= fftSize)
        {
            std::copy (popBuffer.begin() + (n - fftSize), popBuffer.begin() + n,
                       s.ring.begin());
        }
        else
        {
            std::move (s.ring.begin() + n, s.ring.end(), s.ring.begin());
            std::copy (popBuffer.begin(), popBuffer.begin() + n, s.ring.end() - n);
        }
        gotNew = true;
    }

    if (frozen)
        return;

    if (gotNew)
    {
        s.hasData = true;
        runFFT (s);
    }
    else
    {
        const float fall = analyzerFallFor (prefs.speedIndex);
        for (auto& l : s.levelsDb)
            l = juce::jmax (-140.0f, l - fall);
    }
}

void ReverbDisplay::runFFT (Spectrum& s)
{
    std::copy (s.ring.begin(), s.ring.end(), s.fftData.begin());
    std::fill (s.fftData.begin() + fftSize, s.fftData.end(), 0.0f);

    window.multiplyWithWindowingTable (s.fftData.data(), fftSize);
    fft.performFrequencyOnlyForwardTransform (s.fftData.data());

    // Hann coherent gain 0.5, so a full-scale sine reads about 0 dB.
    const float scale = 4.0f / (float) fftSize;
    const float fall = analyzerFallFor (prefs.speedIndex);

    for (int bin = 0; bin < fftSize / 2; ++bin)
    {
        const float mag = s.fftData[(size_t) bin] * scale;
        const float db = juce::Decibels::gainToDecibels (mag, -140.0f);

        auto& level = s.levelsDb[(size_t) bin];
        level = db > level ? db : juce::jmax (-140.0f, level - fall);
    }
}

float ReverbDisplay::spectrumBinY (const Spectrum& s, int bin) const
{
    const double sr = processor.getSampleRate() > 0 ? processor.getSampleRate() : 48000.0;
    const float freq = (float) (bin * sr / fftSize);
    const float tilt = 4.5f * std::log2 (juce::jmax (1.0f, freq) / 1000.0f);
    const float db = s.levelsDb[(size_t) bin] + tilt;
    return juce::jmap (juce::jlimit (analyzerMinDb, analyzerMaxDb, db),
                       analyzerMinDb, analyzerMaxDb, plot.getBottom(), plot.getY());
}

void ReverbDisplay::buildAnalyzerPath (const Spectrum& s, juce::Path& path) const
{
    const double sr = processor.getSampleRate() > 0 ? processor.getSampleRate() : 48000.0;
    const float bottom = plot.getBottom() + 2.0f;
    const int numBins = fftSize / 2;

    path.startNewSubPath (plot.getX() - 2.0f, bottom);
    float lastX = plot.getX() - 10.0f;

    for (int bin = 1; bin < numBins; ++bin)
    {
        const float freq = (float) (bin * sr / fftSize);
        if (freq < displayMinFreq)
            continue;
        if (freq > displayMaxFreq)
            break;

        const float x = freqToX (freq);
        if (x - lastX < 1.0f)
            continue;
        lastX = x;

        path.lineTo (x, spectrumBinY (s, bin));
    }

    path.lineTo (plot.getRight() + 2.0f, bottom);
    path.closeSubPath();
}

// Painting -------------------------------------------------------------------

void ReverbDisplay::paint (juce::Graphics& g)
{
    g.fillAll (colors::background);
    paintGrid (g);

    if (prefs.showInput && inputSpec.hasData)
    {
        juce::Path p;
        buildAnalyzerPath (inputSpec, p);
        g.setColour (textCol.withAlpha (0.16f));
        g.fillPath (p);
    }

    if (prefs.showWet && wetSpec.hasData)
    {
        juce::Path p;
        buildAnalyzerPath (wetSpec, p);
        g.setColour (colors::level.withAlpha (0.22f));
        g.fillPath (p);
    }

    // The section being edited draws last, so where the two curves cross it is
    // the one you can follow.
    if (section == Section::decayEq)
    {
        paintPostCurve (g);
        paintDecayCurves (g);
    }
    else
    {
        paintDecayCurves (g);
        paintPostCurve (g);
    }

    paintAxisLabels (g);
    paintTriggerMeter (g);
    paintHandles (g);
    paintCursorReadout (g);

    if (prefs.pianoScale)
        paintPiano (g);

    paintReadout (g);
}

void ReverbDisplay::paintGrid (juce::Graphics& g) const
{
    g.setFont (juce::Font (juce::FontOptions (11.0f)));

    for (int f : labelFreqs)
    {
        const float x = freqToX ((float) f);
        if (x < plot.getX() || x > plot.getRight())
            continue;

        const bool strong = (f == 100 || f == 1000 || f == 10000);
        g.setColour (strong ? gridStrongCol : gridCol);
        g.drawVerticalLine ((int) x, plot.getY(), plot.getBottom());

        g.setColour (textCol);
        const juce::String label = f >= 1000 ? juce::String (f / 1000) + "k" : juce::String (f);

        // Clamped into the plot rather than centred on the line: the decades at
        // either end sit within a few pixels of the edge, and a label half off
        // the canvas is worse than one a little out of place.
        const int lx = juce::jlimit ((int) plot.getX(), (int) plot.getRight() - 40,
                                     (int) x - 20);
        g.drawText (label, lx, (int) plot.getBottom() + 1, 40, freqLabelHeight,
                    juce::Justification::centred);
    }

    // One horizontal grid, two scales. Each line is a doubling of decay time
    // on the left and 8 dB of Post EQ gain on the right; the centre is the
    // room's own decay and 0 dB at once.
    const int octaves = (int) decayOctaves;

    for (int o = -octaves; o <= octaves; ++o)
    {
        const float y = multToY (std::pow (2.0f, (float) o));
        if (y < plot.getY() - 1.0f || y > plot.getBottom() + 1.0f)
            continue;

        const bool centre = (o == 0);
        g.setColour (centre ? gridStrongCol.interpolatedWith (colors::accent, 0.35f)
                            : gridCol);
        g.drawHorizontalLine ((int) y, plot.getX(), plot.getRight());
    }
}

// Drawn after the curves rather than with the grid. A decay curve that has
// bottomed out runs along the axis, and a seconds label underneath it is
// unreadable exactly where you most want to know what line you are looking at.
void ReverbDisplay::paintAxisLabels (juce::Graphics& g) const
{
    g.setFont (juce::Font (juce::FontOptions (11.0f)));

    const int octaves = (int) decayOctaves;
    const int step = (int) (dbRange
                            / decayOctaves);

    for (int o = -octaves; o <= octaves; ++o)
    {
        const float y = multToY (std::pow (2.0f, (float) o));
        if (y < plot.getY() - 1.0f || y > plot.getBottom() + 1.0f)
            continue;

        // The topmost and bottommost lines are the edges of the plot, so their
        // labels are pushed inside rather than sitting above and below them.
        const int labelY = juce::jlimit ((int) plot.getY() + 1,
                                         (int) plot.getBottom() - 13, (int) y - 13);

        // Seconds down the left in the Decay Rate EQ's hue, dB down the right
        // in the Post EQ's, so a glance at an axis says which curve it belongs
        // to without a legend.
        auto label = [&g, labelY] (const juce::String& text, juce::Colour colour,
                                   juce::Rectangle<int> area, juce::Justification just)
        {
            g.setColour (colors::background.withAlpha (0.72f));
            g.fillRect (area.withY (labelY).withHeight (12));
            g.setColour (colour.withAlpha (0.8f));
            g.drawText (text, area.withY (labelY).withHeight (12), just);
        };

        // "x2" above the line, "/2" below it, and the centre spelled out in
        // full: 1.00x is the number the parameter shows, and the line means
        // "this band rings for exactly as long as the room does".
        const juce::String multText =
            o == 0 ? juce::String ("1.00") + juce::String::fromUTF8 ("\xc3\x97")
                   : (o > 0 ? juce::String::fromUTF8 ("\xc3\x97") + juce::String (1 << o)
                            : juce::String ("/") + juce::String (1 << -o));

        label (multText, colors::decayEq,
               juce::Rectangle<int> ((int) plot.getX() + 3, 0, 50, 12),
               juce::Justification::left);

        if (o != 0)
        {
            const int db = o * step;   // up the canvas is a longer decay and a
                                       // louder band alike
            label (juce::String (db > 0 ? "+" : "") + juce::String (db), colors::postEq,
                   juce::Rectangle<int> ((int) plot.getRight() - 44, 0, 41, 12),
                   juce::Justification::right);
        }
    }
}

void ReverbDisplay::paintPiano (juce::Graphics& g) const
{
    const auto strip = juce::Rectangle<float> (plot.getX(),
                                               plot.getBottom() + (float) freqLabelHeight,
                                               plot.getWidth(), (float) pianoStripHeight);

    g.setColour (colors::panel);
    g.fillRect (strip);

    // Keys are laid out by frequency, so they line up with the log axis rather
    // than with each other — which is the point of the option: a ringing mode
    // can be read off as a note.
    for (int note = 12; note <= 120; ++note)
    {
        const float f0 = (float) midiNoteToFreq (note);
        const float f1 = (float) midiNoteToFreq (note + 1);
        if (f1 < displayMinFreq || f0 > displayMaxFreq)
            continue;

        const float x0 = freqToX (f0), x1 = freqToX (f1);
        if (x1 - x0 < 1.2f)
            continue;

        const auto key = juce::Rectangle<float> (x0, strip.getY(),
                                                 juce::jmax (1.0f, x1 - x0 - 0.6f),
                                                 strip.getHeight());
        g.setColour (isBlackKey (note) ? colors::background
                                       : colors::text.withAlpha (0.55f));
        g.fillRect (key.getIntersection (strip));

        if (note % 12 == 0 && x1 - x0 > 16.0f)
        {
            g.setColour (colors::background.withAlpha (0.8f));
            g.setFont (juce::Font (juce::FontOptions (9.0f)));
            g.drawText ("C" + juce::String (note / 12 - 1), key.getIntersection (strip),
                        juce::Justification::centredBottom);
        }
    }
}

void ReverbDisplay::paintDecayCurves (juce::Graphics& g) const
{
    const int steps = juce::jmax (2, (int) plot.getWidth() / 2);
    const bool activeSection = (section == Section::decayEq);

    // What the section is achieving, as a ratio to the room's own decay: the
    // engine's delivered curve over its neutral one. The reference curve is
    // therefore the centre line and needs no drawing of its own, which is the
    // whole reason this axis is a ratio.
    juce::Path achieved;

    for (int i = 0; i <= steps; ++i)
    {
        const float x = plot.getX() + plot.getWidth() * (float) i / (float) steps;
        const float hz = xToFreq (x);
        const float ratio = deliveredSecondsAt (hz)
                            / juce::jmax (1.0e-4f, neutralSecondsAt (hz));

        const float y = multToY (ratio);
        if (i == 0) achieved.startNewSubPath (x, y);
        else        achieved.lineTo (x, y);
    }

    g.setColour (colors::decayEq.withAlpha (activeSection ? 1.0f : 0.45f));
    g.strokePath (achieved, juce::PathStrokeType (activeSection ? 2.6f : 1.8f));
}

void ReverbDisplay::paintPostCurve (juce::Graphics& g) const
{
    const double sr = processor.getSampleRate() > 0 ? processor.getSampleRate() : 48000.0;
    const bool activeSection = (section == Section::postEq);

    std::vector<BandCoeffs> on, selected;

    for (int b = 0; b < numEqBands; ++b)
    {
        const auto shape = (FilterShape) (int) value (Section::postEq, b, pShape);
        auto c = computeBandCoeffs (shape, value (Section::postEq, b, pFreq),
                                    value (Section::postEq, b, pLevel),
                                    value (Section::postEq, b, pQ),
                                    (int) value (Section::postEq, b, pSlope), sr);

        // A band that is switched off still shows its own faint curve while it
        // is the selected one, so you can aim it before committing it.
        if (activeSection && b == selectedBand && ! postBandOn (b))
            selected.push_back (c);

        if (postBandOn (b))
            on.push_back (std::move (c));
    }

    auto build = [&] (const std::vector<BandCoeffs>& sets, juce::Path& path)
    {
        const int steps = juce::jmax (2, (int) plot.getWidth() / 2);
        for (int i = 0; i <= steps; ++i)
        {
            const float x = plot.getX() + plot.getWidth() * (float) i / (float) steps;
            const double hz = juce::jmin ((double) xToFreq (x), sr * 0.499);

            double db = 0.0;
            for (const auto& c : sets)
                db += bandMagnitudeDb (c, hz, sr);

            const float y = dbToY ((float) juce::jlimit (-96.0, 96.0, db));
            if (i == 0) path.startNewSubPath (x, y);
            else        path.lineTo (x, y);
        }
    };

    if (! selected.empty())
    {
        juce::Path p;
        build (selected, p);
        g.setColour (colors::postEq.withAlpha (0.35f));
        g.strokePath (p, juce::PathStrokeType (1.2f));
    }

    if (on.empty())
        return;

    juce::Path sum;
    build (on, sum);
    g.setColour (colors::postEq.withAlpha (activeSection ? 1.0f : 0.4f));
    g.strokePath (sum, juce::PathStrokeType (activeSection ? 2.2f : 1.5f));
}

void ReverbDisplay::paintHandles (juce::Graphics& g) const
{
    g.setFont (juce::Font (juce::FontOptions (11.0f, juce::Font::bold)));

    // The inactive section's handles are drawn small and dim: visible, so you
    // can see where its bands are while working on the other one, and clickable
    // as a shortcut into that section, but never in the way of a drag.
    for (int pass = 0; pass < 2; ++pass)
    {
        const auto s = (pass == 0) ? (section == Section::decayEq ? Section::postEq
                                                                 : Section::decayEq)
                                   : section;
        const bool activeSection = (pass == 1);
        const auto hue = (s == Section::decayEq) ? colors::decayEq : colors::postEq;

        for (int b = 0; b < numEqBands; ++b)
        {
            const bool off = (s == Section::postEq) && ! postBandOn (b);
            const auto pos = handlePosition (s, b);

            if (! plot.expanded (14.0f).contains (pos))
                continue;

            const bool isSelected = activeSection && b == selectedBand;
            const bool isHovered = activeSection && b == hoveredBand;
            const float r = activeSection ? handleRadius + (isSelected ? 2.0f : 0.0f)
                                          : handleRadius * 0.55f;

            auto col = off ? colors::textDim : hue;
            if (! activeSection)
                col = col.withAlpha (0.4f);
            else if (! isSelected && ! isHovered)
                col = col.withAlpha (0.82f);

            g.setColour (col);
            g.fillEllipse (pos.x - r, pos.y - r, r * 2.0f, r * 2.0f);

            if (isSelected)
            {
                g.setColour (colors::text.withAlpha (0.9f));
                g.drawEllipse (pos.x - r, pos.y - r, r * 2.0f, r * 2.0f, 1.5f);
            }

            if (activeSection)
            {
                g.setColour (colors::background.withAlpha (off ? 0.55f : 0.85f));
                g.drawText (juce::String (b + 1), (int) (pos.x - r), (int) (pos.y - r),
                            (int) (r * 2.0f), (int) (r * 2.0f), juce::Justification::centred);
            }
        }
    }
}

// The trigger, in the teal reserved for it since v0.4 (feature 7, v0.5).
//
// It is a meter down the right-hand edge and not a curve across the plot,
// because the plot's vertical axis is already two things — decay multiplier
// for one section, dB for the other — and a third meaning laid over it would
// be readable in neither. What the strip shows is the pair of numbers that
// actually explain what Ducking and the Auto Gate are doing: how loud the
// trigger is against the threshold it is being judged by, and how much is
// being taken off the wet path because of it.
//
// It appears only while one of the two is switched on. A meter that is always
// there and usually flat teaches people to stop looking at it.
void ReverbDisplay::paintTriggerMeter (juce::Graphics& g) const
{
    const auto on = [this] (const juce::String& paramId)
    {
        return processor.apvts.getRawParameterValue (paramId)->load() > 0.5f;
    };

    const bool ducking = on (id::duckOn);
    const bool gating = on (id::gateOn);

    if (! ducking && ! gating)
        return;

    auto strip = juce::Rectangle<float> (plot.getRight() - 13.0f, plot.getY() + 6.0f,
                                         9.0f, plot.getHeight() - 12.0f);

    g.setColour (colors::background.withAlpha (0.72f));
    g.fillRoundedRectangle (strip, 2.0f);
    g.setColour (colors::sidechain.withAlpha (0.30f));
    g.drawRoundedRectangle (strip, 2.0f, 1.0f);

    auto levelY = [&strip] (float db)
    {
        // -60 dB at the foot, 0 dB at the head: the range a threshold control
        // covers, so the tick and the level are read against the same ruler.
        const float t = juce::jlimit (0.0f, 1.0f, (db + 60.0f) / 60.0f);
        return strip.getBottom() - t * strip.getHeight();
    };

    // Trigger level, filled from the foot.
    const float triggerDb = processor.triggerLevelDb.load();
    if (triggerDb > -60.0f)
    {
        const float y = levelY (triggerDb);
        g.setColour (colors::sidechain.withAlpha (0.55f));
        g.fillRect (strip.getX() + 1.0f, y,
                    strip.getWidth() - 2.0f, strip.getBottom() - y);
    }

    // The threshold each active stage is judging it against, as a tick that
    // overhangs the strip on the left so it reads against the fill.
    auto threshold = [&] (const juce::String& paramId)
    {
        const float y = levelY (processor.apvts.getRawParameterValue (paramId)->load());
        g.fillRect (strip.getX() - 3.0f, y - 0.5f, strip.getWidth() + 3.0f, 1.0f);
    };

    g.setColour (colors::sidechain);
    if (ducking) threshold (id::duckThreshold);
    if (gating)  threshold (id::gateThreshold);

    // Gain reduction, hanging from the head — the ducker's depth and whatever
    // the gate has closed, which is the one number that says what the wet path
    // is actually losing.
    const float gate = juce::jlimit (0.0f, 1.0f, processor.gateGain.load());
    const float reductionDb = processor.duckDepthDb.load()
                            + (gate > 0.0f ? -juce::Decibels::gainToDecibels (gate, -60.0f)
                                           : 60.0f);

    if (reductionDb > 0.1f)
    {
        const float depth = juce::jlimit (0.0f, 1.0f, reductionDb / 30.0f)
                            * strip.getHeight();
        g.setColour (colors::warn.withAlpha (0.85f));
        g.fillRect (strip.getX() + 1.0f, strip.getY(), strip.getWidth() - 2.0f, depth);
    }
}

void ReverbDisplay::paintReadout (juce::Graphics& g) const
{
    const int band = hoveredBand >= 0 ? hoveredBand : selectedBand;
    if (band < 0)
        return;

    const bool decay = (section == Section::decayEq);
    const float freq = value (section, band, pFreq);

    juce::String text;
    text << "BAND " << (band + 1) << "   " << formatFrequency (freq, 0);

    if (decay)
    {
        // The multiplier is what the parameter says; the seconds are what it
        // comes to on this room at this decay time. The axis can only show the
        // first, so the second has to be written down.
        text << "   " << formatDecayMultiplier (value (section, band, pLevel), 0);
        text << "   " << secondsText (deliveredSecondsAt (bandControlFreq (section, band)));
    }
    else
    {
        const auto shape = (FilterShape) (int) value (section, band, pShape);
        text << "   " << juce::String (value (section, band, pLevel), 1) << " dB";
        text << "   " << filterShapeNames[(int) shape];
        if (! postBandOn (band))
            text << "   (off)";
    }

    text << "   Q " << juce::String (value (section, band, pQ), 2);

    g.setFont (juce::Font (juce::FontOptions (11.0f, juce::Font::bold)));
    g.setColour ((decay ? colors::decayEq : colors::postEq).withAlpha (0.9f));
    g.drawText (text, (int) plot.getX() + 74, (int) plot.getY() + 2,
                (int) plot.getWidth() - 130, 14, juce::Justification::left);

    if (frozen)
    {
        g.setColour (colors::level);
        g.drawText ("FREEZE", (int) plot.getRight() - 64, (int) plot.getY() + 2, 60, 14,
                    juce::Justification::right);
    }
}

// Feature 24 asks to see the tail's length at a frequency, and a ratio axis
// cannot show it. Following the cursor does, and does it better than an axis
// would: point at anywhere on the spectrum and read how long the reverb rings
// there, including the part the Decay Rate EQ could not deliver.
void ReverbDisplay::paintCursorReadout (juce::Graphics& g) const
{
    if (! cursorInside || dragging)
        return;

    const float hz = xToFreq (cursor.x);

    g.setColour (colors::outline.withAlpha (0.7f));
    g.drawVerticalLine ((int) cursor.x, plot.getY(), plot.getBottom());

    const auto text = formatFrequency (hz, 0) + juce::String::fromUTF8 ("   \xc2\xb7   ")
                      + secondsText (deliveredSecondsAt (hz));

    auto box = juce::Rectangle<int> ((int) plot.getRight() - 168,
                                     (int) plot.getBottom() - 17, 164, 14);
    g.setColour (colors::background.withAlpha (0.78f));
    g.fillRect (box);

    g.setFont (juce::Font (juce::FontOptions (11.0f, juce::Font::bold)));
    g.setColour (colors::decayEq.withAlpha (0.9f));
    g.drawText (text, box, juce::Justification::right);
}

// Interaction ----------------------------------------------------------------

void ReverbDisplay::mouseMove (const juce::MouseEvent& e)
{
    cursor = e.position;
    cursorInside = plot.contains (cursor);

    const int hit = hitTest (e.position, section);
    if (hit != hoveredBand)
        setMouseCursor (hit >= 0 ? juce::MouseCursor::PointingHandCursor
                                 : juce::MouseCursor::NormalCursor);

    hoveredBand = hit;
    repaint();
}

void ReverbDisplay::mouseExit (const juce::MouseEvent&)
{
    hoveredBand = -1;
    cursorInside = false;
    repaint();
}

void ReverbDisplay::beginGesture (Section s, int band)
{
    if (auto* f = param (s, band, pFreq))  f->beginChangeGesture();
    if (auto* l = param (s, band, pLevel)) l->beginChangeGesture();
}

void ReverbDisplay::endGesture()
{
    if (draggedBand < 0)
        return;

    if (auto* f = param (section, draggedBand, pFreq))  f->endChangeGesture();
    if (auto* l = param (section, draggedBand, pLevel)) l->endChangeGesture();
}

void ReverbDisplay::mouseDown (const juce::MouseEvent& e)
{
    const int hit = hitTest (e.position, section);

    if (e.mods.isPopupMenu())
    {
        if (hit >= 0)
        {
            selectBand (hit);
            showBandMenu (section, hit);
        }
        else
        {
            showDisplayMenu();
        }
        return;
    }

    if (hit < 0)
    {
        // Clicking the other section's handle is how you get to it without
        // reaching for the tabs — the curve you want to edit is the one you
        // are already pointing at.
        const auto other = (section == Section::decayEq) ? Section::postEq
                                                         : Section::decayEq;
        const int otherHit = hitTest (e.position, other);
        if (otherHit >= 0)
        {
            setSection (other);
            selectBand (otherHit);
        }
        return;
    }

    selectBand (hit);

    draggedBand = hit;
    dragging = true;
    dragFine = e.mods.isShiftDown();
    dragAnchorMouse = e.position;
    dragAnchorHandle = handlePosition (section, hit);
    beginGesture (section, hit);
}

void ReverbDisplay::mouseDrag (const juce::MouseEvent& e)
{
    if (! dragging || draggedBand < 0)
        return;

    // Shift is pressable and releasable mid-gesture, which needs the anchor
    // re-taken at the moment it changes or the handle jumps by the accumulated
    // difference in sensitivity. Same correction PrecisionSlider carries.
    if (e.mods.isShiftDown() != dragFine)
    {
        dragFine = e.mods.isShiftDown();
        dragAnchorMouse = e.position;
        dragAnchorHandle = handlePosition (section, draggedBand);
    }

    const float gain = dragFine ? 0.2f : 1.0f;
    const auto target = dragAnchorHandle + (e.position - dragAnchorMouse) * gain;

    setValue (section, draggedBand, pFreq,
              juce::jlimit (minEqFreq, maxEqFreq, xToFreq (target.x)));

    if (section == Section::decayEq)
    {
        setValue (section, draggedBand, pLevel,
                  juce::jlimit (minDecayMult, maxDecayMult, yToMult (target.y)));
        return;
    }

    const auto shape = (FilterShape) (int) value (section, draggedBand, pShape);
    if (shape == FilterShape::bell || shape == FilterShape::lowShelf
        || shape == FilterShape::highShelf)
        setValue (section, draggedBand, pLevel,
                  juce::jlimit (minPostGainDb, maxPostGainDb, yToDb (target.y)));
}

void ReverbDisplay::mouseUp (const juce::MouseEvent&)
{
    endGesture();
    dragging = false;
    draggedBand = -1;
}

void ReverbDisplay::mouseDoubleClick (const juce::MouseEvent& e)
{
    const int hit = hitTest (e.position, section);

    if (hit >= 0)
    {
        showValueEditor (section, hit);
        return;
    }

    // The two sections differ here, and they should. The Decay Rate EQ's six
    // bands are permanent — the shapes are fixed and 1.00x *is* "off" — so
    // there is nothing to create. A Post EQ band, on the other hand, starts
    // switched off, so the empty-space gesture is Equinox's: bring the next
    // unused one up right where you clicked.
    if (section != Section::postEq)
        return;

    const int band = firstUnusedPostBand();
    if (band < 0)
        return;

    setValue (Section::postEq, band, pFreq,
              juce::jlimit (minEqFreq, maxEqFreq, xToFreq (e.position.x)));
    setValue (Section::postEq, band, pLevel,
              juce::jlimit (minPostGainDb, maxPostGainDb, yToDb (e.position.y)));
    setValue (Section::postEq, band, pShape, (float) (int) FilterShape::bell);
    setValue (Section::postEq, band, pOn, 1.0f);
    selectBand (band);
}

void ReverbDisplay::mouseWheelMove (const juce::MouseEvent& e,
                                    const juce::MouseWheelDetails& wheel)
{
    int target = hitTest (e.position, section);
    if (target < 0)
        target = selectedBand;
    if (target < 0)
        return;

    const bool decay = (section == Section::decayEq);
    const float factor = std::pow (e.mods.isShiftDown() ? 1.12f : 1.6f, wheel.deltaY);
    const float q = value (section, target, pQ) * factor;

    if (auto* p = param (section, target, pQ))
    {
        p->beginChangeGesture();
        p->setValueNotifyingHost (p->convertTo0to1 (
            decay ? juce::jlimit (minDecayQ, maxDecayQ, q)
                  : juce::jlimit (minPostQ, maxPostQ, q)));
        p->endChangeGesture();
    }
}

void ReverbDisplay::showValueEditor (Section s, int band)
{
    auto content = std::make_unique<BandValueEditor> (processor, s, band);
    const auto handle = handlePosition (s, band).toInt();
    juce::CallOutBox::launchAsynchronously (
        std::move (content),
        localAreaToGlobal (juce::Rectangle<int> (handle.x - 8, handle.y - 8, 16, 16)),
        nullptr);
}

// Clipboard ------------------------------------------------------------------

void ReverbDisplay::copyBandToClipboard (Section s, int band) const
{
    const bool decay = (s == Section::decayEq);
    juce::XmlElement xml (decay ? decayClipTag : postClipTag);

    xml.setAttribute ("freq", value (s, band, pFreq));
    xml.setAttribute ("level", value (s, band, pLevel));
    xml.setAttribute ("q", value (s, band, pQ));

    if (! decay)
    {
        xml.setAttribute ("shape", value (s, band, pShape));
        xml.setAttribute ("slope", value (s, band, pSlope));
        xml.setAttribute ("chan", value (s, band, pChannel));
    }

    juce::SystemClipboard::copyTextToClipboard (xml.toString());
}

std::unique_ptr<juce::XmlElement> ReverbDisplay::getClipboardBand (Section s) const
{
    auto xml = juce::parseXML (juce::SystemClipboard::getTextFromClipboard());

    // Deliberately not interchangeable between the sections: a multiplier and a
    // gain in dB are different quantities, and pasting one into the other would
    // silently mean something else.
    if (xml != nullptr
        && xml->hasTagName (s == Section::decayEq ? decayClipTag : postClipTag))
        return xml;
    return nullptr;
}

void ReverbDisplay::applyBandXml (Section s, int band, const juce::XmlElement& xml) const
{
    setValue (s, band, pFreq, (float) xml.getDoubleAttribute ("freq"));
    setValue (s, band, pLevel, (float) xml.getDoubleAttribute ("level"));
    setValue (s, band, pQ, (float) xml.getDoubleAttribute ("q"));

    if (s == Section::postEq)
    {
        setValue (s, band, pShape, (float) xml.getDoubleAttribute ("shape"));
        setValue (s, band, pSlope, (float) xml.getDoubleAttribute ("slope"));
        setValue (s, band, pChannel, (float) xml.getDoubleAttribute ("chan"));
        setValue (s, band, pOn, 1.0f);
    }
}

void ReverbDisplay::resetSection (Section s) const
{
    for (int b = 0; b < numEqBands; ++b)
    {
        if (s == Section::decayEq)
        {
            setValue (s, b, pFreq, decayEqDefaultFreq[(size_t) b]);
            setValue (s, b, pLevel, 1.0f);
            setValue (s, b, pQ, decayEqDefaultQ[(size_t) b]);
        }
        else
        {
            setValue (s, b, pOn, 0.0f);
            setValue (s, b, pShape, (float) (int) postEqDefaultShape[(size_t) b]);
            setValue (s, b, pFreq, postEqDefaultFreq[(size_t) b]);
            setValue (s, b, pLevel, 0.0f);
            setValue (s, b, pQ, postEqDefaultQ[(size_t) b]);
            setValue (s, b, pChannel, (float) (int) ChannelMode::stereo);
        }
    }
}

// Menus ----------------------------------------------------------------------

void ReverbDisplay::showBandMenu (Section s, int band)
{
    const bool decay = (s == Section::decayEq);
    juce::PopupMenu menu;

    // Feature 14's Speakers setting shares the Post EQ's channel submenu: the
    // groups and the L/R and mid/side modes are alternatives to one another
    // rather than two settings that compose.
    const auto& layout = processor.speakerLayout();
    auto groupAvailable = [&layout] (SpeakerGroup g)
    {
        return g == SpeakerGroup::all || layout.hasGroup (g);
    };

    if (! decay)
    {
        menu.addItem (menuBandOn, "Band On", true, postBandOn (band));

        const int currentShape = (int) value (s, band, pShape);
        juce::PopupMenu shapes;
        for (int t = 0; t < filterShapeNames.size(); ++t)
            shapes.addItem (menuShapeBase + t, filterShapeNames[t], true, t == currentShape);
        menu.addSubMenu ("Shape", shapes);

        const auto shape = (FilterShape) currentShape;
        const bool hasSlope = (shape == FilterShape::lowCut || shape == FilterShape::highCut);
        const int currentSlope = (int) value (s, band, pSlope);
        juce::PopupMenu slopes;
        for (int i = 0; i < slopeNames.size(); ++i)
            slopes.addItem (menuSlopeBase + i, slopeNames[i], true, i == currentSlope);
        menu.addSubMenu ("Slope", slopes, hasSlope);

        const int currentChan = (int) value (s, band, pChannel);
        juce::PopupMenu chans;
        for (int i = 0; i < channelModeNames.size(); ++i)
        {
            const auto mode = (ChannelMode) i;
            const bool enabled =
                ! isSpeakerGroupMode (mode)
                || groupAvailable ((SpeakerGroup) (i - (int) ChannelMode::front + 1));
            chans.addItem (menuChanBase + i, channelModeNames[i], enabled, i == currentChan);
        }
        menu.addSubMenu ("Channel", chans);

        menu.addSeparator();
    }

    menu.addItem (menuCopyBand, "Copy Band");
    menu.addItem (menuPasteBand, "Paste Band", getClipboardBand (s) != nullptr);
    menu.addItem (menuSnapToNote, "Snap to Note ("
        + freqToNoteName (snapFreqToNote (value (s, band, pFreq))) + ")");
    menu.addItem (menuResetBand, decay ? "Reset to 1.00x" : "Reset Gain");

    if (! decay)
    {
        menu.addSeparator();
        menu.addItem (menuDeleteBand, "Delete Band");
    }

    juce::Component::SafePointer<ReverbDisplay> safeThis (this);
    menu.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (this),
        [safeThis, s, band] (int result)
        {
            if (safeThis == nullptr || result == 0)
                return;

            auto& self = *safeThis;

            if (result == menuBandOn)
                self.setValue (s, band, pOn, self.postBandOn (band) ? 0.0f : 1.0f);
            else if (result == menuCopyBand)
                self.copyBandToClipboard (s, band);
            else if (result == menuPasteBand)
            {
                if (auto xml = self.getClipboardBand (s))
                    self.applyBandXml (s, band, *xml);
            }
            else if (result == menuSnapToNote)
                self.setValue (s, band, pFreq,
                               (float) snapFreqToNote (self.value (s, band, pFreq)));
            else if (result == menuResetBand)
                self.setValue (s, band, pLevel, s == Section::decayEq ? 1.0f : 0.0f);
            else if (result == menuDeleteBand)
                self.setValue (s, band, pOn, 0.0f);
            else if (result >= menuShapeBase && result < menuShapeBase + 100)
                self.setValue (s, band, pShape, (float) (result - menuShapeBase));
            else if (result >= menuSlopeBase && result < menuSlopeBase + 100)
                self.setValue (s, band, pSlope, (float) (result - menuSlopeBase));
            else if (result >= menuChanBase && result < menuChanBase + 100)
                self.setValue (s, band, pChannel, (float) (result - menuChanBase));

            self.repaint();
        });
}

void ReverbDisplay::showDisplayMenu()
{
    juce::PopupMenu analyzer;
    analyzer.addItem (menuShowInput, "Input", true, prefs.showInput);
    analyzer.addItem (menuShowWet, "Reverb", true, prefs.showWet);
    analyzer.addItem (menuFreeze, "Freeze", true, frozen);
    analyzer.addSeparator();
    analyzer.addItem (menuSpeedSlow, "Slow", true, prefs.speedIndex == 0);
    analyzer.addItem (menuSpeedMedium, "Medium", true, prefs.speedIndex == 1);
    analyzer.addItem (menuSpeedFast, "Fast", true, prefs.speedIndex == 2);

    juce::PopupMenu menu;
    menu.addSubMenu ("Analyzer", analyzer);
    menu.addItem (menuPiano, "Piano Scale", true, prefs.pianoScale);
    menu.addSeparator();
    menu.addItem (menuResetSection, section == Section::decayEq ? "Reset Decay Rate EQ"
                                                                : "Reset Post EQ");

    juce::Component::SafePointer<ReverbDisplay> safeThis (this);
    menu.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (this),
        [safeThis] (int result)
        {
            if (safeThis == nullptr || result == 0)
                return;

            auto& self = *safeThis;

            switch (result)
            {
                case menuShowInput: self.prefs.showInput = ! self.prefs.showInput; break;
                case menuShowWet:   self.prefs.showWet = ! self.prefs.showWet; break;
                case menuPiano:     self.prefs.pianoScale = ! self.prefs.pianoScale; break;
                case menuSpeedSlow:   self.prefs.speedIndex = 0; break;
                case menuSpeedMedium: self.prefs.speedIndex = 1; break;
                case menuSpeedFast:   self.prefs.speedIndex = 2; break;

                // Freeze holds the analyzer, so it is a momentary tool rather
                // than a preference and is the one thing here that is not saved.
                case menuFreeze: self.frozen = ! self.frozen; self.repaint(); return;

                case menuResetSection: self.resetSection (self.section); self.repaint(); return;

                default:
                    break;
            }

            self.prefsChanged();
        });
}

} // namespace hrzn
