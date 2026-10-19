#pragma once

#include <jdm/ui/LookAndFeel.h>

// Horizon's face of the shared family design language (jdm_shared): the
// common near-black blue-grey surfaces with Horizon's soft purple control
// accent.
//
// Horizon draws *two* curves, so the hue roles split accordingly: the accent
// is the Decay Rate EQ (the curve that IS the product, same reasoning as
// Equinox's summed curve) and the reserved hue is the Post EQ, in amber for
// maximum separation from purple. Level metering is the family cyan.
//
// Note the sidechain slot moves off the family placeholder (#b287d8) — that
// purple would be indistinguishable from Horizon's accent. Here it is teal
// and means the ducking trigger (v0.5).

namespace hrzn
{

inline const jdm::colors::Palette horizonPalette {
    juce::Colour (0xff15181e), juce::Colour (0xff1d2129),
    juce::Colour (0xff242a34), juce::Colour (0xff313845),
    juce::Colour (0xffd8dce4), juce::Colour (0xff828a9a),
    juce::Colour (0xffa88cd8), juce::Colour (0xff5b4a80),
    juce::Colour (0xffe0a44c), juce::Colour (0xff46b5dc),
    juce::Colour (0xffd96a5a), juce::Colour (0xff56b58c)
};

inline const jdm::colors::Palette horizonPaletteHighContrast {
    juce::Colour (0xff090b0e), juce::Colour (0xff11141a),
    juce::Colour (0xff1b2029), juce::Colour (0xff5a6478),
    juce::Colour (0xfff4f7fc), juce::Colour (0xffbcc5d6),
    juce::Colour (0xffc0a8f0), juce::Colour (0xff7a63ad),
    juce::Colour (0xffffb45c), juce::Colour (0xff62cdf4),
    juce::Colour (0xffff8071), juce::Colour (0xff6bd9a6)
};

namespace colors
{
    using namespace jdm::colors;

    // Horizon's reserved hue means the Post EQ curve (v0.3).
    inline juce::Colour& postEq = jdm::colors::reserved;

    // ...and the accent doubles as the Decay Rate EQ curve.
    inline juce::Colour& decayEq = jdm::colors::accent;
}

struct LookAndFeel : jdm::LookAndFeel
{
    LookAndFeel() : jdm::LookAndFeel (horizonPalette, horizonPaletteHighContrast) {}
};

} // namespace hrzn
