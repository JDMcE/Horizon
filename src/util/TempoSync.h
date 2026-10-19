#pragma once

#include <juce_core/juce_core.h>
#include <array>

namespace hrzn
{

// Host-tempo note divisions for the synced Predelay (feature 18), v0.5.
//
// **Promotion candidate** — the family's other products will want the same
// table the moment any of them syncs anything, and the only product-specific
// thing here is which parameter it feeds.
//
// Lengths are in quarter-note beats, which is the unit the host reports tempo
// in, so a division is one multiply away from milliseconds. The list is sorted
// by actual length rather than grouped by variant: the control is a stepped
// one and stepping it should always make the predelay longer or shorter,
// never jump from a dotted eighth back past a triplet quarter.
struct NoteDivision
{
    const char* name;
    float beats;      // in quarter notes
};

inline constexpr std::array<NoteDivision, 19> noteDivisions { {
    { "1/64",   0.0625f },
    { "1/32T",  1.0f / 12.0f },
    { "1/64.",  0.09375f },
    { "1/32",   0.125f },
    { "1/16T",  1.0f / 6.0f },
    { "1/32.",  0.1875f },
    { "1/16",   0.25f },
    { "1/8T",   1.0f / 3.0f },
    { "1/16.",  0.375f },
    { "1/8",    0.5f },
    { "1/4T",   2.0f / 3.0f },
    { "1/8.",   0.75f },
    { "1/4",    1.0f },
    { "1/2T",   4.0f / 3.0f },
    { "1/4.",   1.5f },
    { "1/2",    2.0f },
    { "1/1T",   8.0f / 3.0f },
    { "1/2.",   3.0f },
    { "1/1",    4.0f }
} };

inline constexpr int numNoteDivisions = (int) noteDivisions.size();

// 1/16 — short enough to be a predelay rather than a slap at any tempo anyone
// works at, which is the setting the control is reached for.
inline constexpr int defaultDivisionIndex = 6;

inline juce::StringArray noteDivisionNames()
{
    juce::StringArray names;
    for (const auto& d : noteDivisions)
        names.add (d.name);
    return names;
}

// Milliseconds for a division at a tempo. Unclamped: the caller decides what
// the parameter it feeds can actually reach, and for Predelay that limit is
// low enough (500 ms) that the long divisions run into it below about 120 BPM.
// Clamping is the honest response — the alternative is either a predelay
// control that reaches a second and a half, which is a different feature, or a
// division list with the useful end sawn off — and the readout shows the time
// actually delivered so a capped setting is visible rather than mysterious.
inline float divisionMilliseconds (int index, double bpm) noexcept
{
    const auto i = (size_t) juce::jlimit (0, numNoteDivisions - 1, index);
    const double safeBpm = juce::jlimit (20.0, 999.0, bpm > 0.0 ? bpm : 120.0);
    return (float) (60000.0 / safeBpm * (double) noteDivisions[i].beats);
}

} // namespace hrzn
