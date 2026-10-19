#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <array>
#include <cmath>

namespace hrzn
{

// Where the speakers are (v0.7).
//
// Everything surround in Horizon is a function of *direction*, not of channel
// index: which lines of the tank a speaker reads, how strongly two speakers
// cross-feed, which speakers a Post EQ band acts on, which side of the room a
// mid/side band folds about. Channel indices differ between 5.1, 7.1, 7.1.4 and
// 9.1.6 and between the host conventions inside each; directions do not. So the
// engine is handed a table of directions and never sees a layout name.
//
// The positions are the ITU/Dolby nominal ones. They do not have to be exactly
// where anybody's speakers are — nothing here is a panner. What they have to be
// is *ordered and separated* correctly, because what reads them is an angular
// distance: two speakers 30 degrees apart must cross-feed more than two 140
// degrees apart, and a height speaker must sit above the layer rather than in
// it. Nominal angles do that; measured ones would not do it better.
//
// Derived per channel from `juce::AudioChannelSet::getTypeOfChannel`, so a
// layout Horizon has never heard of still resolves as long as its channels have
// names — which is what keeps `isBusesLayoutSupported`'s whitelist a policy
// decision rather than a structural limit.

// The speaker groups feature 14's Speakers setting selects between. `all` is
// the default and means the band is not a per-speaker band at all.
//
// Four groups rather than one per speaker, because the useful controls are
// "hold the height layer back", "shorten the surrounds", "keep the dialogue
// channel dry" — nobody wants to tune the left rear on its own, and a selector
// with sixteen entries that mean different things in every layout is a selector
// nobody can write a preset against.
enum class SpeakerGroup { all = 0, front, centre, surround, height };

inline const juce::StringArray speakerGroupNames { "All", "Front", "Centre",
                                                   "Surround", "Height" };

constexpr int numSpeakerGroups = 5;   // including `all`

// The groups that actually partition the speakers — `all` is not one of them.
constexpr int numRealSpeakerGroups = 4;

// 9.1.6 is sixteen channels, which is also the number of lines in the tank.
// That is not a coincidence any more: from v0.7 every non-LFE speaker reads the
// tank through its own Walsh row, and a 16-line tank has exactly fifteen
// non-trivial ones — see ReverbEngine::buildOutputVectors.
constexpr int maxSpeakers = 16;

struct Speaker
{
    float azimuth = 0.0f;    // radians, 0 = straight ahead, positive to the right
    float elevation = 0.0f;  // radians above the listening plane
    bool  lfe = false;
    SpeakerGroup group = SpeakerGroup::front;

    // Index of this speaker's mirror image across the median plane, or -1 for
    // the ones that are their own mirror (centre, top centre, mono). Mid/side
    // Post EQ bands fold about these pairs — see PostEq.
    int mirror = -1;
};

// One entry per `juce::AudioChannelSet::ChannelType` Horizon can place.
// Degrees here and radians in the table, because the numbers are worth being
// able to read.
struct SpeakerPosition
{
    juce::AudioChannelSet::ChannelType type;
    float azimuthDegrees;
    float elevationDegrees;
};

inline constexpr std::array<SpeakerPosition, 24> speakerPositions { {
    { juce::AudioChannelSet::left,              -30.0f,   0.0f },
    { juce::AudioChannelSet::right,              30.0f,   0.0f },
    { juce::AudioChannelSet::centre,              0.0f,   0.0f },
    { juce::AudioChannelSet::LFE,                 0.0f,   0.0f },
    { juce::AudioChannelSet::LFE2,                0.0f,   0.0f },
    { juce::AudioChannelSet::leftCentre,        -15.0f,   0.0f },
    { juce::AudioChannelSet::rightCentre,        15.0f,   0.0f },
    { juce::AudioChannelSet::wideLeft,          -60.0f,   0.0f },
    { juce::AudioChannelSet::wideRight,          60.0f,   0.0f },
    { juce::AudioChannelSet::leftSurroundSide,  -90.0f,   0.0f },
    { juce::AudioChannelSet::rightSurroundSide,  90.0f,   0.0f },
    { juce::AudioChannelSet::leftSurround,     -110.0f,   0.0f },
    { juce::AudioChannelSet::rightSurround,     110.0f,   0.0f },
    { juce::AudioChannelSet::leftSurroundRear, -150.0f,   0.0f },
    { juce::AudioChannelSet::rightSurroundRear, 150.0f,   0.0f },
    { juce::AudioChannelSet::centreSurround,    180.0f,   0.0f },
    { juce::AudioChannelSet::topFrontLeft,      -45.0f,  35.0f },
    { juce::AudioChannelSet::topFrontRight,      45.0f,  35.0f },
    { juce::AudioChannelSet::topFrontCentre,      0.0f,  35.0f },
    { juce::AudioChannelSet::topSideLeft,       -90.0f,  45.0f },
    { juce::AudioChannelSet::topSideRight,       90.0f,  45.0f },
    { juce::AudioChannelSet::topRearLeft,      -135.0f,  35.0f },
    { juce::AudioChannelSet::topRearRight,      135.0f,  35.0f },
    { juce::AudioChannelSet::topRearCentre,     180.0f,  35.0f },
} };

// A resolved layout: the directions, the groups and the mirror pairing, plus
// the two counts everything downstream is written against.
//
// Deliberately a plain value with a fixed array rather than anything that
// allocates: it is rebuilt in prepareToPlay and read on the audio thread.
class SpeakerLayout
{
public:
    SpeakerLayout() { setStereo(); }

    // Resolves a host channel set. Anything unnamed falls back to a horizontal
    // fan, so a layout with `unknown` channel types still produces a room
    // rather than sixteen speakers stacked at zero degrees, all reading the
    // same direction and cross-feeding at unity.
    void set (const juce::AudioChannelSet& channelSet)
    {
        count = juce::jlimit (0, maxSpeakers, channelSet.size());

        if (count == 0)
        {
            setStereo();
            return;
        }

        // Mono is a direction, not a missing one: a single speaker is straight
        // ahead and is the whole front group.
        if (count == 1)
        {
            speakers[0] = { 0.0f, 0.0f, false, SpeakerGroup::front, -1 };
            finish();
            return;
        }

        for (int c = 0; c < count; ++c)
        {
            const auto type = channelSet.getTypeOfChannel (c);
            const auto* found = find (type);

            if (found != nullptr)
            {
                speakers[(size_t) c] = {
                    degToRad (found->azimuthDegrees),
                    degToRad (found->elevationDegrees),
                    type == juce::AudioChannelSet::LFE || type == juce::AudioChannelSet::LFE2,
                    groupFor (found->azimuthDegrees, found->elevationDegrees,
                              type == juce::AudioChannelSet::centre),
                    -1
                };
            }
            else
            {
                // Spread the unnameable ones evenly across the front half,
                // alternating sides so a pair still mirrors.
                const float spread = juce::jmap ((float) c, 0.0f,
                                                 (float) juce::jmax (1, count - 1),
                                                 -60.0f, 60.0f);
                speakers[(size_t) c] = { degToRad (spread), 0.0f, false,
                                         SpeakerGroup::front, -1 };
            }
        }

        finish();
    }

    void setStereo()
    {
        count = 2;
        speakers[0] = { degToRad (-30.0f), 0.0f, false, SpeakerGroup::front, 1 };
        speakers[1] = { degToRad (30.0f), 0.0f, false, SpeakerGroup::front, 0 };
        finish();
    }

    int size() const noexcept { return count; }
    const Speaker& operator[] (int c) const noexcept
    {
        return speakers[(size_t) juce::jlimit (0, maxSpeakers - 1, c)];
    }

    // How many speakers carry reverb. The LFE does not — see the note in
    // ReverbEngine on why it is passed through untouched — so this is the
    // number every wet-side array is sized and normalized by, and it is what
    // "one Walsh row per speaker" has to fit inside.
    int numWet() const noexcept { return wetCount; }

    bool isSurround() const noexcept { return wetCount > 2; }
    bool hasLfe() const noexcept { return wetCount != count; }

    // True when the layout contains any speaker of this group, which is what
    // decides whether feature 14's Speakers setting has anything to offer. In
    // stereo only `front` is populated, so the control is inert and the UI says
    // so rather than pretending.
    bool hasGroup (SpeakerGroup g) const noexcept
    {
        if (g == SpeakerGroup::all)
            return true;
        for (int c = 0; c < count; ++c)
            if (! speakers[(size_t) c].lfe && speakers[(size_t) c].group == g)
                return true;
        return false;
    }

    // Angle between two speakers, in radians, on the sphere. This is the
    // quantity the cross-feed kernel is a function of: what makes two channels
    // of one room correlated is how close together they are, and in an
    // immersive layout "close together" has to include the height layer or the
    // ceiling ends up decorrelated from the wall directly under it.
    float angleBetween (int a, int b) const noexcept
    {
        const auto& p = speakers[(size_t) juce::jlimit (0, maxSpeakers - 1, a)];
        const auto& q = speakers[(size_t) juce::jlimit (0, maxSpeakers - 1, b)];

        const float dot = std::cos (p.elevation) * std::cos (q.elevation)
                              * std::cos (p.azimuth - q.azimuth)
                          + std::sin (p.elevation) * std::sin (q.elevation);
        return std::acos (juce::jlimit (-1.0f, 1.0f, dot));
    }

    bool operator== (const SpeakerLayout& other) const noexcept
    {
        if (count != other.count)
            return false;
        for (int c = 0; c < count; ++c)
        {
            const auto& a = speakers[(size_t) c];
            const auto& b = other.speakers[(size_t) c];
            if (! juce::approximatelyEqual (a.azimuth, b.azimuth)
                || ! juce::approximatelyEqual (a.elevation, b.elevation)
                || a.lfe != b.lfe || a.group != b.group)
                return false;
        }
        return true;
    }

    bool operator!= (const SpeakerLayout& other) const noexcept { return ! (*this == other); }

private:
    static float degToRad (float deg) noexcept
    {
        return deg * juce::MathConstants<float>::pi / 180.0f;
    }

    static const SpeakerPosition* find (juce::AudioChannelSet::ChannelType type) noexcept
    {
        for (const auto& p : speakerPositions)
            if (p.type == type)
                return &p;
        return nullptr;
    }

    // The group boundaries. 65 degrees puts the 9.x wide pair at 60 in the
    // front group and the side surrounds at 90 out of it, which is where a
    // mixer would draw the line; 20 degrees of elevation separates a height
    // layer nominally at 35 from a listening plane nominally at 0 with room for
    // either to be stated a little differently.
    static SpeakerGroup groupFor (float azDeg, float elDeg, bool isCentre) noexcept
    {
        if (elDeg >= 20.0f)
            return SpeakerGroup::height;
        if (isCentre)
            return SpeakerGroup::centre;
        return std::abs (azDeg) <= 65.0f ? SpeakerGroup::front : SpeakerGroup::surround;
    }

    // Pairs each speaker with its mirror image and counts the wet ones. Mirrors
    // are matched on direction rather than on channel type, so a layout that
    // spells its side pair `leftSurround`/`rightSurround` pairs up exactly as
    // one that spells it `leftSurroundSide`/`rightSurroundSide`.
    void finish() noexcept
    {
        wetCount = 0;
        for (int c = 0; c < count; ++c)
        {
            speakers[(size_t) c].mirror = -1;
            if (! speakers[(size_t) c].lfe)
                ++wetCount;
        }

        for (int c = 0; c < count; ++c)
        {
            auto& a = speakers[(size_t) c];
            if (a.lfe || a.mirror >= 0)
                continue;

            for (int k = c + 1; k < count; ++k)
            {
                auto& b = speakers[(size_t) k];
                if (b.lfe || b.mirror >= 0)
                    continue;

                if (std::abs (a.azimuth + b.azimuth) < 1.0e-3f
                    && std::abs (a.elevation - b.elevation) < 1.0e-3f
                    && std::abs (a.azimuth) > 1.0e-3f)
                {
                    a.mirror = k;
                    b.mirror = c;
                    break;
                }
            }
        }
    }

    std::array<Speaker, maxSpeakers> speakers {};
    int count = 0;
    int wetCount = 0;
};

// The layouts Horizon offers the host (v0.7). The whitelist is a policy about
// what has been tested rather than a structural limit — the engine is written
// against directions and channel counts, so anything up to sixteen channels
// works — but a plugin that reports every layout JUCE can name is a plugin that
// gets instantiated at 22.2 and has never been listened to at 22.2.
inline std::array<juce::AudioChannelSet, 6> supportedLayouts()
{
    return { juce::AudioChannelSet::mono(),
             juce::AudioChannelSet::stereo(),
             juce::AudioChannelSet::create5point1(),
             juce::AudioChannelSet::create7point1(),
             juce::AudioChannelSet::create7point1point4(),
             juce::AudioChannelSet::create9point1point6() };
}

inline bool isSupportedLayout (const juce::AudioChannelSet& set)
{
    for (const auto& s : supportedLayouts())
        if (s == set)
            return true;
    return false;
}

// A short name for the panel: "7.1.4" rather than JUCE's descriptive string,
// which runs to a comma-separated list of every channel and is longer than the
// header. Derived from the counts so it stays right for a layout not on the
// whitelist above.
inline juce::String layoutShortName (const SpeakerLayout& layout)
{
    if (layout.size() <= 1)
        return "MONO";
    if (layout.size() == 2 && ! layout.hasLfe())
        return "STEREO";

    int bed = 0, height = 0, lfe = 0;
    for (int c = 0; c < layout.size(); ++c)
    {
        if (layout[c].lfe)                                 ++lfe;
        else if (layout[c].group == SpeakerGroup::height)  ++height;
        else                                               ++bed;
    }

    auto s = juce::String (bed) + "." + juce::String (lfe);
    if (height > 0)
        s += "." + juce::String (height);
    return s;
}

} // namespace hrzn
