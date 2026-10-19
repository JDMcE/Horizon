#pragma once

#include <juce_core/juce_core.h>
#include <array>
#include <cmath>

namespace hrzn
{

// Which of the three engine cores a room model is built on (feature 3).
// The cores differ in how energy is distributed between discrete early
// reflections and the diffuse late field, and in how the late field is
// generated.
//
// From v0.2 this is a *label*, not a switch: Space morphs steplessly through
// the table, so nothing in the engine may branch on a discrete core. What
// used to be "which core" is now carried by the interpolatable geometry
// fields below — chiefly `diffuseStages` and `diffusion` (how deep and how
// strong the diffusion network runs), `erTaps`/`erLevel` (how much discrete
// early energy there is) and `sizeScale`. A plate is a plate because it has
// eight deep diffusers and almost no discrete reflections, not because an
// enum said so. The name survives for the UI readout.
enum class Core
{
    sparse = 0, // ambiences and small rooms: tap-bank ERs, light late field
    dense,      // plates and chambers: deep allpass diffusion, few discrete ERs
    hall        // halls and cathedrals: full FDN late field behind real ERs
};

inline const juce::StringArray coreNames { "Sparse", "Dense", "Hall" };

// The shape of a room's early reflection pattern, as distinct from its size.
//
// Through v0.4.2 every model shared one generator — `pow(u, 0.75)` tap spacing
// under an `exp(-2.6 t)` envelope, uniform panning — and differed only in seed,
// count, span and level. That is why the small rooms did not sound small: Small
// Room was sixteen taps across 28 ms, a tap every 1.8 ms at 1/sqrt(16) each with
// alternating polarity, which is a noise burst. What tells the ear it is in a
// small box is a *handful of strong, isolated* reflections with audible gaps
// between them, and no amount of retuning the count alone produces that while
// the envelope and the amplitude spread stay fixed.
//
// All four fields are baked into the precomputed tap tables at prepare() time,
// so they cost nothing at runtime and — unlike the late field — carry no
// stability risk whatsoever. The neutral values in the comment reproduce the
// v0.4.2 generator exactly, which is how this change was verified one step at a
// time.
struct ErShape
{
    float clustering;   // tap spacing exponent; <1 crowds the onset (v0.4.2: 0.75)
    float decayRate;    // envelope decay across the span (v0.4.2: 2.6)
    // Contrast between the loudest and quietest taps, as an exponent on the
    // per-tap jitter. 0 is the v0.4.2 even spray; 1 stretches a [0.55, 1] jitter
    // to [0.05, 1], which is 26 dB between the strong bounces and the fill and
    // is what reads as discrete reflections rather than as diffusion.
    float discreteness; // 0 .. 1
    float panSpread;    // 1 = hard across the image (v0.4.2), 0 = centred
};

// How the late field's *structure* varies from room to room, as opposed to how
// large it is. These are the fields that stop the table being one box resized.
struct LateShape
{
    // Exponent on the spread of the sixteen FDN line lengths about their
    // geometric mean. 1.0 is the base set (see ReverbEngine::baseDelayMs).
    //
    // Until v0.5 the only per-room quantity was `sizeScale`, which multiplies
    // the whole set — so the *ratios* between lines, and therefore the modal
    // pattern the ear identifies a space by, were identical in all fourteen
    // rooms. A cathedral was an ambience transposed down. This exponent is what
    // gives each room its own modal fingerprint instead of a transposition.
    //
    // Total delay is renormalized away afterwards, so this changes the
    // distribution of the modes and never their density (an FDN has one mode per
    // 1/totalDelay hertz — see ReverbEngine::totalLineDelaySeconds, which stays
    // exact for every model precisely because of that renormalization). Above 1
    // the lines spread wider and arrival times grow distinct; below 1 they
    // cluster, modes pile up, and the result is the metallic ring that a plate
    // wants and that everything else must be kept away from. The coloration
    // bound in the harness is what says how far down each room may go.
    float lineSpread;

    // The input diffuser cascade, per room at last. v0.4.2 shared one set of
    // eight lengths scaled by `jlimit(0.35, 1.25, sizeScale * 1.4)`, and that
    // clamp floored at 0.35 for every model up to Chamber — so the six smallest
    // rooms ran *identical* diffusers, which is a large part of why they shared
    // a voice. Stated directly, a plate can have the short tight cascade that is
    // most of what a plate is and a hall can have a long loose one.
    float diffMeanMs;   // geometric-mean diffuser length
    float diffSpread;   // exponent on their spread, as lineSpread is for the tank

    // Skew of the feedback matrix's reflection vector. 0 is the classic
    // I - (2/N)J that every room used through v0.4.2.
    //
    // The matrix stays a Householder reflection H = I - 2vv^T, which is
    // orthogonal for *any* unit v — so the lossless-tank invariant Freeze
    // depends on, and the rule that every bit of decay comes from the feedback
    // gains, hold at every value of this field. What changes is how evenly the
    // lines couple: v uniform mixes all sixteen equally and maximally, which is
    // the smooth wash v0.4.2 gave every room, while a skewed v makes the
    // large-weight lines hubs and leaves the small-weight ones nearly
    // independent, so their modes stay individually audible.
    //
    // Its range is deliberately small, and finding out why cost a round of
    // measurements. Uneven coupling buys individually audible modes — character —
    // but it buys them by *reducing inter-line interaction*, and inter-line
    // interaction is what modulated lines beating against one another is made of.
    // That is the same mechanism as the tank's shimmer and as Character's chorus
    // extreme. The first cut of this table skewed the large rooms hard, and the
    // harness caught the consequence exactly: Boundless's held-tone concentration
    // went from 0.67 to 0.97 — it stopped shimmering — and Character 100% stopped
    // reaching a chorus at all. So the large, lush rooms stay near uniform, where
    // their lushness comes from, and what skew there is belongs to the small and
    // medium rooms, which want to sound like a place with walls and have the
    // coloration headroom to pay for it.
    float mixSkew;
};

// Per-room stereo behaviour. Both of these were hardcoded constants shared by
// every model, which is why a vocal booth and an arena had the same image.
struct StereoShape
{
    float diffDetune;      // L/R input diffuser length ratio (v0.4.2: 1.061)
    // How correlated the two late-field taps are. 0 is the v0.4.2 pair of
    // orthogonal Walsh rows — fully decorrelated; 1 collapses them to one row,
    // i.e. a mono tail. This is not the Width control wearing a different hat:
    // Width scales the side signal after the fact, while this changes which
    // lines each output actually sums, so a small room's tail is *intrinsically*
    // narrower rather than turned down.
    float lateCorrelation; // 0 .. 1
};

// One room model: a geometry record, not a preset. Space (feature 8) is a
// stepless position through this table, and v0.2 interpolates every field
// here to morph between neighbours without clicks — which is why they are all
// plain interpolatable scalars and why `erTaps` is the only integer (it
// selects how many entries of a precomputed tap table are live; the tap
// times and gains themselves interpolate).
//
// The table must stay ordered by increasing size: Space combines room size
// and decay time in one control, so monotonicity is what makes the sweep
// feel like one continuous space growing around you.
struct RoomModel
{
    const char* name;
    Core core;

    float sizeScale;      // multiplies the base FDN delay set (see ReverbEngine)
    // The room's *natural* decay. From v0.4.1 this no longer reaches the engine
    // — Decay Time is an absolute control and Space does not touch it — so it
    // survives as documentation: what this room would ring for if it were a
    // real one, which is what the factory presets are authored against and
    // what the panel offers as the room's own figure.
    float baseRt60;
    float diffusion;      // 0..1 input allpass depth
    float diffuseStages;  // 1..8 how many input diffusers the model runs
    float erSpreadMs;     // time span of the early reflection pattern
    int   erTaps;         // number of discrete early reflections
    float erLevel;        // linear gain of the ER bank
    float lateLevel;      // linear gain of the late field
    // FDN delay modulation, as a fraction of each line's own length rather
    // than as an absolute time (v0.4.2).
    //
    // A loop of length D resonates at n*fs/D, so varying D by +/-depth detunes
    // every mode in it by depth/D. That ratio — not the depth — is what the ear
    // hears, and specifying the depth in milliseconds made it a function of
    // room size: the same 0.22 ms that is a gentle 5 cents on a cathedral's
    // 1705-sample line is 25% of an ambience's 43-sample one, which is nearly
    // four semitones of warble. Every small room in the table was audibly
    // pitch-modulated, worst at the small end where the modal field is sparse
    // enough to hear individual modes move.
    //
    // As a ratio the figure means the same thing on every room: 0.004 is about
    // seven cents of movement whatever the room's size. Character still scales
    // it, so its chorus extreme reaches a real chorus.
    float modDetune;      // peak modal detune, as a fraction (0.004 = ~7 cents)
    float modRateHz;      // FDN delay modulation rate

    // The room's spectral decay character, as *ratios* to the mid band, which
    // is what a room's materials actually determine (v0.4.1). `hfDecayRatio`
    // 0.40 means the top octave rings for 40% as long as 1 kHz does; the
    // corners say where each shelf starts working.
    //
    // These replace what used to be a lowpass and a highpass cutoff with a
    // broadband loss attached. Expressing them as ratios is what lets Decay
    // Time be an absolute number of seconds on every room, and what makes a
    // room's voicing survive a change of decay length rather than track it.
    float hfDampHz;       // high shelf corner (air absorption)
    float hfDecayRatio;   // HF decay / mid decay at neutral Brightness
    float lfDampHz;       // low shelf corner
    float lfDecayRatio;   // LF decay / mid decay. Capped below 1 — see below.

    unsigned int seed;    // deterministic ER pattern seed

    // v0.5: the fields that make the rooms differ in kind rather than in size.
    // Appended rather than interleaved so the fifteen columns above keep the
    // layout they have had since v0.1.
    ErShape     er;
    LateShape   late;
    StereoShape stereo;
};

// Fourteen models spanning tight ambience to a cathedral. Sizes and decay
// times rise monotonically; the core changes underneath without the user
// needing to know, which is the whole point of a single Space control.
//
// A note on the low shelf. Through v0.4 the low end was a highpass with a
// subsonic cutoff, for a good reason: a one-pole highpass has a very gentle
// skirt, so a corner up at 60 Hz still cost measurable energy per pass at
// 500 Hz, and a line traversed hundreds of times a second turns that into a
// mid-band decay noticeably shorter than the model claims. The consequence was
// that every room held its low end for exactly as long as its mid, which is
// not what any real room does and is why a big setting always sounded muddy.
//
// A *shelf* is unity above its corner, so it can sit at 300 Hz without costing
// the mid band anything. The small rooms now let go of their low end quickly
// (a small room physically cannot sustain a long low mode) and the large ones
// hold onto it, which is most of what makes the two ends of this table sound
// like different places rather than like one place resized.
//
// The ratios stay below 1: a low end that rings *longer* than the mid would
// need the feedback loop to lift, which is what the Decay Rate EQ is for and
// what it takes care to do safely (see DecayEq.h). Keeping every model on the
// attenuating side also means DC can never build up, which is the job the old
// highpass was really doing.
//
// A note on diffuseStages: the plates and the chamber run the network almost
// to its full depth (cascaded allpasses are what a plate *is*), the small
// rooms barely diffuse at all so their discrete reflections stay audible, and
// the halls sit in the middle — they get their density from the tank rather
// than from the input stage. Thickness (feature 7) scales this whole column,
// and Character (feature 13) pulls against it.
// The v0.5 revoicing: what each column is being asked to do.
//
// **er.clustering / er.decayRate / er.discreteness.** The small end of the table
// gets few, strong, evenly spaced reflections with 20-odd dB between the loud
// ones and the fill; the large end keeps the dense crowded spray it always had.
// This is the change you hear first and it is most of what was wrong: Small Room
// was sixteen taps across 28 ms at equal amplitude, which is a noise burst, and
// its `erTaps` column has come down accordingly. Close walls also mean the early
// reflections arrive nearly as loud as one another, so `decayRate` rises with
// size rather than staying at 2.6 for a broom cupboard and a cathedral alike.
//
// **late.lineSpread.** Below 1 for the two plates and nowhere else. Clustering
// the line lengths piles the modes up, and a dense colored mode field is exactly
// what a plate is and exactly what every other room has to be kept away from.
// The halls open out above 1, which spreads their modes further apart and, at the
// same total delay, makes their arrival times distinctly different from one
// another instead of being one pattern transposed.
//
// Ambience is the one room that could not take the treatment its size suggests.
// It has the tightest coloration budget in the table (5.6 dB against a 7 dB
// bound) because at sizeScale 0.045 its shortest line is only 43 samples, whose
// modes are already 1.1 kHz apart and individually audible; widening the spread
// shortens that line further and rings. So it stays just under 1, and its
// identity is carried by its early reflections, which is where a tiny room's
// identity actually lives.
//
// **late.diffMeanMs / diffSpread.** Stated per room at last. The plates get the
// short tight cascade that makes a plate shimmer; the halls get long loose ones,
// though not as long as the arithmetic wanted — a 30 ms allpass in front of the
// tank smears the onset of a large room into mush, which is the trap the old
// size-derived clamp was clumsily avoiding.
//
// **late.mixSkew.** Positive weights the short lines as the hubs energy crosses
// through, which leaves the long lines — the ones carrying the slow low modes —
// nearly independent and individually audible: a room with a character rather
// than a wash. Negative does the reverse, leaving the *short* lines free to ring,
// which reads as metallic and is why the plates are the only negative entries.
// The halls sit mildly positive, small rooms further up.
//
// **stereo.** A vocal booth and an arena shared a 1.061 diffuser detune and a
// fully decorrelated tail through v0.4.2. Now the small rooms are intrinsically
// narrow and correlated, as a small room is, and the image opens with size.
//                                       size    rt60   diff  stages  erMs  taps   erLv  lateLv detune modHz    hfHz  hfRatio    lfHz  lfRatio  seed
//                                                       er: clust  decay  discr   pan      late: spread  diffMs  dSprd   skew     stereo: detune  corr
inline constexpr std::array<RoomModel, 14> roomModels { {
    { "Ambience",     Core::sparse, 0.045f,  0.28f, 0.56f, 4.4f,  16.0f,  8, 1.05f, 0.30f, 0.0030f, 1.20f, 11000.0f, 0.62f, 260.0f, 0.56f, 0x1234u,
      { 1.00f, 1.5f, 0.85f, 0.55f }, { 0.96f,   1.3f, 0.70f,  0.34f }, { 1.020f, 0.40f } },
    { "Small Room",   Core::sparse, 0.085f,  0.45f, 0.54f, 4.0f,  28.0f,  9, 0.92f, 0.52f, 0.0035f, 1.05f,  9500.0f, 0.55f, 235.0f, 0.60f, 0x2f19u,
      { 0.95f, 1.7f, 0.80f, 0.65f }, { 1.04f,   1.8f, 0.78f,  0.38f }, { 1.028f, 0.33f } },
    { "Studio",       Core::sparse, 0.130f,  0.65f, 0.60f, 4.2f,  38.0f, 12, 0.80f, 0.70f, 0.0045f, 0.88f,  9000.0f, 0.52f, 210.0f, 0.65f, 0x51a3u,
      { 0.90f, 2.0f, 0.65f, 0.75f }, { 1.09f,   2.4f, 0.85f,  0.36f }, { 1.038f, 0.25f } },
    { "Live Room",    Core::hall,   0.190f,  0.90f, 0.66f, 4.6f,  48.0f, 16, 0.72f, 0.86f, 0.0060f, 0.80f,  8800.0f, 0.60f, 190.0f, 0.71f, 0x7c42u,
      { 0.85f, 2.2f, 0.50f, 0.85f }, { 1.13f,   3.2f, 0.92f,  0.30f }, { 1.048f, 0.17f } },
    { "Chamber",      Core::dense,  0.250f,  1.25f, 0.77f, 6.8f,  55.0f, 14, 0.50f, 1.00f, 0.0100f, 0.72f,  8000.0f, 0.46f, 170.0f, 0.77f, 0x9d31u,
      { 0.80f, 2.4f, 0.25f, 0.90f }, { 1.00f,   2.6f, 0.76f,  0.18f }, { 1.055f, 0.12f } },
    { "Bright Plate", Core::dense,  0.220f,  1.60f, 0.86f, 7.8f,  26.0f,  4, 0.24f, 1.06f, 0.0110f, 1.45f, 13000.0f, 0.74f, 215.0f, 0.63f, 0xb60fu,
      { 0.72f, 2.8f, 0.20f, 0.95f }, { 0.92f,   1.6f, 0.56f, -0.20f }, { 1.076f, 0.05f } },
    { "Warm Plate",   Core::dense,  0.260f,  2.00f, 0.87f, 8.0f,  32.0f,  4, 0.24f, 1.06f, 0.0110f, 1.20f,  6800.0f, 0.34f, 180.0f, 0.67f, 0xd7e8u,
      { 0.72f, 2.7f, 0.20f, 0.95f }, { 0.94f,   1.9f, 0.58f, -0.18f }, { 1.076f, 0.05f } },
    { "Small Hall",   Core::hall,   0.380f,  1.80f, 0.72f, 5.4f,  70.0f, 20, 0.58f, 1.00f, 0.0115f, 0.64f,  7800.0f, 0.46f, 148.0f, 0.83f, 0xf2b7u,
      { 0.78f, 2.5f, 0.18f, 0.95f }, { 1.11f,   4.2f, 1.00f,  0.14f }, { 1.062f, 0.10f } },
    { "Concert Hall", Core::hall,   0.560f,  2.40f, 0.74f, 5.6f,  92.0f, 24, 0.48f, 1.00f, 0.0140f, 0.54f,  7200.0f, 0.41f, 125.0f, 0.88f, 0x1a95u,
      { 0.75f, 2.6f, 0.12f, 1.00f }, { 1.05f,   5.2f, 1.04f,  0.10f }, { 1.070f, 0.06f } },
    { "Large Hall",   Core::hall,   0.720f,  3.20f, 0.77f, 5.8f, 112.0f, 28, 0.41f, 1.00f, 0.0170f, 0.47f,  6800.0f, 0.36f, 108.0f, 0.92f, 0x3c73u,
      { 0.75f, 2.6f, 0.10f, 1.00f }, { 1.13f,   6.0f, 1.08f,  0.10f }, { 1.078f, 0.04f } },
    { "Arena",        Core::hall,   0.900f,  4.20f, 0.80f, 6.0f, 140.0f, 30, 0.36f, 1.00f, 0.0200f, 0.41f,  6400.0f, 0.32f,  95.0f, 0.94f, 0x5e51u,
      { 0.75f, 2.6f, 0.08f, 1.00f }, { 1.03f,   6.8f, 1.10f,  0.09f }, { 1.085f, 0.03f } },
    { "Cathedral",    Core::hall,   1.150f,  6.00f, 0.82f, 6.2f, 172.0f, 32, 0.31f, 1.00f, 0.0230f, 0.35f,  6000.0f, 0.27f,  85.0f, 0.95f, 0x802fu,
      { 0.75f, 2.6f, 0.06f, 1.00f }, { 1.12f,   7.6f, 1.12f,  0.08f }, { 1.092f, 0.02f } },
    { "Vast Space",   Core::hall,   1.450f,  9.00f, 0.84f, 6.4f, 205.0f, 32, 0.27f, 1.00f, 0.0250f, 0.29f,  5600.0f, 0.23f,  78.0f, 0.95f, 0xa20du,
      { 0.75f, 2.6f, 0.05f, 1.00f }, { 1.06f,   8.2f, 1.14f,  0.07f }, { 1.098f, 0.02f } },
    { "Boundless",    Core::hall,   1.800f, 14.00f, 0.86f, 6.6f, 245.0f, 32, 0.22f, 1.00f, 0.0280f, 0.25f,  5200.0f, 0.19f,  70.0f, 0.95f, 0xc3ebu,
      { 0.75f, 2.6f, 0.04f, 1.00f }, { 1.08f,   8.8f, 1.15f,  0.06f }, { 1.105f, 0.02f } },
} };

inline constexpr int numRoomModels = (int) roomModels.size();

// Space (0..1) -> continuous table position. From v0.2 the fractional part is
// the morph: the engine sits *between* two models and interpolates their
// geometry rather than snapping to the nearer one.
inline float spaceToPosition (float space01) noexcept
{
    return juce::jlimit (0.0f, 1.0f, space01) * (float) (numRoomModels - 1);
}

// Still needed by the UI readout, the factory presets and the tests, which
// address models by index.
inline int spaceToModelIndex (float space01) noexcept
{
    return juce::jlimit (0, numRoomModels - 1,
                         (int) std::lround (spaceToPosition (space01)));
}

inline int positionToModelIndex (float position) noexcept
{
    return juce::jlimit (0, numRoomModels - 1, (int) std::lround (position));
}

// The scalar fields of the model the engine is actually running: a point
// between two table entries. Delay lengths, tap times and diffuser lengths do
// *not* live here — those move every sample and are kept in the engine (see
// ReverbEngine::evaluateGeometry). These are the ones that only need to be
// refreshed at control rate.
struct MorphedModel
{
    float sizeScale;
    float rt60;
    float diffusion;
    float diffuseStages;
    float erLevel;
    float lateLevel;
    float modDetune;
    float modRateHz;
    float hfDampHz;
    float hfDecayRatio;
    float lfDampHz;
    float lfDecayRatio;

    // The two v0.5 fields that are genuinely control-rate. The rest of the new
    // geometry — the ER shape, the line spread, the diffuser lengths and the L/R
    // diffuser detune — is baked into tables precomputed per model at prepare()
    // time and interpolated per sample, so none of it needs a scalar here.
    float mixSkew;
    float lateCorrelation;
};

// Geometric interpolation, for the quantities the ear hears as ratios: decay
// times spanning 0.28 s to 14 s and cutoffs spanning 3.3 kHz to 12 kHz are
// perceptually uniform in log space, not in linear space. Linear interpolation
// of a 14 s decay against a 9 s one would spend most of the knob's travel in
// the top second.
//
// `sizeScale` deliberately stays *linear*: it is evaluated per sample (it sets
// every FDN delay length) and the table is dense enough that the difference is
// inaudible, so it is not worth a transcendental in the sample loop.
inline float lerpGeometric (float a, float b, float f) noexcept
{
    if (a <= 0.0f || b <= 0.0f)
        return a + (b - a) * f;
    return a * std::pow (b / a, f);
}

inline float lerpLinear (float a, float b, float f) noexcept
{
    return a + (b - a) * f;
}

inline MorphedModel morphBetween (int lowIndex, int highIndex, float f) noexcept
{
    const auto& A = roomModels[(size_t) juce::jlimit (0, numRoomModels - 1, lowIndex)];
    const auto& B = roomModels[(size_t) juce::jlimit (0, numRoomModels - 1, highIndex)];
    f = juce::jlimit (0.0f, 1.0f, f);

    MorphedModel m {};
    m.sizeScale     = lerpLinear (A.sizeScale, B.sizeScale, f);
    m.rt60          = lerpGeometric (A.baseRt60, B.baseRt60, f);
    m.diffusion     = lerpLinear (A.diffusion, B.diffusion, f);
    m.diffuseStages = lerpLinear (A.diffuseStages, B.diffuseStages, f);
    m.erLevel       = lerpLinear (A.erLevel, B.erLevel, f);
    m.lateLevel     = lerpLinear (A.lateLevel, B.lateLevel, f);
    m.modDetune     = lerpLinear (A.modDetune, B.modDetune, f);
    m.modRateHz     = lerpGeometric (A.modRateHz, B.modRateHz, f);
    m.hfDampHz      = lerpGeometric (A.hfDampHz, B.hfDampHz, f);
    m.lfDampHz      = lerpGeometric (A.lfDampHz, B.lfDampHz, f);

    // The ratios interpolate geometrically for the same reason the cutoffs do:
    // they are heard as proportions, and a 0.19 blended linearly against a
    // 0.95 spends most of the morph sounding like the brighter of the two.
    m.hfDecayRatio  = lerpGeometric (A.hfDecayRatio, B.hfDecayRatio, f);
    m.lfDecayRatio  = lerpGeometric (A.lfDecayRatio, B.lfDecayRatio, f);

    // Both are signed or zero-based quantities describing a shape rather than a
    // ratio a listener hears proportionally, so both interpolate linearly. A
    // geometric blend of `mixSkew` would not even be defined at 0, which is the
    // value most of the table sits at.
    m.mixSkew         = lerpLinear (A.late.mixSkew, B.late.mixSkew, f);
    m.lateCorrelation = lerpLinear (A.stereo.lateCorrelation, B.stereo.lateCorrelation, f);
    return m;
}

inline MorphedModel morphAtPosition (float position) noexcept
{
    const float p = juce::jlimit (0.0f, (float) (numRoomModels - 1), position);
    const int i0 = juce::jlimit (0, numRoomModels - 1, (int) p);
    const int i1 = juce::jmin (i0 + 1, numRoomModels - 1);
    return morphBetween (i0, i1, p - (float) i0);
}

// The decay time the table alone asks for at a given Space position, before
// Decay Rate. Used by the UI and by the tests to state the contract.
inline float nominalRt60At (float position) noexcept
{
    return morphAtPosition (position).rt60;
}

} // namespace hrzn
