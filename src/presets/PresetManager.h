#pragma once

#include <jdm/presets/PresetManager.h>
#include "../Parameters.h"

namespace hrzn
{

using jdm::PresetManager;
using jdm::PresetInfo;

// Horizon's preset ecosystem configuration: file format identity plus the
// factory library, handed to the shared jdm::PresetManager. State schema
// mirrors HorizonAudioProcessor::stateVersion.
//
// The library was revisited whole at v0.8, because it had been written against
// v0.3 and the plugin had gained five phases of controls since: absolute decay
// time (v0.4.1), Ducking, the Auto Gate and tempo-synced Predelay (v0.5), IR
// matching (v0.6), surround out to 9.1.6 with speaker-scoped Post EQ bands
// (v0.7), and the modal density floor that revoiced the four small rooms
// (v0.7.1). Four things came out of the pass, and each is a rule the harness
// now enforces in `testPresetLibraryQuality`:
//
//  1. **Every room model is somebody's starting point.** Space is a continuum
//     and the browser is the only signpost through it, so a model with no
//     preset behind it is a model nobody will ever hear. Warm Plate and Vast
//     Space had none.
//
//  2. **Every control a preset can carry is carried by one.** A user finds out
//     that this reverb ducks, gates, syncs to tempo or scopes an EQ band to the
//     height layer by loading something that does it. Through v0.7 nothing in
//     the library touched any of them.
//
//  3. **The decay times are chosen, not inherited.** Through v0.4 decay was a
//     percentage of the room's own RT60, so the v0.4.1 migration multiplied
//     each preset's percentage by its model's decay and left the product:
//     1.36 s, 4.32 s, 6.72 s, 5.49 s. Those are arithmetic, not decisions. They
//     are stated in round numbers now, which also means changing a room model's
//     `baseRt60` no longer silently moves a preset's tail.
//
//  4. **No preset ships with Freeze engaged.** Freeze seals the tank *and holds
//     the input out*, so a room frozen before anything was played into it has
//     nothing to sustain — the old "Freeze Pad" loaded to the dry signal and
//     nothing else. A Freeze preset is the setting you engage Freeze *from*.
inline jdm::PresetProductConfig makePresetConfig()
{
    jdm::PresetProductConfig c;
    c.productName = "Horizon";
    c.extension = "hrzn";
    c.presetTag = juce::Identifier ("HorizonPreset");
    c.stateTag = juce::Identifier ("HorizonState");
    c.stateVersion = 2;

    // Lock Mix (feature 17, v0.5) is the user's, not the preset's: it says how
    // loading should behave, so a preset that carried it could switch it off
    // on the way in and the setting would defeat itself. Listing it here
    // strips it from everything written and preserves the session's own value
    // through everything loaded.
    c.sessionProperties.add (prop::lockMix.toString());

    // MIDI Learn mappings (feature 26, v0.8). A preset that carries its own
    // MIDIMAP applies it; one that does not inherits the session's, because
    // loading a sound should never silently unplug a controller. That is the
    // family rule — Meridian names the same tree — and it is why the mappings
    // live in a child tree rather than in properties: a preset either brings a
    // controller layout or it brings none, and "none" has to be
    // distinguishable from "all controllers unmapped".
    c.inheritChildTrees.add ("MIDIMAP");

    // Space values are percentages through the room model table; the helper
    // keeps the presets readable and pinned to the model they mean, so
    // v0.2's stepless morph can retune the table without silently moving
    // every factory preset to a different room.
    auto atModel = [] (int index)
    {
        return 100.0f * (float) index / (float) (numRoomModels - 1);
    };

    // The two enum-valued Post EQ controls, spelled once. A band's shape and
    // its channel scope are stored as choice *indices*, and writing
    // `(float) (int) FilterShape::lowCut` at every use reads as a cast rather
    // than as a setting.
    auto shape = [] (FilterShape s) { return (float) (int) s; };
    auto scope = [] (ChannelMode m) { return (float) (int) m; };

    c.factory = {
        { "Init", "Basics", "default;clean",
          "Neutral starting point: a hall between Small and Concert, two "
          "seconds of decay, 30% mix.",
          {} },

        // --- Vocals ---------------------------------------------------------

        { "Vocal Plate", "Vocals", "plate;vocal;bright;pop",
          "Bright plate with a little predelay to keep the words in front of "
          "the tail. The classic pop vocal sheen.",
          { { id::space, atModel (5) }, { id::decayTime, 1.40f },
            { id::predelay, 28.0f }, { id::mix, 24.0f },
            { id::brightness, 66.0f }, { id::distance, 22.0f },
            { id::character, 36.0f }, { id::thickness, 66.0f } } },

        { "Warm Vocal Chamber", "Vocals", "chamber;vocal;warm;ballad",
          "Darker chamber sitting further back — for ballads where the plate "
          "would be too glassy.",
          { { id::space, atModel (4) }, { id::decayTime, 1.50f },
            { id::predelay, 35.0f }, { id::mix, 28.0f },
            { id::brightness, 36.0f }, { id::distance, 48.0f } } },

        { "Wide Air Vocal", "Vocals", "post eq;mid side;vocal;air;wide",
          "Post EQ doing what only it can: the tail's mid is rolled off below "
          "300 Hz so it never crowds the vocal, and the sides get an air lift "
          "the centre does not. The reverb opens up around the voice rather "
          "than in front of it.",
          { { id::space, atModel (7) }, { id::decayTime, 1.60f },
            { id::predelay, 30.0f }, { id::mix, 26.0f },
            { id::distance, 45.0f }, { id::width, 135.0f },
            { id::postEqOn (0), 1.0f }, { id::postEqType (0), shape (FilterShape::lowCut) },
            { id::postEqFreq (0), 300.0f }, { id::postEqSlope (0), 1.0f },
            { id::postEqChan (0), scope (ChannelMode::mid) },
            { id::postEqOn (5), 1.0f }, { id::postEqType (5), shape (FilterShape::highShelf) },
            { id::postEqFreq (5), 9000.0f }, { id::postEqGain (5), 5.0f },
            { id::postEqChan (5), scope (ChannelMode::side) } } },

        // Ducking's own preset (v0.5). The setting is chosen so the control
        // *is* the preset: a hall big enough that it would bury the vocal if
        // it sat still, pulled 10 dB out of the way while the line is being
        // sung and let back in between phrases.
        { "Ducked Pop Vocal", "Vocals", "ducking;vocal;pop;big;send",
          "A hall you could not otherwise get away with on a lead vocal. "
          "Ducking pulls the tail 10 dB down while the line is being sung and "
          "lets it back up in the gaps, so the room is only ever heard where "
          "there is room for it.",
          { { id::space, atModel (8) }, { id::decayTime, 2.20f },
            { id::predelay, 20.0f }, { id::mix, 36.0f },
            { id::brightness, 58.0f }, { id::distance, 52.0f },
            { id::duckOn, 1.0f }, { id::duckThreshold, -26.0f },
            { id::duckAmount, 10.0f }, { id::duckAttack, 6.0f },
            { id::duckRelease, 240.0f } } },

        // Tempo-synced Predelay's own preset (v0.5), and the Warm Plate's only
        // outing in the library — a plate dark enough that a predelay this long
        // reads as a slap rather than as a mistake.
        { "Warm Slap Vocal", "Vocals", "plate;warm;slap;tempo;sync;retro",
          "Warm plate behind a tempo-synced sixteenth of predelay, so the tail "
          "starts on the beat instead of on the word. Follows the host tempo; "
          "the readout shows the delay it is actually delivering.",
          { { id::space, atModel (6) }, { id::decayTime, 1.20f },
            { id::predelaySync, 1.0f }, { id::predelayDiv, 6.0f },
            { id::mix, 26.0f }, { id::brightness, 42.0f },
            { id::distance, 30.0f }, { id::character, 44.0f } } },

        // --- Drums ----------------------------------------------------------

        { "Drum Room", "Drums", "drums;room;tight;punch",
          "Tight live room, no predelay, close in: adds size to a kit without "
          "washing out the transients.",
          { { id::space, atModel (3) }, { id::decayTime, 0.65f },
            { id::predelay, 0.0f }, { id::mix, 22.0f },
            { id::brightness, 55.0f }, { id::distance, 18.0f },
            { id::width, 120.0f }, { id::character, 68.0f },
            { id::thickness, 32.0f } } },

        { "Snare Plate", "Drums", "drums;snare;plate;bright",
          "Short bright plate tuned to sit under a snare and disappear before "
          "the next hit.",
          { { id::space, atModel (5) }, { id::decayTime, 0.70f },
            { id::predelay, 12.0f }, { id::mix, 26.0f },
            { id::brightness, 72.0f }, { id::distance, 15.0f } } },

        // The Auto Gate's own preset (v0.5), and the one effect in the library
        // that is the *dynamics* rather than the room: a chamber set to ring
        // for two and a half seconds and then cut off at 110 ms, which is a
        // sound nothing but a gate has ever made.
        { "Gated Snare", "Drums", "gate;drums;snare;eighties;effect",
          "A big chamber shut off before you hear it is one. The Auto Gate "
          "holds the tail for a tenth of a second after the hit and then closes "
          "in twenty-five milliseconds — the sound the eighties built a decade "
          "on, and still the fastest way to make a snare enormous.",
          { { id::space, atModel (4) }, { id::decayTime, 2.40f },
            { id::predelay, 8.0f }, { id::mix, 40.0f },
            { id::brightness, 62.0f }, { id::distance, 20.0f },
            { id::thickness, 80.0f },
            { id::gateOn, 1.0f }, { id::gateThreshold, -32.0f },
            { id::gateHold, 110.0f }, { id::gateRelease, 25.0f } } },

        { "Kit Bloom", "Drums", "drums;parallel;big;wide;bus",
          "The other drum reverb: a live room heard from across it, wide and "
          "long enough to bloom behind the kit. Best on a parallel send with "
          "the fader riding the chorus.",
          { { id::space, atModel (3) }, { id::decayTime, 1.40f },
            { id::predelay, 18.0f }, { id::mix, 34.0f },
            { id::brightness, 50.0f }, { id::distance, 58.0f },
            { id::width, 150.0f }, { id::thickness, 62.0f } } },

        // --- Instruments ----------------------------------------------------

        { "Amp Room", "Instruments", "guitar;amp;small;room;character",
          "A small room with its reflections left audible — the sound of a "
          "cabinet in a space rather than a cabinet with reverb on it. "
          "Character is up, so the early bounces stay discrete.",
          { { id::space, atModel (1) }, { id::decayTime, 0.45f },
            { id::predelay, 6.0f }, { id::mix, 20.0f },
            { id::brightness, 48.0f }, { id::distance, 25.0f },
            { id::character, 62.0f } } },

        { "Piano Hall", "Instruments", "piano;acoustic;hall;natural;wide",
          "The concert hall from the middle of the stalls. Long enough to be a "
          "hall, far enough back to be believable, and voiced flat because an "
          "acoustic instrument brings its own top end.",
          { { id::space, atModel (8) }, { id::decayTime, 2.20f },
            { id::predelay, 24.0f }, { id::mix, 28.0f },
            { id::brightness, 54.0f }, { id::distance, 55.0f },
            { id::width, 120.0f } } },

        { "Synth Expanse", "Instruments", "synth;pad;huge;wide;ambient",
          "Vast Space at eight seconds: past the size of any hall and not yet "
          "the endless wash. Pads, arpeggios and anything that wants to sound "
          "like it is happening somewhere very large.",
          { { id::space, atModel (12) }, { id::decayTime, 8.00f },
            { id::predelay, 50.0f }, { id::mix, 42.0f },
            { id::brightness, 46.0f }, { id::distance, 70.0f },
            { id::width, 165.0f }, { id::thickness, 72.0f } } },

        // --- Halls ----------------------------------------------------------

        { "Concert Hall", "Halls", "hall;orchestral;natural;wide",
          "The reference hall at its own decay, far enough back to hear the "
          "room rather than the source.",
          { { id::space, atModel (8) }, { id::decayTime, 2.40f },
            { id::predelay, 22.0f }, { id::mix, 32.0f },
            { id::distance, 58.0f }, { id::width, 115.0f } } },

        { "Grand Hall", "Halls", "hall;large;orchestral;lush",
          "Large hall stretched a little longer, wide and lush — strings, "
          "pads, anything that wants to bloom.",
          { { id::space, atModel (9) }, { id::decayTime, 4.00f },
            { id::predelay, 30.0f }, { id::mix, 34.0f },
            { id::brightness, 44.0f }, { id::distance, 65.0f },
            { id::width, 140.0f }, { id::thickness, 70.0f } } },

        { "Cathedral", "Halls", "cathedral;huge;dark;choir",
          "Long, dark and enormous. Choirs, drones and anything that wants to "
          "sound sacred.",
          { { id::space, atModel (11) }, { id::decayTime, 6.50f },
            { id::predelay, 45.0f }, { id::mix, 38.0f },
            { id::brightness, 32.0f }, { id::distance, 72.0f },
            { id::width, 135.0f } } },

        // --- Utility --------------------------------------------------------

        { "Ambience Dust", "Utility", "ambience;short;subtle;glue",
          "Barely-there ambience: the room tone you notice only when you "
          "bypass it. Good glue on a bus.",
          { { id::space, atModel (0) }, { id::decayTime, 0.25f },
            { id::mix, 14.0f }, { id::brightness, 52.0f },
            // 60 rather than v0.8's 30: at a quarter second the early cluster
            // is most of the room, and up close it, not the tail, set the decay
            // the preset delivered (0.21 s for the 0.25 it asks; 0.24 from here).
            { id::distance, 60.0f } } },

        { "Studio Room", "Utility", "room;natural;neutral;acoustic",
          "Neutral mid-size studio room — a believable space rather than an "
          "effect. Sits well on acoustic sources.",
          { { id::space, atModel (2) }, { id::decayTime, 0.65f },
            { id::mix, 20.0f }, { id::distance, 40.0f } } },

        { "Mono Ambience", "Utility", "mono;ambience;narrow;dialogue",
          "Collapsed to mono for dialogue and anything that must survive a "
          "fold-down. Short and unobtrusive.",
          { { id::space, atModel (1) }, { id::decayTime, 0.36f },
            { id::mix, 18.0f }, { id::width, 0.0f },
            { id::distance, 35.0f }, { id::character, 34.0f } } },

        // The two v0.2 controls, each with a preset that is about it. The Space
        // values here sit deliberately *between* table entries — the morph
        // means a room no longer has to be one of the fourteen.
        { "Dense Chamber", "Utility", "thickness;dense;smooth;glue;bus",
          "Thickness at the top of its range: a chamber-into-plate morph with "
          "no discrete reflections left in it at all. Smooth glue on a bus.",
          { { id::space, 36.0f }, { id::decayTime, 1.40f },
            { id::predelay, 14.0f }, { id::mix, 26.0f },
            { id::brightness, 46.0f }, { id::distance, 44.0f },
            { id::character, 18.0f }, { id::thickness, 100.0f } } },

        { "Dark Chamber Bus", "Utility", "post eq;dark;bus;glue;vintage",
          "A chamber with the top end taken off the tail rather than out of "
          "the room — the decay stays as long as it was, it just stops being "
          "bright. The old trick of an EQ after the send, built in.",
          { { id::space, atModel (4) }, { id::decayTime, 1.30f },
            { id::predelay, 18.0f }, { id::mix, 28.0f },
            { id::distance, 50.0f },
            { id::postEqOn (5), 1.0f }, { id::postEqType (5), shape (FilterShape::highCut) },
            { id::postEqFreq (5), 4500.0f }, { id::postEqSlope (5), 1.0f },
            { id::postEqOn (2), 1.0f }, { id::postEqFreq (2), 900.0f },
            { id::postEqGain (2), -3.5f }, { id::postEqQ (2), 1.2f } } },

        { "Tight Low End", "Utility", "decay eq;clean;bass;mix;bus",
          "The single most useful thing a Decay Rate EQ does: the low end "
          "rings for a third as long as the rest of the room, so a hall stops "
          "turning a bass part to mud while keeping all of its size.",
          { { id::space, atModel (8) }, { id::decayTime, 2.40f },
            { id::mix, 30.0f }, { id::distance, 55.0f },
            { id::predelay, 20.0f },
            { id::decayEqFreq (0), 180.0f }, { id::decayEqMult (0), 0.35f },
            { id::decayEqFreq (1), 400.0f }, { id::decayEqMult (1), 0.7f } } },

        // The two send presets. Mix 100% is not a louder setting of the same
        // thing — it says the plugin is on an aux and the dry path belongs to
        // the channel it was sent from — so these are the only two entries the
        // harness leaves out of its level comparison.
        { "Send Hall", "Utility", "send;aux;bus;100% wet;hall",
          "For an aux: 100% wet, so the return fader is the mix control and "
          "every channel sent to it lands in the same room. A concert hall at "
          "a length that suits most things at once.",
          { { id::space, atModel (8) }, { id::decayTime, 2.60f },
            { id::predelay, 25.0f }, { id::mix, 100.0f },
            { id::distance, 60.0f }, { id::width, 120.0f } } },

        { "Sidechain Duck Send", "Utility", "ducking;sidechain;send;aux;bus",
          "The same aux, keyed from the sidechain input instead of from its "
          "own. Feed it the dry bus and the return ducks against the whole mix "
          "rather than against whatever happens to be sent to it — which is "
          "what you want the moment more than one channel is using the send.",
          { { id::space, atModel (9) }, { id::decayTime, 3.20f },
            { id::predelay, 20.0f }, { id::mix, 100.0f },
            { id::distance, 60.0f }, { id::width, 130.0f },
            { id::duckOn, 1.0f },
            { id::triggerSource, (float) (int) TriggerSource::sidechain },
            { id::duckThreshold, -24.0f }, { id::duckAmount, 12.0f },
            { id::duckAttack, 8.0f }, { id::duckRelease, 300.0f } } },

        // --- Surround (v0.7) ------------------------------------------------
        //
        // The three presets that are about a room having speakers in it. All
        // three open on a stereo track unchanged: a band scoped to a group the
        // layout has not is exactly inert, and Front in stereo is the whole
        // room, so nothing here misbehaves at two channels — it just stops
        // being the point.

        { "Dialogue Room", "Surround", "dialogue;post;film;speakers;centre",
          "A small neutral room for dialogue, with the surrounds rolled off "
          "above 4 kHz so the rear layer stays a sense of space rather than a "
          "second source of consonants. Mono-safe by construction.",
          { { id::space, atModel (2) }, { id::decayTime, 0.55f },
            { id::predelay, 10.0f }, { id::mix, 16.0f },
            { id::distance, 38.0f }, { id::width, 85.0f },
            { id::postEqOn (0), 1.0f }, { id::postEqType (0), shape (FilterShape::lowCut) },
            { id::postEqFreq (0), 180.0f }, { id::postEqSlope (0), 1.0f },
            { id::postEqOn (5), 1.0f }, { id::postEqType (5), shape (FilterShape::highCut) },
            { id::postEqFreq (5), 4000.0f }, { id::postEqSlope (5), 1.0f },
            { id::postEqChan (5), scope (ChannelMode::surround) } } },

        { "Immersive Bed", "Surround", "immersive;atmos;height;bed;wide",
          "An arena with the height layer lifted: four decibels of air above "
          "8 kHz that only the top speakers get, which is what makes a ceiling "
          "read as a ceiling rather than as more surround. Atmospheres, "
          "crowds, anything that has to be above you.",
          { { id::space, atModel (10) }, { id::decayTime, 4.50f },
            { id::predelay, 35.0f }, { id::mix, 40.0f },
            { id::brightness, 48.0f }, { id::distance, 75.0f },
            { id::width, 150.0f },
            { id::postEqOn (5), 1.0f }, { id::postEqType (5), shape (FilterShape::highShelf) },
            { id::postEqFreq (5), 8000.0f }, { id::postEqGain (5), 4.0f },
            { id::postEqChan (5), scope (ChannelMode::height) } } },

        { "Score Stage", "Surround", "score;orchestral;film;centre;speakers",
          "A scoring stage with the centre speaker's share of the room thinned "
          "below 250 Hz. The centre carries the dialogue in a film mix and the "
          "reverb it needs is a different reverb from the one the orchestra "
          "wants, which is precisely what a per-speaker band is for.",
          { { id::space, atModel (9) }, { id::decayTime, 2.80f },
            { id::predelay, 26.0f }, { id::mix, 34.0f },
            { id::distance, 62.0f }, { id::width, 130.0f },
            { id::postEqOn (0), 1.0f }, { id::postEqType (0), shape (FilterShape::lowShelf) },
            { id::postEqFreq (0), 250.0f }, { id::postEqGain (0), -6.0f },
            { id::postEqChan (0), scope (ChannelMode::centre) } } },

        // --- Special --------------------------------------------------------

        { "Chorus Bloom", "Special", "character;chorus;modulated;pad;wide",
          "Character pushed into its over-modulated extreme on a hall halfway "
          "to an arena. Pads and guitars bloom and drift; not for drums.",
          { { id::space, 74.0f }, { id::decayTime, 5.50f },
            { id::predelay, 24.0f }, { id::mix, 42.0f },
            { id::brightness, 58.0f }, { id::distance, 55.0f },
            { id::character, 92.0f }, { id::thickness, 40.0f },
            { id::width, 150.0f } } },

        { "Long Bloom", "Special", "decay eq;pad;bloom;ambient;low",
          "The Decay Rate EQ the other way up. Everything below 250 Hz rings "
          "four times as long as the rest, so the room blooms underneath the "
          "source instead of on top of it. Pads, drones, film beds.",
          { { id::space, atModel (10) }, { id::decayTime, 5.00f },
            { id::mix, 44.0f }, { id::predelay, 40.0f },
            { id::distance, 70.0f }, { id::width, 145.0f },
            { id::decayEqFreq (0), 250.0f }, { id::decayEqMult (0), 4.0f },
            { id::decayEqFreq (4), 5000.0f }, { id::decayEqMult (4), 0.6f },
            { id::decayEqFreq (5), 9000.0f }, { id::decayEqMult (5), 0.45f } } },

        { "Distant Choir", "Special", "far;dark;pad;atmosphere",
          "Pushed all the way to the back wall with the highs absorbed on the "
          "way — a source that has left the room.",
          { { id::space, atModel (10) }, { id::decayTime, 6.50f },
            { id::predelay, 60.0f }, { id::mix, 46.0f },
            { id::brightness, 22.0f }, { id::distance, 100.0f },
            { id::width, 150.0f } } },

        // Freeze ships *off*, and the description says why in the imperative.
        // See the note at the top of this file: a preset that arrives frozen
        // arrives holding nothing.
        { "Freeze Pad", "Special", "freeze;infinite;drone;sound design",
          "The setting to freeze from. Play a chord, engage Freeze, and "
          "whatever is in the tank at that moment sustains forever — then mute "
          "the source and play over the top of it.",
          { { id::space, atModel (9) }, { id::decayTime, 6.00f },
            { id::predelay, 12.0f }, { id::mix, 52.0f },
            { id::brightness, 50.0f }, { id::distance, 60.0f },
            { id::width, 140.0f } } },

        { "Endless", "Special", "huge;long;pad;ambient;drone",
          "The largest room at the longest decay the control reaches — an "
          "ambient wash that takes half a minute to leave.",
          { { id::space, atModel (13) }, { id::decayTime, 30.0f },
            { id::predelay, 80.0f }, { id::mix, 48.0f },
            { id::brightness, 40.0f }, { id::distance, 80.0f },
            { id::width, 160.0f } } },
    };

    return c;
}

} // namespace hrzn
