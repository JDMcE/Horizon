#pragma once

#include <juce_core/juce_core.h>

namespace hrzn::help
{

// One string per control, used three ways (v0.8): the hover tooltip, the
// help-hint bar along the bottom of the window, and the description a screen
// reader announces. One source, so the three can never disagree — which is the
// whole reason this file exists rather than the strings living at each control.
//
// House style, followed rather than described: name the control, say what it
// does in one sentence, then say the thing that is not obvious — usually how it
// interacts with another control. Two sentences, occasionally three.

// --- The main controls ------------------------------------------------------

inline constexpr const char* space =
    "Space: steplessly selects the room, from a tight ambience to a cathedral — "
    "size, density, early reflection pattern and absorption together. Decay Time "
    "is separate and stays where you put it. Shift-drag for fine adjustment; "
    "right-click to jump straight to a named room.";
inline constexpr const char* decayTime =
    "Decay Time: how long the reverb rings, 100 ms to 30 s, independent of Space. "
    "Changing the room changes its size and character, not its length. Right-click "
    "for the room's own natural decay.";
inline constexpr const char* predelay =
    "Predelay: delay before the reverb starts, 0-500 ms. Keeps the source in front "
    "of its own tail — the single most useful control for keeping a vocal intelligible "
    "in a big room.";
inline constexpr const char* mix =
    "Mix: dry/wet blend. On an aux send set it to 100% and let the return fader be "
    "the control.";
inline constexpr const char* width =
    "Width: stereo width of the reverb — 0% mono, 100% the room's own image, 200% "
    "wide. The mono fold-down is identical at every setting, so widening can never "
    "cost you the mix.";
inline constexpr const char* brightness =
    "Brightness: overall tone and how fast high frequencies decay. 50% is the room's "
    "own voicing; below darkens and shortens the HF tail, above opens it up.";
inline constexpr const char* distance =
    "Distance: how far the source is from you inside the room. Moves the early/late "
    "balance, the high-frequency absorption and the level together, the way stepping "
    "back in a real room does.";
inline constexpr const char* character =
    "Character: from a clean transparent decay at 0%, through the room's own voicing "
    "at 50%, to a lively reverb with pronounced reflections and finally an "
    "over-modulated chorus at 100%. Not for drums above about 70%.";
inline constexpr const char* thickness =
    "Thickness: how densely the late field builds up behind the onset. 50% is the "
    "room's own density; above it packs the reflections together, below it opens "
    "them out.";
inline constexpr const char* freeze =
    "Freeze: seals the reverb tank. The current tail sustains indefinitely and no "
    "new input enters, so engage it while something is ringing — a room frozen "
    "before anything was played into it has nothing to hold.";
inline constexpr const char* bypass =
    "Bypass: click-free bypass, crossfaded and latency-aligned.";
inline constexpr const char* lockMix =
    "Lock Mix: presets and A/B recalls leave the Mix control where you put it, and "
    "the rest of the preset still loads. Saved with the session, never with a preset.";

// --- Predelay sync (v0.5) ---------------------------------------------------

inline constexpr const char* predelaySync =
    "Sync: locks Predelay to the host tempo as a note length instead of a time.";
inline constexpr const char* predelayDiv =
    "Predelay division: the note length, with dotted and triplet variants. The "
    "readout below is the time it actually comes to at the current tempo — long "
    "divisions clamp at the 500 ms the delay line reaches.";

// --- Ducking and the Auto Gate (v0.5) ---------------------------------------

inline constexpr const char* duckOn =
    "Ducking: pulls the reverb down while the trigger is playing and lets it bloom "
    "in the gaps. The dry signal is never touched, which is what makes it safe to "
    "leave on.";
inline constexpr const char* gateOn =
    "Auto Gate: cuts the tail off once the trigger stops. The gated-reverb sound, "
    "and the way to keep a long tail out of quiet passages.";
inline constexpr const char* triggerSource =
    "Trigger: what drives Ducking and the Auto Gate — this plugin's own input, or "
    "the external sidechain input. Both stages read the same trigger.";
inline constexpr const char* duckThreshold =
    "Duck Threshold: trigger level at which ducking starts. Full depth is reached "
    "about 12 dB above it.";
inline constexpr const char* duckAmount =
    "Duck Amount: how far the reverb is pulled down at full trigger, in dB.";
inline constexpr const char* duckAttack =
    "Duck Attack: how quickly the duck arrives once the trigger does. Fast values "
    "catch consonants; slow ones let the transient through first.";
inline constexpr const char* duckRelease =
    "Duck Release: how quickly the reverb comes back up once the trigger stops. "
    "This is the control that decides whether ducking sounds like a mix move or "
    "like an effect.";
inline constexpr const char* gateThreshold =
    "Gate Threshold: the trigger level that holds the gate open.";
inline constexpr const char* gateHold =
    "Gate Hold: how long the tail stays open after the trigger drops below the "
    "threshold, before the gate starts closing. This is the length of a gated snare.";
inline constexpr const char* gateRelease =
    "Gate Release: how long the gate takes to close once the hold has run out. "
    "Short is the eighties sound; longer reads as a room that stops politely.";

// --- The display and the two EQ sections (v0.3, v0.4) -----------------------

inline constexpr const char* displayText =
    "The room, drawn: the impulse response with the early reflections and the late "
    "field, the two EQ curves over it, and the analyzer behind. Drag a band handle "
    "to move it, drag vertically for gain or decay, use the wheel for Q, and "
    "double-click a handle to reset it.";
inline constexpr const char* decayEqTab =
    "Decay Rate EQ: how long each part of the spectrum rings, as a multiplier on "
    "the room's own decay. It works inside the feedback path, so 0.35x at 180 Hz "
    "means the low end really does stop sooner — it is not an EQ on the output.";
inline constexpr const char* postEqTab =
    "Post EQ: tone of the wet signal after the room, with the family's full shape "
    "set and a channel scope per band. It changes what the tail sounds like without "
    "changing how long it lasts.";
inline constexpr const char* sectionOn =
    "Switches the whole section in and out, so you can hear what it is doing "
    "without losing where you put it.";
inline constexpr const char* bandOn =
    "Switches this band in and out. Crossfaded over about 20 ms, so it is safe to "
    "automate.";
inline constexpr const char* bandFreq =
    "Band frequency.";
inline constexpr const char* bandLevel =
    "Decay Rate EQ: the decay multiplier at this frequency, 0.2x to 6.0x. Post EQ: "
    "the band's gain in dB.";
inline constexpr const char* bandQ =
    "Band Q: how wide the band reaches either side of its centre.";
inline constexpr const char* bandShape =
    "Band shape: bell, shelves, cuts up to 48 dB/oct, or a notch. The Decay Rate "
    "EQ's shapes are fixed — a low shelf, four bells and a high shelf — because a "
    "decay curve is a broad, smooth thing.";
inline constexpr const char* bandSlope =
    "Cut slope, 6 to 48 dB per octave. Only applies to the two cut shapes.";
inline constexpr const char* bandChannel =
    "Which channels this band works on: the whole image, one side, mid or side, or "
    "one speaker group of a surround layout. A band scoped to speakers the layout "
    "does not have costs nothing and changes nothing, so a 7.1.4 preset opens "
    "unchanged on a stereo track.";

// --- Impulse response import (v0.6) -----------------------------------------

inline constexpr const char* importIr =
    "Import IR: analyse a WAV, AIFF or FLAC impulse response and set the reverb to "
    "match it as closely as it can. The result is one undo away, and the match "
    "figure says how close it got.";
inline constexpr const char* irCompare =
    "Compare: A/B the fitted reverb against convolution with the imported response "
    "itself, level-matched and latency-free, so the difference you hear is the "
    "difference between them. Switching to the convolution takes about a tenth "
    "of a second while it warms up.";

// --- Header, meters, shell --------------------------------------------------

inline constexpr const char* undo =
    "Undo the last change (Ctrl+Z). Each A/B slot keeps its own history.";
inline constexpr const char* redo = "Redo (Ctrl+Shift+Z or Ctrl+Y).";
inline constexpr const char* slotA =
    "A/B comparison, state A. Each slot is a complete independent setting of every "
    "control, with its own undo history.";
inline constexpr const char* slotB =
    "A/B comparison, state B. Switch freely; nothing is lost on either side.";
inline constexpr const char* copyAb =
    "Copy the current state onto the other A/B slot. The copy is undoable in that "
    "slot's history.";
inline constexpr const char* presetName =
    "Preset browser: search, tags, favourites, factory and user presets (P).";
inline constexpr const char* presetPrev = "Previous preset ([).";
inline constexpr const char* presetNext = "Next preset (]).";
inline constexpr const char* compact =
    "Compact (K): hides the display, the EQ sections and the footer, and keeps the "
    "controls you reach for while something is playing.";
inline constexpr const char* fullScreen =
    "Full screen (F): scales the interface to fill the display. Drag any edge for "
    "any size in between.";
inline constexpr const char* helpButton =
    "Help: the manual, keyboard shortcuts (F1), hover hints (H), high contrast and "
    "text size.";
inline constexpr const char* meterStrip =
    "Input and output level, peak-holding. The reverb adds energy, so the output "
    "sitting above the input is the plugin working rather than a fault.";
inline constexpr const char* roomReadout =
    "The room Space is currently sitting on. Between two models it names both and "
    "how far through it is, because a stepless control has no single answer.";
inline constexpr const char* decayReadout =
    "The decay the engine is actually delivering at 1 kHz — after the Decay Rate "
    "EQ, Brightness and the room's own absorption, not the number the knob asked "
    "for.";

// Keyboard shortcuts, as data. The F1 overlay and the manual's shortcut table
// are both generated from this array, so the two cannot drift apart — which
// they will otherwise do on the first shortcut anybody adds.
struct Shortcut { const char* keys; const char* action; };

inline constexpr Shortcut shortcuts[] = {
    { "Ctrl+Z",                "Undo (active A/B slot)" },
    { "Ctrl+Shift+Z / Ctrl+Y", "Redo" },
    { "A / B",                 "Switch to state A / B" },
    { "C",                     "Copy current state to the other slot" },
    { "P",                     "Open / close the preset browser" },
    { "[ / ]",                 "Previous / next preset" },
    { "E",                     "Switch the display between the two EQ sections" },
    { "K",                     "Toggle the Compact layout" },
    { "F",                     "Toggle full screen" },
    { "H",                     "Toggle the help hints bar" },
    { "F1",                    "Keyboard shortcuts overlay" },
    { "Esc",                   "Close overlay / browser / full screen" },
    { "Tab / Shift+Tab",       "Move keyboard focus between controls" },
    { "Arrow keys",            "Adjust the focused control" },
    { "Shift+drag",            "Fine adjustment on any knob" },
    { "Right-click",           "MIDI Learn (and Space's room list)" },
};

// The shortcut table as HTML rows, so the manual's table and the F1 overlay are
// the same list rather than two lists that agree today.
inline juce::String shortcutTableHtml()
{
    juce::String rows ("<table><tr><th>Keys</th><th>Action</th></tr>\n");

    for (const auto& s : shortcuts)
        rows << "<tr><td><kbd>" << juce::String (s.keys).replace (" / ", "</kbd> / <kbd>")
             << "</kbd></td><td>" << s.action << "</td></tr>\n";

    return rows + "</table>\n";
}

// The manual ships inside the plugin binary — there is no installer until v0.9
// — so "Open Manual" materializes it next to the settings file and opens the
// default browser. Purple-accented, the family's per-product convention.
inline juce::String manualHtml()
{
    const char* body = R"HTML(<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Horizon Manual</title>
<style>
 body { font-family: Segoe UI, Helvetica, Arial, sans-serif; background:#15181e; color:#d8dce4;
        max-width: 60em; margin: 2em auto; padding: 0 1.5em; line-height: 1.55; }
 h1, h2, h3 { color: #a88cd8; font-weight: 600; }
 h1 { border-bottom: 1px solid #313845; padding-bottom: .3em; }
 code, kbd { background:#242a34; border-radius:3px; padding:.1em .4em; color:#d8dce4; }
 table { border-collapse: collapse; margin: 1em 0; }
 th, td { border: 1px solid #313845; padding: .35em .7em; text-align: left; }
 th { background:#1d2129; }
 .decay { color:#a88cd8; } .post { color:#e0a44c; } .sc { color:#56b58c; }
</style></head><body>

<h1>Horizon</h1>
<p>Horizon is an algorithmic reverb: fourteen room models on one stepless
<b>Space</b> control, an absolute decay time independent of the room, a six-band
decay-rate EQ inside the feedback path, a six-band post EQ on the wet output,
ducking and an auto gate, impulse-response matching, and channel layouts up to
9.1.6. This manual covers every control; hover anything in the plugin — or press
<kbd>H</kbd> for the hints bar — for the short version in place.</p>

<h2>Quick start</h2>
<ol>
<li>Start from a preset (<kbd>P</kbd> opens the browser) — the library is organised
    by what you are putting the reverb on rather than by what the reverb is.</li>
<li>Set <b>Space</b> for the kind of room and <b>Decay Time</b> for how long it rings.
    They are independent: you can audition the same three-second tail in an ambience
    and in a cathedral.</li>
<li>Set <b>Predelay</b> to keep the source in front of its tail, and <b>Distance</b>
    for how far inside the room you are standing.</li>
<li>Blend with <b>Mix</b> — or set it to 100% and use the plugin on an aux.</li>
<li>If the low end is crowding the mix, pull band 1 of the <span class="decay">Decay
    Rate EQ</span> down to about 0.35x rather than reaching for a high-pass.</li>
</ol>

<h2>The room</h2>
<p><b>Space</b> runs through fourteen models — Ambience, Small Room, Studio, Live
Room, Chamber, Bright Plate, Warm Plate, Small Hall, Concert Hall, Large Hall, Arena,
Cathedral, Vast Space, Boundless — and sits genuinely <i>between</i> two of them
anywhere in its travel, interpolating size, early-reflection pattern, diffusion,
absorption and modal structure. Right-click it to jump to a named room. The readout
under the display names the pair it is between and how far through.</p>
<p><b>Decay Time</b> is an absolute 100 ms to 30 s, and does not move when Space does.
Right-click it for the room's own natural decay — the figure a real hall that size
would ring for.</p>
<p><b>Distance</b> moves you inside the room: early/late balance, high-frequency
absorption and level together. <b>Character</b> runs from a clean transparent decay
(0%) through the room's own voicing (50%) to an over-modulated chorus (100%).
<b>Thickness</b> sets how densely the late field packs in behind the onset, and
<b>Brightness</b> the overall tone and how fast the highs die away.</p>
<p><b>Width</b> goes from mono through the room's own image to twice it. The fold-down
to mono is <i>identical</i> at every setting — Width scales only what differs between
the speakers, never their mean — so widening cannot cost you a mono compatibility
check.</p>
<p><b>Freeze</b> seals the tank: the tail sustains forever and no new input enters.
Engage it while something is ringing, then mute the source and play over the top.</p>

<h2>Predelay, and syncing it</h2>
<p><b>Predelay</b> delays the whole reverb by up to 500 ms. <b>SYNC</b> reinterprets it
as a note length locked to the host tempo, with dotted and triplet divisions; the
readout under the selector shows the time it actually comes to, so a division that has
run into the 500 ms ceiling is visible rather than mysterious.</p>

<h2>The two EQ sections</h2>
<p>They are not two versions of the same thing.</p>
<p>The <span class="decay">Decay Rate EQ</span> lives <i>inside</i> the feedback path
and changes how long each part of the spectrum rings, as a multiplier from 0.2x to
6.0x on the room's own decay. Six bands — a low shelf, four bells and a high shelf.
The most useful single move in the plugin is band 1 at about 180 Hz and 0.35x: the low
end stops ringing while the room keeps all of its size.</p>
<p>The <span class="post">Post EQ</span> sits on the wet output and changes the tone of
the tail without touching its length. Six bands with the full shape set — bells,
shelves, cuts up to 48 dB/oct, notch — and a <b>channel scope</b> per band: the whole
image, one side, mid, side, or one speaker group of a surround layout. Rolling the mid
off below 300 Hz while lifting air on the sides is what makes a reverb open up
<i>around</i> a vocal rather than in front of it.</p>
<p>Both sections have a bypass switch on their band row, and both are genuinely inert
when neutral — they cost no CPU at all until a band asks for something.</p>

<h2>Ducking and the Auto Gate</h2>
<p><b>Ducking</b> pulls the reverb down while the trigger plays and lets it bloom in
the gaps; <b>Auto Gate</b> cuts the tail off once the trigger stops. Both act on the
wet path alone — the dry signal arrives unaltered whatever they do — and both read the
same <span class="sc">Trigger</span>, which is either this plugin's input or the
external sidechain bus. Feed the sidechain from the dry mix bus when several channels
share the send.</p>

<h2>Impulse response matching</h2>
<p><b>IMPORT IR</b> analyses a WAV, AIFF or FLAC impulse response — its decay across
frequency, its early-reflection pattern, its density and its stereo image — and fits
the reverb's own controls to it. The result is an ordinary set of parameter values you
can then edit, and it is one undo away. <b>COMPARE</b> A/Bs the fitted reverb against
convolution with the response itself, level-matched and latency-free. Switching
over to the convolution takes about a tenth of a second: it starts only when you ask
for it, and the crossfade waits until it is running so the switch never clicks.</p>

<h2>Surround and immersive</h2>
<p>Horizon runs mono, stereo, 5.1, 7.1, 7.1.4 and 9.1.6. It is one room with a window
per speaker rather than N reverbs: every speaker reads the same sixteen-line tank
through its own decorrelated row, so they share every mode, every decay time and every
change to the sample. Neighbouring speakers are cross-fed by an amount that falls off
with the angle between them, the way two microphones a few degrees apart in a real
room hear nearly the same field.</p>
<p>The <b>LFE is passed through untouched</b>. It is a band-limited effects channel
rather than a position, so there is no direction for the room to place it at; the
consequence is that at Mix 100% the LFE carries dry signal.</p>

<h2>Presets, A/B and undo</h2>
<p>The browser (<kbd>P</kbd>, or click the preset name) searches names, tags and
categories; star anything to collect it under Favourites. Factory presets are built in;
user presets are plain XML <code>.hrzn</code> files in
<code>Documents/Horizon/Presets</code> — friendly to backups and sync, and portable
with IMPORT/EXPORT.</p>
<p><b>A/B</b> keeps two complete states with independent undo histories
(<kbd>Ctrl+Z</kbd>/<kbd>Ctrl+Shift+Z</kbd>); <b>COPY</b> pushes the current state onto
the other slot. <b>LOCK</b> beside Mix keeps your blend through preset loads and A/B
recalls — it is a session setting and is never written into a preset.</p>

<h2>MIDI Learn</h2>
<p>Right-click any knob or switch &rarr; <i>MIDI Learn</i>, then move a controller. One
CC per parameter and one parameter per CC, so relearning releases whatever held it
before. Mappings persist with the session, and a preset that carries none inherits the
session's — loading a sound never unplugs a controller.</p>

<h2>Keyboard shortcuts</h2>
<!--SHORTCUTS-->

<h2>Accessibility</h2>
<p>Every control is keyboard-reachable (<kbd>Tab</kbd> to focus, arrow keys to adjust)
and carries a name and a description for screen readers. The HELP menu offers a
<b>high-contrast</b> palette and <b>text size</b> up to 130%. The window is freely
resizable by dragging any edge (0.5x&ndash;2x), independent of system DPI scaling.</p>

</body></html>
)HTML";

    return juce::String::fromUTF8 (body).replace ("<!--SHORTCUTS-->", shortcutTableHtml());
}

} // namespace hrzn::help
