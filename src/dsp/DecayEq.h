#pragma once

#include "FilterCoeffs.h"
#include "DelayLine.h"
#include "SpeakerLayout.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <array>
#include <cmath>

namespace hrzn
{

// The six-band Decay Rate EQ (feature 19).
//
// This is not an EQ on the reverb's output — it changes how long each part of
// the spectrum *rings*, which means it has to live inside the FDN's feedback
// path where every trip round the loop compounds it. A band set to 2.0x makes
// the decay at that frequency twice as long as the room model's own; 0.5x
// halves it. Post EQ (`PostEq.h`) is the other thing entirely: it shapes the
// tone of what comes out without touching the decay at all.
//
// ---------------------------------------------------------------------------
// Lifting inside a feedback loop, safely
// ---------------------------------------------------------------------------
// An FDN with an orthogonal mixing matrix is stable exactly when every line's
// round-trip gain is below unity at every frequency. A band told to ring
// *longer* has to reduce that line's total attenuation at those frequencies —
// there is no other way to make something decay more slowly — so the section
// has to lift, and the only question is how much headroom there is to lift
// into.
//
// The headroom is exactly known, per line and per frequency: it is the
// attenuation already there, which is the line's feedback gain plus whatever
// the damping filter is taking. So the design leaves the feedback gain alone —
// it stays bit-for-bit the value v0.2 computed from the decay time — and every
// band's multiplier becomes a target for the cascade:
//
//     needed attenuation at fc = baseline attenuation * (1/multiplier - 1)
//
// which is a cut below 1.0x and a lift above it. The cascade is then checked
// against the real condition,
//
//     feedback + damping + cascade must stay on the attenuating side of a
//     240-second decay, at every frequency
//
// evaluated on a log grid plus each band's own centre, and scaled back towards
// neutral if it is not met. That check is what makes the section safe, and
// because the required lift for a multiplier M is always `baseline*(1-1/M)`
// against a headroom of `baseline`, it is met with room to spare for any
// single band at any setting — the scaling only ever bites when several
// adjacent bands are all lifting hard at once, which is the one case where
// the loop really is running out of room.
//
// Two earlier designs are worth recording because both looked safer and were
// worse. Forbidding lifts entirely and lowering the whole line's feedback gain
// instead is stable by construction, but one band asking for more than the
// loop can give (a high shelf fighting the model's own air absorption) drags
// the feedback gain to its floor and forces every *other* band to be cut to
// compensate — so a request that could not be met wrecked the five bands that
// could. And bounding the cascade by the sum of its sections' maxima, rather
// than evaluating it, is rigorous and about three times too pessimistic: it
// discards the damping filter's headroom and counts lifts an octave apart as
// though they stacked, which cost the cut direction most of its range.
//
// One thing both of those got right and is kept: the section maxima have to be
// *measured*, never inferred from the nominal gains. An RBJ shelf with a Q
// above 0.71 overshoots its own asymptote, so "every gain is <= 0 dB" is not
// sufficient for stability — which is how the first version of this let a
// random-settings sweep run away.
//
// ---------------------------------------------------------------------------
// Hitting the numbers: band interaction
// ---------------------------------------------------------------------------
// Six sections spread over the audible range overlap heavily — a bell at
// 800 Hz with Q 0.9 still puts a tenth of its gain at 300 Hz — so designing
// each band to its own target independently misses by 20-30% wherever two
// neighbouring bands are doing different things. Worse, when several bands
// ask for a longer decay the errors accumulate in the direction that matters.
//
// The fix is the standard graphic-EQ one: build the interaction matrix W,
// where W(i,j) is the attenuation band j delivers at band i's control
// frequency per dB of its own gain, and solve W y = target. It is a 6x6 solve
// on a matrix that only changes when a band's frequency, Q or shape does, so
// it costs nothing worth measuring.
//
// ---------------------------------------------------------------------------
// Refresh strategy
// ---------------------------------------------------------------------------
// The per-line design depends on the line's delay length, which glides
// continuously through the Space morph, so it cannot be computed once. It also
// cannot be computed for all sixteen lines every control tick — that is 96
// biquad designs and a stability sweep each, at 1.5 kHz. Instead two lines are
// redesigned per tick, round-robin, so every line is current within eight
// ticks (about 5 ms at 48 kHz).
//
// The gap that leaves is covered from the other end: the *feedback* gains are
// refreshed for every line on every tick and carry a floor equal to the lift
// the line's cascade is actually applying, so a design a few milliseconds out
// of date can make the decay briefly wrong but can never make the loop unsafe.
// Everything the design produces is slew limited in dB as well, which is what
// keeps a coefficient set from stepping between refreshes.
//
// When every band sits at 1.0x the whole section designs out to nothing: no
// stages, no per-sample cost, and a feedback gain identical to the one the
// engine used before this existed.
class DecayEq
{
public:
    static constexpr int numBands = 6;
    static constexpr int maxLines = 16;

    // Delivered decay is reported on a fixed half-octave grid, 31.25 Hz to
    // 16 kHz. The 1 kHz point is the figure the front panel shows; the whole
    // curve is what v0.4's decay-time overlay draws, and half-octave steps are
    // the coarsest spacing on which a band with a Q around 1 still reads as a
    // bump rather than as a kink.
    static constexpr int numReportPoints = 19;
    static constexpr int midReportPoint = 10;   // 31.25 * 2^5 = 1000 Hz

    // The decay ceiling. Nothing here can be unstable, so this is purely a
    // "long enough to be indistinguishable from Freeze" number.
    static constexpr float maxDecaySeconds = 240.0f;
    static constexpr float minDecaySeconds = 0.02f;

    struct Band
    {
        bool on = true;
        FilterShape shape = FilterShape::bell;
        float freq = 1000.0f;
        float mult = 1.0f;   // decay multiplier at this band's control frequency
        float q = 0.9f;
    };

    static float reportFrequency (int point) noexcept
    {
        return 31.25f * std::pow (2.0f,
                                  0.5f * (float) juce::jlimit (0, numReportPoints - 1, point));
    }

    void prepare (double newSampleRate, int lineCount)
    {
        sampleRate = newSampleRate;
        numLines = juce::jlimit (1, maxLines, lineCount);

        for (int k = 0; k < numReportPoints; ++k)
        {
            const double w = juce::MathConstants<double>::twoPi
                             * juce::jmin ((double) reportFrequency (k), sampleRate * 0.49)
                             / sampleRate;
            reportPoint[(size_t) k] = { std::cos (w), std::sin (w),
                                        std::cos (2.0 * w), std::sin (2.0 * w) };
        }

        for (int b = 0; b < numBands; ++b)
        {
            smoothed[(size_t) b] = target[(size_t) b];
            smoothing[(size_t) b].snap (target[(size_t) b]);
        }

        matrixDirty = dampDirty = true;
        reset();
    }

    void reset() noexcept
    {
        for (auto& c : cascade)
            c.clear();
        for (int l = 0; l < maxLines; ++l)
        {
            attenDb[(size_t) l] = 0.0f;
            boostHeadroom[(size_t) l] = 0.0f;
            flatAttenDb[(size_t) l] = 0.0f;
            for (int k = 0; k < numReportPoints; ++k)
                lineDecay[(size_t) l][(size_t) k] = lineNeutral[(size_t) l][(size_t) k] = 1.0f;
        }
        lastGain.fill (0.0f);
        nextLine = 0;
        designedAll = false;
    }

    // Control-rate target set. Smoothing happens in update(), so a knob move
    // never steps a coefficient.
    void setBands (const std::array<Band, numBands>& bands) noexcept
    {
        target = bands;
    }


    // A preset load, an A/B switch or an undo is a discontinuity by
    // definition, and the engine relocates rather than glides for it. The EQ
    // arrives with it rather than sliding in behind it.
    void snapToTargets() noexcept
    {
        smoothed = target;
        for (int b = 0; b < numBands; ++b)
            smoothing[(size_t) b].snap (target[(size_t) b]);

        matrixDirty = dampDirty = true;
        designedAll = false;
    }

    // True when the section is doing something — i.e. when the engine has to
    // run the cascades at all.
    bool isActive() const noexcept { return active; }

    // Called once per control block. `delaySamples` and `rt60` come from the
    // morphing geometry; the damping coefficients are the model's own air
    // absorption as voiced by Brightness, which is the baseline the band
    // multipliers are defined against.
    // The shelf gains arrive per line from v0.7.1. They have to: the damping
    // filter takes a fixed loss per pass, so once the lines span forty to one
    // the *baseline* every band multiplier is stated against is forty times
    // larger on the shortest line than on the longest. Designing all sixteen
    // against a single figure would put the delivered decay ratios wrong in
    // opposite directions at the two ends of the set.
    void update (const float* delaySamples, float rt60,
                 float hfCoeff, const float* hfGains,
                 float lfCoeff, const float* lfGains) noexcept
    {
        smoothTargets();

        bool gainsMoved = ! juce::approximatelyEqual (hfCoeff, lastLp)
                          || ! juce::approximatelyEqual (lfCoeff, lastHp);

        for (int l = 0; l < numLines && ! gainsMoved; ++l)
            gainsMoved = ! juce::approximatelyEqual (hfGains[(size_t) l], lastHfGain[(size_t) l])
                         || ! juce::approximatelyEqual (lfGains[(size_t) l], lastLfGain[(size_t) l]);

        if (gainsMoved)
        {
            lastLp = hfCoeff;
            lastHp = lfCoeff;
            for (int l = 0; l < numLines; ++l)
            {
                lastHfGain[(size_t) l] = hfGains[(size_t) l];
                lastLfGain[(size_t) l] = lfGains[(size_t) l];
            }
            dampDirty = true;
        }

        // Order matters: the matrix rebuild is what places the control
        // frequencies, and the damping attenuation is sampled at them.
        if (matrixDirty)
        {
            buildInteractionMatrix();
            matrixDirty = false;
        }

        if (dampDirty)
        {
            computeDampingAttenuation();
            dampDirty = false;
        }

        currentRt60 = juce::jlimit (minDecaySeconds, 3600.0f, rt60);

        // Feedback attenuation is refreshed for *every* line on every tick,
        // not only for the line being redesigned, and it carries a floor.
        //
        // The floor is the important half. A cascade that lifts needs the
        // feedback gain to have room for it, and how much room there is
        // depends on the decay time and the line's length — both of which move
        // continuously while a design is up to eleven milliseconds old. Left
        // alone, automating Decay Rate upwards under a lifted band lets the
        // round-trip gain pass unity, and an FDN that passes unity does not
        // come back. So each line remembers how much lift its cascade actually
        // applies, and the feedback attenuation is never allowed below it.
        //
        // The cost of the floor binding is that the decay is briefly shorter
        // than asked for, for as long as it takes the design to catch up.
        // What the shelves still cost at the frequency Decay Time is stated at.
        //
        // Both damping shelves are unity in their passbands, but a one-pole
        // shelf's skirt is gentle, so a low shelf cornered at 320 Hz — which a
        // small room wants — is not quite finished by 1 kHz. Left uncorrected
        // that is a decay-time error of up to 20% on the smallest rooms, which
        // an absolute control in seconds cannot have. Subtracting it here makes
        // the *total* attenuation at 1 kHz come to exactly the figure the decay
        // time asks for, on every room.
        //
        // This is safe in a way that the same correction against a broadband
        // lowpass would not have been: the shelves only ever attenuate, so
        // taking their 1 kHz cost out of the feedback gain still leaves every
        // other frequency on the attenuating side. Doing it with v0.4's
        // lowpass would have handed the bottom octave a decay measured in
        // minutes.
        for (int l = 0; l < numLines; ++l)
        {
            const auto li = (size_t) l;
            const double d = juce::jmax (1.0, (double) delaySamples[li]);
            const double floorDb = 60.0 * d / (sampleRate * (double) maxDecaySeconds);
            const double midDampDb = (double) reportDampDb[li][(size_t) midReportPoint];
            const double flat = juce::jmax (
                floorDb, 60.0 * d / (sampleRate * (double) currentRt60) - midDampDb);

            lineDelay[li] = (float) d;
            attenDb[li] = (float) juce::jmax (flat, (double) boostHeadroom[li] + floorDb);

            // The same figure with the section taken out of the picture: the
            // per-pass attenuation this line would carry from its feedback gain
            // alone. It is what every band's multiplier is stated against, so
            // it — not the delivered curve — is where v0.4's display puts the
            // band handles (see reportDecay).
            flatAttenDb[li] = (float) juce::jmax (flat, floorDb);
        }

        // A fresh design (or one that just became active) refreshes every line
        // at once: round-robin is for keeping up with a glide, not for
        // arriving somewhere.
        const int refresh = designedAll ? linesPerTick : numLines;
        for (int i = 0; i < refresh; ++i)
        {
            designLine (nextLine);
            nextLine = (nextLine + 1) % numLines;
        }
        designedAll = true;
    }

    // The per-pass feedback attenuation this line should use, in dB (positive).
    // The engine turns it into the scalar feedback gain. It is deliberately the
    // value the line's *own* cascade was designed against rather than the
    // freshest one, so a line mid-refresh is never louder than its filter
    // expects.
    float feedbackAttenDb (int line) const noexcept
    {
        return attenDb[(size_t) juce::jlimit (0, maxLines - 1, line)];
    }

    double process (int line, double x) noexcept
    {
        return cascade[(size_t) line].process (x);
    }

    // Delivered decay at report point k, averaged over the lines. This
    // replaces v0.2's single-filter estimate: with six sections in the loop
    // there is no one cutoff to derive a number from, so the number comes from
    // the design itself — the total per-pass attenuation each line actually
    // has at that frequency.
    float decaySecondsAt (int point) const noexcept
    {
        const int k = juce::jlimit (0, numReportPoints - 1, point);
        double acc = 0.0;
        for (int l = 0; l < numLines; ++l)
            acc += lineDecay[(size_t) l][(size_t) k];
        return (float) (acc / (double) numLines);
    }

    float deliveredMidDecay() const noexcept { return decaySecondsAt (midReportPoint); }

    // What the room would deliver at report point k with this section neutral:
    // the feedback gain and the damping filter, and nothing else.
    //
    // This is the curve a band multiplier is defined against — `designLine`
    // builds each band's target as `baseline * (1/mult - 1)` against exactly
    // this attenuation — so it is the honest place to put a handle labelled
    // "2.00x". Drawing the handles on the *delivered* curve instead would be
    // circular: every band would sit on the curve by construction and a
    // request the loop could not meet would look like it had been met.
    float neutralDecaySecondsAt (int point) const noexcept
    {
        const int k = juce::jlimit (0, numReportPoints - 1, point);
        double acc = 0.0;
        for (int l = 0; l < numLines; ++l)
            acc += lineNeutral[(size_t) l][(size_t) k];
        return (float) (acc / (double) numLines);
    }

    // The frequency a band's multiplier is stated at. For a bell that is the
    // band frequency; a shelf's gain is only fully developed away from its
    // corner, so its control point sits a log-and-a-half out into the shelf.
    static float controlFrequencyFor (const Band& b, double sampleRate) noexcept
    {
        // 1.5x rather than something further out into the shelf: a high shelf
        // parked at 10 kHz would otherwise state its multiplier at 30 kHz,
        // where nobody can hear it and the harness cannot measure it. At 1.5x
        // the shelf has developed most of its gain, and what it overshoots
        // above that lands in the top octave where a reverb has nothing left.
        constexpr float shelfRatio = 1.5f;
        float f = b.freq;
        if (b.shape == FilterShape::lowShelf)  f /= shelfRatio;
        if (b.shape == FilterShape::highShelf) f *= shelfRatio;
        return juce::jlimit (20.0f, juce::jmin (16000.0f, (float) (sampleRate * 0.45)), f);
    }

    float controlFrequency (int band) const noexcept
    {
        return controlFreq[(size_t) juce::jlimit (0, numBands - 1, band)];
    }


    bool allFinite() const noexcept
    {
        for (int l = 0; l < numLines; ++l)
            if (! cascade[(size_t) l].isFinite())
                return false;
        return true;
    }

private:
    // Per-sample smoothing lives in the engine; here a one-pole at control
    // rate is enough, because what these values feed is a coefficient design
    // that is itself refreshed gradually. The snap-when-close keeps a parked
    // EQ from redesigning forever on the last ulp.
    // Returns true when the value moved, which is the signal to redesign.
    //
    // The underflow check is not defensive tidying, it is the whole reason
    // this is a function. An exponential glide's increment shrinks as it
    // arrives, and at a float magnitude around seven (which is what log(1100)
    // is) the increment drops below the ULP while the value is still further
    // than `epsilon` from its goal. It then stops moving and never compares
    // equal, so every tick reports a change forever. That had two of the six
    // bands rebuilding the interaction matrix, the stability grid and six
    // biquad designs fifteen hundred times a second with every control parked
    // — half again the CPU of the reverb itself, spent arriving somewhere it
    // had already got to. Snapping on a step that cannot land is the fix, and
    // it holds at any magnitude rather than at the ones a tuned epsilon
    // happens to cover.
    static bool approach (float& v, float goal, float rate, float epsilon,
                          float maxStep = 1.0e9f) noexcept
    {
        const float delta = goal - v;

        if (std::abs (delta) <= epsilon)
        {
            if (juce::approximatelyEqual (v, goal))
                return false;
            v = goal;
            return true;
        }

        const float next = v + juce::jlimit (-maxStep, maxStep, delta * rate);
        v = juce::approximatelyEqual (next, v) ? goal : next;
        return true;
    }

    void smoothTargets() noexcept
    {
        // ~90 ms at a 1.5 kHz control rate — slower than the engine's other
        // smoothing, deliberately. A band's multiplier does not move a gain,
        // it moves a filter *design*, and a design is only refreshed once
        // every eight ticks; the glide has to be gentle enough that eight
        // ticks of it is a small change to a coefficient set rather than a
        // step in one. A high shelf sitting where the model's own absorption
        // is already several dB per pass is the worst case — its target swings
        // by tens of dB across the multiplier's range — and it is the control
        // that sets this number.
        constexpr float rate = 0.035f;
        bool anyActive = false;

        for (int b = 0; b < numBands; ++b)
        {
            const auto& t = target[(size_t) b];
            auto& s = smoothed[(size_t) b];

            if (s.on != t.on || s.shape != t.shape)
            {
                s.on = t.on;
                s.shape = t.shape;
                matrixDirty = true;
            }

            // Frequency, Q and multiplier are all heard as ratios, so they
            // glide geometrically — a sweep from 100 Hz to 10 kHz should spend
            // as long in each octave.
            //
            // Frequency and Q carry a slew cap as well as a time constant, for
            // the same reason the designed gains do: an exponential glide
            // covers most of its distance in its first few ticks, and the
            // design is only refreshed every eighth tick, so the *first* step
            // of a full-range jump is the one that would be heard. The caps
            // work out at about a quarter of an octave per refresh.
            //
            // The log domain is where the smoothed value *lives*, rather than
            // somewhere it visits: the obvious version takes the log of the
            // stored frequency, glides it, and exponentiates it back, and that
            // round trip is not the identity. A settled band then drifts by an
            // ulp every tick, never compares equal to its target, and marks
            // the design dirty forever — which had a plugin with both sections
            // switched off rebuilding a 6x6 interaction matrix, a 48-point
            // frequency grid and six biquad designs fifteen hundred times a
            // second. It cost more than half of what the reverb itself did.
            auto& sm = smoothing[(size_t) b];
            sm.retarget (t);

            if (approach (sm.logFreq, sm.targetLogFreq, rate, 1.0e-5f, 0.02f))
            {
                s.freq = std::exp (sm.logFreq);
                matrixDirty = true;
            }

            if (approach (sm.logQ, sm.targetLogQ, rate, 1.0e-5f, 0.03f))
            {
                s.q = std::exp (sm.logQ);
                matrixDirty = true;
            }

            if (approach (sm.logMult, sm.targetLogMult, rate, 1.0e-6f))
                s.mult = std::exp (sm.logMult);

            if (s.on && std::abs (s.mult - 1.0f) > 1.0e-4f)
                anyActive = true;
        }

        active = anyActive;
    }

    void computeDampingAttenuation() noexcept
    {
        // Sixteen lines times seventy-odd frequencies of closed-form magnitude,
        // and only when a damping coefficient has actually moved. It is the one
        // place the per-line shelves cost anything, and it is worth checking
        // against the CPU audit whenever the grid grows.
        auto attenAt = [this] (int line, float hz)
        {
            const float cosW = (float) std::cos (juce::MathConstants<double>::twoPi
                                                 * (double) hz / sampleRate);
            const float mag = DampingFilter::magnitudeAt (lastLp,
                                                          lastHfGain[(size_t) line],
                                                          lastHp,
                                                          lastLfGain[(size_t) line], cosW);
            return -20.0f * std::log10 (juce::jmax (1.0e-6f, mag));
        };

        for (int l = 0; l < numLines; ++l)
        {
            const auto li = (size_t) l;

            for (int b = 0; b < numBands; ++b)
                bandDampDb[li][(size_t) b] = attenAt (l, controlFreq[(size_t) b]);

            for (int k = 0; k < numReportPoints; ++k)
                reportDampDb[li][(size_t) k] = attenAt (l, reportFrequency (k));

            for (int k = 0; k < gridCount; ++k)
                gridDampDb[li][(size_t) k] = attenAt (l, gridFreq[(size_t) k]);
        }
    }

    // W(i,j): the attenuation, in dB, that band j delivers at band i's control
    // frequency, per dB of band j's own gain. Probed at -6 dB rather than -1
    // because the shelves' response is very slightly gain-dependent and the
    // interesting range is a few dB.
    void buildInteractionMatrix() noexcept
    {
        constexpr double probeDb = -6.0;

        numActive = 0;
        for (int b = 0; b < numBands; ++b)
        {
            controlFreq[(size_t) b] = controlFrequencyFor (smoothed[(size_t) b], sampleRate);
            if (smoothed[(size_t) b].on)
                activeIndex[(size_t) numActive++] = b;
        }

        for (int j = 0; j < numActive; ++j)
        {
            const auto& band = smoothed[(size_t) activeIndex[(size_t) j]];
            const auto probe = computeBandCoeffs (band.shape, band.freq, probeDb,
                                                  band.q, 0, sampleRate);
            for (int i = 0; i < numActive; ++i)
                interaction[(size_t) i][(size_t) j] =
                    bandMagnitudeDb (probe, controlFreq[(size_t) activeIndex[(size_t) i]],
                                     sampleRate) / probeDb;
        }

        buildGrid();
        dampDirty = true;   // the control and grid frequencies may have moved
    }

    // Gaussian elimination with partial pivoting. n <= 6, so this is a
    // handful of flops and does not need to be clever.
    static bool solveSquare (double a[numBands][numBands], const double* rhs,
                             double* out, int n) noexcept
    {
        double m[numBands][numBands + 1];
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
                m[i][j] = a[i][j];
            m[i][n] = rhs[i];
        }

        for (int col = 0; col < n; ++col)
        {
            int pivot = col;
            for (int r = col + 1; r < n; ++r)
                if (std::abs (m[r][col]) > std::abs (m[pivot][col]))
                    pivot = r;

            if (std::abs (m[pivot][col]) < 1.0e-9)
                return false;

            if (pivot != col)
                for (int j = col; j <= n; ++j)
                    std::swap (m[col][j], m[pivot][j]);

            for (int r = col + 1; r < n; ++r)
            {
                const double f = m[r][col] / m[col][col];
                for (int j = col; j <= n; ++j)
                    m[r][j] -= f * m[col][j];
            }
        }

        for (int i = n - 1; i >= 0; --i)
        {
            double acc = m[i][n];
            for (int j = i + 1; j < n; ++j)
                acc -= m[i][j] * out[j];
            out[i] = acc / m[i][i];
        }
        return true;
    }

    // A band's glide state, kept in the log domain so that a settled band is
    // exactly settled. `retarget` recomputes the target logs only when the
    // parameter behind them actually moved.
    struct BandSmoothing
    {
        float logFreq = 0.0f, logQ = 0.0f, logMult = 0.0f;
        float targetLogFreq = 0.0f, targetLogQ = 0.0f, targetLogMult = 0.0f;
        float seenFreq = -1.0f, seenQ = -1.0f, seenMult = -1.0f;

        void retarget (const Band& t) noexcept
        {
            if (! juce::approximatelyEqual (seenFreq, t.freq))
            {
                seenFreq = t.freq;
                targetLogFreq = std::log (juce::jmax (1.0f, t.freq));
            }
            if (! juce::approximatelyEqual (seenQ, t.q))
            {
                seenQ = t.q;
                targetLogQ = std::log (juce::jmax (0.01f, t.q));
            }
            if (! juce::approximatelyEqual (seenMult, t.mult))
            {
                seenMult = t.mult;
                targetLogMult = std::log (juce::jmax (1.0e-3f, t.mult));
            }
        }

        void snap (const Band& t) noexcept
        {
            retarget (t);
            logFreq = targetLogFreq;
            logQ = targetLogQ;
            logMult = targetLogMult;
        }
    };

    // One frequency on the stability grid, with its trigonometry precomputed.
    // Evaluating a section here is a dozen multiply-adds — no transcendentals
    // in the inner loop, which is what makes checking the whole cascade at
    // fifty frequencies, per line, at control rate, affordable.
    struct GridPoint { double cw = 1.0, sw = 0.0, c2w = 1.0, s2w = 0.0; };

    static double stageDbAt (const BiquadCoeffs& st, const GridPoint& g) noexcept
    {
        const double nr = st.b0 + st.b1 * g.cw + st.b2 * g.c2w;
        const double ni = -(st.b1 * g.sw + st.b2 * g.s2w);
        const double dr = 1.0 + st.a1 * g.cw + st.a2 * g.c2w;
        const double di = -(st.a1 * g.sw + st.a2 * g.s2w);
        return 10.0 * std::log10 (juce::jmax (1.0e-30, nr * nr + ni * ni)
                                  / juce::jmax (1.0e-30, dr * dr + di * di));
    }

    // The frequencies the stability bound is checked at: a log sweep, plus
    // each active band's own frequency, because a bell peaks exactly there and
    // a sweep alone can step over a narrow one.
    void buildGrid() noexcept
    {
        gridCount = 0;

        auto add = [this] (double hz)
        {
            if (gridCount >= maxGridPoints)
                return;
            const double w = juce::MathConstants<double>::twoPi
                             * juce::jlimit (10.0, sampleRate * 0.49, hz) / sampleRate;
            grid[(size_t) gridCount] = { std::cos (w), std::sin (w),
                                         std::cos (2.0 * w), std::sin (2.0 * w) };
            gridFreq[(size_t) gridCount] = (float) hz;
            ++gridCount;
        };

        constexpr int sweep = 40;
        const double lo = 20.0, hi = sampleRate * 0.49;
        for (int i = 0; i < sweep; ++i)
            add (lo * std::pow (hi / lo, (double) i / (double) (sweep - 1)));

        for (int i = 0; i < numActive; ++i)
            add ((double) smoothed[(size_t) activeIndex[(size_t) i]].freq);
    }

    void designLine (int line) noexcept
    {
        const auto li = (size_t) line;
        const double fs = sampleRate;
        const double d = (double) lineDelay[li];

        // Per-pass attenuations, all in dB and all positive. The feedback
        // attenuation was set for this line in update(); with a neutral
        // section it is exactly the decay time's own figure, which is what
        // makes a neutral engine bit-for-bit the v0.2 one.
        const double feedback = (double) attenDb[li];
        const double floorDb = 60.0 * d / (fs * (double) maxDecaySeconds);

        if (! active || numActive == 0)
        {
            cascade[li].setStages (nullptr, 0);
            boostHeadroom[li] = 0.0f;
            reportDecay (line, d, feedback, nullptr);
            return;
        }

        // What the filter has to add at each band's control frequency:
        // negative is a lift, for a band that should ring longer.
        double need[numBands] {}, gains[numBands] {};
        double worstNeed = 0.0;
        for (int i = 0; i < numActive; ++i)
        {
            const int b = activeIndex[(size_t) i];
            const double mult = juce::jmax (1.0e-3, (double) smoothed[(size_t) b].mult);
            const double baseline = feedback + (double) bandDampDb[li][(size_t) b];

            need[(size_t) i] = baseline * (1.0 / mult - 1.0);
            worstNeed = juce::jmax (worstNeed, std::abs (need[(size_t) i]));
        }

        // Solved as regularised least squares, (W'W + lambda I) x = W' need,
        // rather than as a straight inverse.
        //
        // Two bands dragged onto the same frequency make W singular, and the
        // first version of this detected that and fell back to designing each
        // band to its own target. That fallback is a *discontinuity*: a band
        // sweeping past another flips the whole gain vector between two
        // different answers, and back again, which the interpolation
        // regression caught as a step twenty times the family's click bound.
        // A ridge term cannot flip, because there is nothing to flip to — it
        // degrades smoothly towards a damped solution exactly where the
        // inverse would have blown up.
        double ata[numBands][numBands] {}, atb[numBands] {}, solution[numBands] {};
        double trace = 0.0;

        for (int i = 0; i < numActive; ++i)
        {
            for (int j = 0; j < numActive; ++j)
            {
                double acc = 0.0;
                for (int k = 0; k < numActive; ++k)
                    acc += interaction[(size_t) k][(size_t) i]
                           * interaction[(size_t) k][(size_t) j];
                ata[i][j] = acc;
            }
            trace += ata[i][i];

            double acc = 0.0;
            for (int k = 0; k < numActive; ++k)
                acc += interaction[(size_t) k][(size_t) i] * need[(size_t) k];
            atb[i] = acc;
        }

        const double lambda = 1.0e-3 * trace / juce::jmax (1, numActive) + 1.0e-12;
        for (int i = 0; i < numActive; ++i)
            ata[i][i] += lambda;

        if (! solveSquare (ata, atb, solution, numActive))
            for (int i = 0; i < numActive; ++i)
                solution[i] = need[(size_t) i];

        for (int i = 0; i < numActive; ++i)
        {
            const double x = std::isfinite (solution[i]) ? solution[i] : need[(size_t) i];
            gains[(size_t) i] = -juce::jlimit (-(4.0 * worstNeed + 12.0),
                                               4.0 * worstNeed + 12.0, x);
        }

        // Slew limit, in the units that matter.
        //
        // Everything upstream of here is smoothed, but what has to not step is
        // the *coefficient set*, and the map from a band's multiplier to its
        // designed gain is wildly non-uniform: a high shelf sitting where the
        // model's own absorption is already several dB per pass swings tens of
        // dB across the multiplier's travel, while the same move on a mid band
        // is worth a fraction of one. Smoothing the multiplier enough to tame
        // the first makes the second sluggish. Limiting the gain directly is
        // the same bound applied where it belongs, and it leaves an ordinary
        // knob move — a change of a few tenths of a dB per refresh — entirely
        // untouched.
        if (designedAll)
            for (int i = 0; i < numActive; ++i)
            {
                const auto slot = li * (size_t) numBands + (size_t) i;
                gains[(size_t) i] = juce::jlimit ((double) lastGain[slot] - maxGainStepDb,
                                                  (double) lastGain[slot] + maxGainStepDb,
                                                  gains[(size_t) i]);
            }

        // The stability check.
        //
        // The solve will ask a band to *lift* whenever its neighbours are
        // cutting across it — a band told to ring twice as long usually needs
        // no attenuation of its own, but does need to undo the skirts of the
        // ones either side. Forbidding lifts outright (the obvious way to stay
        // safe) costs about a third of the control's range in one direction
        // and most of it in the other: a 2.0x band measures 1.3x because its
        // neighbours eat it, and a 0.5x band cannot cut at all without
        // dragging an octave either side down with it.
        //
        // So lifts are allowed, and the loop's actual round-trip gain is
        // checked instead of bounded. The condition is the exact one —
        //
        //     feedback attenuation + damping attenuation + cascade
        //         must stay on the attenuating side of a 240-second decay,
        //         at every frequency
        //
        // — evaluated on a log grid plus each band's own centre. An earlier
        // version bounded the cascade by the sum of its sections' maxima,
        // which is rigorous and roughly three times too pessimistic: it threw
        // away the damping filter's headroom and counted lifts an octave apart
        // as though they stacked.
        //
        // Note that the section maxima have to be *measured* either way: an
        // RBJ shelf with a Q above 0.71 overshoots its own asymptote, so
        // "every gain is <= 0 dB" is not sufficient for stability. That is how
        // the first version of this let a random-settings sweep run away.
        std::array<BiquadCoeffs, numBands> stages {};
        int numStages = 0;
        double wanted = 0.0;

        for (int attempt = 0; ; ++attempt)
        {
            // One stage per active band, always, in band order — never packed
            // to skip the ones that happen to be sitting at 0 dB. The cascade
            // carries filter *state*, so dropping a section shifts every
            // section after it onto the wrong state, and a gain wobbling
            // across the threshold does that repeatedly. A bell at 0 dB is an
            // exact unity response anyway, so leaving it in costs arithmetic
            // and nothing else.
            numStages = 0;
            for (int i = 0; i < numActive; ++i)
            {
                const auto& band = smoothed[(size_t) activeIndex[(size_t) i]];
                const auto c = computeBandCoeffs (band.shape, band.freq, gains[(size_t) i],
                                                  band.q, 0, sampleRate);
                for (int s = 0; s < c.numStages && numStages < numBands; ++s)
                    stages[(size_t) numStages++] = c.stages[(size_t) s];
            }

            if (numStages == 0)
            {
                wanted = 0.0;
                break;
            }

            // How much feedback attenuation this cascade needs to have under
            // it, taken at the frequency where it is most demanding: the
            // cascade's own gain there, less whatever the damping filter is
            // already giving up.
            wanted = 0.0;
            double atPeak = 0.0;
            for (int k = 0; k < gridCount; ++k)
            {
                double cascadeDb = 0.0;
                for (int s = 0; s < numStages; ++s)
                    cascadeDb += stageDbAt (stages[(size_t) s], grid[(size_t) k]);

                const double demand = cascadeDb - (double) gridDampDb[li][(size_t) k];
                if (demand > wanted)
                {
                    wanted = demand;
                    atPeak = cascadeDb;
                }
            }

            const double excess = wanted - (feedback - floorDb);
            if (excess <= 0.0)
                break;

            if (attempt >= 4)
            {
                // Nothing should ever reach this, but "the reverb ran away" is
                // not a failure mode worth leaving a path to.
                numStages = 0;
                wanted = 0.0;
                break;
            }

            // Scaling the whole gain vector moves the design towards neutral,
            // which is always the safe direction, and the response is close
            // enough to linear in the gains that one pass usually lands it.
            const double alpha = juce::jlimit (0.0, 0.99,
                                               0.9 * (atPeak - excess) / juce::jmax (1.0e-9, atPeak));
            for (int i = 0; i < numActive; ++i)
                gains[(size_t) i] *= alpha;
        }

        boostHeadroom[li] = (float) juce::jmax (0.0, wanted);
        for (int i = 0; i < numActive; ++i)
            lastGain[li * (size_t) numBands + (size_t) i] = (float) gains[(size_t) i];

        cascade[li].setStages (stages.data(), numStages);
        reportDecay (line, d, feedback, numStages > 0 ? &cascade[li] : nullptr);
    }

    void reportDecay (int line, double delaySamples, double feedbackDb,
                      const BiquadCascade<numBands>* casc) noexcept
    {
        const double flat = (double) flatAttenDb[(size_t) line];

        for (int k = 0; k < numReportPoints; ++k)
        {
            const double damp = (double) reportDampDb[(size_t) line][(size_t) k];

            double atten = feedbackDb + damp;
            if (casc != nullptr)
                for (int s = 0; s < casc->numStages; ++s)
                    atten -= stageDbAt (casc->stages[(size_t) s], reportPoint[(size_t) k]);

            const double seconds = 60.0 * delaySamples
                                   / (sampleRate * juce::jmax (1.0e-9, atten));
            lineDecay[(size_t) line][(size_t) k] =
                (float) juce::jlimit (0.0, (double) maxDecaySeconds, seconds);

            // Same arithmetic without the cascade, and against the feedback
            // attenuation the decay time asks for rather than the one the
            // lift floor may have raised it to. With every band at 1.00x the
            // two are the same number, which the harness asserts.
            const double neutral = 60.0 * delaySamples
                                   / (sampleRate * juce::jmax (1.0e-9, flat + damp));
            lineNeutral[(size_t) line][(size_t) k] =
                (float) juce::jlimit (0.0, (double) maxDecaySeconds, neutral);
        }
    }

    double sampleRate = 48000.0;
    int numLines = maxLines;

    std::array<Band, numBands> target {}, smoothed {};
    std::array<BandSmoothing, numBands> smoothing {};
    std::array<int, numBands> activeIndex {};
    int numActive = 0;
    bool active = false;

    std::array<float, numBands> controlFreq {};
    std::array<std::array<float, numBands>, maxLines> bandDampDb {};
    std::array<std::array<float, numReportPoints>, maxLines> reportDampDb {};
    std::array<GridPoint, numReportPoints> reportPoint {};
    std::array<std::array<double, numBands>, numBands> interaction {};

    static constexpr int maxGridPoints = 48;
    std::array<GridPoint, maxGridPoints> grid {};
    std::array<float, maxGridPoints> gridFreq {};
    std::array<std::array<float, maxGridPoints>, maxLines> gridDampDb {};
    int gridCount = 0;

    // Sixteen lines refreshed two at a time: every line is current within
    // eight control ticks, about five milliseconds at 48 kHz, which is short
    // enough that a coefficient set is never more than a hair out of date and
    // long enough that the design cost is an eighth of doing them all.
    static constexpr int linesPerTick = 2;

    // ...which puts a refresh about 5 ms apart, so this is a 70 dB/second
    // ceiling on any one section's gain. Far faster than a hand, fast enough
    // for automation, slow enough that a coefficient set never jumps.
    static constexpr double maxGainStepDb = 0.35;

    std::array<BiquadCascade<numBands>, maxLines> cascade {};
    std::array<float, maxLines> attenDb {}, lineDelay {}, boostHeadroom {}, flatAttenDb {};
    std::array<float, (size_t) maxLines * numBands> lastGain {};
    std::array<std::array<float, numReportPoints>, maxLines> lineDecay {}, lineNeutral {};

    float currentRt60 = 1.0f;
    float lastLp = -1.0f, lastHp = -1.0f;
    std::array<float, maxLines> lastHfGain {}, lastLfGain {};
    bool matrixDirty = true, dampDirty = true, designedAll = false;
    int nextLine = 0;
};

} // namespace hrzn
