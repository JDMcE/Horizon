#pragma once

#include <juce_audio_formats/juce_audio_formats.h>
#include <vector>

namespace hrzn::ir
{

// Reading an impulse response off disk (feature 6), v0.6.
//
// WAV/AIFF/FLAC, mono through multichannel, any sample rate — which is the
// whole of the feature's file-handling promise, and none of it is interesting
// except for the three places it can quietly go wrong:
//
//   1. **Sample rate.** An IR recorded at 96 kHz loaded into a 44.1 kHz
//      session and used unresampled is a room 2.2 times the size, decaying 2.2
//      times as slowly. Everything downstream measures time in samples, so the
//      conversion has to happen here, once, rather than being carried as a
//      caveat.
//   2. **Channel count.** Everything the analysis measures is a property of
//      the room, not of the microphone array, so channels are summed. Summed,
//      not averaged-then-normalised by peak: a true-stereo IR whose channels
//      are decorrelated sums to roughly the same energy per channel, and
//      scaling by 1/n would make a four-channel capture read 6 dB quieter than
//      the same room captured in mono. The analysis is scale-invariant in every
//      measurement except the peak, but the convolution A/B is not.
//   3. **Length.** A file is whatever somebody put in it. A 90-minute WAV
//      dropped on the import button should be refused, not loaded.
struct LoadResult
{
    bool valid = false;
    juce::String error;

    std::vector<float> mono;        // summed, resampled to `sampleRate`
    juce::AudioBuffer<float> audio; // up to two channels, resampled: the A/B source
    double sampleRate = 0.0;

    double sourceSampleRate = 0.0;  // what the file said, for the report
    int sourceChannels = 0;
    double sourceSeconds = 0.0;
};

// Long enough for any real space — the largest cathedral IRs run about 12
// seconds — with room to spare, and short enough that a mistake costs a
// message rather than a minute of resampling.
inline constexpr double maxIrSeconds = 30.0;

inline LoadResult loadImpulseResponse (const juce::File& file, double targetSampleRate)
{
    LoadResult r;
    r.sampleRate = targetSampleRate;

    if (! file.existsAsFile())
    {
        r.error = "File not found.";
        return r;
    }

    juce::AudioFormatManager formats;
    formats.registerBasicFormats();

    std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (file));

    if (reader == nullptr)
    {
        r.error = "Not an audio file this build can read.";
        return r;
    }

    r.sourceSampleRate = reader->sampleRate;
    r.sourceChannels = (int) reader->numChannels;
    r.sourceSeconds = reader->sampleRate > 0.0
                        ? (double) reader->lengthInSamples / reader->sampleRate : 0.0;

    if (reader->lengthInSamples <= 0 || reader->numChannels <= 0 || reader->sampleRate <= 0.0)
    {
        r.error = "The file contains no audio.";
        return r;
    }

    if (r.sourceSeconds > maxIrSeconds)
    {
        r.error = "Longer than " + juce::String ((int) maxIrSeconds)
                    + " seconds — that is not an impulse response.";
        return r;
    }

    const int sourceLength = (int) reader->lengthInSamples;
    const int sourceChannels = (int) reader->numChannels;

    juce::AudioBuffer<float> source (sourceChannels, sourceLength);
    reader->read (&source, 0, sourceLength, 0, true, true);

    // Fold to at most two channels for the A/B convolution, and to one for the
    // analysis. A multichannel capture's extra channels are summed into both
    // sides rather than dropped: they are part of the room's response and
    // discarding them would change the tail the user is comparing against.
    juce::AudioBuffer<float> stereo (2, sourceLength);
    stereo.clear();

    for (int ch = 0; ch < sourceChannels; ++ch)
    {
        const int destination = sourceChannels == 1 ? -1 : (ch % 2);
        if (destination < 0)
        {
            stereo.addFrom (0, 0, source, ch, 0, sourceLength);
            stereo.addFrom (1, 0, source, ch, 0, sourceLength);
        }
        else
        {
            stereo.addFrom (destination, 0, source, ch, 0, sourceLength);
        }
    }

    // Resample if the file and the session disagree. Lagrange is more than
    // enough here: the thing being resampled is a decaying noise-like tail, and
    // the analysis that follows measures energy over octave-wide bands.
    const double ratio = reader->sampleRate / targetSampleRate;

    if (std::abs (ratio - 1.0) > 1.0e-9)
    {
        const int resampledLength = juce::jmax (1, (int) std::ceil (sourceLength / ratio));
        juce::AudioBuffer<float> resampled (2, resampledLength);
        resampled.clear();

        for (int ch = 0; ch < 2; ++ch)
        {
            juce::LagrangeInterpolator interpolator;
            interpolator.reset();
            interpolator.process (ratio, stereo.getReadPointer (ch),
                                  resampled.getWritePointer (ch), resampledLength,
                                  sourceLength, 0);
        }

        r.audio = std::move (resampled);
    }
    else
    {
        r.audio = std::move (stereo);
    }

    const int length = r.audio.getNumSamples();
    r.mono.resize ((size_t) length);
    for (int i = 0; i < length; ++i)
        r.mono[(size_t) i] = 0.5f * (r.audio.getSample (0, i) + r.audio.getSample (1, i));

    // Whatever came back has to be finite before anything downstream trusts it.
    for (auto& v : r.mono)
        if (! std::isfinite (v))
            v = 0.0f;

    float peak = 0.0f;
    for (float v : r.mono)
        peak = juce::jmax (peak, std::abs (v));

    if (peak <= 1.0e-7f)
    {
        r.error = "The file is silent.";
        return r;
    }

    r.valid = true;
    return r;
}

} // namespace hrzn::ir
