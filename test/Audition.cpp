// Listening and measurement tool for the room models.
//
// For every room in the table this renders, at the room's own decay time:
//
//   <dir>/NN_<room>_ir.wav     the stereo impulse response, 100% wet
//   <dir>/NN_<room>_clap.wav   a synthetic hand clap through the reverb at 35% mix
//
// and prints the measurements that tell a plate from a room:
//
//   NED@t    normalized echo density (Abel & Huang) in a 21 ms window centred
//            on t ms. Gaussian noise reads 1.0; a handful of discrete echoes reads
//            well below. A room starts sparse and fills in; a plate starts full.
//   mix ms   when NED first reaches 0.9: the mixing time.
//   C50      clarity, early (0-50 ms) over late energy in dB. Positive means the
//            onset carries the energy, as it does in a real room's IR.
//   ripple   fine-resolution spectral ripple of the late field (dB, std about a
//            third-octave trend). High reads as metallic, discrete modes.
//
// Usage: horizon_audition [output dir]   (default: ./audition)

#include "../src/PluginProcessor.h"
#include <juce_dsp/juce_dsp.h>
#include <cstdio>
#include <vector>
#include <cmath>

namespace
{

constexpr double sr = 48000.0;
constexpr int blockSize = 512;

void setParam (HorizonAudioProcessor& p, const juce::String& id, float plainValue)
{
    auto* param = p.apvts.getParameter (id);
    jassert (param != nullptr);
    param->setValueNotifyingHost (param->convertTo0to1 (plainValue));
}

std::unique_ptr<HorizonAudioProcessor> makeRoom (int model, float mixPercent)
{
    auto p = std::make_unique<HorizonAudioProcessor>();
    p->setPlayConfigDetails (2, 2, sr, blockSize);
    p->prepareToPlay (sr, blockSize);
    setParam (*p, hrzn::id::mix, mixPercent);
    setParam (*p, hrzn::id::predelay, 0.0f);
    setParam (*p, hrzn::id::space,
              100.0f * (float) model / (float) (hrzn::numRoomModels - 1));
    setParam (*p, hrzn::id::decayTime, hrzn::roomModels[(size_t) model].baseRt60);

    // Let the smoothers and the Space morph arrive before anything is measured.
    juce::AudioBuffer<float> buf (2, blockSize);
    juce::MidiBuffer midi;
    const float target = (float) model;
    for (int b = 0; b < 4000; ++b)
    {
        buf.clear();
        p->processBlock (buf, midi);
        if (b > 32 && std::abs (p->displaySpacePosition.load() - target) <= 1.0e-3f)
            break;
    }
    for (int b = 0; b < 16; ++b)
    {
        buf.clear();
        p->processBlock (buf, midi);
    }
    return p;
}

juce::AudioBuffer<float> render (HorizonAudioProcessor& p, const juce::AudioBuffer<float>& input,
                                 double seconds)
{
    const int total = (int) (seconds * sr);
    juce::AudioBuffer<float> out (2, total);
    out.clear();

    juce::AudioBuffer<float> buf (2, blockSize);
    juce::MidiBuffer midi;
    for (int start = 0; start < total; start += blockSize)
    {
        const int n = juce::jmin (blockSize, total - start);
        buf.clear();
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < n; ++i)
                if (start + i < input.getNumSamples())
                    buf.setSample (ch, i, input.getSample (ch, start + i));
        p.processBlock (buf, midi);
        for (int ch = 0; ch < 2; ++ch)
            out.copyFrom (ch, start, buf, ch, 0, n);
    }
    return out;
}

// Three quick noise bursts a few ms apart, the way a real clap is several
// palms' worth of transients rather than one click.
juce::AudioBuffer<float> makeClap()
{
    juce::AudioBuffer<float> x (2, (int) (0.05 * sr));
    x.clear();
    juce::Random rng (0xc1a9);
    for (double onsetMs : { 0.0, 7.0, 15.0 })
    {
        const int start = (int) (onsetMs * 0.001 * sr);
        for (int i = 0; start + i < x.getNumSamples(); ++i)
        {
            const float env = 0.5f * std::exp (-(float) i / (float) (0.004 * sr));
            const float v = env * (rng.nextFloat() * 2.0f - 1.0f);
            x.addSample (0, start + i, v);
            x.addSample (1, start + i, v);
        }
    }
    return x;
}

bool writeWav (const juce::File& file, const juce::AudioBuffer<float>& buf)
{
    file.deleteFile();
    juce::WavAudioFormat wav;
    std::unique_ptr<juce::AudioFormatWriter> writer (
        wav.createWriterFor (new juce::FileOutputStream (file), sr,
                             (unsigned int) buf.getNumChannels(), 24, {}, 0));
    return writer != nullptr && writer->writeFromAudioSampleBuffer (buf, 0, buf.getNumSamples());
}

std::vector<float> channel (const juce::AudioBuffer<float>& b, int ch)
{
    return { b.getReadPointer (ch), b.getReadPointer (ch) + b.getNumSamples() };
}

constexpr int nedWin = 1024;

double nedAt (const std::vector<float>& h, size_t centre)
{
    if (centre < (size_t) nedWin / 2 || centre + nedWin / 2 >= h.size())
        return 0.0;
    const size_t s = centre - nedWin / 2;

    double mean = 0.0;
    for (int i = 0; i < nedWin; ++i)
        mean += h[s + (size_t) i];
    mean /= nedWin;

    double var = 0.0;
    for (int i = 0; i < nedWin; ++i)
        var += juce::square (h[s + (size_t) i] - mean);
    const double sd = std::sqrt (var / nedWin);
    if (sd <= 1.0e-12)
        return 0.0;

    int over = 0;
    for (int i = 0; i < nedWin; ++i)
        if (std::abs (h[s + (size_t) i] - mean) > sd)
            ++over;
    return (double) over / nedWin / 0.3173;
}

double mixingTimeMs (const std::vector<float>& h)
{
    for (size_t c = nedWin / 2; c + nedWin / 2 < h.size(); c += 64)
        if (nedAt (h, c) >= 0.9)
            return (double) c * 1000.0 / sr;
    return -1.0;
}

double clarityDb (const std::vector<float>& h, double splitMs)
{
    const size_t split = (size_t) (splitMs * 0.001 * sr);
    double early = 0.0, late = 0.0;
    for (size_t i = 0; i < h.size(); ++i)
        (i < split ? early : late) += (double) h[i] * h[i];
    return 10.0 * std::log10 ((early + 1.0e-30) / (late + 1.0e-30));
}

// Std of the fine magnitude spectrum about its third-octave average, 200 Hz -
// 8 kHz, over the late field (50 ms onwards).
double lateRippleDb (const std::vector<float>& h)
{
    constexpr int order = 15, n = 1 << order;
    const size_t start = (size_t) (0.05 * sr);
    std::vector<float> data (2 * (size_t) n, 0.0f);
    for (int i = 0; i < n && start + (size_t) i < h.size(); ++i)
    {
        const float w = 0.5f - 0.5f * std::cos (juce::MathConstants<float>::twoPi * (float) i / (float) n);
        data[(size_t) i] = h[start + (size_t) i] * w;
    }
    juce::dsp::FFT fft (order);
    fft.performFrequencyOnlyForwardTransform (data.data());

    const double binHz = sr / n;
    std::vector<double> db ((size_t) n / 2);
    for (size_t k = 1; k < db.size(); ++k)
        db[k] = 20.0 * std::log10 ((double) data[k] + 1.0e-12);

    double acc = 0.0;
    int count = 0;
    for (size_t k = (size_t) (200.0 / binHz); k < (size_t) (8000.0 / binHz); ++k)
    {
        const double f = (double) k * binHz;
        const size_t lo = (size_t) (f * std::pow (2.0, -1.0 / 6.0) / binHz);
        const size_t hi = (size_t) (f * std::pow (2.0, 1.0 / 6.0) / binHz);
        double m = 0.0;
        for (size_t j = lo; j <= hi; ++j)
            m += db[j];
        m /= (double) (hi - lo + 1);
        acc += juce::square (db[k] - m);
        ++count;
    }
    return std::sqrt (acc / juce::jmax (1, count));
}

} // namespace

int main (int argc, char** argv)
{
    juce::ScopedJuceInitialiser_GUI juceInit;

    const juce::File dir = juce::File::getCurrentWorkingDirectory()
                               .getChildFile (argc > 1 ? argv[1] : "audition");
    dir.createDirectory();

    std::printf ("Writing renders to %s\n\n", dir.getFullPathName().toRawUTF8());
    std::printf ("%-13s %6s %6s %6s %6s %6s %6s %7s %7s %6s\n", "room", "NED@16", "NED@30",
                 "NED@60", "NED@120", "NED@250", "mix ms", "C50 dB", "ripple", "RT s");

    const auto clap = makeClap();

    for (int m = 0; m < hrzn::numRoomModels; ++m)
    {
        const auto& model = hrzn::roomModels[(size_t) m];
        const double seconds = juce::jlimit (1.5, 16.0, 1.3 * model.baseRt60 + 0.5);
        const auto name = juce::String (m).paddedLeft ('0', 2) + "_"
                          + juce::String (model.name).replaceCharacter (' ', '_');

        juce::AudioBuffer<float> impulse (2, 1);
        impulse.setSample (0, 0, 1.0f);
        impulse.setSample (1, 0, 1.0f);

        auto wetRoom = makeRoom (m, 100.0f);
        const auto ir = render (*wetRoom, impulse, seconds);
        writeWav (dir.getChildFile (name + "_ir.wav"), ir);

        auto mixRoom = makeRoom (m, 35.0f);
        writeWav (dir.getChildFile (name + "_clap.wav"), render (*mixRoom, clap, seconds));

        const auto h = channel (ir, 0);
        auto at = [&] (double ms) { return nedAt (h, (size_t) (ms * 0.001 * sr)); };

        std::printf ("%-13s %6.2f %6.2f %6.2f %6.2f %6.2f %6.1f %7.2f %7.2f %6.2f\n",
                     model.name, at (16.0), at (30.0), at (60.0), at (120.0), at (250.0),
                     mixingTimeMs (h), clarityDb (h, 50.0), lateRippleDb (h),
                     (double) model.baseRt60);
    }

    return 0;
}
