#pragma once

#include <juce_core/juce_core.h>
#include <vector>

namespace hrzn
{

// Lock-free single-producer/single-consumer FIFO carrying a mono mix of the
// audio for the spectrum analyzer. Audio thread pushes, GUI thread pops.
//
// Ported unchanged from Equinox's `src/dsp/AnalyzerFifo.h`, where it has been
// feeding the family's analyzer since that plugin's v0.2. **Promotion
// candidate** on the same footing as `ui/PrecisionSlider.h` and
// `dsp/FilterCoeffs.h`: two products now carry an identical copy.
class AnalyzerFifo
{
public:
    AnalyzerFifo() : fifo (bufferSize), buffer ((size_t) bufferSize, 0.0f) {}

    void push (const float* l, const float* r, int numSamples) noexcept
    {
        int start1, size1, start2, size2;
        fifo.prepareToWrite (numSamples, start1, size1, start2, size2);

        for (int i = 0; i < size1; ++i)
            buffer[(size_t) (start1 + i)] = 0.5f * (l[i] + r[i]);
        for (int i = 0; i < size2; ++i)
            buffer[(size_t) (start2 + i)] = 0.5f * (l[size1 + i] + r[size1 + i]);

        fifo.finishedWrite (size1 + size2);
    }

    int getNumReady() const noexcept { return fifo.getNumReady(); }

    // Pops up to numSamples into dest; returns the number actually popped.
    int pop (float* dest, int numSamples) noexcept
    {
        int start1, size1, start2, size2;
        fifo.prepareToRead (numSamples, start1, size1, start2, size2);

        for (int i = 0; i < size1; ++i)
            dest[i] = buffer[(size_t) (start1 + i)];
        for (int i = 0; i < size2; ++i)
            dest[size1 + i] = buffer[(size_t) (start2 + i)];

        fifo.finishedRead (size1 + size2);
        return size1 + size2;
    }

    void clear() noexcept { fifo.reset(); }

private:
    static constexpr int bufferSize = 32768;

    juce::AbstractFifo fifo;
    std::vector<float> buffer;
};

} // namespace hrzn
