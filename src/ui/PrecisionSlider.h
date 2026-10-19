#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <functional>

namespace hrzn
{

// A rotary that can be placed exactly.
//
// Horizon needs this more than the rest of the family does. Space runs through
// fourteen room models over its travel, so at JUCE's default sensitivity one
// whole room is about nineteen pixels of drag, and deliberately parking the
// morph a third of the way between two of them is guesswork. Three things fix
// that, and the first two apply to every knob rather than only to Space,
// because a control that behaves differently from the ones either side of it
// is worse than one that is merely coarse:
//
//   - holding Shift makes a drag eight times finer, and the modifier may be
//     pressed and released *during* a gesture
//   - the scroll wheel steps by 4% of the range instead of JUCE's 15%, or by
//     0.5% with Shift held
//   - a right-click hook, so the owner can offer somewhere exact to jump to
//
// Deliberately product-agnostic: worth promoting into `jdm_shared` when
// Equinox, Meridian and Zenith adopt the same gestures.
class PrecisionSlider : public juce::Slider
{
public:
    static constexpr int coarseDragPixels = 250;    // JUCE's own default
    static constexpr int fineDragPixels = 2000;

    static constexpr double coarseWheelStep = 0.04; // proportion of the range
    static constexpr double fineWheelStep = 0.005;

    // Right-click hook. The slider stays ignorant of what is on the menu; the
    // editor owns that.
    std::function<void()> onPopupMenu;

    // Wheel hook. The owner routes this through the parameter so a scroll is
    // wrapped in a proper begin/end change gesture — JUCE's built-in wheel
    // path does that via an internal drag notification we cannot reach from
    // outside, and a value that reaches the host without gesture markers is a
    // value the host may decline to record.
    std::function<void (double)> onWheelValue;

    PrecisionSlider() { setMouseDragSensitivity (coarseDragPixels); }

    void mouseDown (const juce::MouseEvent& e) override
    {
        if (e.mods.isPopupMenu() && onPopupMenu != nullptr)
        {
            onPopupMenu();
            return;
        }

        setFine (e.mods.isShiftDown());
        juce::Slider::mouseDown (e);
    }

    void mouseDrag (const juce::MouseEvent& e) override
    {
        // JUCE measures an absolute drag from wherever the mouse went down, so
        // changing the sensitivity part-way through would rescale everything
        // dragged so far and jump the value. Ending the drag and starting a
        // new one from the current position re-anchors both the origin and the
        // value, which is the only way to let Shift be pressed and released
        // mid-move — which is how people actually reach for it.
        if (e.mods.isShiftDown() != fine)
        {
            juce::Slider::mouseUp (e);
            setFine (e.mods.isShiftDown());
            juce::Slider::mouseDown (e);
        }

        juce::Slider::mouseDrag (e);
    }

    void mouseWheelMove (const juce::MouseEvent& e,
                         const juce::MouseWheelDetails& wheel) override
    {
        if (! isEnabled() || e.mods.isAnyMouseButtonDown())
            return;

        const double amount = (std::abs (wheel.deltaX) > std::abs (wheel.deltaY)
                                   ? -(double) wheel.deltaX : (double) wheel.deltaY)
                              * (wheel.isReversed ? -1.0 : 1.0);

        if (juce::approximatelyEqual (amount, 0.0))
            return;

        const double step = e.mods.isShiftDown() ? fineWheelStep : coarseWheelStep;
        const double target = proportionOfLengthToValue (
            juce::jlimit (0.0, 1.0, valueToProportionOfLength (getValue()) + amount * step));

        if (juce::approximatelyEqual (target, getValue()))
            return;

        if (onWheelValue != nullptr)
            onWheelValue (target);
        else
            setValue (target, juce::sendNotificationSync);
    }

private:
    void setFine (bool shouldBeFine)
    {
        fine = shouldBeFine;
        setMouseDragSensitivity (fine ? fineDragPixels : coarseDragPixels);
    }

    bool fine = false;
};

} // namespace hrzn
