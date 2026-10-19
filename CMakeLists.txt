cmake_minimum_required(VERSION 3.22)

project(horizon VERSION 0.8.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# JUCE is vendored in libs/JUCE (cloned at tag 8.0.14)
add_subdirectory(libs/JUCE)

# Family-shared components (design language, presets, state).
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../jdm_shared" jdm_shared)

# NEEDS_MIDI_INPUT went TRUE at v0.8 with MIDI Learn (same order the rest of
# the family took: Zenith added the port at its v0.6). An effect that declares
# a MIDI input gets one in every host; a plugin that maps controllers without
# declaring one gets no messages at all in several of them.
juce_add_plugin(horizon
    COMPANY_NAME "Equinox Audio"
    BUNDLE_ID com.equinoxaudio.horizon
    PLUGIN_MANUFACTURER_CODE Eeqa
    PLUGIN_CODE Hrzn
    FORMATS VST3 Standalone
    PRODUCT_NAME "Horizon"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE
    COPY_PLUGIN_AFTER_BUILD FALSE
    VST3_CATEGORIES Fx Reverb)

target_sources(horizon PRIVATE
    src/PluginProcessor.cpp
    src/PluginEditor.cpp
    src/ui/ReverbDisplay.cpp)

target_include_directories(horizon PRIVATE src)

target_compile_definitions(horizon PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)

target_link_libraries(horizon
    PRIVATE
        jdm_shared
        juce::juce_audio_utils
        juce::juce_dsp
        juce::juce_opengl
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)

# Offline DSP test harness (console app around the processor sources).
juce_add_console_app(horizon_tests PRODUCT_NAME "horizon_tests")

target_sources(horizon_tests PRIVATE
    test/DspTests.cpp
    src/PluginProcessor.cpp
    src/PluginEditor.cpp
    src/ui/ReverbDisplay.cpp)

target_include_directories(horizon_tests PRIVATE src)

target_compile_definitions(horizon_tests PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_MODAL_LOOPS_PERMITTED=0)

target_link_libraries(horizon_tests
    PRIVATE
        jdm_shared
        juce::juce_audio_utils
        juce::juce_dsp
        juce::juce_opengl
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)

# Listening/measurement tool: renders every room's IR and a clap to WAV and
# prints echo density, clarity and late-field ripple per room.
juce_add_console_app(horizon_audition PRODUCT_NAME "horizon_audition")

target_sources(horizon_audition PRIVATE
    test/Audition.cpp
    src/PluginProcessor.cpp
    src/PluginEditor.cpp
    src/ui/ReverbDisplay.cpp)

target_include_directories(horizon_audition PRIVATE src)

target_compile_definitions(horizon_audition PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_MODAL_LOOPS_PERMITTED=0)

target_link_libraries(horizon_audition
    PRIVATE
        jdm_shared
        juce::juce_audio_utils
        juce::juce_dsp
        juce::juce_opengl
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)
