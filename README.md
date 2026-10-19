# Horizon

An algorithmic reverb plugin (VST3 / Standalone) built with JUCE 8. Part of the Equinox
Audio family alongside [Equinox](https://github.com/JDMcE/Equinox) (EQ),
[Meridian](https://github.com/JDMcE/Meridian) (compressor) and
[Zenith](https://github.com/JDMcE/Zenith) (limiter).

Runs mono, stereo, 5.1, 7.1, 7.1.4 and 9.1.6. One feedback delay network serves every layout and
each speaker reads it through its own Walsh row, so an immersive session is one room heard from
fifteen places rather than fifteen reverbs — and costs about 2.6x a stereo one rather than fifteen
times it. The LFE is passed through untouched.

Host integration: MIDI Learn on every control (right-click), independent undo history per A/B
slot, an embedded manual with an F1 shortcuts overlay, hover hints on the H key, a high-contrast
palette and scalable UI text.

## Building

Requirements: CMake 3.22+ and a C++20 compiler.

Horizon shares code with the rest of the family through
[jdm_shared](https://github.com/JDMcE/jdm_shared), which must be checked out next to
this repository:

```
some-folder/
  Horizon/
  jdm_shared/
```

```
git clone https://github.com/JDMcE/Horizon.git
git clone https://github.com/JDMcE/jdm_shared.git
cd Horizon
git clone --depth 1 --branch 8.0.14 https://github.com/juce-framework/JUCE.git libs/JUCE
cmake -B build
cmake --build build --config Release
```

Targets:

- `horizon_All` — everything below
- `horizon_VST3` / `horizon_Standalone` — the plugin
- `horizon_tests` — offline DSP verification harness (run the built exe; exits non-zero on failure)

## License

Horizon is free software, released under the GNU Affero General Public License v3.0
— see [LICENSE](LICENSE). It is built on [JUCE](https://juce.com), used under JUCE's
AGPLv3 open-source licence.
