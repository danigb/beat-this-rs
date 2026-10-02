# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [1.1.0] - unreleased

Bounded memory for long inputs. Memory figures are macOS `peak memory footprint` (MiB, median of
3) on an Apple M4 Pro, full model `beat_this.onnx`, synthetic mono input at 48 kHz unless a figure
is marked 44.1 kHz.

### Added

- `BeatThis::stream` / `BeatStream::{push, finish, finish_timed}`: a streaming front end. Push
  mono `f32` chunks of any size as they are decoded; the whole signal never has to be in memory.
  Peak memory on 60 min of 48 kHz mono: **620.5 MiB** (vs 5 651.0 MiB through 1.0.0's
  `analyze_audio`, see the notes), growing linearly by **93.8 MiB per hour** of audio
  (5 / 10 / 20 / 60 min: 533.7 / 542.2 / 558.0 / 620.5 MiB). Of that, about 3 MB is front-end
  state, independent of the input length; the growth is the mel spectrogram (88 MiB per hour) and
  the beat logits. Output: bit-identical to `analyze_audio` on the same samples, for every chunk
  size; against 1.0.0, bit-identical for 22 050 Hz input and 22 050·2^k Hz sources
  (11.025 / 44.1 / 88.2 kHz), and at other rates different as described under the resampler below.
  The stream mutably borrows the `BeatThis`; with the rten model it is `Send`.
- `BeatThis::analyze_owned` / `analyze_owned_timed`: take the caller's `Vec<f32>` and free it before
  the beat model runs. Bit-identical to `analyze_audio`.

### Changed

- `analyze_audio`, `analyze_audio_timed`, `analyze_owned(_timed)` and `analyze_file` run through
  `BeatStream` (one push of the whole signal). Bit-identical to the whole-buffer pipeline they
  replace; the slice is no longer copied. 60 min at 48 kHz: 1 278.5 MiB through `analyze_audio`
  (1.0.0: 5 651.0), of which 659 MiB is the caller's own native-rate buffer.
- The rten backend no longer clones model inputs or outputs. Bit-identical (identity suite). At
  20 min this removed 202 MiB of PCM copies from the mel stage.
- The mel spectrogram is computed in 64-frame-aligned windows of 1 536 frames (about 30.7 s)
  instead of one whole-signal graph run. Bit-identical on rten (identity suite; see the platform
  notes for where it ran; the alignment avoids a rounding difference in rten-gemm's partial column
  tiles). At 60 min the mel stage went from 4 709.8 to 2 519.0 MiB. A custom mel model passed to
  `BeatThis::from_models` must use hop 441; a mismatch is reported as an error.
- The resampler runs in 8 192-frame chunks at every input rate. Memory, `resample` stage at
  60 min: 2 376.5 → 1 717.7 MiB at 48 kHz, 2 215.8 → 1 610.0 MiB at 44.1 kHz. What this does to the
  output depends on the source/target ratio and on the input length:
  - **Exact ratios: bit-identical.** When rubato's step (source rate / target rate) is a short
    dyadic fraction, chunking cannot change any read position. That is the case for 22 050·2^k Hz
    sources resampled to 22 050 Hz (tested at 11.025, 44.1 and 88.2 kHz), which is what the
    pipeline does. Identity is a property of the pair, not of the source: `load_audio` of the
    44.1 kHz test MP3 is bit-identical at targets 22 050 and 88 200 Hz, but differs from 1.0.0 at
    48 000 Hz (max |Δ| 6.29e-4), 32 000 Hz (3.66e-4) and 16 000 Hz (6.13e-5); 44.1 → 48 kHz on
    the synthetic signal differs by 8.02e-3 at 20 min and 5.66e-2 at 60 min.
  - **Other ratios (48 kHz etc.): 1.0.0's one-shot resampler loses precision on long inputs, 1.1.0
    does not.** 1.0.0 resampled the whole input in one rubato call, which accumulates the read
    position by repeated `f64` additions over the whole input, so the position error grows with the
    length; how fast depends on the ratio's binary digits and changes each time the position passes
    a power of two. At 48 kHz it grows slowly up to 2^27 input frames (46.6 min) and much faster
    after; at 96 kHz, whose step is exactly twice the 48 kHz one, the change falls at 2^28 frames,
    the same 46.6 min (measured). 1.1.0 renormalises the position after every chunk, so its rounding
    never grows. Measured against an analytic 1 kHz tone over 60 min (`resampler_accuracy_long`),
    per-minute rms error: 1.1.0 stays at 3.24e-6 to 3.30e-6 at 48 kHz; 1.0.0 goes from 3.65e-6
    (minute 1) to 1.19e-4 (minute 20), 2.80e-4 (minute 46) and 1.14e-2 (minute 60). At 96 kHz:
    1.1.0 6.89e-6 throughout, 1.0.0 up to 1.14e-2; 44.1 → 48 kHz: 1.1.0 at most 3.09e-6, 1.0.0 up
    to 1.53e-2.
  - **So at those rates the output differs from 1.0.0.** Resampled PCM at 48 kHz, max |Δ| against
    1.0.0: 4.24e-5 (5 min), 8.89e-5 (10 min), 1.86e-4 (20 min), 3.90e-4 (40 min), 4.47e-3
    (50 min), 1.83e-2 (60 min). Output length and leading delay are unchanged.
  - **Short inputs: a small drift, beats unchanged** (decision D1, accepted on 2026-10-02 by Dani).
    Beats and downbeats were bit-identical to 1.0.0 on every input tested: the committed MP3
    upsampled to 48 kHz (small and full models), a 23-file corpus decoded at 48 kHz (full model,
    23 of 23, re-run on the final 1.1.0 code),
    and the synthetic 48 kHz signal at 20 min (small and full models) and at 40 and 50 min (full
    model). On the corpus and on inputs up to 20 min, mel values moved by at most 3.47e-4 and
    logits by at most 2.10e-4; at 40 min the logits moved by 3.11e-4, at 50 min by 4.35e-3.
  - **Long inputs: beats can move by one frame.** On 60 min of the synthetic 48 kHz signal (full
    model), 2 of 7 200 beats and 2 of 5 886 downbeats moved one frame (20 ms) earlier, from
    3 063.02 to 3 063.00 s and from 3 182.52 to 3 182.50 s (each downbeat with its beat, as
    downbeats are snapped to the nearest beat); the beat-logit drift reaches 1.63e-2 and the mel
    drift 8.46e-3 (minutes 55–60). At both places the beat logits of the two adjacent frames were
    within 5.1e-4 of each other in 1.0.0 (7.06051 / 7.06101 and 7.23161 / 7.23191), so the drift
    swapped which frame peaks. This follows from the precision
    loss of 1.0.0's resampler shown above, which 1.1.0 fixes; it is not part of the short-input
    drift that D1 accepted.
- A sample rate of 0 is an error from `analyze_audio`, `analyze_owned` and `stream` at every input
  length. 1.0.0 returned an error for up to 2 samples and panicked inside
  rubato from 3 samples on.
- Error messages changed for inputs that already failed in 1.0.0: an empty signal at a rate other
  than 22 050 Hz now gives "cannot resample empty input" (1.0.0: rubato's "Invalid chunk_size
  provided: 0"); an empty signal at 22 050 Hz, and signals too short to give any 22 050 Hz sample
  (1–3 samples at 44.1 kHz, 1–4 at 48 kHz), give "cannot compute a mel spectrogram of empty input"
  (1.0.0: an rten `Pad` operator error). Which inputs fail is unchanged.
- `load_audio` frees the interleaved decode buffer before resampling instead of at the end of the
  call. Output unchanged. The saving was not visible in the CLI's peak on a 20-min 48 kHz stereo
  WAV (1 322.5 vs 1 323.3 MiB `ru_maxrss`), where the peak lies outside the resample.

### Notes

- Identity is checked by `tests/identity_v1_0_0.rs`, which runs a verbatim copy of the 1.0.0
  pipeline live on the same inputs and compares with `to_bits()`. Inputs at non-exact ratios are
  checked against drift bounds (PCM 2e-3, mel 5e-4, logits 5e-4) with beats and downbeats
  bit-identical; those bounds hold only for the inputs they are applied to, at most 20 min long,
  and are not a general bound (the logit drift is 4.35e-3 at 50 min). The Python goldens and the
  rten/ort cross-runtime test pass unchanged.
- Platforms. Everything above was run on arm64 (Apple M4 Pro, NEON). On x86_64 with AVX2+FMA,
  under QEMU emulation (Docker `linux/amd64`), only the identity test binary ran, at two commits
  of the branch: `5e70cfe` (windowed mel added, not yet routed) and `cddc70b` (mel routed through
  the windowed path; that pass is a rerun after a QEMU segfault that also occurs with the older
  binary). At both commits the resampler still ran 48 kHz input in one shot. The drift tests for
  chunked resampling at every rate, all `stream` tests and the long-input tests have **not** run on
  x86_64. The AVX-512 path (rten-gemm NR 32) has never been executed. Memory has not been measured
  on Linux.
- Bit identity is verified against this repository's `Cargo.lock` (rten 0.24.0 / rten-gemm 0.24.0,
  rubato 3.0.0, rustfft 6.4.1). The dependency ranges in `Cargo.toml` are unchanged, so a consumer
  that resolves a newer patch release could get different kernels and lose bit identity.
- The smallest |logit| seen in the corpus (1.97e-6) is below the largest logit drift (8.96e-5), so
  at non-exact ratios a beat could in principle cross the threshold on some short input, though
  none did on the inputs above; on long inputs beats did move (60 min, above).
- The 1.0.0 figure of 5 651.0 MiB at 60 min comes from a noisy measurement: on that host the
  60-min 1.0.0 cells varied widely between runs (`ru_maxrss` of `analyze_audio` from 2 203 to
  5 479 MiB over 7 runs, as macOS compressed pages), and footprint and `ru_maxrss` disagreed at
  that size. Treat it as an approximate upper value; the 5/10/20-min figures are stable.
- On macOS a buffer freed by the caller (or by `analyze_owned`) can stay in the process footprint
  for a while after `free`, which is why `analyze_owned` peaks about as high as `analyze_audio`;
  use `stream` to stay at O(chunk).
- `beat_this::__probe` (hidden from the docs) is unstable test and measurement infrastructure,
  not public API; it can change in any release.

## [1.0.0] - 2026-05-30

Parity-with-the-reference release. Parity with the Python
[`beat_this`](https://github.com/CPJKU/beat_this) reference is now **verified by a
committed golden test** (`tests/python_parity.rs`), not just argued by construction:
F-measure == 1.0 (standard FP32 model) and ≥ 0.99 (small model) at the ±70 ms MIR
window for both beats and downbeats.

### Added

- Golden parity test against the Python reference (`tests/python_parity.rs` +
  `scripts/gen_golden.py`), runnable on a fresh clone with the committed small model
- "Parity with the Python reference" section in the README documenting the verification
  and the remaining known divergence

### Changed

- `deduplicate_peaks` keeps **fractional** merged-peak frame positions instead of
  rounding to an integer frame, matching the Python reference (removes a ≤10 ms
  divergence on merged adjacent peaks)
- `beat_counts` now ports the reference's `infer_beat_numbers`: pickup-measure
  (anacrusis) beats are numbered so they lead _into_ the first downbeat, matching the
  Python `.beats`/JSON count column (beat/downbeat **times** are unchanged)
- The `ort` (ONNX Runtime) backend is now behind an off-by-default `ort` Cargo
  feature. The default build is pure-Rust `rten` only — no `libonnxruntime` needed.
  Build/test the ort backend (and its `--runtime ort`, cross-runtime parity tests, and
  op-level `--profile`) with `--features ort`.

### Notes

- **Known remaining divergence:** the resampler (`rubato` sinc vs Python `soxr`) differs
  sub-perceptually for inputs **not** already at 22050 Hz; inputs at 22050 Hz resample
  exactly. Decode precision (f32 + symphonia vs float64 + torchaudio) differs negligibly.
- **Post-processing is "minimal" only** — the optional `--dbn` path (madmom DBN) is
  intentionally not implemented; the model is designed to be accurate without it, and
  Python's default is also "minimal", so default-vs-default output matches.
- The "identical timestamps" claim is scoped to the two Rust backends (rten vs ort),
  verified by `tests/cross_runtime.rs`.

## [0.3.0] - 2026-03-11

### Changed

- Simplified public API: internal modules (`inference`, `mel`, `postprocessing`, `output`) are now private
- Renamed core types: `InferenceRuntime` → `Runtime`, `InferenceSession` → `Model`, `BeatInference` → `BeatPredictor`, `MelProcessor` → `MelExtractor`, `PostProcessor` → `PeakPicker`
- Renamed methods: `process` → `predict` (BeatPredictor), `process` → `extract` (MelExtractor), `process` → `decode` (PeakPicker)
- Re-exported `beat_counts` and `calculate_bpm` from the crate root

### Added

- `BeatThis::from_models` constructor for building from pre-loaded models
- `BeatThis::beat_model_mut` accessor for runtime-specific operations (e.g. ORT profiling)
- `analyze_audio_timed` method with per-stage timing via `TimedAnalysis` and `AnalysisTiming`

### Fixed

- Audio chunk padding: correctly handle last chunk border trimming instead of always using full `CHUNK_SIZE`, fixing potential out-of-bounds access on short audio

## [0.2.0] - 2026-03-05

### Changed

- Replaced `process_audio` / `process_file` with `analyze_audio` / `analyze_file` returning a richer `BeatAnalysis` type that includes mel spectrogram and raw logits
- Removed `BeatResult` from the public API; `PostProcessor::process` now returns `(Vec<f32>, Vec<f32>)`

### Added

- `BeatAnalysis` struct with `beats`, `downbeats`, `mel`, `beat_logits`, and `downbeat_logits` fields
- `--mel` CLI flag to write mel spectrogram as numpy `.npy` file
- `write_mel_npy` function in `output` module

## [0.1.0] - 2025-03-03

Initial release.

### Added

- Beat and downbeat detection from audio files (WAV, MP3, FLAC, OGG)
- Two runtime backends: `rten` (pure Rust, default) and `ort` (ONNX Runtime with CoreML on macOS)
- Multiple output formats: JSON, plain text `.beats`, click track WAV, mixed audio WAV
- BPM estimation from beat timestamps
- Batch processing of directories with summary statistics
- Rust library API (`BeatThis` struct) for embedding in other applications
- Standard (~83 MB) and small (~10 MB) model variants
- Docker image support
