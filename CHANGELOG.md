# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

Parity-with-the-reference release (targeting v1.0.0). Parity with the Python
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
  (anacrusis) beats are numbered so they lead *into* the first downbeat, matching the
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
