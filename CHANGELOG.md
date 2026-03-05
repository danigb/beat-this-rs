# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

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
