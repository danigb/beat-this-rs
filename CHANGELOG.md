# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

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
