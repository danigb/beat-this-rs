# [Beat This! Rust](https://github.com/danigb/beat-this-rs)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/danigb/beat-this-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/danigb/beat-this-rs/actions/workflows/ci.yml)

A Rust port of the "Beat This!" AI-powered beat tracking system from Johannes Kepler University Linz.

The port was made with [Claude](https://claude.ai/)

## Overview

This is a Rust port of the Beat This! inteference mechanism, originally published at ISMIR 2024. The goal is to generate beat information from any audio without any external dependency (except the model weights themselves)

The original system uses a transformer-based neural network to detect musical beats and downbeats in audio files with high accuracy.

- **Original Paper**: ["Beat This! Accurate and Generalizable Beat Tracking"](https://arxiv.org/pdf/2407.21658)
- **Original Repository**: https://github.com/CPJKU/beat_this
- **C++ Port**: https://github.com/mosynthkey/beat_this_cpp

## Features

- **Two Runtime Backends**: Choose between `rten` (pure Rust, zero external dependencies) or `ort` (ONNX Runtime)
- **Multiple Output Formats**: JSON, plain text `.beats` files, click track WAV, or mixed audio
- **Batch Processing**: Process entire directories of audio files with summary statistics
- **BPM Estimation**: Automatic tempo detection from beat timestamps
- **Rust Library**: Clean public API for embedding in other applications
- **Multiple Model Variants**: Standard and small model sizes

## Architecture

The system consists of four main components:

1. **Audio Processing**: Load audio files (WAV, MP3, FLAC, OGG via symphonia) and resample to 22050 Hz mono
2. **Mel Spectrogram**: Convert audio to 128-dimensional Mel spectrograms using an ONNX model
3. **Beat Inference**: Run the trained transformer model in overlapping 30-second chunks
4. **Post-processing**: Extract beat and downbeat timestamps via peak detection, deduplication, and downbeat snapping

## Dependencies

- **rten**: Pure-Rust ONNX runtime (default backend, no external dependencies)
- **ort**: ONNX Runtime bindings with CoreML support on macOS (optional backend)
- **symphonia**: Audio decoding (MP3, WAV, FLAC, OGG)
- **rubato**: High-quality sinc resampling
- **hound**: WAV file writing
- **ndarray**: N-dimensional array operations

## Installation

### From Source

Requires [uv](https://docs.astral.sh/uv/) for Python model management scripts. The scripts will download and convert models into onnx format (they are published as [ckpt files](https://cloud.cp.jku.at/index.php/s/7ik4RrBKTS273gp))

```bash
git clone git@github.com:danigb/beat-this-rs.git
cd beat-this-rs
./scripts/download-models.sh    # Downloads model via uv run
cargo build --release
```

The release binary will be at `target/release/beat-this`. Release mode enables LTO and stripping for optimized performance.

### Using the ORT Runtime

The default `rten` runtime requires no external dependencies. If you want to use the `ort` runtime, install ONNX Runtime first:

```bash
# macOS
brew install onnxruntime

# Or download from https://github.com/microsoft/onnxruntime/releases
```

Then run with `--runtime ort`.

## Model Setup

The mel spectrogram model (`models/mel_spectrogram.onnx`) is included in the repository. To download and convert the beat tracking model:

```bash
./scripts/download-models.sh
```

This uses [uv](https://docs.astral.sh/uv/) internally to manage Python dependencies. Alternatively, download and convert manually:

```bash
uv run scripts/ckpt2onnx.py final0
mv models/final0.onnx models/beat_this.onnx
```

For the small model (~10 MB, optional):

```bash
uv run scripts/ckpt2onnx.py small0
mv models/small0.onnx models/beat_this_small.onnx
```

After setup, the `models/` directory should contain:

```
models/
├── mel_spectrogram.onnx    # Included in repo (~270 KB)
├── beat_this.onnx          # Standard model (~83 MB)
└── beat_this_small.onnx    # Small model (~10 MB, optional)
```

Visit [original repo](https://github.com/CPJKU/beat_this?tab=readme-ov-file#available-models) for available models

## Usage

### Command Line Interface

**Default output (JSON to stdout)**:

```bash
beat-this input.wav
```

**Write JSON and beats files** (auto-named from input):

```bash
beat-this input.wav --json --beats
# → input.json, input.beats
```

**Write with explicit paths**:

```bash
beat-this input.wav --json=results.json --click=clicks.wav
```

**Use the small model**:

```bash
beat-this input.wav --model models/beat_this_small.onnx
```

**Batch processing a directory**:

```bash
beat-this ./music-folder/ -r --json --beats
```

### CLI Options

| Option                  | Description                                             |
| ----------------------- | ------------------------------------------------------- |
| `<input>`               | Audio file or directory                                 |
| `--json [FILE]`         | Write JSON output (default ext: `.json`)                |
| `--beats [FILE]`        | Write beats text file (default ext: `.beats`)           |
| `--click [FILE]`        | Write click-track WAV (default ext: `.click.wav`)       |
| `--mix [FILE]`          | Write mixed audio WAV (default ext: `.mix.wav`)         |
| `--overwrite`           | Overwrite existing output files                         |
| `--model <PATH>`        | Beat model path (default: `models/beat_this.onnx`)      |
| `--mel-model <PATH>`    | Mel model path (default: `models/mel_spectrogram.onnx`) |
| `--runtime <rten\|ort>` | Inference backend (default: `rten`)                     |
| `-r, --recursive`       | Recurse into subdirectories                             |
| `-v, --verbose`         | Print timing for each stage                             |
| `--profile <PREFIX>`    | ORT profiling trace output                              |

### Rust API

```rust
use std::path::Path;
use beat_this::{BeatThis, runtime::rten::RtenRuntime};

// Initialize with the pure-Rust runtime
let mut bt = BeatThis::new(
    &RtenRuntime,
    Path::new("models/mel_spectrogram.onnx"),
    Path::new("models/beat_this.onnx"),
)?;

// Process an audio file
let result = bt.process_file(Path::new("input.wav"))?;

println!("Found {} beats, {} downbeats", result.beats.len(), result.downbeats.len());
for (i, &time) in result.beats.iter().enumerate() {
    println!("Beat {}: {:.3}s", i, time);
}
```

## Output Formats

### JSON (default)

```json
{
  "beats": [0.34, 0.68, 1.02, 1.36, 1.7, 2.04],
  "downbeats": [0.34, 1.7],
  "bpm": 120.0
}
```

### Plain Text Beats (`--beats`)

Tab-separated values with beat time and position within measure:

```
0.340	4
0.681	1
1.023	2
1.364	3
1.705	4
2.047	1
```

- **Column 1**: Beat time in seconds
- **Column 2**: Beat number (1 = downbeat, 2-4 = other beats)

### Click Track (`--click`)

Generated 44100 Hz mono WAV with:

- **Downbeats**: 880 Hz sine wave
- **Other beats**: 440 Hz sine wave
- ADSR envelope shaping

### Mixed Audio (`--mix`)

Combines the original audio with the click track:

- **Original music**: 70% volume
- **Click track**: 30% volume

## Project Structure

```
beat-this-rs/
├── src/
│   ├── lib.rs                # Public API (BeatThis struct)
│   ├── main.rs               # CLI application
│   ├── audio.rs              # Audio loading and resampling
│   ├── mel.rs                # Mel spectrogram computation
│   ├── inference.rs          # Chunked beat/downbeat inference
│   ├── postprocessing.rs     # Peak detection, deduplication, snapping
│   ├── output.rs             # JSON, WAV, click track generation
│   └── runtime/
│       ├── mod.rs            # Runtime trait abstractions
│       ├── ort.rs            # ONNX Runtime backend (CoreML on macOS)
│       └── rten.rs           # Pure-Rust backend
├── models/                   # ONNX model files (mel model included, beat models downloaded)
├── tests/                    # Integration tests
├── references/               # Reference implementations
└── Cargo.toml
```

## Model Details

- **Architecture**: Transformer-based neural network
- **Input**: 128-dimensional Mel spectrograms
- **Output**: Beat and downbeat probability logits
- **Processing**: Overlapping chunks of 1500 frames (30 seconds at 50 fps)
- **Standard model**: ~83 MB (32-bit float)
- **Small model**: ~10 MB

## Performance

Benchmarked on Apple M4 MacBook Pro (2024), comparing against the Python reference implementation (PyTorch, CPU mode, no DBN post-processing):

| File      | Duration | Python | Rust (rten) | Rust (ort) |
| --------- | -------: | -----: | ----------: | ---------: |
| short.wav |       9s |   1.8s |        0.7s |       1.2s |
| test1.mp3 |     4:32 |   5.1s |        4.6s |       4.6s |
| test2.mp3 |    13:48 |  11.9s |       12.1s |      11.9s |

Both Rust runtimes produce identical beat timestamps. The `rten` backend (pure Rust, no external dependencies) performs on par with `ort` (ONNX Runtime). Processing time is dominated by beat inference, which scales linearly with audio duration.

To run the benchmarks yourself:

```bash
uv run scripts/integration-test.py
```

## Acknowledgments

- **Original Beat This! Implementation**: This Rust port is based on the Beat This! model from Johannes Kepler University Linz. See the [original repository](https://github.com/CPJKU/beat_this) for the research paper and licensing terms.
- **C++ Port**: [beat_this_cpp](https://github.com/mosynthkey/beat_this_cpp) by mosynthkey, which served as a reference for this implementation.
- **Dependencies**: rten, ort, symphonia, rubato, and the broader Rust ecosystem.
