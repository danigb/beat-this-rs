# [Beat This! Rust](https://github.com/danigb/beat-this-rs)

A Rust port of the "Beat This!" AI-powered beat tracking system from Johannes Kepler University Linz.

The port was made with [Claude](https://claude.ai/)

## Overview

This is a Rust implementation of the Beat This! model, originally published at ISMIR 2024. The system uses a transformer-based neural network to detect musical beats and downbeats in audio files with high accuracy.

**Original Paper**: "Beat This! Accurate and Generalizable Beat Tracking"
**Original Repository**: https://github.com/CPJKU/beat_this
**C++ Port**: https://github.com/mosynthkey/beat_this_cpp

## Features

- **Two Runtime Backends**: Choose between `rten` (pure Rust, zero external dependencies) or `ort` (ONNX Runtime with CoreML acceleration on macOS)
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

## Building

### Prerequisites

- Rust toolchain (1.70+)
- ONNX model files in the `models/` directory (see [Model Setup](#model-setup))

### Development Build

```bash
git clone git@github.com:danigb/beat-this-rs.git
cd beat-this-rs
cargo build
```

### Production Build

For optimized performance, build in release mode:

```bash
cargo build --release
```

The release binary will be at `target/release/beat-this`. Release mode enables compiler optimizations (LTO, native CPU features) that significantly improve inference speed.

### Using the ORT Runtime

The default `rten` runtime requires no external dependencies. If you want to use the `ort` runtime (for CoreML acceleration on Apple Silicon), install ONNX Runtime first:

```bash
# macOS
brew install onnxruntime

# Or download from https://github.com/microsoft/onnxruntime/releases
```

Then run with `--runtime ort`.

## Model Setup

The mel spectrogram model (`models/mel_spectrogram.onnx`) is included in the repository. To download and convert the beat tracking model, run:

```bash
./scripts/download-models.sh
```

This requires Python 3.10+ and [uv](https://docs.astral.sh/uv/). Alternatively, download and convert manually:

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

## Usage

### Command Line Interface

**Default output (JSON to stdout)**:

```bash
beat-this input.wav
```

**Plain text beats**:

```bash
beat-this input.wav --output-beats
```

**Generate a click track WAV**:

```bash
beat-this input.wav --output-click clicks.wav
```

**Mixed audio (original + click track)**:

```bash
beat-this input.wav --output-mixed mixed.wav
```

**Show BPM**:

```bash
beat-this input.wav --bpm
```

**Use the small model**:

```bash
beat-this input.wav --model-variant small
```

**Use the ORT runtime with verbose timing**:

```bash
beat-this input.wav --runtime ort -v
```

**Batch processing a directory**:

```bash
beat-this ./music-folder/ -r
```

### CLI Options

| Option                              | Description                                             |
| ----------------------------------- | ------------------------------------------------------- |
| `<input>`                           | Audio file or directory                                 |
| `--model <PATH>`                    | Beat model path (default: `models/beat_this.onnx`)      |
| `--mel-model <PATH>`                | Mel model path (default: `models/mel_spectrogram.onnx`) |
| `--model-variant <standard\|small>` | Model size variant                                      |
| `--runtime <rten\|ort>`             | Inference backend (default: `rten`)                     |
| `--output-beats`                    | Print plain text instead of JSON                        |
| `--output-click <PATH>`             | Write click track WAV                                   |
| `--output-mixed <PATH>`             | Write mixed audio WAV                                   |
| `--bpm`                             | Print estimated BPM                                     |
| `-r, --recursive`                   | Recurse into subdirectories                             |
| `-v, --verbose`                     | Print timing for each stage                             |
| `--threads <N>`                     | ORT intra-op threads (0 = auto)                         |
| `--profile <PREFIX>`                | ORT profiling trace output                              |

### Rust API

```rust
use std::path::Path;
use beat_this::{BeatThis, runtime::rten::RtenRuntime};

// Initialize with the pure-Rust runtime
let runtime = RtenRuntime;
let mut bt = BeatThis::new(
    &runtime,
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

### Plain Text Beats (`--output-beats`)

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

### Click Track (`--output-click`)

Generated 44100 Hz mono WAV with:

- **Downbeats**: 880 Hz sine wave
- **Other beats**: 440 Hz sine wave
- ADSR envelope shaping

### Mixed Audio (`--output-mixed`)

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

## Acknowledgments

- **Original Beat This! Implementation**: This Rust port is based on the Beat This! model from Johannes Kepler University Linz. See the [original repository](https://github.com/CPJKU/beat_this) for the research paper and licensing terms.
- **C++ Port**: [beat_this_cpp](https://github.com/mosynthkey/beat_this_cpp) by mosynthkey, which served as a reference for this implementation.
- **Dependencies**: rten, ort, symphonia, rubato, and the broader Rust ecosystem.
