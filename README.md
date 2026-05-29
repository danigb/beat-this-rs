# [Beat This! Rust](https://github.com/danigb/beat-this-rs)

[![Crates.io](https://img.shields.io/crates/v/beat-this.svg)](https://crates.io/crates/beat-this)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/danigb/beat-this-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/danigb/beat-this-rs/actions/workflows/ci.yml)

A Rust port of the ["Beat This!"](https://github.com/CPJKU/beat_this) AI beat-tracking
system (ISMIR 2024, Johannes Kepler University Linz). It detects musical **beats** and
**downbeats** in audio with no runtime dependencies beyond the model weights.

Ported with [Claude](https://claude.ai/).

- **Paper**: ["Beat This! Accurate and Generalizable Beat Tracking"](https://arxiv.org/pdf/2407.21658)
- **Original repo**: https://github.com/CPJKU/beat_this · **C++ port**: https://github.com/mosynthkey/beat_this_cpp

## Features

- **Pure-Rust by default** — the `rten` backend needs no system libraries. An `ort`
  (ONNX Runtime) backend is available behind the `ort` Cargo feature
  (`--features ort`) for cross-runtime validation and profiling; it requires
  `libonnxruntime` at runtime.
- **Bundled small model** — clone and run with zero setup; download the full-accuracy
  model when you need it.
- **CLI + library** — a `beat-this` binary and a clean `beat_this` crate API.
- **Many outputs** — JSON, plain-text `.beats`, click-track WAV, mixed audio, and mel
  spectrogram `.npy`.
- **Batch mode** — process directories or globs with a summary file.

## Quick start

No Python toolchain required. The mel model and a small beat model are committed to the
repo, so you can run inference immediately after building:

```bash
git clone git@github.com:danigb/beat-this-rs.git
cd beat-this-rs
cargo build --release

# Run with the bundled small model (zero setup):
./target/release/beat-this input.mp3 --model models/beat_this_small.onnx

# For best accuracy, fetch the full model once — then it's the default:
./scripts/download-models.sh
./target/release/beat-this input.mp3
```

## Install

**As a CLI tool** (from crates.io):

```bash
cargo install beat-this
```

The published crate does **not** bundle the model files. The committed mel + small models
live in the repo's [`models/`](models/) directory, and the full beat model is on
[Releases](https://github.com/danigb/beat-this-rs/releases) (see [Models](#models)). Pass
their paths explicitly:

```bash
beat-this input.mp3 --model beat_this.onnx --mel-model mel_spectrogram.onnx
```

Two models are committed to the repository, so the test suite and a basic run work
with **no setup**:

- `models/mel_spectrogram.onnx` (~270 KB) — log-mel front end.
- `models/beat_this_small.onnx` (~10 MB) — small beat model used by the test suite
  and a good default for quick runs.

The full-accuracy FP32 beat model (`beat_this.onnx`, ~83 MB) is **not** committed.
Fetch it from GitHub Releases with `curl` (no Python):

```bash
./scripts/download-models.sh        # downloads + checksum-verifies models/beat_this.onnx
```

<details>
<summary>Maintainers: regenerating model assets</summary>

The ONNX files are converted from the official Beat This! checkpoints with
`scripts/ckpt2onnx.py`, which needs [uv](https://docs.astral.sh/uv/) (torch + onnx +
onnxscript). End users never run this — it only (re)generates the committed small model
and the release asset:

```bash
uv run scripts/ckpt2onnx.py final0   # FP32 -> upload models/final0.onnx as the beat_this.onnx release asset
uv run scripts/ckpt2onnx.py small1   # small -> commit  models/small1.onnx as beat_this_small.onnx
```

See the [original repo](https://github.com/CPJKU/beat_this#available-models) for available checkpoints.

Every maintainer script under `scripts/*.py` is a self-contained
[PEP 723](https://peps.python.org/pep-0723/) script with its dependencies declared
inline, run as `uv run scripts/<name>.py …` — no virtualenv, `pip`, or
`requirements.txt`. Installing [uv](https://docs.astral.sh/uv/) is the only
prerequisite.

</details>

### Optional: the `ort` backend

The default build uses the pure-Rust `rten` runtime and needs no system libraries. To
also build the ONNX Runtime backend (for `--runtime ort` and `--profile`), enable the
`ort` feature:

```bash
cargo build --release --features ort
```

It loads `libonnxruntime` at runtime — install it (`brew install onnxruntime` on macOS)
or point `ORT_DYLIB_PATH` at the shared library.

## Command-line usage

```bash
# Default: print JSON to stdout
beat-this input.wav

# Write auto-named output files (input.json, input.beats)
beat-this input.wav --json --beats

# Explicit output paths
beat-this input.wav --json=results.json --click=clicks.wav

# Batch a directory (per-file JSON + a beat_this.json summary)
beat-this ./music/ -r --json --beats --mix

# Glob (quote it so the shell doesn't expand it)
beat-this "music/**/*.mp3" --json
```

| Option                  | Description                                                       |
| ----------------------- | ----------------------------------------------------------------- |
| `<input>`               | Audio file, directory, or glob pattern                            |
| `--json [=FILE]`        | Write JSON output (default ext: `.json`)                          |
| `--beats [=FILE]`       | Write beats text file (default ext: `.beats`)                     |
| `--click [=FILE]`       | Write click-track WAV (default ext: `.click.wav`)                 |
| `--mix [=FILE]`         | Write mixed audio WAV (default ext: `.mix.wav`)                   |
| `--mel [=FILE]`         | Write mel spectrogram as numpy `.npy` (default ext: `.mel.npy`)   |
| `--model <PATH>`        | Beat model path (default: `models/beat_this.onnx`)                |
| `--mel-model <PATH>`    | Mel model path (default: `models/mel_spectrogram.onnx`)           |
| `--runtime <rten\|ort>` | Inference backend (default: `rten`; `ort` needs `--features ort`) |
| `--overwrite`           | Overwrite existing output files                                   |
| `-r, --recursive`       | Recurse into subdirectories (batch mode)                          |
| `-v, --verbose`         | Print timing for each stage                                       |
| `--profile <PREFIX>`    | Write ORT profiling trace (requires `--features ort`)             |

## Library usage

```rust
use std::path::Path;
use beat_this::{BeatThis, RtenRuntime};

// Initialize with the pure-Rust runtime and your model paths.
let mut bt = BeatThis::new(
    &RtenRuntime,
    Path::new("models/mel_spectrogram.onnx"),
    Path::new("models/beat_this.onnx"),
)?;

// Analyze an audio file.
let analysis = bt.analyze_file(Path::new("input.wav"))?;

println!("{} beats, {} downbeats", analysis.beats.len(), analysis.downbeats.len());
for (i, &t) in analysis.beats.iter().enumerate() {
    println!("beat {i}: {t:.3}s");
}
```

`analysis.beats` / `analysis.downbeats` are beat times in seconds; `analysis.mel.shape`
is `[1, T, 128]`. To use the ONNX Runtime backend instead, swap `&RtenRuntime` for
`&OrtRuntime::default()` (requires the ONNX Runtime dylib — see [Install](#install)).

## Output formats

### JSON (default)

```json
{
  "beats": [0.34, 0.68, 1.02, 1.36, 1.7, 2.04],
  "downbeats": [0.34, 1.7],
  "bpm": 120.0
}
```

### Plain text (`--beats`)

Tab-separated `time<TAB>beat-number`, where `1` is a downbeat and `2`–`4` are other beats:

```
0.340	4
0.681	1
1.023	2
```

### Click track (`--click`)

44100 Hz mono WAV: 880 Hz sine on downbeats, 440 Hz on other beats, with ADSR shaping.

### Mixed audio (`--mix`)

Original music (70%) blended with the click track (30%).

## How it works

The pipeline has four stages:

1. **Audio** — decode (symphonia: WAV/MP3/FLAC/OGG) and resample (rubato) to 22050 Hz mono.
2. **Mel spectrogram** — a 128-band log-mel front end computed by an ONNX model.
3. **Beat inference** — a transformer runs over overlapping 1500-frame (30 s) chunks.
4. **Post-processing** — peak picking, deduplication, and downbeat-to-beat snapping.

The beat model takes 128-dim mel spectrograms and emits beat/downbeat logits. The
**standard** model is ~83 MB (FP32); the **small** model is ~10 MB.

## Parity with the Python reference

Parity with the original Python [`beat_this`](https://github.com/CPJKU/beat_this) is
**verified by a committed test** (`tests/python_parity.rs`), not just argued by
construction. It runs the full Rust pipeline on a shared audio file and compares the
beat/downbeat times to a golden generated from the Python reference on the matching
model, scored with the standard ±70 ms MIR F-measure:

- **Standard FP32 model:** F-measure == **1.0** for both beats and downbeats — the
  Rust port is faithful to the reference within MIR tolerance.
- **Small model (always-on, runs on a fresh clone with no download):** F-measure ≥
  **0.99**. The small structural model has a handful of logit peaks sitting right at
  the decision threshold, where the pure-Rust `rten` backend and PyTorch differ by an
  epsilon and tip a peak in or out — an irreducible, sub-MIR float difference, not a
  pipeline divergence.

Known, bounded, sub-MIR divergences from the reference: the resampler (`rubato` sinc
vs Python `soxr`, only affects inputs not already at 22050 Hz) and a ≤10 ms rounding
of merged adjacent peaks. Regenerate the golden fixtures with
`scripts/gen_golden.py` (maintainer-only) if the checkpoint or the mel/inference
graph changes; see `tests/fixtures/README.md` for provenance.

## Performance

Apple M4 MacBook Pro (2024), vs. the Python reference (PyTorch, CPU, no DBN):

| File      | Duration | Python | Rust (rten) | Rust (ort) |
| --------- | -------: | -----: | ----------: | ---------: |
| short.wav |      9 s |  1.8 s |       0.7 s |      1.2 s |
| test1.mp3 |     4:32 |  5.1 s |       4.6 s |      4.6 s |
| test2.mp3 |    13:48 | 11.9 s |      12.1 s |     11.9 s |

The two Rust backends agree on timestamps within MIR tolerance (verified on a real
signal by `tests/cross_runtime.rs`) and perform on par with each other.
Runtime is dominated by beat inference, which scales linearly with audio duration.

## Acknowledgments

- **Beat This!** by Johannes Kepler University Linz — see the [original repo](https://github.com/CPJKU/beat_this) for the paper and licensing.
- **[beat_this_cpp](https://github.com/mosynthkey/beat_this_cpp)** by mosynthkey, a reference for this port.
- Built on rten, ort, symphonia, rubato, and the broader Rust ecosystem.

## License

[MIT](LICENSE). This is a Rust port of "Beat This!", which is also MIT-licensed; the
[LICENSE](LICENSE) file retains both the original (Institute of Computational Perception,
JKU Linz) and the port's copyright notices.
