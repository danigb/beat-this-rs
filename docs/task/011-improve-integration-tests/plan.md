# Plan: Integration Test Script

## Goal

Create `integration-test.py` that compares the Python and Rust beat-this implementations on real audio files, measuring both correctness and performance.

## Script: `integration-test.py`

Single Python script at project root. Run with `uv run integration-test.py`.

### Step 1: Build Rust binary

Run `cargo build --release`. Fail fast if build fails. Binary at `target/release/beat-this`.

### Step 2: Discover audio files

Recursively walk `integration_test_files/` for `.mp3`, `.wav`, `.flac`, `.ogg` files. Print count and list.

### Step 3: Process each file

For each audio file `<dir>/<name>.mp3`:

**Python version:**

- Run: `python -m beat_this.cli <file> -o <dir>/<name>.python.beats --dbn false`
- Use `PYTHONPATH=references` so it finds the `beat_this` package
- Measure wall-clock time, write to `<dir>/<name>.python-time.txt`
- The `--dbn false` flag disables DBN post-processing to match Rust behavior

**Rust version:**

- Run: `target/release/beat-this <file> -o <dir>/<name>.rust.beats`
- Uses default models (`models/beat_this.onnx` + `models/mel_spectrogram.onnx`)
- Measure wall-clock time, write to `<dir>/<name>.rust-time.txt`

**Model alignment:** Python `final0` checkpoint and Rust `models/beat_this.onnx` are the same model (converted via `ckpt2onnx.py`). Both use the standard (non-small) variant.

### Step 4: Compare outputs

For each file, compare `.python.beats` and `.rust.beats`:

- Parse both as lists of `(time: float, beat: int)` tuples
- Allow a timestamp tolerance of ±0.01s for floating-point differences
- If beat counts differ or timestamps diverge beyond tolerance, write `<dir>/<name>.differences.txt` with:
  - Line-by-line comparison showing mismatched entries
  - Summary stats (total beats in each, number of differences)

### Step 5: Print summary

After all files:

- Total files processed
- Files with differences vs files matching
- Average/total time for Python vs Rust
- Performance speedup ratio

## Output file structure

```
integration_test_files/
  album_folder/
    track.mp3
    track.python.beats
    track.python-time.txt
    track.rust.beats
    track.rust-time.txt
    track.differences.txt    # only if differences found
```

## Idempotency

The script overwrites existing result files on each run, so it can be re-run at any time to get fresh results.

## Dependencies

- Python with `beat_this` dependencies (torch, soundfile, etc.) available
- Rust toolchain for `cargo build`
- ONNX models in `models/` directory

## Implementation

Single file: `integration-test.py` (~150 lines). Uses only stdlib (`subprocess`, `time`, `pathlib`, `os`).
