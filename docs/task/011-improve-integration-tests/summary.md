# Task 011: Integration Tests & Performance — Summary

## What was done

### 1. Integration test script: `scripts/integration-test.py`

Self-contained Python script with uv inline metadata (PEP 723). Dependencies (torch, torchaudio, numpy, soundfile, etc.) are resolved automatically by uv.

```bash
uv run scripts/integration-test.py
```

For each audio file in `integration_test_files/`:

- Runs Python `beat_this` CLI (`--no-dbn --gpu -1` for fair comparison)
- Runs Rust `beat-this` binary (release build)
- Compares `.beats` output with ±0.02s tolerance
- Writes `.python-time.txt`, `.rust-time.txt`, and `.differences.txt` (if any)
- Prints summary with match count, failure count, and speedup ratio

Both versions use the same model: `final0` checkpoint → `models/beat_this.onnx` (converted via `ckpt2onnx.py`). Verified byte-identical.

### 2. Verbose timing mode: `--verbose` / `-v`

Added `--verbose` flag to the Rust CLI that prints per-stage timing to stderr:

```
[info] CoreML available: no
[timing] Model loading: 0.374s
[timing] Audio loading: 0.533s (5998399 samples, 272.0s duration)
[timing] Mel spectrogram: 0.228s (13602 frames)
[timing] Beat inference: 14.956s
[timing] Post-processing: 0.000s
[timing] Total: 16.092s
```

Required making `BeatThis` struct fields public (`mel`, `inference`, `post`) to call each stage individually from `main.rs`.

### 3. Dynamic linking of ONNX Runtime

Switched from static linking to `load-dynamic` feature in the `ort` crate. The binary now loads `libonnxruntime.dylib` at runtime via `ORT_DYLIB_PATH`.

```toml
ort = { version = "=2.0.0-rc.10", features = ["ndarray", "load-dynamic", "coreml"] }
```

Benefits: faster compilation, smaller binary, can swap ORT versions without recompiling.

Homebrew's `onnxruntime` (1.24.2) works. The `ort` crate targets 1.22.0 but is ABI-compatible.

### 4. CoreML evaluation

Added CoreML execution provider support. The runtime automatically tries CoreML when available, falling back to CPU.

Tested with the official ORT 1.22.0 macOS ARM64 release (which includes CoreML):

| Config                  | Model Load | Beat Inference | Total |
| ----------------------- | ---------- | -------------- | ----- |
| CPU only (homebrew ORT) | 0.37s      | **15.2s**      | 16.3s |
| CoreML NeuralNetwork    | 2.4s       | 28.7s          | 31.8s |
| CoreML MLProgram        | 11.1s      | 62.9s          | 74.7s |

**CoreML is slower for this model.** The transformer architecture with dynamic axes doesn't map well — CoreML adds compilation overhead without accelerating compute. CPU remains the best path.

## Performance profile

For a 4.5-minute track

| Stage              | Time      | % of total |
| ------------------ | --------- | ---------- |
| Model loading      | 0.37s     | 2%         |
| Audio loading      | 0.53s     | 3%         |
| Mel spectrogram    | 0.23s     | 1%         |
| **Beat inference** | **15.0s** | **93%**    |
| Post-processing    | <0.001s   | 0%         |

The bottleneck is transformer inference in ONNX Runtime's CPU kernels. Python (PyTorch CPU) completes the same track in ~5s — roughly 3x faster, likely due to more optimized attention/matmul kernels.

## Files changed

- `scripts/integration-test.py` — new integration test script
- `src/main.rs` — `--verbose` flag with per-stage timing
- `src/lib.rs` — made `BeatThis` fields public
- `src/runtime/ort.rs` — dynamic linking, CoreML EP support
- `Cargo.toml` — `load-dynamic` and `coreml` features
