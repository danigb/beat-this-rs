# Task 002: Project Scaffold — Summary

Implements **Step 1** from [plan.md](../001-research/plan.md): project scaffold + runtime trait + ort backend.

## What was done

### Project structure

```
beat-this-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API: re-exports Tensor, InferenceSession, InferenceRuntime
│   ├── main.rs             # Placeholder CLI binary
│   ├── runtime.rs          # Tensor struct + trait definitions
│   └── runtime/
│       └── ort.rs          # OrtRuntime + OrtSession (ort backend)
└── tests/
    └── ort_integration.rs  # Integration tests with real ONNX models
```

### Runtime abstraction (`src/runtime.rs`)

- `Tensor` — simple owned f32 tensor (shape + flat data, row-major)
- `InferenceSession` trait — `run(&mut self, inputs) -> HashMap<String, Tensor>`
- `InferenceRuntime` trait — factory: `load_model(path) -> Session`

### Ort backend (`src/runtime/ort.rs`)

- `OrtRuntime` with configurable optimization level and thread count
- `OrtSession` wrapping `ort::Session`
- Converts `Tensor` ↔ `ndarray::ArrayD` ↔ `ort::Value` for I/O

### Dependencies

```toml
ort = { version = "=2.0.0-rc.10", features = ["ndarray"] }
ndarray = "0.16"
anyhow = "1"
```

Note: uses static linking (ort downloads onnxruntime at build time) rather than `load-dynamic`, which would require a pre-installed `libonnxruntime.dylib`.

### Tests (4 passing)

- `test_load_mel_model` — loads mel spectrogram ONNX model
- `test_mel_inference` — runs mel model on 1s silence, verifies output shape `[1, T, 128]`
- `test_load_beat_model` — loads beat tracking ONNX model
- `test_beat_inference` — runs beat model on fake spectrogram, verifies output keys

Tests use models from `references/remixatron_rust/` and skip gracefully if not present.
