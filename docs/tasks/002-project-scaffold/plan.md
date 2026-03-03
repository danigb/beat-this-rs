# Plan: Project Scaffold + Runtime Trait + Ort Backend

## Context

This implements **Step 1** from [plan.md](docs/task/001-research/plan.md) — the foundational scaffold for `beat-this-rs`. The project currently has no Rust code (no `Cargo.toml`, no `src/`). This step establishes the project structure, defines the runtime abstraction traits, and implements the first (default) ort backend. Everything else in the pipeline (audio, mel, inference, postprocessing) will build on this foundation.

## Files to Create

### 1. `Cargo.toml`

- `[package]` with name `beat-this`, edition 2021
- Binary target: `src/main.rs`
- Library target: `src/lib.rs`
- Dependencies: `ort = "=2.0.0-rc.10"` with features `["load-dynamic", "ndarray"]`, `ndarray = "0.16"`, `anyhow = "1"`
- Note: `load-dynamic` is used by the reference project and allows runtime linking of onnxruntime shared library

### 2. `src/lib.rs`

- Declare modules: `pub mod runtime;`
- Re-export key types: `Tensor`, `InferenceSession`, `InferenceRuntime`

### 3. `src/main.rs`

- Minimal placeholder binary that prints a message
- Will be expanded in Step 8 (CLI)

### 4. `src/runtime.rs` — Core trait definitions

Define the runtime abstraction exactly as specified in [plan.md:37-58](docs/task/001-research/plan.md):

```rust
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub trait InferenceSession {
    fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Result<HashMap<String, Tensor>>;
}

pub trait InferenceRuntime {
    type Session: InferenceSession;
    fn load_model(&self, path: &Path) -> Result<Self::Session>;
}
```

One change from plan: `inputs` takes `&Tensor` (borrowed) instead of owned `Tensor` to avoid unnecessary cloning — the ort backend needs to copy into ndarray anyway.

Also declare: `pub mod ort;` (the backend submodule).

### 5. `src/runtime/ort.rs` — Ort backend implementation

**`OrtRuntime`** struct with config fields:

- `optimization_level: GraphOptimizationLevel` (default: Level3)
- `intra_threads: usize` (default: 1)

Implements `InferenceRuntime`:

- `load_model()`: `Session::builder()?.with_optimization_level(...)?.with_intra_threads(...)?.commit_from_file(path)?`

**`OrtSession`** struct wrapping `ort::session::Session`.

Implements `InferenceSession`:

- Convert each input `Tensor` → `ndarray::ArrayD<f32>` → `Value::from_array()`
- Build named inputs using `ort::inputs!` macro
- Run session, extract outputs via `try_extract_tensor::<f32>()`
- Convert output shapes and data back to `Tensor`

Reference patterns from [remixatron_rust mel.rs and inference.rs](references/remixatron_rust/src-tauri/src/beat_tracker/).

### 6. `tests/ort_integration.rs` — Integration test

- Test loads an ONNX model from `references/remixatron_rust/src-tauri/` (the mel spectrogram model)
- If model file doesn't exist, skip test gracefully (references/ is gitignored)
- Verify: session loads, inference runs, output tensor has expected shape
- This validates the full Tensor ↔ ort conversion round-trip

## Verification

1. `cargo build` — project compiles without errors
2. `cargo test` — unit tests pass (if ONNX models are available, integration test also passes)
3. `cargo run` — binary prints placeholder message
