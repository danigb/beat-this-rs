# Task 013: Multiple ONNX Runtimes — Development Plan

Add `rten` and `tract-onnx` as alternative inference backends behind the existing `InferenceRuntime` / `InferenceSession` traits, selectable via Cargo features and a CLI flag.

## Goal

A user can run:

```
cargo run --features rten -- audio.mp3 --runtime rten
cargo run --features tract -- audio.mp3 --runtime tract
cargo run -- audio.mp3                           # ort, the default
```

All three backends produce identical beat/downbeat output for the same audio file (within floating-point tolerance). The `ort` crate remains the default and only mandatory dependency.

---

## Design Decisions

### 1. Cargo features for backend selection

Each non-default backend is gated behind a feature flag so users don't pull unnecessary dependencies:

```toml
[features]
default = []
rten = ["dep:rten"]
tract = ["dep:tract-onnx"]
```

`ort` stays unconditional for now (it's the production default). The `rten` and `tract` modules compile only when their feature is enabled.

### 2. Runtime selection at the CLI level

The CLI gets a `--runtime <ort|rten|tract>` flag. The binary dispatches to the selected backend. Since `BeatThis<S>` is generic over the session type, each branch constructs the right runtime and calls the same pipeline code. No dynamic dispatch (`Box<dyn>`) needed — the match arms are monomorphized.

### 3. No trait changes needed

The existing trait surface is sufficient for all three backends:

```rust
pub trait InferenceSession {
    fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Result<HashMap<String, Tensor>>;
}

pub trait InferenceRuntime {
    type Session: InferenceSession;
    fn load_model(&self, path: &Path) -> Result<Self::Session>;
}
```

All three runtimes support: runtime `.onnx` loading, named tensor I/O, f32 inference, dynamic input shapes. The `Tensor` type (shape + flat f32 data) is the common currency at the trait boundary; each backend converts internally.

### 4. Backend-specific session extras

`OrtSession` currently exposes `end_profiling()` outside the trait. Each backend may have its own extras (e.g., rten's `model.op_count()`). These live on the concrete session type, not on the trait. The CLI uses downcasting or conditional compilation to access them.

---

## Implementation Steps

### Step 1: Add `rten` backend (`src/runtime/rten.rs`)

**Crate**: `rten = "0.15"` (or latest stable that supports `.onnx` loading — verify version)

**Implementation**:

```rust
pub struct RtenRuntime;

pub struct RtenSession {
    model: rten::Model,
    // name→NodeId mappings built at load time
    input_ids: HashMap<String, NodeId>,
    output_ids: HashMap<String, NodeId>,
}
```

- `RtenRuntime::load_model`: load the `.onnx` file with `rten::Model::load_file(path)`. Build name→NodeId maps from `model.input_ids()` / `model.output_ids()` and their names.
- `RtenSession::run`: for each named input, look up `NodeId`, convert `Tensor` → `rten_tensor::NdTensor<f32, D>`, pass as `(NodeId, InputOrOutput)`. Run `model.run(...)`. Convert output `Tensor` back to our `Tensor` type, reverse-map NodeId → name.
- No threading config needed — rten uses Rayon internally.

**Feature gate**: `#[cfg(feature = "rten")]` on the module and `pub mod rten;` in `runtime.rs`.

### Step 2: Add `tract` backend (`src/runtime/tract.rs`)

**Crate**: `tract-onnx = "0.21"` (or latest stable)

**Implementation**:

```rust
pub struct TractRuntime;

pub struct TractSession {
    plan: TypedSimplePlan<TypedModel>,
    input_names: Vec<String>,   // ordered by model input index
    output_names: Vec<String>,  // ordered by model output index
}
```

- `TractRuntime::load_model`: load with `tract_onnx::onnx().model_for_path(path)`. Read input/output names from the model metadata. Use `into_optimized()?.into_runnable()` to get a `TypedSimplePlan`. Store ordered name vectors.
- `TractSession::run`: reorder inputs by name→index mapping, convert `Tensor` → `tract_ndarray::Array` → `tract_data::Tensor`. Run `plan.run(tvec!(...))`. Convert output tensors back, map index→name.
- **Dynamic shapes**: tract wants concrete shapes at optimization time. At load time, read input facts from the ONNX file. For dynamic dims, use `to_dim()` symbolic values or set concrete shapes with `with_input_fact()` using the model's own metadata. Alternatively, optimize with streaming dims.

**Feature gate**: `#[cfg(feature = "tract")]` on the module.

### Step 3: Wire feature-gated modules in `src/runtime.rs`

```rust
pub mod ort;

#[cfg(feature = "rten")]
pub mod rten;

#[cfg(feature = "tract")]
pub mod tract;
```

No changes to trait definitions or `Tensor`.

### Step 4: Add `--runtime` CLI flag in `src/main.rs`

Add a `Runtime` enum and `--runtime` flag:

```rust
#[derive(Clone, clap::ValueEnum)]
enum Runtime {
    Ort,
    #[cfg(feature = "rten")]
    Rten,
    #[cfg(feature = "tract")]
    Tract,
}
```

In `main()`, match on the selected runtime and construct the appropriate `BeatThis<S>`:

```rust
match cli.runtime {
    Runtime::Ort => {
        let runtime = OrtRuntime { ... };
        let mut bt = BeatThis::new(&runtime, &mel_path, &beat_path)?;
        run_pipeline(&mut bt, &cli)?;
    }
    #[cfg(feature = "rten")]
    Runtime::Rten => {
        let runtime = RtenRuntime;
        let mut bt = BeatThis::new(&runtime, &mel_path, &beat_path)?;
        run_pipeline(&mut bt, &cli)?;
    }
    // ...
}
```

Extract the common pipeline logic (audio loading, mel, inference, postprocessing, output) into a generic `run_pipeline<S: InferenceSession>(bt: &mut BeatThis<S>, cli: &Cli)` function to avoid duplicating it per backend. The profiling code (ORT-specific) is conditionally compiled inside the ort branch.

### Step 5: Update `Cargo.toml`

```toml
[dependencies]
ort = { version = "=2.0.0-rc.10", features = ["ndarray", "load-dynamic", "coreml"] }
ndarray = "0.16"
rten = { version = "0.15", optional = true }
tract-onnx = { version = "0.21", optional = true }
# ... rest unchanged

[features]
default = []
rten = ["dep:rten"]
tract = ["dep:tract-onnx"]
```

### Step 6: Integration tests

Add `tests/rten_integration.rs` and `tests/tract_integration.rs`:

- Each is gated with `#![cfg(feature = "rten")]` / `#![cfg(feature = "tract")]`
- Mirror the existing `tests/ort_integration.rs` pattern: load model, run inference on test audio, verify output shape and basic properties
- **Cross-runtime consistency test**: a shared test (in `tests/cross_runtime.rs` or similar) that runs the same audio through all available backends and asserts that beat timestamps match within ±1 frame (20ms at 50 fps)

### Step 7: Verify and benchmark

- `cargo test` — default (ort) tests pass, no regressions
- `cargo test --features rten` — rten tests pass
- `cargo test --features tract` — tract tests pass
- `cargo test --features "rten,tract"` — all tests pass including cross-runtime
- Manual benchmark: run the CLI on the same audio file with each runtime, compare wall-clock times (verbose mode prints per-stage timing)

---

## Cargo.toml Changes

```toml
# New optional dependencies
rten = { version = "0.15", optional = true }
tract-onnx = { version = "0.21", optional = true }

# New feature flags
[features]
default = []
rten = ["dep:rten"]
tract = ["dep:tract-onnx"]
```

## File Changes

| File | Action |
|------|--------|
| `src/runtime.rs` | Add feature-gated `pub mod rten` and `pub mod tract` |
| `src/runtime/rten.rs` | **New** — `RtenRuntime` + `RtenSession` |
| `src/runtime/tract.rs` | **New** — `TractRuntime` + `TractSession` |
| `src/main.rs` | Add `--runtime` flag, extract `run_pipeline` helper, dispatch per backend |
| `Cargo.toml` | Add optional deps + feature flags |
| `tests/rten_integration.rs` | **New** — rten backend tests |
| `tests/tract_integration.rs` | **New** — tract backend tests |

No changes to: `runtime.rs` traits, `Tensor`, `lib.rs`, `inference.rs`, `mel.rs`, `postprocessing.rs`, `audio.rs`, `output.rs`.

---

## Risk Notes

1. **rten `.onnx` loading**: direct ONNX loading was added in rten v0.23. Earlier versions require a custom `.rten` format conversion. Verify the version supports `Model::load_file("model.onnx")` directly.
2. **tract dynamic shapes**: tract prefers known shapes at optimization time. If the mel spectrogram model has a dynamic time axis, tract may need `with_input_fact()` with a symbolic dim or per-invocation model re-optimization. Test with both models (mel and beat) to confirm it handles variable input lengths.
3. **Operator coverage**: the Beat This! model uses attention layers (LayerNorm, MatMul, Softmax, etc.). Both rten and tract claim good transformer support, but test with the actual `.onnx` files early — if an operator is missing, that backend is blocked until upstream adds it.
4. **Numerical differences**: different backends may produce slightly different f32 results due to operator implementations, fused ops, or threading. The cross-runtime test should use a tolerance (e.g., ±1 frame after post-processing) rather than exact equality.

---

## Implementation Order

Start with **rten** (Step 1) — it's the closest to ort in API style (runtime loading, named tensors) and has the best transformer operator coverage among pure-Rust options. Once rten works end-to-end, add **tract** (Step 2). The CLI wiring (Steps 3-4) can happen in parallel with either backend since the trait surface doesn't change.
