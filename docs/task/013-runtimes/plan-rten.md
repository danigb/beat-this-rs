# Task 013: Add rten Backend — Development Plan

Add `rten` as a pure-Rust alternative to `ort`, selectable via a Cargo feature and CLI flag.

## Goal

```
cargo run -- audio.mp3                                  # ort (default)
cargo run --features rten -- audio.mp3 --runtime rten   # rten (pure Rust)
```

Both backends produce identical beat/downbeat output for the same audio (within floating-point tolerance). The `rten` dependency is optional — users who don't need it pay no cost.

---

## Why rten

- **Pure Rust**: no C++ dylib, no `load-dynamic` headaches, ~20 MB smaller
- **Near-ort CPU performance**: ~80–100% on transformers (Whisper, BERT benchmarked)
- **Direct `.onnx` loading**: since v0.23, no format conversion needed
- **WASM-ready**: compiles anywhere Rust compiles
- **Transformer-proven**: tested on Whisper, Llama 3, GPT-2, ModernBERT, CLIP

---

## rten API Reference (v0.24)

The key types we need:

```rust
// Loading
Model::load_file("model.onnx") -> Result<Model, LoadError>

// Node inspection
model.input_ids()  -> &[NodeId]
model.output_ids() -> &[NodeId]
model.node_info(id) -> Option<NodeInfo>  // .name() gives the string name
model.node_id("name") -> Result<NodeId, RunError>

// Inference
model.run(
    inputs: Vec<(NodeId, ValueOrView<'_>)>,
    outputs: &[NodeId],
    opts: Option<RunOptions>,
) -> Result<Vec<Value>, RunError>

// Tensor I/O
Value::from_shape([1, 1500, 128], data_vec) -> Result<Value>
value.into_shape_vec::<f32, 2>() -> Result<([usize; N], Vec<f32>)>
value.into_tensor::<f32>() -> Option<Tensor<f32>>
```

---

## Implementation Steps

### Step 1: Update `Cargo.toml`

```toml
[dependencies]
rten = { version = "0.24", optional = true }

[features]
default = []
rten = ["dep:rten"]
```

No other dependency changes. The `rten` crate is pure Rust — no system libs, no build scripts.

### Step 2: Create `src/runtime/rten.rs`

```rust
use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use rten::{Model, NodeId, Value as RtenValue};

use super::{InferenceRuntime, InferenceSession, Tensor};

/// Pure-Rust ONNX inference runtime backed by rten.
pub struct RtenRuntime;

impl InferenceRuntime for RtenRuntime {
    type Session = RtenSession;

    fn load_model(&self, path: &Path) -> Result<RtenSession> {
        let model = Model::load_file(path)?;

        // Build name→NodeId maps from model metadata
        let input_map: HashMap<String, NodeId> = model
            .input_ids()
            .iter()
            .filter_map(|&id| {
                model.node_info(id).and_then(|info| {
                    info.name().map(|n| (n.to_string(), id))
                })
            })
            .collect();

        let output_map: HashMap<String, NodeId> = model
            .output_ids()
            .iter()
            .filter_map(|&id| {
                model.node_info(id).and_then(|info| {
                    info.name().map(|n| (n.to_string(), id))
                })
            })
            .collect();

        let output_ids: Vec<NodeId> = model.output_ids().to_vec();

        Ok(RtenSession {
            model,
            input_map,
            output_map,
            output_ids,
        })
    }
}

pub struct RtenSession {
    model: Model,
    input_map: HashMap<String, NodeId>,    // "mel_spectrogram" → NodeId
    output_map: HashMap<String, NodeId>,   // "beat" → NodeId
    output_ids: Vec<NodeId>,               // ordered output node IDs
}

impl InferenceSession for RtenSession {
    fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Result<HashMap<String, Tensor>> {
        // Convert named inputs to (NodeId, ValueOrView) pairs
        let rten_inputs: Vec<(NodeId, RtenValue)> = inputs
            .iter()
            .map(|(name, tensor)| {
                let node_id = self.input_map.get(*name)
                    .ok_or_else(|| anyhow!("unknown input: {}", name))?;
                let value = RtenValue::from_shape(
                    tensor.shape.as_slice(),
                    tensor.data.clone(),
                )?;
                Ok((*node_id, value))
            })
            .collect::<Result<Vec<_>>>()?;

        // Run inference
        let inputs_with_views: Vec<(NodeId, _)> = rten_inputs
            .iter()
            .map(|(id, val)| (*id, val.into()))
            .collect();

        let outputs = self.model.run(
            inputs_with_views,
            &self.output_ids,
            None,
        )?;

        // Convert outputs to named Tensor map
        // Reverse-map: iterate output_ids in order, match with output Values
        let mut result = HashMap::new();
        for (id, value) in self.output_ids.iter().zip(outputs.into_iter()) {
            // Find the name for this output NodeId
            let name = self.output_map.iter()
                .find(|(_, nid)| *nid == id)
                .map(|(name, _)| name.clone())
                .unwrap_or_else(|| format!("output_{}", id));

            // Extract f32 tensor data
            let rten_tensor = value.into_tensor::<f32>()
                .ok_or_else(|| anyhow!("output {} is not f32", name))?;
            let shape: Vec<usize> = rten_tensor.shape().to_vec();
            let data: Vec<f32> = rten_tensor.into_data();

            result.insert(name, Tensor { shape, data });
        }

        Ok(result)
    }
}
```

**Key details**:
- Name→NodeId maps are built once at load time (not per inference call)
- `Value::from_shape` accepts `&[usize]` + `Vec<f32>` — matches our `Tensor` layout exactly
- `into_tensor::<f32>()` + `into_data()` extracts the flat Vec back out
- No threading config — rten uses Rayon internally, defaults to physical core count
- `&self` on `Model::run` (rten) vs `&mut self` on our trait — compatible since `&mut` is more restrictive

### Step 3: Wire the module in `src/runtime.rs`

Add after `pub mod ort;`:

```rust
#[cfg(feature = "rten")]
pub mod rten;
```

No trait changes, no changes to `Tensor`.

### Step 4: Add `--runtime` CLI flag in `src/main.rs`

**4a.** Add a `Runtime` enum:

```rust
#[derive(Clone, Default, clap::ValueEnum)]
enum Runtime {
    #[default]
    Ort,
    #[cfg(feature = "rten")]
    Rten,
}
```

**4b.** Add the flag to `Cli`:

```rust
/// Inference runtime to use
#[arg(long = "runtime", default_value_t = Runtime::Ort)]
runtime: Runtime,
```

**4c.** Extract common pipeline into a generic helper:

```rust
fn run_pipeline<S: InferenceSession>(bt: &mut BeatThis<S>, cli: &Cli) -> Result<()> {
    // Audio loading, mel, inference, postprocessing, output
    // (move the existing body of main() here, minus model loading)
}
```

**4d.** Dispatch in `main()`:

```rust
match cli.runtime {
    Runtime::Ort => {
        let runtime = OrtRuntime { intra_threads: cli.threads, ..Default::default() };
        let mut bt = BeatThis::new(&runtime, &mel_path, &beat_path)?;
        run_pipeline(&mut bt, &cli)?;
    }
    #[cfg(feature = "rten")]
    Runtime::Rten => {
        let runtime = RtenRuntime;
        let mut bt = BeatThis::new(&runtime, &mel_path, &beat_path)?;
        run_pipeline(&mut bt, &cli)?;
    }
}
```

ORT-specific code (profiling, CoreML info) stays in the `Runtime::Ort` branch only.

### Step 5: Integration test (`tests/rten_integration.rs`)

```rust
#![cfg(feature = "rten")]

use beat_this::runtime::rten::RtenRuntime;
use beat_this::runtime::InferenceRuntime;
use beat_this::BeatThis;

#[test]
fn test_rten_full_pipeline() {
    // Skip if model files missing
    let mel_path = std::path::Path::new("models/mel_spectrogram.onnx");
    let beat_path = std::path::Path::new("models/beat_this.onnx");
    if !mel_path.exists() || !beat_path.exists() {
        eprintln!("Skipping: model files not found");
        return;
    }

    let runtime = RtenRuntime;
    let mut bt = BeatThis::new(&runtime, mel_path, beat_path).unwrap();
    let result = bt.process_file("tests/fixtures/test.mp3".as_ref()).unwrap();

    assert!(!result.beats.is_empty(), "should detect beats");
    assert!(!result.downbeats.is_empty(), "should detect downbeats");
}
```

Additional tests mirroring `tests/ort_integration.rs`:
- Model loading (mel + beat separately)
- Inference on short/long synthetic spectrograms
- Shape validation

### Step 6: Cross-runtime consistency test

In `tests/cross_runtime.rs` (runs only when both are available):

```rust
#![cfg(feature = "rten")]

// Run same audio through ort and rten, compare beat timestamps
// Tolerance: ±1 frame (20ms at 50 fps)
```

This catches operator implementation differences. If beats match within 1 frame, the backends are interchangeable.

### Step 7: Verify

- `cargo build` — compiles without rten (default features)
- `cargo build --features rten` — compiles with rten
- `cargo test` — ort tests pass, no regressions
- `cargo test --features rten` — rten tests pass
- Manual: `cargo run --features rten -- test.mp3 --runtime rten -v` — prints per-stage timing

---

## File Changes

| File | Action |
|------|--------|
| `Cargo.toml` | Add `rten` optional dep + feature flag |
| `src/runtime.rs` | Add `#[cfg(feature = "rten")] pub mod rten;` |
| `src/runtime/rten.rs` | **New** — `RtenRuntime` + `RtenSession` |
| `src/main.rs` | Add `--runtime` flag, extract `run_pipeline`, dispatch |
| `tests/rten_integration.rs` | **New** — rten backend tests |
| `tests/cross_runtime.rs` | **New** — ort vs rten consistency |

No changes to: traits, `Tensor`, `lib.rs`, `inference.rs`, `mel.rs`, `postprocessing.rs`, `audio.rs`, `output.rs`.

---

## Risks

1. **Operator coverage**: the Beat This! model uses attention layers (LayerNorm, MatMul, Softmax, multi-head attention). rten has tested on Whisper/BERT/Llama but not this specific model. **Mitigation**: try loading the model early (Step 2) — if an op is missing, rten will error with the op name and we'll know immediately.

2. **`Value::from_shape` with dynamic rank**: rten's `from_shape` accepts `&[usize]`, which should handle our 3D tensors (`[1, T, 128]`). Verify that dynamic-rank values work with `model.run()` (vs fixed-rank `NdTensor`).

3. **Numerical drift**: rten's MatMul/LayerNorm may differ from ort by small epsilon. The cross-runtime test uses post-processed beat timestamps (not raw logits) with ±1 frame tolerance, which is generous enough to absorb this.

4. **rten version pinning**: use `rten = "0.24"` (latest). Direct `.onnx` loading requires ≥0.23. Pin to avoid accidental downgrades.
