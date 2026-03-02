# Beat This! Rust Port — Implementation Plan

## Architecture

```
beat-this-rs/
├── Cargo.toml
├── models/                     # ONNX model files (gitignored, downloaded)
│   ├── mel_spectrogram.onnx
│   ├── beat_this.onnx          # standard (~83 MB)
│   └── beat_this_small.onnx    # small (~8 MB)
│
└── src/
    ├── lib.rs                  # Public API: BeatThis struct
    ├── audio.rs                # Audio loading + resampling
    ├── mel.rs                  # Mel spectrogram (ONNX-based)
    ├── inference.rs            # Chunked beat/downbeat inference
    ├── postprocessing.rs       # Peak picking → timestamps
    ├── runtime.rs              # Tensor, InferenceSession, InferenceRuntime traits
    ├── runtime/
    │   ├── ort.rs              # ort backend (default)
    │   ├── rten.rs             # rten backend (future: pure Rust)
    │   └── tract.rs            # tract backend (future: ARM-optimized)
    ├── output.rs               # Beat file + click track WAV output
    └── main.rs                 # CLI binary
```

---

## Runtime Abstraction (first-class concern)

The inference backend must be swappable. Both the mel model and the beat model go through the same trait.

### Trait design

```rust
/// Simple f32 tensor with shape (row-major / C-order).
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// A loaded model session ready for inference.
pub trait InferenceSession {
    /// Run inference with named inputs, return named outputs.
    fn run(
        &mut self,
        inputs: &[(&str, Tensor)],
    ) -> Result<HashMap<String, Tensor>>;
}

/// Factory for creating sessions from ONNX model files.
pub trait InferenceRuntime {
    type Session: InferenceSession;

    fn load_model(&self, path: &Path) -> Result<Self::Session>;
}
```

### Design rationale

- **`Tensor`** is a simple owned type (shape + flat f32 data). No ndarray/rten-tensor/tract dependency at the trait boundary. Each backend converts internally.
- **Named I/O** (`&str` keys) matches ONNX semantics. Backends that use positional I/O (tract, rten) maintain internal name↔index mappings built from model metadata at load time.
- **`&mut self` on `run`** — all three target backends use `&self` internally, but `&mut self` is less restrictive for implementors and our pipeline is sequential.
- **`InferenceRuntime`** is the factory. Runtime-specific config (threads, optimization level) is set on the concrete runtime struct before it's used as a factory, not in the trait.

### Backend adapter notes

**ort** (straightforward):
- Named inputs/outputs map directly to ort's API (`ort::inputs!["name" => value]`, `outputs["name"]`).
- `Tensor.data` → `ndarray::Array` → `ort::Value::from_array()` for input.
- `value.try_extract_tensor::<f32>()` → copy into `Tensor` for output.
- Config: `OrtRuntime` struct holds optimization level, thread count, passed to `Session::builder()`.

**rten** (name→NodeId mapping):
- At load time, call `model.node_id("name")` for all model inputs/outputs, store as `HashMap<String, NodeId>`.
- On `run`: look up `NodeId` from input name, create `rten_tensor::NdTensor` from `Tensor.data`, pass as `(NodeId, ValueOrView)`.
- Output `Value` → `into_tensor::<f32>()` → copy shape + data into `Tensor`. Reverse-map `NodeId` → name for output keys.
- Config: `RtenRuntime` struct holds optimization/prepacking flags, passed to `ModelOptions`.

**tract-onnx** (name→index mapping + input facts):
- At load time: read ONNX model metadata to build name→index mapping for inputs and outputs. Use the model's own shape annotations (including dynamic dims as symbolic) for `with_input_fact()`.
- On `run`: order input tensors by index, call `plan.run(tvec!(...))`. Map output indices back to names.
- Config: `TractRuntime` struct holds thread count, passed via `PlanOptions` / `Executor`.
- **Special concern**: tract wants input shapes at optimization time. The backend reads these from the ONNX file's input metadata. Dynamic dimensions (like variable sequence length) become symbolic dims, which tract handles natively.

### Runtime-specific config

Each concrete runtime struct carries its own config. This stays out of the trait:

```rust
// Example: ort backend
pub struct OrtRuntime {
    pub optimization_level: GraphOptimizationLevel,  // Level1..Level3
    pub intra_threads: usize,                         // default: 1
}

// Example: rten backend
pub struct RtenRuntime {
    pub enable_optimization: bool,
    pub prepack_weights: bool,
}

// Example: tract backend
pub struct TractRuntime {
    pub num_threads: Option<usize>,  // None = single-threaded
}
```

Callers configure the runtime, then use it generically:

```rust
fn process<R: InferenceRuntime>(runtime: &R, mel_path: &Path, beat_path: &Path) {
    let mel_session = runtime.load_model(mel_path)?;
    let beat_session = runtime.load_model(beat_path)?;
    // ...
}
```

---

## Implementation Steps

### Step 1: Project scaffold + runtime trait

- `cargo init --lib` with binary target
- Define `Tensor`, `InferenceSession`, `InferenceRuntime` in `runtime.rs`
- Implement `OrtRuntime` + `OrtSession` in `runtime/ort.rs`
- Deps: `ort`, `anyhow`
- Test: load a small ONNX model, run inference, verify output shape

### Step 2: Audio loading

- `audio.rs`: `load_audio(path, target_sr) -> Result<AudioData>`
- `AudioData`: `{ samples: Vec<f32>, sample_rate: u32 }`
- Decode with symphonia (MP3/WAV/FLAC/OGG)
- Convert to mono (average channels)
- Resample to target_sr with rubato (sinc, high quality)
- Deps: `symphonia`, `rubato`
- Reference: `remixatron_rust/.../audio/loader.rs`
- Test: load a WAV file, verify sample count and rate

### Step 3: Mel spectrogram (ONNX model)

- `mel.rs`: `MelProcessor` struct
- Takes `&mut S: InferenceSession` (session loaded externally)
- Input: `&[f32]` PCM samples → `Tensor` shape `[1, N]`
- Output: `Tensor` shape `[1, T, 128]` → `Vec<Vec<f32>>` or flat with shape
- Input name: `"audio_pcm"`, output name: `"mel_spectrogram"`
- Reference: `remixatron_rust/.../beat_tracker/mel.rs`
- Test: compare output shape against expected frame count (samples / hop_length)

### Step 4: Chunked inference

- `inference.rs`: `BeatProcessor` struct
- Takes `&mut S: InferenceSession`
- Implements chunk splitting (1500 frames, 6 border, 1488 stride)
- For each chunk: zero-pad → run model → extract logits → trim borders
- Aggregate in reverse order (keep_first)
- Input name: `"mel_spectrogram"`, output names: `"beat"`, `"downbeat"`
  - Note: verify actual output names from the ONNX model; may differ between mel-model and beat-model variants
- Returns: `(Vec<f32>, Vec<f32>)` — beat logits, downbeat logits
- Reference: `remixatron_rust/.../beat_tracker/inference.rs`, `beat_this_cpp/Source/InferenceProcessor.cpp`
- Test: verify output length matches input spectrogram length

### Step 5: Post-processing

- `postprocessing.rs`: `PostProcessor` struct
- Max-pool with kernel=7, stride=1, padding=3
- Threshold at logit > 0
- Deduplicate adjacent peaks (running mean within 1 frame)
- Convert frame indices to seconds (÷ fps)
- Snap downbeats to nearest beat
- Remove duplicate downbeat times
- Returns: `BeatResult { beats: Vec<f32>, downbeats: Vec<f32> }`
- Reference: `remixatron_rust/.../beat_tracker/post_processor.rs`, `beat_this_cpp/Source/Postprocessor.cpp`
- Test: known logit sequence → expected beat times

### Step 6: Public API

- `lib.rs`: `BeatThis` struct
- Constructor: `BeatThis::new(mel_model_path, beat_model_path)` — loads both sessions via runtime
- `process_file(path) -> Result<BeatResult>` — full pipeline
- `process_audio(samples, sample_rate) -> Result<BeatResult>` — from raw audio
- `BeatResult`: `{ beats: Vec<f32>, downbeats: Vec<f32> }`
- Optionally: `beat_counts: Vec<i32>` (1=downbeat, 2-N=beat in measure)

### Step 7: Output generation

- `output.rs`:
  - `write_beats_file(path, result)` — tab-separated `time\tbeat_count` (`.beats` format)
  - `write_click_track(path, result, duration, sample_rate)` — WAV with click sounds
    - 880 Hz sine for downbeats, 440 Hz for beats
    - ADSR envelope (10ms attack, 50ms decay, 100ms duration)
    - Use `hound` for WAV writing
  - `write_mixed_audio(path, original_audio, result, sample_rate)` — original + clicks mixed
- Reference: `beat_this_cpp/Source/main.cpp`
- Deps: `hound`

### Step 8: CLI

- `main.rs`: clap-based CLI
- Args: `<audio_file>`, `--model <path>`, `--mel-model <path>`
- Output flags: `--output-beats <file>`, `--output-audio <file>`, `--output-mixed <file>`
- Optional: `--calc-bpm` (median inter-beat interval → BPM)
- Model variant selection: `--model-variant <standard|small>` or auto-detect from path
- Deps: `clap`

---

## Crate Dependencies

```toml
[dependencies]
# Runtime abstraction + default backend
ort = "=2.0.0-rc.10"
ndarray = "0.16"            # Internal use in ort backend only

# Audio
symphonia = { version = "0.5", features = ["all"] }
rubato = "0.14"

# Output
hound = "3.5"

# CLI
clap = { version = "4", features = ["derive"] }

# Core
anyhow = "1"
```

---

## Validation Plan

1. **Unit tests per module**: each step above includes test criteria
2. **Integration test**: full pipeline on a reference audio file, compare beat timestamps against Python/C++ output (tolerance: ±20ms / 1 frame)
3. **Cross-runtime test**: once a second backend exists, run same audio through both, verify identical results
4. **Performance benchmark**: time each pipeline stage separately (audio load, mel, inference, postprocessing) using `criterion`

---

## Future Work (not in initial scope)

- `rten` backend — pure Rust, near-ort CPU performance, best candidate for second backend and benchmarking
- `tract` backend — pure Rust, ARM-optimized, proven at Sonos
- Custom mel spectrogram implementation (realfft-based, Option B)
- GPU acceleration (ort CoreML/CUDA execution providers)
- Cross-runtime benchmarks (ort vs rten vs tract on same audio, per-stage timing)
- WASM target (rten or tract backend)
- Streaming/real-time processing
- Python bindings (PyO3)
