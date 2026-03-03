# Task 007: Public API — Development Plan

Implements **Step 6** from [plan.md](../001-research/plan.md): create a high-level `BeatThis` struct in `src/lib.rs` that composes audio loading, mel spectrogram, inference, and post-processing into a single easy-to-use pipeline.

## Goal

Add a `BeatThis<S>` struct to `src/lib.rs` that owns a `MelProcessor`, `BeatInference`, and `PostProcessor`, exposing two entry points: `process_file` (from audio file path) and `process_audio` (from raw PCM samples). This is the primary public interface for library consumers.

---

## Public API

```rust
/// Target sample rate expected by the mel spectrogram model.
const TARGET_SAMPLE_RATE: u32 = 22050;

/// High-level beat tracker composing the full pipeline.
///
/// Owns the mel spectrogram model, the beat inference model, and the
/// post-processor. Generic over the inference session type, so it works
/// with any backend (ort, rten, tract).
pub struct BeatThis<S: InferenceSession> {
    mel: MelProcessor<S>,
    inference: BeatInference<S>,
    post: PostProcessor,
}

impl<S: InferenceSession> BeatThis<S> {
    /// Create a new beat tracker by loading both ONNX models via the given runtime.
    ///
    /// - `runtime`: any `InferenceRuntime` (e.g. `OrtRuntime::default()`)
    /// - `mel_model_path`: path to the mel spectrogram ONNX model
    /// - `beat_model_path`: path to the beat tracking ONNX model
    pub fn new<R: InferenceRuntime<Session = S>>(
        runtime: &R,
        mel_model_path: &Path,
        beat_model_path: &Path,
    ) -> Result<Self>;

    /// Run the full pipeline on an audio file.
    ///
    /// Loads the file, resamples to 22050 Hz mono, computes mel spectrogram,
    /// runs beat inference, and post-processes into beat/downbeat timestamps.
    pub fn process_file(&mut self, path: &Path) -> Result<BeatResult>;

    /// Run the full pipeline on raw audio samples.
    ///
    /// The samples are resampled to 22050 Hz if `sample_rate` differs.
    /// Input should be mono f32 PCM.
    pub fn process_audio(&mut self, samples: &[f32], sample_rate: u32) -> Result<BeatResult>;
}
```

### Usage

```rust
use beat_this::{BeatThis, InferenceRuntime};
use beat_this::runtime::ort::OrtRuntime;
use std::path::Path;

let runtime = OrtRuntime::default();
let mut bt = BeatThis::new(
    &runtime,
    Path::new("models/mel_spectrogram.onnx"),
    Path::new("models/beat_this.onnx"),
)?;

// From file
let result = bt.process_file(Path::new("song.mp3"))?;
println!("{} beats, {} downbeats", result.beats.len(), result.downbeats.len());

// From raw audio
let result = bt.process_audio(&samples, 44100)?;
```

---

## Implementation Steps

### 1. Add `BeatThis` struct to `src/lib.rs`

The struct and its impl go directly in `lib.rs` since it's the primary public API type.

#### 1a. `BeatThis::new`

1. Load mel session: `runtime.load_model(mel_model_path)?`.
2. Load beat session: `runtime.load_model(beat_model_path)?`.
3. Construct: `MelProcessor::new(mel_session)`, `BeatInference::new(beat_session)`, `PostProcessor::default()`.
4. Return `BeatThis { mel, inference, post }`.

#### 1b. `BeatThis::process_audio`

The core pipeline, called by both entry points:

1. Resample if needed: if `sample_rate != TARGET_SAMPLE_RATE`, call `load_audio`-style resampling. Since `load_audio` only works from files, extract the resampling logic into a helper or handle inline:
   - If `sample_rate == TARGET_SAMPLE_RATE`: use samples directly.
   - If `sample_rate != TARGET_SAMPLE_RATE`: resample using `AudioData` / rubato (reuse the existing `audio` module's internal resampling by calling `load_audio` indirectly — but since we already have raw samples, we need direct access to resampling).

   **Decision**: expose a `resample` function from `audio.rs` (currently private). Make `audio::resample(samples, source_sr, target_sr) -> Result<Vec<f32>>` public. This is a minimal change — the function already exists, just needs `pub` visibility.

2. Compute mel spectrogram: `self.mel.process(&resampled)?`.
3. Run inference: `self.inference.process(&mel)?`.
4. Post-process: `self.post.process(&beat_logits, &downbeat_logits)?`.
5. Return `BeatResult`.

#### 1c. `BeatThis::process_file`

1. Load audio: `load_audio(path, TARGET_SAMPLE_RATE)?`.
2. Delegate: `self.process_audio(&audio.samples, audio.sample_rate)?`.

Since `load_audio` already resamples to `TARGET_SAMPLE_RATE`, the `process_audio` call will skip the resampling step (rates match).

### 2. Make `audio::resample` public

Change `fn resample(...)` to `pub fn resample(...)` in `src/audio.rs`. This allows `process_audio` to resample raw samples without going through file loading.

### 3. Update `lib.rs` re-exports

Add `BeatThis` to the public exports. The module declarations and existing re-exports stay unchanged.

```rust
// Existing re-exports remain:
pub use audio::{load_audio, AudioData};
pub use inference::BeatInference;
pub use mel::MelProcessor;
pub use postprocessing::{BeatResult, PostProcessor};
pub use runtime::{InferenceRuntime, InferenceSession, Tensor};

// New: the BeatThis struct is defined directly in lib.rs, so it's
// automatically public — no re-export needed.
```

### 4. Add unit tests

In `src/lib.rs` (`#[cfg(test)] mod tests`):

- **`test_process_audio_resampling_bypass`** — verify that when `sample_rate == 22050`, no resampling occurs (mock or use a tiny synthetic signal, confirm it passes through to mel unchanged in length).

Note: most meaningful tests require ONNX models, so the bulk of testing is in the integration test below.

### 5. Add integration test

Create `tests/public_api_integration.rs`:

- **`test_process_file`** — load a real audio file via `BeatThis::process_file`. Verify:
  - Result has non-empty beats and downbeats.
  - All times are non-negative and within audio duration.
  - Times are sorted.
  - Every downbeat appears in the beats vector.
  - Median beat interval is musically plausible (0.2s–2.0s).

- **`test_process_audio`** — load audio manually with `load_audio`, pass samples to `BeatThis::process_audio`. Verify:
  - Same invariants as above.
  - Results match `process_file` on the same audio (identical `BeatResult`).

- **`test_process_audio_different_sample_rate`** — load audio at 44100 Hz (without resampling), pass to `process_audio(samples, 44100)`. Verify:
  - Pipeline completes without error.
  - Results are non-empty and musically plausible.

All tests skip gracefully if model files or test audio aren't present (same pattern as existing tests).

---

## File Changes

| File | Action |
|------|--------|
| `src/lib.rs` | Add `BeatThis` struct, `new`, `process_file`, `process_audio`, constant, tests |
| `src/audio.rs` | Make `resample` function `pub` (single keyword change) |
| `tests/public_api_integration.rs` | **New** — integration tests for the public API |

No new dependencies needed.

---

## Design Decisions

1. **Generic over `InferenceSession` (not a concrete backend)**: `BeatThis<S>` works with any backend. The caller chooses the runtime at construction time. This matches the project's first-class runtime abstraction and avoids coupling `lib.rs` to ort.

2. **Constructor takes a runtime reference**: `BeatThis::new(&runtime, mel_path, beat_path)` uses the runtime as a factory to load both sessions. The runtime is borrowed only during construction — not stored — so it doesn't infect the struct's lifetime. This matches the pattern shown in the architecture plan.

3. **`process_audio` accepts any sample rate**: callers don't need to pre-resample to 22050 Hz. The method handles it internally, making the API more forgiving. If the rate already matches, no resampling is performed (zero cost).

4. **`process_file` delegates to `process_audio`**: avoids duplicating the mel → inference → post-processing pipeline. `load_audio` handles decoding + resampling, then `process_audio` handles the rest. Since `load_audio` already resamples to the target rate, the resampling check in `process_audio` is a no-op for this path.

5. **Expose `audio::resample` as public**: the function already exists and is well-tested. Making it public is a one-keyword change that enables `process_audio` to resample raw samples without file I/O. This is a minimal, non-breaking API surface addition.

6. **`BeatThis` defined in `lib.rs` (not a separate module)**: it's the crate's primary public type and composes other modules. Placing it in `lib.rs` makes the import path `beat_this::BeatThis` — clean and discoverable. The implementation is short (~30 lines) and doesn't warrant its own module.

7. **`&mut self` on pipeline methods**: both `MelProcessor::process` and `BeatInference::process` require `&mut self` (because `InferenceSession::run` is `&mut self`). This propagates to `BeatThis` — the tracker cannot be shared across threads without external synchronization, which is fine for a sequential pipeline.

8. **No builder pattern**: the constructor takes exactly three arguments (runtime, mel path, beat path) with no optional configuration. A builder would add complexity for no benefit. If configuration is needed in the future (e.g., custom fps, custom target sample rate), it can be added via a `BeatThisBuilder` without breaking the existing `new` API.

---

## Validation

1. **Compilation**: `BeatThis` struct compiles with the existing module types and trait bounds.
2. **Integration test**: `process_file` produces correct results on real audio (non-empty, sorted, snapped, plausible BPM).
3. **Consistency**: `process_file` and `process_audio` (on the same audio) produce identical `BeatResult`.
4. **Resampling**: `process_audio` handles both matching and non-matching sample rates correctly.
5. **Cross-reference**: results match the existing `postprocessing_integration.rs` test (which manually wires the pipeline) — confirming that `BeatThis` is a faithful composition of the same steps.
