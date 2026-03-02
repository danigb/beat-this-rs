# Task 004: Mel Spectrogram (ONNX Model) — Development Plan

Implements **Step 3** from [plan.md](../001-research/plan.md): compute a log-mel spectrogram from raw PCM audio using a pre-exported ONNX model.

## Goal

Create `src/mel.rs` with a `MelProcessor` struct that takes mono f32 PCM samples (from `audio.rs`) and returns a mel spectrogram tensor ready for the beat-tracking model. This uses the **ONNX mel model approach** (Option A from the research) — the torchaudio mel computation exported as a standalone ONNX model, guaranteeing exact numerical parity with the Python training pipeline.

---

## Public API

```rust
/// Computes log-mel spectrograms via an ONNX model.
pub struct MelProcessor<S: InferenceSession> {
    session: S,
}

impl<S: InferenceSession> MelProcessor<S> {
    /// Wrap an already-loaded inference session for the mel spectrogram model.
    pub fn new(session: S) -> Self;

    /// Compute mel spectrogram from mono PCM samples at 22050 Hz.
    ///
    /// Input: mono f32 samples (any length).
    /// Output: Tensor with shape [1, time_frames, 128].
    ///
    /// The number of time frames depends on sample count:
    /// time_frames ≈ samples.len() / hop_length (hop_length = 441 for 50 fps at 22050 Hz).
    pub fn process(&mut self, samples: &[f32]) -> Result<Tensor>;
}
```

Key design choices:

- **Generic over `S: InferenceSession`** — uses the runtime trait abstraction, not hardcoded to ort. The session is loaded externally by the caller via `InferenceRuntime::load_model()`, keeping `mel.rs` backend-agnostic.
- **Takes `&[f32]` not `AudioData`** — the mel processor doesn't need to know about sample rates or the audio module. The caller is responsible for providing 22050 Hz mono samples.
- **Returns `Tensor`** not ndarray — stays within the project's own type system. Downstream consumers (chunked inference) also work with `Tensor`.

---

## ONNX Model Details

| Property       | Value                                     |
|----------------|-------------------------------------------|
| File           | `MelSpectrogram_Ultimate.onnx`            |
| Input name     | `audio_pcm`                               |
| Input shape    | `[1, N]` (batch=1, N=sample count)        |
| Input dtype    | float32                                   |
| Output name    | `mel_spectrogram`                         |
| Output shape   | `[1, T, 128]` (batch=1, T=time frames, 128 mel bins) |
| Output dtype   | float32                                   |

The model internally performs: STFT (n_fft=1024, hop=441, Hann window) → mel filterbank (128 bins, 30–11000 Hz, Slaney) → log1p(1000 × mel) compression.

---

## Implementation Steps

### 1. Create `src/mel.rs`

The implementation is straightforward — it's a thin wrapper around `InferenceSession`:

```rust
pub fn process(&mut self, samples: &[f32]) -> Result<Tensor> {
    // 1. Wrap samples in a Tensor with shape [1, N]
    let input = Tensor {
        shape: vec![1, samples.len()],
        data: samples.to_vec(),
    };

    // 2. Run inference
    let mut outputs = self.session.run(&[("audio_pcm", &input)])?;

    // 3. Extract the mel spectrogram output
    let mel = outputs
        .remove("mel_spectrogram")
        .ok_or_else(|| anyhow!("Model missing 'mel_spectrogram' output"))?;

    // 4. Validate shape: expect [1, T, 128]
    anyhow::ensure!(
        mel.shape.len() == 3 && mel.shape[0] == 1 && mel.shape[2] == 128,
        "Unexpected mel shape: {:?}",
        mel.shape
    );

    Ok(mel)
}
```

Also add a helper to extract frame count:

```rust
/// Number of time frames in a mel spectrogram tensor.
/// Assumes shape [1, T, 128].
pub fn num_frames(mel: &Tensor) -> usize {
    mel.shape[1]
}
```

### 2. Wire into `src/lib.rs`

Add `pub mod mel;` and re-export `MelProcessor`:

```rust
pub mod mel;
pub use mel::MelProcessor;
```

### 3. Add integration test

Extend `tests/ort_integration.rs` (or create `tests/mel_integration.rs`) with tests that exercise `MelProcessor` through the runtime abstraction:

**`test_mel_processor_basic`** — load mel model, process 1 second of silence:
- Verify output shape is `[1, T, 128]`
- Verify `T` is approximately `22050 / 441 ≈ 50` (±1 frame for padding)
- Verify all values are finite

**`test_mel_processor_duration_scaling`** — process 2 seconds, then 5 seconds:
- Verify frame count scales linearly: `T ≈ samples / 441`
- Verify the ratio `T_5s / T_2s ≈ 2.5`

**`test_mel_processor_with_audio`** — full pipeline test (load audio file → mel):
- Load test audio with `load_audio(path, 22050)`
- Process through `MelProcessor`
- Verify output shape `[1, T, 128]` where `T ≈ duration * 50`
- Verify mel values are in a reasonable range (not all zeros for real audio)
- Skip if test audio or model not present

Tests skip gracefully if model/audio files aren't present (same pattern as existing tests).

---

## File Changes

| File | Action |
|------|--------|
| `src/mel.rs` | **New** — `MelProcessor` struct + `process` method |
| `src/lib.rs` | Add `pub mod mel` + re-export |
| `tests/mel_integration.rs` | **New** — integration tests for mel processing |

---

## Design Decisions

1. **ONNX model over custom FFT**: The mel model guarantees bit-exact parity with the Python training pipeline. A custom implementation (realfft + mel filterbank) would be ~200-300 lines and risk numerical differences that could degrade beat tracking accuracy. The tradeoff is an additional ~4 MB model file, which is negligible compared to the beat model (~83 MB).

2. **Generic over InferenceSession**: `MelProcessor<S>` works with any backend (ort, rten, tract). The session is created externally, so the mel module has zero knowledge of specific runtimes. This follows the same pattern established in the architecture plan.

3. **Session owned, not borrowed**: `MelProcessor` owns the session rather than borrowing it. This simplifies lifetime management and matches the expected usage — one session per processor, created at startup, used throughout the pipeline.

4. **No internal batching or chunking**: The ONNX mel model handles the entire audio in one pass. Unlike the beat model (which needs chunked inference for memory reasons), the mel spectrogram computation is lightweight enough to process the full audio at once. A 5-minute song is ~6.6M samples — the input tensor is ~25 MB and the output is ~1.5 MB.

5. **Validation in `process`**: The shape assertion catches model mismatches early (e.g., wrong model file loaded). This is a cheap check that prevents confusing errors downstream in the inference stage.

---

## Validation

- Output shape `[1, T, 128]` for any input length
- Frame count `T ≈ samples.len() / 441` (within ±2 frames)
- Output feeds correctly into the beat model in Step 4 (input name `"mel_spectrogram"`, shape `[1, T, 128]`)
- All output values are finite (no NaN/Inf)
