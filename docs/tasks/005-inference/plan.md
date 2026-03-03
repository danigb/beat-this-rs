# Task 005: Chunked Inference — Development Plan

Implements **Step 4** from [plan.md](../001-research/plan.md): split a mel spectrogram into overlapping chunks, run the beat-tracking ONNX model on each chunk, and aggregate the beat/downbeat logits into full-length output vectors.

## Goal

Create `src/inference.rs` with a `BeatInference` struct that takes a mel spectrogram `Tensor` (shape `[1, T, 128]`), splits it into overlapping chunks of 1500 frames, runs each chunk through the beat model, trims border frames, and aggregates the results using "keep_first" overlap mode. Returns `(Vec<f32>, Vec<f32>)` — beat logits and downbeat logits, one value per spectrogram frame.

---

## Background: Chunked Inference Algorithm

The Beat This! model was trained on fixed-length 1500-frame segments (30 seconds at 50 fps). For longer audio, inference uses a sliding-window approach with overlap to avoid edge artifacts.

### Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `chunk_size` | 1500 | Frames per chunk (30s at 50 fps) |
| `border_size` | 6 | Frames discarded from each edge of predictions |
| `stride` | 1488 | `chunk_size - 2 * border_size` — effective step between chunks |
| `overlap_mode` | `keep_first` | Earlier chunks take priority in overlapping regions |

### Algorithm Steps

1. **Generate chunk starts**: `starts = [-border, -border + stride, -border + 2*stride, ...]` while `start < T - border`.
2. **Avoid short end**: if `T > stride`, adjust the last start to `T - (chunk_size - border)` so the final chunk aligns with the spectrogram end rather than producing a short trailing chunk.
3. **Extract and pad each chunk**: for each start position, extract `spect[max(start,0) .. min(start+chunk_size, T)]` and zero-pad on the left (`max(0, -start)` frames) and right (`max(0, min(border, start+chunk_size-T))` frames) to reach exactly `chunk_size` frames.
4. **Run inference**: pass each padded chunk as `[1, chunk_size, 128]` through the beat ONNX model. The model returns beat and downbeat logits, each of length `chunk_size`.
5. **Aggregate (keep_first)**: initialize output buffers with `-1000.0` (a sentinel "no prediction" value). Process chunks in **reverse** order. For each chunk, strip `border_size` frames from both ends of the prediction, then write the valid predictions to `output[start+border .. start+chunk_size-border]`. Because earlier chunks are written last, they overwrite later chunks in overlapping regions — hence "keep_first".

### Edge Cases

- **Short audio** (T ≤ stride): produces a single chunk, padded to `chunk_size`. No overlap aggregation needed.
- **First chunk** (start = -border): left-padded with `border` zero frames. After border trimming, predictions start at frame 0.
- **Last chunk** (avoid_short_end): shifted left to align with the spectrogram end, potentially overlapping more than `border` frames with the previous chunk. The keep_first mode ensures the earlier chunk's predictions dominate.

---

## ONNX Model I/O

The beat model input/output names depend on which ONNX export was used. The implementation must handle both naming conventions:

| Convention | Input | Outputs |
|-----------|-------|---------|
| remixatron ONNX | `"mel_spectrogram"` | `"beat"`, `"downbeat"` |
| beat_this_cpp ONNX | `"input_spectrogram"` | `"beat"`, `"downbeat"` |

**Strategy**: use `"mel_spectrogram"` as the input name (matching remixatron's models, which are used in our tests). For outputs, look for `"beat"` and `"downbeat"`. If either is missing, try `"beat_logits"` / `"downbeat_logits"` as fallbacks.

The outputs are 1D logits per frame (shape `[1, chunk_size]`). Each value is an unbounded real number (logit); positive values indicate a beat/downbeat, negative values indicate no beat.

---

## Public API

```rust
/// Constants for the chunking algorithm (matching the Python training pipeline).
const CHUNK_SIZE: usize = 1500;
const BORDER_SIZE: usize = 6;
const STRIDE: usize = CHUNK_SIZE - 2 * BORDER_SIZE; // 1488

/// Runs chunked beat/downbeat inference on a mel spectrogram.
pub struct BeatInference<S: InferenceSession> {
    session: S,
}

impl<S: InferenceSession> BeatInference<S> {
    /// Wrap an already-loaded inference session for the beat model.
    pub fn new(session: S) -> Self;

    /// Run inference on a full mel spectrogram.
    ///
    /// Input: mel spectrogram Tensor with shape `[1, T, 128]`.
    /// Returns: `(beat_logits, downbeat_logits)` — each `Vec<f32>` of length T.
    pub fn process(&mut self, mel: &Tensor) -> Result<(Vec<f32>, Vec<f32>)>;
}
```

This follows the same pattern as `MelProcessor<S>` — generic over the inference session type, constructed with an already-loaded session.

---

## Implementation Steps

### 1. Create `src/inference.rs`

Three internal functions composed in `BeatInference::process`:

```
mel [1, T, 128] → [generate_starts] → Vec<i32>
                → [extract_chunk] → padded Tensor [1, chunk_size, 128]  (per start)
                → [run model] → (beat_logits, downbeat_logits)          (per chunk)
                → [aggregate] → (Vec<f32>, Vec<f32>) of length T
```

#### 1a. `generate_starts(full_time: usize) -> Vec<i32>`

Generate chunk start positions:

1. Start from `-BORDER_SIZE as i32`, step by `STRIDE`, while `start < full_time as i32 - BORDER_SIZE as i32`.
2. If `full_time > STRIDE` (avoid short end), adjust the last start to `full_time as i32 - (CHUNK_SIZE as i32 - BORDER_SIZE as i32)`.
3. Return the list of starts.

Uses `i32` because the first start is negative (`-6`).

#### 1b. `extract_chunk(mel: &Tensor, start: i32) -> Tensor`

Extract and zero-pad a single chunk from the mel spectrogram:

1. Compute actual slice bounds: `actual_start = max(0, start)`, `actual_end = min(start + CHUNK_SIZE, full_time)`.
2. Compute padding: `pad_left = max(0, -start)`, `pad_right = max(0, min(BORDER_SIZE, start + CHUNK_SIZE - full_time))`.
3. Allocate a zeroed `Tensor` of shape `[1, CHUNK_SIZE, 128]`.
4. Copy the mel slice `[actual_start..actual_end, :]` into the chunk at offset `pad_left`.
5. Return the padded chunk.

The mel spectrogram `Tensor` is stored as flat row-major data with shape `[1, T, 128]`. Element `(0, t, f)` is at index `t * 128 + f`.

#### 1c. `process` method (run + aggregate)

1. Validate input shape: must be `[1, T, 128]`, extract `full_time = T`.
2. Call `generate_starts(full_time)` to get start positions.
3. Initialize output buffers: `beat_logits = vec![-1000.0; full_time]`, `downbeat_logits = vec![-1000.0; full_time]`.
4. Reverse the starts list (for keep_first aggregation).
5. For each start:
   a. Extract padded chunk via `extract_chunk`.
   b. Run inference: `session.run(&[("mel_spectrogram", &chunk)])`.
   c. Extract beat and downbeat output tensors from the results (try `"beat"` then `"beat_logits"`, same for downbeat).
   d. Strip border frames: take `output[BORDER_SIZE .. CHUNK_SIZE - BORDER_SIZE]` (length = STRIDE = 1488).
   e. Write valid predictions to `beat_logits[write_start..write_end]` where `write_start = (start + BORDER_SIZE) as usize` and `write_end = min((start + CHUNK_SIZE - BORDER_SIZE) as usize, full_time)`.
6. Return `(beat_logits, downbeat_logits)`.

### 2. Wire into `src/lib.rs`

Add `pub mod inference;` and re-export `BeatInference`.

### 3. Add integration test

Create `tests/inference_integration.rs`:

- **`test_beat_inference_short`** — create a short mel spectrogram (e.g., 100 frames of zeros, shape `[1, 100, 128]`), run through `BeatInference`. Verify:
  - Output lengths both equal 100
  - All values are finite (no NaN/Inf)
  - Values are not all `-1000.0` (the sentinel was overwritten)

- **`test_beat_inference_long`** — create a mel spectrogram longer than one chunk (e.g., 3000 frames), run through `BeatInference`. Verify:
  - Output lengths both equal 3000
  - All values are finite
  - No `-1000.0` sentinels remain (every frame was covered by at least one chunk)

- **`test_beat_inference_with_real_audio`** — load the test MP3, compute mel via `MelProcessor`, run through `BeatInference`. Verify:
  - Output length matches mel frame count
  - Some beat logits are positive (real music should trigger beats)
  - Beat and downbeat logits differ (they're separate predictions)

All tests skip gracefully if model files aren't present (same pattern as existing tests).

### 4. Add unit tests

In `src/inference.rs`:

- **`test_generate_starts_short`** — 100 frames → single start at `-6`.
- **`test_generate_starts_exact_chunk`** — 1500 frames → single start at `-6`.
- **`test_generate_starts_two_chunks`** — 2000 frames → verify start positions and that the last start is adjusted for avoid_short_end.
- **`test_generate_starts_long`** — 5000 frames → verify correct number of chunks, first start = `-6`, stride = 1488.
- **`test_extract_chunk_first`** — first chunk (start = -6) has left padding of 6.
- **`test_extract_chunk_middle`** — middle chunk has no padding.

---

## File Changes

| File | Action |
|------|--------|
| `src/inference.rs` | **New** — `BeatInference` struct with chunked inference |
| `src/lib.rs` | Add `pub mod inference` + re-export |
| `tests/inference_integration.rs` | **New** — integration tests |

No dependency changes needed — the implementation uses only the existing `runtime::Tensor` and `runtime::InferenceSession` abstractions.

---

## Design Decisions

1. **Generic over `InferenceSession` (not `InferenceRuntime`)**: follows the same pattern as `MelProcessor<S>` — the caller loads the session and passes it in. This keeps the struct simple and testable (could use a mock session for unit tests).

2. **`i32` for start positions**: the first start is `-BORDER_SIZE` (negative), which is used to zero-pad the beginning. Using `i32` avoids unsigned arithmetic issues. The values are small enough that `i32` is sufficient for any practical spectrogram length.

3. **Flat `Vec<f32>` output (not `Tensor`)**: the downstream postprocessor works with 1D logit vectors, not tensors. Returning `(Vec<f32>, Vec<f32>)` is the simplest interface. If a `Tensor` is ever needed, wrapping is trivial.

4. **Output name fallback**: the beat model's ONNX output names may vary between exports (`"beat"` vs `"beat_logits"`). The implementation tries the canonical name first, then falls back. This avoids hard-coding a single model variant.

5. **Constants (not configurable)**: `CHUNK_SIZE`, `BORDER_SIZE`, and `STRIDE` are fixed to match the Python training pipeline. Making them configurable would add complexity with no benefit — changing them would produce incorrect results.

6. **No parallelism**: chunks are processed sequentially. ONNX Runtime already uses multiple threads internally (controlled by `OrtRuntime::intra_threads`). Parallelizing at the chunk level would add complexity and contend with the runtime's own thread pool. Can be revisited if profiling shows it's a bottleneck.

---

## Validation

1. **Shape preservation**: output length equals input mel frame count for any input length.
2. **Coverage**: no `-1000.0` sentinels remain in the output (every frame is covered by at least one chunk).
3. **Real audio**: beat logits show positive values at musically meaningful positions.
4. **Cross-reference**: compare output against remixatron_rust or beat_this_cpp on the same audio file (tolerance: ±1e-5 per frame, or identical after peak picking).
