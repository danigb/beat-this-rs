# Task 006: Post-Processing — Development Plan

Implements **Step 5** from [plan.md](../001-research/plan.md): take raw beat/downbeat logits from inference and produce final beat and downbeat timestamps via peak picking, deduplication, and downbeat alignment.

## Goal

Create `src/postprocessing.rs` with a `PostProcessor` struct that takes beat and downbeat logit vectors (one value per spectrogram frame), applies max-pool peak picking, thresholding, deduplication, and downbeat-to-beat alignment, and returns `BeatResult { beats: Vec<f32>, downbeats: Vec<f32> }` — sorted timestamps in seconds.

---

## Background: Minimal Post-Processing Algorithm

The post-processing pipeline converts raw per-frame logits into discrete beat/downbeat event times. This follows the "minimal" strategy from the Python reference (as opposed to the DBN strategy which requires madmom and is out of scope).

### Pipeline Overview

```
beat_logits [T]  ──┐
                    ├──→ find_peaks ──→ deduplicate ──→ frame→time ──→ BeatResult
downbeat_logits [T]─┘                                       │
                                                             └──→ snap downbeats to nearest beat
                                                             └──→ deduplicate downbeat times
```

### Algorithm Steps

1. **Peak picking (per logit vector)**:
   - Max-pool with kernel=7, stride=1, padding=3: for each frame `i`, compute the max over `logits[i-3..i+4]` (clamped to bounds).
   - A frame is a peak if `logits[i] == max_pool[i]` (it equals the local maximum) AND `logits[i] > 0.0` (logit > 0 corresponds to probability > 0.5 after sigmoid).
   - This is equivalent to the Python: `pred_logits != F.max_pool1d(pred_logits, 7, 1, 3)` masked to -1000, then `> 0`.

2. **Deduplication of adjacent peaks**:
   - Groups of peak frame indices where consecutive peaks are ≤ `width` frames apart (width=1) are merged into a single peak at their running mean position.
   - Running mean update: `p += (p2 - p) / c` where `c` is the count of peaks in the current group.
   - Final position is rounded to the nearest integer frame index.

3. **Frame-to-time conversion**:
   - `time = frame_index / fps` where `fps = 50.0` (the model's frame rate).

4. **Snap downbeats to nearest beat**:
   - For each downbeat time, find the beat time with the smallest absolute difference and replace the downbeat time with that beat time.
   - This ensures every downbeat coincides exactly with a beat.

5. **Deduplicate downbeat times**:
   - After snapping, multiple downbeats may map to the same beat. Remove duplicates (keep unique sorted values).

### Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `fps` | 50.0 | Frames per second (model output rate) |
| `max_pool_kernel` | 7 | Peak picking window size (±3 frames = ±60ms) |
| `max_pool_padding` | 3 | Symmetric padding for max-pool (`(kernel-1)/2`) |
| `threshold` | 0.0 | Logit threshold (> 0 ≈ probability > 0.5) |
| `dedup_width` | 1 | Maximum frame gap to merge adjacent peaks |

---

## Public API

```rust
/// Default frames per second for the beat model.
const FPS: f32 = 50.0;

/// Beat detection results: sorted timestamps in seconds.
pub struct BeatResult {
    /// Beat times in seconds (sorted, deduplicated).
    pub beats: Vec<f32>,
    /// Downbeat times in seconds (sorted, deduplicated, snapped to nearest beat).
    pub downbeats: Vec<f32>,
}

/// Post-processes raw beat/downbeat logits into timestamped events.
pub struct PostProcessor {
    fps: f32,
}

impl PostProcessor {
    /// Create a new post-processor with the given frame rate.
    pub fn new(fps: f32) -> Self;

    /// Create a post-processor with the default fps (50.0).
    pub fn default() -> Self;

    /// Process beat and downbeat logits into a BeatResult.
    ///
    /// Both input slices must have the same length (one value per spectrogram frame).
    /// Returns beat and downbeat times in seconds.
    pub fn process(&self, beat_logits: &[f32], downbeat_logits: &[f32]) -> Result<BeatResult>;
}
```

The `BeatResult` struct is defined in this module and will be re-exported from `lib.rs`. It's the return type for the full pipeline (used by the public API in step 6).

---

## Implementation Steps

### 1. Create `src/postprocessing.rs`

Four internal functions composed in `PostProcessor::process`:

#### 1a. `find_peaks(logits: &[f32]) -> Vec<usize>`

Identify local maxima that exceed the threshold:

1. For each frame `i` in `0..len`:
   - Skip if `logits[i] <= 0.0` (below threshold).
   - Compute the local max over `logits[max(0, i-3) .. min(len, i+4)]`.
   - If `logits[i]` equals the local max (is a peak), record `i`.
2. Pass the peak indices through `deduplicate_peaks`.
3. Return deduplicated peak frame indices.

Note: when multiple frames share the same maximum value within a window, all of them pass the `logits[i] == local_max` check. The deduplication step merges these adjacent co-maxima into a single peak.

#### 1b. `deduplicate_peaks(peaks: &[usize], width: usize) -> Vec<usize>`

Merge adjacent peak frame indices using a running mean:

1. If empty, return empty.
2. Initialize: `p = peaks[0] as f64`, `c = 1.0`.
3. For each subsequent peak `p2`:
   - If `p2 as f64 - p <= width as f64`: merge — `c += 1.0`, `p += (p2 as f64 - p) / c`.
   - Else: emit `p.round() as usize`, reset `p = p2 as f64`, `c = 1.0`.
4. Emit final `p.round() as usize`.
5. Return collected indices.

Uses `f64` for the running mean to match Python's float precision and avoid accumulation errors.

#### 1c. `snap_downbeats_to_beats(beat_times: &[f32], downbeat_times: &mut Vec<f32>)`

Align each downbeat to its nearest beat:

1. If `beat_times` is empty, return (leave downbeats as-is).
2. For each downbeat time `d`:
   - Binary search `beat_times` for `d` to find the insertion point.
   - Compare the beat immediately before and after the insertion point.
   - Replace `d` with the closer beat time.
3. Sort and deduplicate `downbeat_times`.

Uses binary search (`partition_point`) instead of linear scan — O(D log B) instead of O(D × B). The beat_times vector is already sorted from the peak-picking step.

#### 1d. `PostProcessor::process` (main entry point)

1. Validate: `beat_logits.len() == downbeat_logits.len()`, bail if not.
2. `beat_frames = find_peaks(beat_logits)`.
3. `downbeat_frames = find_peaks(downbeat_logits)`.
4. Convert to times: `beats = beat_frames.iter().map(|&f| f as f32 / self.fps).collect()`.
5. Convert to times: `downbeats = downbeat_frames.iter().map(|&f| f as f32 / self.fps).collect()`.
6. `snap_downbeats_to_beats(&beats, &mut downbeats)`.
7. Return `BeatResult { beats, downbeats }`.

### 2. Wire into `src/lib.rs`

- Add `pub mod postprocessing;`.
- Re-export: `pub use postprocessing::{BeatResult, PostProcessor};`.

### 3. Add unit tests

In `src/postprocessing.rs` (`#[cfg(test)] mod tests`):

- **`test_find_peaks_single_peak`** — logits with one clear peak (e.g., `[0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0]`) → peak at index 3.

- **`test_find_peaks_below_threshold`** — logits all negative → no peaks.

- **`test_find_peaks_multiple_peaks`** — two peaks separated by more than 3 frames → both detected.

- **`test_find_peaks_adjacent_equal`** — adjacent frames with equal positive values → deduplicated to one peak at the mean position.

- **`test_deduplicate_peaks_empty`** — empty input → empty output.

- **`test_deduplicate_peaks_no_adjacent`** — peaks far apart → unchanged.

- **`test_deduplicate_peaks_merge`** — `[10, 11, 12, 20]` with width=1 → `[11, 20]` (first three merge to mean ≈ 11).

- **`test_snap_downbeats`** — beats at `[1.0, 2.0, 3.0]`, downbeats at `[1.1, 2.8]` → snapped to `[1.0, 3.0]`.

- **`test_snap_downbeats_dedup`** — two downbeats that snap to the same beat → one unique result.

- **`test_snap_downbeats_empty_beats`** — no beats → downbeats unchanged.

- **`test_process_full`** — construct synthetic logits with known peaks, run `process`, verify beat/downbeat times match expected values.

- **`test_process_empty_logits`** — zero-length input → empty result (no error).

- **`test_process_mismatched_lengths`** — different-length inputs → error.

### 4. Add integration test

Create `tests/postprocessing_integration.rs`:

- **`test_postprocessing_with_real_inference`** — load test audio, compute mel spectrogram, run inference, pass logits to `PostProcessor`. Verify:
  - Beats and downbeats are non-empty (real music produces events).
  - All times are non-negative and ≤ audio duration.
  - Times are sorted in ascending order.
  - Every downbeat time appears in the beats vector (snapping invariant).
  - Beat intervals are musically plausible (e.g., between 0.2s and 2.0s for typical music).

Skips gracefully if model files aren't present (same pattern as existing tests).

---

## File Changes

| File | Action |
|------|--------|
| `src/postprocessing.rs` | **New** — `PostProcessor`, `BeatResult`, peak picking, deduplication |
| `src/lib.rs` | Add `pub mod postprocessing` + re-exports |
| `tests/postprocessing_integration.rs` | **New** — integration test with real audio |

No dependency changes needed — the implementation uses only standard library operations on `Vec<f32>` / `Vec<usize>`.

---

## Design Decisions

1. **`BeatResult` struct (not a tuple)**: provides named fields (`beats`, `downbeats`) for clarity. This struct becomes the return type of the full public API pipeline, so it's worth defining properly. Placed in `postprocessing.rs` since that's where it's produced.

2. **`&[f32]` input (not `Tensor`)**: the inference step already returns `(Vec<f32>, Vec<f32>)`. There's no reason to wrap back into a `Tensor` for post-processing — it's pure 1D signal processing.

3. **Binary search for downbeat snapping**: the Rust reference uses linear scan O(D × B), but since both beat and downbeat times are sorted, `partition_point` gives O(D log B) with minimal extra complexity. For typical audio (hundreds of beats), either approach is fast, but binary search is the idiomatic Rust choice.

4. **`f64` for running mean in deduplication**: the Python reference uses float64 by default. Using `f64` for the intermediate running mean avoids subtle rounding differences when comparing outputs across implementations. The final result is rounded to `usize`, so the extra precision is costless.

5. **Configurable `fps` (with default)**: while fps is always 50 for the current model, accepting it as a parameter makes testing easier and avoids a hardcoded constant that would need to match across modules. The `default()` constructor provides the standard value.

6. **Minimal mode only (no DBN)**: the DBN post-processing requires the `madmom` library (Python/C), which has no Rust equivalent. The minimal approach produces results within a few ms of the DBN approach for most music and matches what both reference Rust/C++ implementations use.

---

## Validation

1. **Unit test coverage**: every internal function has dedicated tests with known inputs and expected outputs.
2. **Invariant checks**: beats are sorted, downbeats are a subset of beat times, no duplicate times.
3. **Integration test**: full pipeline produces musically reasonable results on real audio.
4. **Cross-reference**: compare output against `remixatron_rust` post-processor on the same logit vectors (tolerance: ±1 frame / ±20ms for times).
