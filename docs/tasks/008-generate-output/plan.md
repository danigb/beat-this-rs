# Task 008: Output Generation — Development Plan

Implements **Step 7** from [plan.md](../001-research/plan.md): generate output files from beat detection results — `.beats` text files, click track WAV audio, and mixed audio (original + clicks).

## Goal

Add an `output` module (`src/output.rs`) that converts `BeatResult` into three output formats: a tab-separated `.beats` text file with beat counts, a click-track WAV file with sine-wave clicks at beat positions, and a mixed WAV file that layers clicks over the original audio. Also add a `calculate_bpm` utility function.

---

## Public API

```rust
// src/output.rs

use crate::postprocessing::BeatResult;
use anyhow::Result;
use std::path::Path;

/// Write a `.beats` file: tab-separated `time\tbeat_count` per line.
///
/// Beat count is 1 for downbeats, 2..N for subsequent beats in the measure.
/// Times are formatted with 3 decimal places.
pub fn write_beats_file(path: &Path, result: &BeatResult) -> Result<()>;

/// Generate a click-track WAV file.
///
/// Downbeats get an 880 Hz click, regular beats get 440 Hz.
/// Each click is a 100ms sine wave with ADSR envelope.
/// Output is mono 44100 Hz 32-bit float WAV.
pub fn write_click_track(path: &Path, result: &BeatResult) -> Result<()>;

/// Generate a mixed WAV file: original audio + click track layered on top.
///
/// Original audio is scaled to 0.7 gain, clicks are added at 0.3 gain.
/// Output preserves the original sample rate. Output is mono.
pub fn write_mixed_audio(
    path: &Path,
    result: &BeatResult,
    original_samples: &[f32],
    sample_rate: u32,
) -> Result<()>;

/// Calculate BPM from beat timestamps using median inter-beat interval.
///
/// Filters out unrealistic intervals (<0.1s or >3.0s, i.e. outside 20–600 BPM).
/// Returns `None` if fewer than 2 valid intervals exist.
pub fn calculate_bpm(result: &BeatResult) -> Option<f32>;

/// Compute beat counts from beat and downbeat timestamps.
///
/// Returns a `Vec<i32>` parallel to `result.beats`:
/// - 1 for downbeats
/// - 2, 3, 4, ... for subsequent beats within each measure
///
/// Used internally by `write_beats_file` and available for external use.
pub fn beat_counts(result: &BeatResult) -> Vec<i32>;
```

### Usage

```rust
use beat_this::output;

// After running the pipeline:
let result = bt.process_file(Path::new("song.mp3"))?;

// Save beat timestamps
output::write_beats_file(Path::new("song.beats"), &result)?;

// Generate click track
output::write_click_track(Path::new("clicks.wav"), &result)?;

// Generate mixed audio (original + clicks)
let audio = beat_this::load_audio(Path::new("song.mp3"), 44100)?;
output::write_mixed_audio(Path::new("mixed.wav"), &result, &audio.samples, audio.sample_rate)?;

// Calculate BPM
if let Some(bpm) = output::calculate_bpm(&result) {
    println!("BPM: {:.1}", bpm);
}
```

---

## Implementation Steps

### 1. Add `hound` dependency

Add `hound = "3.5"` to `Cargo.toml` for WAV file writing.

### 2. Implement `beat_counts`

Derive beat position within each measure from the beat and downbeat lists:

1. For each beat time in `result.beats`, check if it also appears in `result.downbeats` (use a tolerance of ~0.001s since both are snapped).
2. If it's a downbeat, assign count = 1 and reset the counter.
3. Otherwise, increment the counter (2, 3, 4, ...).
4. Handle the case where beats appear before the first downbeat — assign incrementing counts starting from 1 (or from an arbitrary value; the C++ reference starts at 1).

### 3. Implement `write_beats_file`

1. Compute beat counts via `beat_counts(result)`.
2. Open the file with `std::fs::File::create`.
3. For each `(beat_time, count)` pair, write `"{time:.3}\t{count}\n"`.
4. Use `BufWriter` for performance.

### 4. Implement click synthesis helpers

Internal (not public) helper functions:

#### 4a. `generate_sine_click`

```rust
/// Generate a single click: ADSR-enveloped sine wave.
fn generate_sine_click(frequency: f32, sample_rate: u32) -> Vec<f32>
```

- Duration: 100ms (0.1s)
- Attack: 10ms linear ramp up
- Sustain: from 10ms to 50ms before end (at full amplitude)
- Decay: 50ms linear ramp down
- Formula: `amplitude * sin(2π * frequency * t)`
- Return: `Vec<f32>` of `(0.1 * sample_rate)` samples

#### 4b. Constants

```rust
const CLICK_SAMPLE_RATE: u32 = 44100;
const CLICK_DURATION: f32 = 0.1;     // 100ms
const CLICK_ATTACK: f32 = 0.01;      // 10ms
const CLICK_DECAY: f32 = 0.05;       // 50ms
const DOWNBEAT_FREQ: f32 = 880.0;    // A5
const BEAT_FREQ: f32 = 440.0;        // A4
```

### 5. Implement `write_click_track`

1. Compute beat counts to distinguish downbeats from regular beats.
2. Determine total duration: `last_beat_time + CLICK_DURATION + CLICK_DECAY`.
3. Allocate a zero-filled `Vec<f32>` of `(total_duration * CLICK_SAMPLE_RATE)` samples.
4. For each beat:
   - Choose frequency: 880 Hz if count == 1 (downbeat), 440 Hz otherwise.
   - Generate sine click at `CLICK_SAMPLE_RATE`.
   - Mix (additive) into the buffer at `(beat_time * CLICK_SAMPLE_RATE)` offset.
5. Normalize: find max absolute value; if > 1.0, scale entire buffer by `1.0 / max`.
6. Write WAV using `hound::WavWriter` with spec: 1 channel, 44100 Hz, 32-bit float.

### 6. Implement `write_mixed_audio`

1. Compute beat counts.
2. Calculate total duration: `max(original_duration, last_beat_time + click_tail)`.
3. Allocate output buffer of `(total_duration * sample_rate)` samples.
4. Copy original audio into buffer scaled by 0.7.
5. For each beat:
   - Generate sine click at the original `sample_rate` (not fixed 44100).
   - Mix into buffer at beat position, scaled by 0.3.
6. Normalize if any sample exceeds ±1.0.
7. Write WAV using `hound::WavWriter`: 1 channel, original sample rate, 32-bit float.

### 7. Implement `calculate_bpm`

1. If `result.beats.len() < 2`, return `None`.
2. Compute inter-beat intervals: `beats[i] - beats[i-1]` for all consecutive pairs.
3. Filter to range 0.1s–3.0s (20–600 BPM).
4. If no valid intervals remain, return `None`.
5. Sort intervals, take the median.
6. Return `Some(60.0 / median)`.

### 8. Wire up module in `lib.rs`

1. Add `pub mod output;` to `src/lib.rs`.
2. The module is used as `beat_this::output::write_beats_file(...)` etc. — no individual re-exports needed.

### 9. Add unit tests in `src/output.rs`

Tests using synthetic `BeatResult` values (no ONNX models needed):

- **`test_beat_counts_basic`** — 4 beats with downbeat at index 0: expect `[1, 2, 3, 4]`.
- **`test_beat_counts_multiple_downbeats`** — two measures: expect `[1, 2, 3, 1, 2, 3]`.
- **`test_beat_counts_no_downbeats`** — all beats, no downbeats: expect `[1, 2, 3, ...]`.
- **`test_write_beats_file`** — write to a temp file, read back, verify format (`time\tcount\n`, 3 decimal places).
- **`test_write_click_track`** — write to temp file, verify WAV header (mono, 44100 Hz, f32), verify non-zero samples at expected beat positions.
- **`test_write_mixed_audio`** — write to temp file with synthetic original audio, verify WAV is valid and longer than original if beats extend beyond.
- **`test_calculate_bpm`** — beats spaced at 0.5s intervals → expect ~120 BPM.
- **`test_calculate_bpm_too_few_beats`** — single beat → `None`.
- **`test_generate_sine_click`** — verify length, starts near zero (attack), ends near zero (decay), peak amplitude ≈ 1.0.

### 10. Add integration test

Create `tests/output_integration.rs`:

- **`test_full_pipeline_to_beats_file`** — run `BeatThis::process_file`, then `write_beats_file`. Verify file is non-empty, lines are properly formatted, beat counts start at 1 for downbeats.
- **`test_full_pipeline_to_click_track`** — run pipeline, generate click track. Verify WAV file exists and has expected sample rate.
- **`test_full_pipeline_bpm`** — run pipeline on test audio, calculate BPM. Verify result is in a plausible range (e.g., 60–200 BPM for typical music).

All tests skip gracefully if model files or test audio aren't present.

---

## File Changes

| File | Action |
|------|--------|
| `Cargo.toml` | Add `hound = "3.5"` dependency |
| `src/output.rs` | **New** — beat file writer, click track generator, mixed audio, BPM calculation |
| `src/lib.rs` | Add `pub mod output;` declaration |
| `tests/output_integration.rs` | **New** — integration tests with real audio pipeline |

---

## Design Decisions

1. **Free functions, not a struct**: the output functions are stateless transformations — they take a `BeatResult` and a path, and write a file. No struct is needed. This keeps the API simple: `output::write_beats_file(path, &result)`.

2. **`beat_counts` as a public helper**: the C++ `BeatResult` includes `beat_counts` as a field, but computing it requires knowing both `beats` and `downbeats`. Rather than adding it to `BeatResult` (which would complicate `PostProcessor` and duplicate data), expose it as a derived function. Callers who need beat counts can call `output::beat_counts(&result)`.

3. **Click track at fixed 44100 Hz**: click tracks are standalone audio files not tied to the original audio. Using 44100 Hz (CD quality) is standard for audio output. The mixed audio function uses the original sample rate instead, to preserve fidelity.

4. **Mono output only**: the C++ reference supports stereo mixing, but the Rust pipeline works with mono audio throughout (`load_audio` converts to mono). Supporting stereo in `write_mixed_audio` would require the caller to provide stereo samples, which the current `AudioData` doesn't carry. Keeping it mono is consistent with the existing pipeline. Stereo support can be added later if needed.

5. **Normalization only when clipping**: rather than always normalizing (which would change the perceived loudness), only normalize when the peak exceeds 1.0. This preserves the original dynamic range when possible.

6. **`calculate_bpm` returns `Option<f32>`**: rather than returning 0.0 for invalid input (as the C++ version does), use Rust's `Option` type to clearly signal when BPM cannot be computed.

7. **3 decimal places for `.beats` format**: matches the C++ reference and provides millisecond precision, which is sufficient for beat timestamps at 50 fps (20ms resolution).

---

## Validation

1. **Unit tests pass**: all synthetic tests in `src/output.rs` pass without ONNX models.
2. **`.beats` file format**: output matches expected tab-separated format with 3 decimal places.
3. **Click track WAV**: valid WAV file, correct sample rate and format, audible clicks at beat positions.
4. **Mixed audio WAV**: valid WAV file, original audio audible with click overlay.
5. **BPM accuracy**: for test audio with known tempo, calculated BPM is within ±2 BPM of expected value.
6. **Integration tests**: full pipeline → output files complete without errors on real audio.
7. **Cross-reference with C++ output**: for the same audio input, `.beats` file content matches the C++ implementation's output (same beat times, same beat counts).
