# Task 003: Audio Loading — Development Plan

Implements **Step 2** from [plan.md](../001-research/plan.md): audio file decoding, mono conversion, and resampling to 22050 Hz.

## Goal

Create `src/audio.rs` with a single public function that loads any common audio file and returns mono f32 samples at a target sample rate. This is the entry point for the beat-tracking pipeline — all downstream modules (mel spectrogram, inference, postprocessing) consume its output.

---

## Public API

```rust
/// Mono audio data at a known sample rate.
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Load an audio file, convert to mono, and resample to `target_sr`.
pub fn load_audio(path: &Path, target_sr: u32) -> Result<AudioData>;
```

The target sample rate for the Beat This! pipeline is **22050 Hz**, but the function accepts it as a parameter so callers remain explicit.

---

## Implementation Steps

### 1. Add dependencies

Add to `Cargo.toml`:

```toml
symphonia = { version = "0.5", features = ["all"] }
rubato = "0.14"
```

- **symphonia** — pure-Rust audio decoder (MP3, WAV, FLAC, OGG, AAC)
- **rubato** — pure-Rust async resampler (sinc interpolation, high quality)

### 2. Create `src/audio.rs`

Three internal stages, composed in `load_audio`:

```
audio file → [decode] → interleaved f32 + sample_rate + channel_count
           → [to_mono] → mono Vec<f32>
           → [resample] → mono Vec<f32> at target_sr
           → AudioData { samples, sample_rate }
```

#### 2a. Decode (`decode`)

Use symphonia to decode the audio file into interleaved f32 samples:

1. Open file, create `MediaSourceStream`
2. Probe format with `symphonia::default::get_probe()` (with file extension hint)
3. Find first audio track with a supported codec
4. Create decoder for that track
5. Loop over packets:
   - Call `decoder.decode(&packet)` to get an `AudioBufferRef`
   - Convert to `SampleBuffer<f32>` and collect interleaved samples
   - Handle `DecodeError` (skip packet), `IoError` / end-of-stream (break)
6. Return `(samples: Vec<f32>, sample_rate: u32, channels: usize)`

Error cases:
- File not found / unreadable → propagate IO error
- No supported audio track → `anyhow::bail!`
- Unsupported codec → `anyhow::bail!`

Reference: `remixatron_rust/.../audio/loader.rs`

#### 2b. Convert to mono (`to_mono`)

```rust
fn to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }
    samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}
```

All three reference implementations (Python, C++, Rust) use simple channel averaging. No weighting needed for beat tracking.

#### 2c. Resample (`resample`)

Use rubato `SincFixedIn` resampler (fixed input block size, variable output):

1. If `source_sr == target_sr`, return samples unchanged
2. Create `SincFixedIn` with:
   - Resample ratio: `target_sr as f64 / source_sr as f64`
   - Max relative ratio: `2.0`
   - Sinc interpolation parameters (matching remixatron reference):
     - `sinc_len`: 256
     - `f_cutoff`: 0.95
     - `interpolation`: `Linear`
     - `oversampling_factor`: 256
     - `window`: `BlackmanHarris2`
   - 1 channel
   - Input block size from `resampler.input_frames_max()`
3. Process samples in blocks:
   - Slice input into chunks of `input_frames_max`
   - Zero-pad final chunk if needed
   - Call `resampler.process(&[chunk])` for each block
   - Collect output into result vec
4. Flush remaining samples with `resampler.process_partial()`
5. Trim output to expected length: `(input_len as f64 * ratio).ceil() as usize`

Reference: `remixatron_rust/.../audio/loader.rs`

### 3. Wire into `src/lib.rs`

Add `pub mod audio;` to `lib.rs` and re-export `AudioData` and `load_audio`.

### 4. Add integration test

Create `tests/audio_integration.rs`:

- **`test_load_wav`** — load a WAV file from references, verify:
  - `sample_rate == 22050`
  - `samples.len()` matches expected duration (±1% tolerance for resampling)
  - Samples are finite (no NaN/Inf)
  - Samples are in reasonable range (not all zeros if source has content)

- **`test_load_mp3`** — same checks for an MP3 file (if available in references)

- **`test_mono_passthrough`** — load a mono file, verify no degradation

- **`test_resample_identity`** — load a 22050 Hz file, verify no resampling artifacts (samples should be nearly identical)

Tests should skip gracefully if reference audio files aren't present (same pattern as `ort_integration.rs`).

---

## File Changes

| File | Action |
|------|--------|
| `Cargo.toml` | Add `symphonia`, `rubato` dependencies |
| `src/audio.rs` | **New** — decode, mono conversion, resampling |
| `src/lib.rs` | Add `pub mod audio` + re-exports |
| `tests/audio_integration.rs` | **New** — integration tests |

---

## Design Decisions

1. **symphonia over rodio/hound**: symphonia is pure Rust, supports all major formats (MP3, WAV, FLAC, OGG), and is the standard for Rust audio decoding. rodio is playback-focused; hound is WAV-only.

2. **rubato sinc over linear interpolation**: beat tracking accuracy depends on correct spectral content. Sinc resampling with 256-point filter preserves frequencies up to the Nyquist limit. The C++ reference uses linear interpolation, but our Rust reference (remixatron) uses sinc — we follow the higher-quality approach.

3. **Function-based API (not struct)**: `load_audio` is stateless — no configuration to carry between calls. A struct would add unnecessary complexity. If configuration (e.g., resampler quality) is needed later, a builder can be added without breaking the existing function.

4. **`target_sr` as parameter**: although the pipeline always uses 22050, keeping it as a parameter makes the function self-documenting and testable at other rates.

5. **No streaming / chunked reading**: the mel spectrogram ONNX model takes all samples at once (`[1, N]`), so chunked audio loading wouldn't reduce peak memory — we'd still accumulate the full `Vec<f32>` before the next stage. True end-to-end streaming (load → mel → infer in one pass) is a future optimization that should be designed holistically across all pipeline stages, not added to audio loading alone. A 5-min song is ~25 MB; even a 1-hour file is ~300 MB — well within typical memory budgets and dwarfed by model weights.

---

## Validation

- Load a reference audio file and verify `sample_rate == 22050`
- Compare sample count against expected: `duration_seconds * 22050` (±1%)
- Verify output feeds correctly into mel spectrogram in the next step (Step 3)
