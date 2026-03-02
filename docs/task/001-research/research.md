# Beat This! Rust Port — Research

## 1. What is Beat This?

Beat This! is a state-of-the-art beat tracking system from the ISMIR 2024 paper _"Beat This! Accurate Beat Tracking Without DBN Postprocessing"_ by Foscarin, Schlüter, and Widmer. Its key innovation: accurate beat and downbeat detection **without** Dynamic Bayesian Network postprocessing, achieved through specialized loss functions and a hybrid CNN-Transformer architecture.

**Outputs**: beat timestamps (seconds) and downbeat timestamps (seconds).

---

## 2. Pipeline Overview

Both the Python original and the C++ port follow the same pipeline:

```
Audio File (WAV/MP3/FLAC/OGG)
    │
    ▼
Load & convert to mono float32
    │
    ▼
Resample to 22050 Hz
    │
    ▼
Compute Log-Mel Spectrogram → [frames, 128]
    │
    ▼
Split into chunks (1500 frames, 6-frame overlap)
    │
    ▼
ONNX Model Inference → beat logits + downbeat logits per frame
    │
    ▼
Aggregate chunks (keep_first mode)
    │
    ▼
Post-processing: peak picking → beat/downbeat timestamps
```

---

## 3. Audio Processing Parameters

| Parameter        | Value                                        |
| ---------------- | -------------------------------------------- |
| Sample rate      | 22050 Hz                                     |
| Frame rate       | 50 fps (hop_length=441 → 20ms per frame)     |
| FFT size         | 1024                                         |
| Window           | Hann, length=1024                            |
| Hop length       | 441 samples                                  |
| Mel bins         | 128                                          |
| Frequency range  | 30 Hz – 11000 Hz                             |
| Mel scale        | Slaney (not HTK)                             |
| Normalization    | frame_length (divide by sqrt(win_length))    |
| Power            | 1 (amplitude spectrum)                       |
| Log transform    | `log1p(1000 * mel_energy)` with floor 1e-10  |
| Padding          | Reflect, 512 samples each side               |

---

## 4. Model Architecture

Hybrid CNN-Transformer (ROFormer variant):

### Frontend (CNN)
- **Stem**: Conv2D(1→32, kernel=(4,3), stride=(4,1)) + BatchNorm + GELU
- **3 blocks**: each doubles channels (32→64→128→256), halves frequency, includes optional PartialFTTransformer (frequency + time attention with rotary embeddings)
- **Projection**: Linear((256×4) → 512)

### Transformer
- 6 layers of ROFormer (rotary position embeddings)
- dim=512, heads=16, head_dim=32, ff_mult=4
- RMSNorm, gated attention, feedforward with GELU

### Output Head (SumHead)
- Linear(512 → 2) producing beat_logit and downbeat_logit
- Final beat prediction: `beat_logit + downbeat_logit` (ensures downbeats are always beats)

### Model sizes
- **Standard**: ~78 MB (PyTorch) / ~83 MB (ONNX)
- **Small**: ~8 MB

---

## 5. Chunked Inference

For processing long audio files:

| Parameter   | Value                          |
| ----------- | ------------------------------ |
| Chunk size  | 1500 frames (30 seconds)       |
| Border      | 6 frames (120ms) each side     |
| Stride      | 1488 frames                    |
| Aggregation | keep_first (reverse-order)     |

Process: split spectrogram into overlapping chunks, run inference on each, discard border predictions, concatenate.

---

## 6. Post-processing (Minimal, no DBN)

1. **Max-pool** logits with kernel=7, stride=1, padding=3 (±70ms window)
2. **Threshold**: keep peaks where logit > 0 (sigmoid > 0.5)
3. **Deduplicate**: average positions of adjacent peaks (within 1 frame)
4. **Convert** frame indices to seconds (÷ 50)
5. **Snap** downbeats to nearest beat timestamp
6. **Remove** duplicate downbeat times

---

## 7. ONNX Model

The C++ port already includes an ONNX conversion script and pre-converted model.

| Property        | Value                              |
| --------------- | ---------------------------------- |
| File            | `beat_this.onnx` (~83 MB)          |
| Opset           | 14                                 |
| Input name      | `input_spectrogram`                |
| Input shape     | `[1, time_frames, 128]` (dynamic)  |
| Output names    | `beat`, `downbeat`                 |
| Output shapes   | `[1, time_frames]` each            |
| Data type       | float32                            |

Source checkpoint: `final0.ckpt` from the official Beat This! project.

---

## 8. Existing Reference Implementations

### 8.1 Remixatron Rust Reference

The Remixatron project (`references/remixatron_rust/src-tauri/src/`) already implements the **exact same Beat This! pipeline in Rust**. It is a music analysis and "Infinite Jukebox" app that uses Beat This! for beat/downbeat detection, then builds on top with feature extraction, structural analysis, and jump graph playback.

**Beat tracking modules** (directly reusable):

| Module                        | Purpose                                        |
| ----------------------------- | ---------------------------------------------- |
| `beat_tracker/mel.rs`         | Mel spectrogram via ONNX model (not custom FFT)|
| `beat_tracker/inference.rs`   | Chunked ONNX inference (1500 frames, 6 border) |
| `beat_tracker/post_processor.rs` | Peak picking, deduplication, downbeat snapping |
| `audio/loader.rs`             | Symphonia decode + Rubato resample to 22050 Hz |

**Key crate versions used**:

| Crate     | Version      | Purpose                    |
| --------- | ------------ | -------------------------- |
| symphonia | 0.5.5        | Audio decoding             |
| rubato    | 0.14.1       | Sinc resampling            |
| ort       | 2.0.0-rc.10  | ONNX Runtime inference     |
| ndarray   | 0.16.1       | Tensor/array operations    |
| rustfft   | 6.4.1        | FFT (used in analysis, not mel) |
| anyhow    | 1.0          | Error handling             |

**Important finding — Mel spectrogram approach**: Remixatron does NOT compute the mel spectrogram with custom FFT code. Instead, it uses a **separate ONNX model** that takes raw PCM audio and outputs the mel spectrogram directly. This avoids the risk of numerical mismatch entirely — the mel computation runs the same torchaudio code exported to ONNX.

**Rubato configuration** (high-quality sinc resampling):
- `sinc_len`: 256, `f_cutoff`: 0.95
- `interpolation`: Linear (Sinc variant)
- `oversampling_factor`: 256
- `window`: BlackmanHarris2

**What Remixatron adds beyond beat tracking** (not needed for our port):
- MFCC, Chroma, CQT feature extraction for beat similarity
- Structural segmentation via novelty curves + clustering (linfa)
- Jump graph construction for infinite playback
- Audio playback engine (kira)
- Tauri desktop app + Chromecast streaming

**Impact on our approach**: The Remixatron codebase validates our recommended crate stack (symphonia + rubato + ort + ndarray) and provides working Rust code for every pipeline stage. The mel-via-ONNX approach is an interesting alternative to custom FFT — see section 9.5.

### 8.2 C++ Port Reference

The C++ port (`references/beat_this_cpp/`) is ~1370 lines across 5 source files:

| Component              | Lines | Purpose                              |
| ---------------------- | ----- | ------------------------------------ |
| `beat_this_api.h/cpp`  | ~290  | Public API with Pimpl pattern        |
| `MelSpectrogram.h/cpp` | ~260  | Mel spectrogram from scratch         |
| `InferenceProcessor`   | ~230  | ONNX Runtime + chunking              |
| `Postprocessor`        | ~175  | Peak picking + snapping              |
| `main.cpp`             | ~410  | CLI + audio output generation        |

Dependencies: ONNX Runtime 1.18.0, miniaudio (audio I/O), PocketFFT (FFT).

---

## 9. Rust Implementation Options

### 9.1 ONNX Inference

| Crate     | ONNX Support   | GPU  | Pure Rust | Maturity  | Verdict                |
| --------- | -------------- | ---- | --------- | --------- | ---------------------- |
| **ort**   | Full (runtime) | Yes  | No (C++)  | High (RC) | **Best choice**        |
| tract     | ~85%           | No   | Yes       | High      | Good fallback          |
| candle    | Partial        | Yes  | Mostly    | Medium    | ONNX support too weak  |
| burn      | Compile-time   | Yes  | Yes       | Medium    | Not suited for ONNX    |

**Recommendation: `ort`** (v2.0.0-rc.11, wraps ONNX Runtime 1.23)

Pros:
- Full ONNX operator coverage (transformer models work out of the box)
- Dynamic input shapes supported natively
- Runtime model loading from `.onnx` files
- GPU acceleration: CUDA, CoreML, TensorRT, DirectML
- Production-proven (HuggingFace TEI, Google Magika, rust-bert)
- Cross-platform with prebuilt binaries

Cons:
- Not pure Rust (wraps C++ ONNX Runtime)
- Still in release candidate (pin exact version)
- Adds ~20 MB binary size from ONNX Runtime dylib

**Alternative: `tract`** if pure Rust / no C deps is critical, or WASM target needed. CPU-only.

### 9.2 Audio Loading

| Crate       | Formats             | Pure Rust | Notes                        |
| ----------- | ------------------- | --------- | ---------------------------- |
| **symphonia** | WAV/MP3/FLAC/OGG+ | Yes       | **Best choice**              |
| symphonium  | Same (wraps above)  | Yes       | Convenience wrapper          |
| hound       | WAV only            | Yes       | Too limited                  |
| rodio       | Same as symphonia   | Yes       | Overkill (playback library)  |

**Recommendation: `symphonia`** — pure Rust, all required formats, battle-tested.

### 9.3 Resampling

| Crate        | Quality  | Pure Rust | Notes                  |
| ------------ | -------- | --------- | ---------------------- |
| **rubato**   | High     | Yes       | **Best choice**        |
| dasp         | Basic    | Yes       | More of a building block |
| samplerate   | High     | No (C)    | C dependency           |

**Recommendation: `rubato`** — high-quality sinc resampling, pure Rust, real-time safe.

Note: The C++ port uses linear interpolation (miniaudio). Rubato's sinc resampler is closer to the Python pipeline's quality (torchaudio uses soxr).

### 9.4 FFT

| Crate       | Type         | Pure Rust | Notes                      |
| ----------- | ------------ | --------- | -------------------------- |
| rustfft     | Complex FFT  | Yes       | SIMD-accelerated           |
| **realfft** | Real FFT     | Yes       | **Best choice** (wraps rustfft, ~2x faster for real input) |

**Recommendation: `realfft`** — optimized for real-valued audio input. 1024→513 complex bins.

### 9.5 Mel Spectrogram

| Option                    | Pros                              | Cons                                |
| ------------------------- | --------------------------------- | ----------------------------------- |
| **ONNX mel model**        | Exact parity guaranteed, zero risk | Extra ONNX model file (~few MB), two inference calls |
| **Custom implementation** | Full control, no extra model file | ~200-300 lines, risk of numerical mismatch |
| mel_spec crate            | Fast, Whisper-compatible          | May not support Slaney / custom params |
| spectrs crate             | Batteries-included, Slaney support | Newer, less tested                   |

**Option A (Remixatron approach): ONNX mel model** — Export the torchaudio mel spectrogram as a separate ONNX model. The mel computation runs the exact same code used during training, eliminating numerical mismatch risk entirely. This is what Remixatron does successfully. Tradeoff: requires distributing an additional ONNX model file.

**Option B (C++ approach): Custom implementation** using `realfft` — The C++ reference (`MelSpectrogram.cpp`) is ~200 lines and translates directly to Rust. Gives full control and no extra model dependency, but requires careful validation against Python output.

**Recommendation**: Start with **Option A** (ONNX mel model) for faster development and guaranteed correctness. Consider Option B later if the extra model file is undesirable.

---

## 10. Recommended Rust Crate Stack

Already validated by Remixatron in production:

| Pipeline Stage        | Crate              | Version      | Pure Rust |
| --------------------- | ------------------ | ------------ | --------- |
| Audio decoding        | symphonia          | 0.5.5        | Yes       |
| Resampling            | rubato             | 0.14+        | Yes       |
| ONNX inference        | ort                | 2.0.0-rc.10+ | No        |
| Array operations      | ndarray            | 0.16+        | Yes       |
| Error handling        | anyhow             | 1.0          | Yes       |
| WAV writing           | hound              | 3.5+         | Yes       |
| CLI                   | clap               | 4.x          | Yes       |
| FFT (custom mel)      | realfft            | 3.5.0        | Yes       |

All crates except `ort` are pure Rust. The `ort` crate bundles prebuilt ONNX Runtime binaries for macOS/Linux/Windows.

---

## 11. Implementation Plan

### Components to build

1. **Audio loader** (~50 lines) — symphonia decode → mono f32 → rubato resample to 22050 Hz. Reference: `remixatron_rust/.../audio/loader.rs`
2. **Mel spectrogram** — either ONNX model (~30 lines, reference: `remixatron_rust/.../beat_tracker/mel.rs`) or custom FFT (~250 lines, reference: `beat_this_cpp/Source/MelSpectrogram.cpp`)
3. **Inference processor** (~150 lines) — chunk splitting, ort session management, chunk aggregation. Reference: `remixatron_rust/.../beat_tracker/inference.rs`
4. **Postprocessor** (~100 lines) — 1D max-pool, peak detection, deduplication, frame-to-time, downbeat snapping. Reference: `remixatron_rust/.../beat_tracker/post_processor.rs`
5. **Public API** (~50 lines) — `BeatThis` struct with `process_audio()` method
6. **CLI** (~100 lines) — clap-based command line interface

Estimated total: ~500-700 lines of Rust.

### Module structure

```
src/
├── lib.rs              # Public API (BeatThis struct)
├── audio.rs            # Audio loading + resampling
├── mel.rs              # Mel spectrogram computation
├── inference.rs        # ONNX model inference + chunking
├── postprocessing.rs   # Peak picking + beat extraction
└── main.rs             # CLI binary
```

### Validation strategy

- Compare mel spectrogram output against Python/C++ on the same audio file
- Compare model inference output (logits) for the same spectrogram input
- Compare final beat timestamps end-to-end
- Use the same test audio files used in the C++ port
- Can also cross-validate against Remixatron's output on the same audio

---

## 12. Risks and Mitigations

| Risk                                      | Impact | Mitigation                                        |
| ----------------------------------------- | ------ | ------------------------------------------------- |
| Mel spectrogram numerical mismatch        | High   | Use ONNX mel model (zero risk); or validate custom impl against Python |
| ort crate API instability (RC)            | Low    | Pin exact version; already used by Remixatron in production |
| ONNX opset 14 operator support            | Low    | ort wraps official runtime; full support           |
| Resampling quality differences            | Medium | Rubato sinc is closer to Python's soxr than C++'s linear |
| Large binary size from ONNX Runtime       | Low    | Acceptable tradeoff for full model support         |

---

## 13. Decisions

1. **Mel spectrogram**: Start with ONNX mel model for guaranteed correctness. Implement custom FFT version later.
2. **GPU acceleration**: Yes. Performance is a goal. Will measure and optimize iteratively.
3. **Model variants**: Support all variants (standard ~83 MB, small ~8 MB, etc.). Same ONNX interface, different weight files.
4. **Audio output**: Yes — beat file output + click track WAV generation (using `hound` for WAV writing), matching C++ port features.
5. **Runtime abstraction**: Design a trait-based abstraction over inference backends from the start. Begin with `ort`, but the trait boundary must allow swapping to `tract` or other runtimes for benchmarking. This is a first-class architectural concern.
6. **Remixatron reuse**: Use as reference only, no code copying. Write fresh implementations informed by its patterns.
