# Task 013: rten Runtime — Performance Results

## Setup

- **Machine**: Apple Silicon M1
- **Rust**: 1.93.1 (release build)
- **ort**: 2.0.0-rc.10 wrapping Homebrew ORT 1.24.2 (CPU only, no CoreML), threads=auto
- **rten**: 0.24.0 (pure Rust, Rayon threading, default core count)
- **Model**: beat_this.onnx (83 MB, standard)

## Benchmark: 3 files, ort vs rten

### Short — 9.3s audio, 468 mel frames, 1 chunk

|          | Model Load | Mel    | Beat Inference | Total  | Beats      |
| -------- | ---------- | ------ | -------------- | ------ | ---------- |
| **ort**  | 0.382s     | 0.008s | **0.364s**     | 0.782s | 16 (13 db) |
| **rten** | 0.020s     | 0.003s | **0.387s**     | 0.423s | 16 (13 db) |

### 01 test1.mp3 — 4.5 min, 13602 mel frames, 10 chunks

|          | Model Load | Mel    | Beat Inference | Total  | Beats        |
| -------- | ---------- | ------ | -------------- | ------ | ------------ |
| **ort**  | 0.348s     | 0.193s | **3.371s**     | 4.295s | 460 (122 db) |
| **rten** | 0.021s     | 0.091s | **3.775s**     | 4.245s | 460 (122 db) |

### 02 Test2.mp3 — 3.4 min, 10144 mel frames, 7 chunks

|          | Model Load | Mel    | Beat Inference | Total  | Beats        |
| -------- | ---------- | ------ | -------------- | ------ | ------------ |
| **ort**  | 0.354s     | 0.147s | **2.369s**     | 3.173s | 584 (121 db) |
| **rten** | 0.021s     | 0.066s | **2.632s**     | 2.990s | 584 (121 db) |

## Output Correctness

All three files produce **byte-identical** `.beats` output between ort and rten. Same beat count, same downbeat count, same BPM, same timestamps.

## Analysis

### Beat inference (the bottleneck)

| File              | ort    | rten   | rten / ort         |
| ----------------- | ------ | ------ | ------------------ |
| Short (1 chunk)   | 0.364s | 0.387s | 1.06x (6% slower)  |
| test1 (10 chunks) | 3.371s | 3.775s | 1.12x (12% slower) |
| Test2 (7 chunks)  | 2.369s | 2.632s | 1.11x (11% slower) |

rten beat inference is ~6–12% slower than Homebrew ORT CPU. This matches the research estimate of "80–100% of ort on CPU."

### Model loading

|         | ort   | rten   | Speedup        |
| ------- | ----- | ------ | -------------- |
| Average | 0.36s | 0.021s | **17x faster** |

rten loads models in ~20ms vs ort's ~360ms. No dynamic library resolution, no session optimization overhead.

### Mel spectrogram

| File  | ort    | rten   | Speedup |
| ----- | ------ | ------ | ------- |
| Short | 0.008s | 0.003s | 2.7x    |
| test1 | 0.193s | 0.091s | 2.1x    |
| Test2 | 0.147s | 0.066s | 2.2x    |

rten is ~2x faster on mel spectrogram computation (STFT + mel filterbank).

### Total wall time

| File  | ort    | rten       | Winner       |
| ----- | ------ | ---------- | ------------ |
| Short | 0.782s | **0.423s** | rten (1.8x)  |
| test1 | 4.295s | **4.245s** | rten (1.01x) |
| Test2 | 3.173s | **2.990s** | rten (1.06x) |

rten wins on total wall time for all three files thanks to faster model loading and mel computation, despite slightly slower beat inference.

## Comparison with Python

From the previous research (task 012):

| Runtime                     | Beat Inference (4.5-min track) |
| --------------------------- | ------------------------------ |
| Python PyTorch CPU          | ~5s                            |
| Rust ort CPU (threads=auto) | 3.4s                           |
| **Rust rten (pure Rust)**   | **3.8s**                       |

Both Rust runtimes are faster than Python. rten is ~25% faster than PyTorch while being pure Rust with no C/C++ dependencies.

## Conclusion

rten is a viable production runtime for this project:

- **Identical results** to ort on all test files
- **~10% slower** on beat inference (the bottleneck), offset by faster loading and mel
- **Total wall time is comparable or faster** than ort
- **Pure Rust**: no dylib management, no CoreML issues, no SIGABRT on exit
- **17x faster model loading**: significant for CLI startup and batch processing
