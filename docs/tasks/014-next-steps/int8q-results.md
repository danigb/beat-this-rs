# Int8 Quantization — Results

**Date:** 2026-03-03
**Machine:** Apple Silicon (M1/M2), macOS, ORT runtime with CoreML
**Quantization:** Dynamic Int8 (MatMul ops only)

## Model Size

| Model | Size | Ratio |
|-------|------|-------|
| `beat_this.onnx` (FP32) | 79.3 MB | 1.0x |
| `beat_this_int8.onnx` (Int8) | 22.6 MB | 3.5x smaller |

## Beat Inference Speedup

Pure beat inference time (excluding model loading, audio loading, mel spectrogram):

| Audio File | Duration | FP32 | Int8 | Speedup |
|------------|----------|------|------|---------|
| short.wav | ~10s | 0.738s | 0.436s | **1.69x** |
| test1.mp3 | ~4.5min | 5.447s | 3.884s | **1.40x** |
| test2.mp3 | ~13min | 17.829s | 12.158s | **1.47x** |

Average inference speedup: **~1.5x**

## Accuracy (Nearest-Match Comparison, 20ms tolerance)

| Audio File | FP32 Beats | Matched | Missed | Spurious | Max Dev | Result |
|------------|-----------|---------|--------|----------|---------|--------|
| short.wav | 16 | 16 | 0 | 0 | 0.0ms | PASS |
| test1.mp3 | 460 | 460 | 0 | 0 | 20.0ms | PASS |
| test2.mp3 | 2259 | 2236 | 23 | 25 | 20.0ms | FAIL |

### test2.mp3 Analysis

- 23 missed + 25 spurious out of 2259 beats = **~2% beat error rate**
- For matched beats, max deviation is 20ms (1 frame) — positional accuracy is good
- The missed/spurious beats may occur at section boundaries or during tempo changes
- Downbeat matching: 947/974 matched, 27 missed, 13 spurious

## Success Criteria Assessment

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Model size < 25 MB | < 25 MB | 22.6 MB | PASS |
| Beats within ±1 frame | ±20ms | ±20ms (matched) | PASS* |
| No missed/spurious beats | 0 | 23/25 on test2 | FAIL* |
| Inference ≥ 1.5x faster | 1.5x | 1.5x avg | PASS |
| Loads on ORT runtime | yes | yes | PASS |

*test2.mp3 has ~2% error rate on a 13-minute track. short.wav and test1.mp3 pass fully.

## Recommendation

The Int8 quantized model is suitable for use where:
- A 3.5x model size reduction is valuable (22.6 MB vs 79.3 MB)
- ~1.5x inference speedup is desired
- A small (~2%) beat error rate on very long tracks is acceptable

For applications requiring bit-exact accuracy on all tracks, continue using the FP32 model.

### Future Work

- **Static quantization** with calibration data may improve both accuracy and speed
- **ONNX model optimization** (graph folding, operator fusion) before quantization
- **Per-layer quantization analysis** to identify which layers cause the test2 errors
