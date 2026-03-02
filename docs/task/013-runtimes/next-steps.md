# Task 013: rten Runtime — Next Steps

## Current State

rten is working and producing identical results to ort. Beat inference is ~10% slower than Homebrew ORT CPU, but total wall time is comparable or faster thanks to 17x faster model loading and 2x faster mel spectrogram.

The bottleneck is beat inference, dominated by MatMul (73%) and Softmax (12%) in the transformer attention layers.

## Optimization Opportunities

### 1. Int8 Quantization — HIGH ROI

**Expected gain**: 2-3x faster beat inference (3.8s → ~1.5-2s on 4.5-min track)

MatMul is 73% of inference time. rten supports int8/uint8 quantized models with hardware-accelerated integer matmul:
- ARM: UDOT / i8mm instructions (Apple Silicon M1+)
- x86: VNNI instructions
- rten explicitly supports `MatMulNBits` operator

**How to do it**:
```bash
# Dynamic quantization (easiest, quantizes weights to int8)
python -m onnxruntime.quantization.quantize \
  --input models/beat_this.onnx \
  --output models/beat_this_int8.onnx \
  --quant_format QDQ
```

Or use the Python API for more control:
```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("beat_this.onnx", "beat_this_int8.onnx", weight_type=QuantType.QInt8)
```

**Validation**: run all 3 test files, compare `.beats` output to f32 baseline. Accept if timestamps match within ±1 frame (20ms).

**Risk**: accuracy may degrade on edge cases (quiet passages, complex rhythms). The model's attention layers may be sensitive to quantization noise.

### 2. Small Model as Default — ZERO EFFORT

**Expected gain**: 37% faster beat inference

Already benchmarked in task 012: small model (10 MB) gives 0.33s vs 0.52s on short.wav with nearly identical beat detection (1 fewer downbeat out of 13).

**How to do it**: change the CLI default from `standard` to `small`, or add a `--fast` flag.

**Risk**: slightly less accurate on some tracks. Could offer both and let users choose.

### 3. Rayon Thread Tuning — LOW EFFORT

**Expected gain**: 0-20%

rten defaults to physical core count via Rayon. ORT's optimal was 10 threads on M1. rten may have a different sweet spot.

**How to do it**: benchmark with `RAYON_NUM_THREADS=N` for N in {4, 6, 8, 10, 12}:
```bash
for n in 4 6 8 10 12; do
  echo "=== threads=$n ==="
  RAYON_NUM_THREADS=$n ./target/release/beat-this audio.mp3 -v --runtime rten
done
```

If there's a meaningful sweet spot, expose it as a `--threads` option for rten (currently only affects ort).

### 4. Memory-Mapped Loading — LOW ROI

**Expected gain**: 20ms → ~5ms model loading

rten supports `Model::load_mmap(path)` (unsafe, requires `mmap` feature). Loading is already 20ms so the absolute gain is negligible for CLI usage. More relevant for library consumers who load/unload models frequently.

## Not Worth Pursuing

| Optimization | Why not |
|---|---|
| FP16 model | Proven no-op on CPU in task 012 — ARM NEON casts back to f32 |
| ONNX graph optimization | Model's non-standard attention (Einsum/Split/Concat) prevents fusions |
| Chunk-level parallelism | Conflicts with rten's internal Rayon pool |
| CoreML / GPU | rten is CPU-only; ort CoreML proven 5x slower for this model |

## Priority Order

1. **Thread tuning** — 30 min benchmark, free improvement if one exists
2. **Int8 quantization** — biggest potential gain, requires model conversion + accuracy validation
3. **Small model default** — product decision, no code needed
4. Memory-mapped loading — defer until needed
