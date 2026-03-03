# Task 012: Performance Investigation — Research

## Context

Python (PyTorch CPU) is ~3x faster than Rust (ONNX Runtime CPU) for beat inference on Apple Silicon (M1). This document captures all investigation and findings.

## 1. CoreML Evaluation

### Approach

Tested CoreML execution provider as an alternative to CPU-only inference, hoping Apple's Neural Engine or GPU could accelerate the transformer model.

### Building ORT from Source

Attempted to build ONNX Runtime 1.25.0 from source with `--use_coreml`:

```bash
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
./build.sh --config Release --build_shared_lib --parallel --use_coreml \
  --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64
```

**Failed** due to:

1. **CMake 4.x incompatibility** — The `psimd` dependency uses `cmake_minimum_required` below 3.5, which CMake 4.0.3 rejects entirely. See [onnxruntime#23556](https://github.com/microsoft/onnxruntime/issues/23556).
2. **Abseil version conflict** — Homebrew's `abseil` package conflicts with ORT's bundled version. `CMAKE_IGNORE_PREFIX_PATH="/opt/homebrew"` avoids the Abseil issue but doesn't fix the CMake version problem.

**Workaround**: Use pre-built official ORT releases from GitHub which include CoreML support.

### CoreML Benchmark Results

Test file: `integration_test_files/short.wav` (9.3s audio, 468 mel frames)

| Config                         | Model Load | Beat Inference | Total |
| ------------------------------ | ---------- | -------------- | ----- |
| Homebrew ORT 1.24.2 (CPU only) | 0.37s      | **1.6s**       | 2.0s  |
| Official ORT 1.24.2 (CoreML)   | 2.1s       | **8.0s**       | 10.1s |

Test file: `integration_test_files/Test1.mp3` (4.5 min, 13602 mel frames)

| Config                                                | Model Load     | Beat Inference | Total |
| ----------------------------------------------------- | -------------- | -------------- | ----- |
| Homebrew ORT 1.24.2 (CPU only)                        | 0.42s          | **15.7s**      | 16.8s |
| Official ORT 1.24.2 (CoreML)                          | >120s (killed) | —              | —     |
| Official ORT 1.22.0 CoreML NeuralNetwork (prev. task) | 2.4s           | **28.7s**      | 31.8s |
| Official ORT 1.22.0 CoreML MLProgram (prev. task)     | 11.1s          | **62.9s**      | 74.7s |
| Python PyTorch CPU                                    | —              | ~5s            | ~5s   |

### CoreML Conclusion

**CoreML is a dead end for this model.** The transformer architecture with dynamic axes doesn't map well to CoreML's compute graph. CoreML adds model compilation overhead (seconds to minutes) and runs inference slower than CPU. The "Context leak detected, msgtracer returned -1" warnings during CoreML inference further indicate poor compatibility.

## 2. ORT Profiling Analysis

### Setup

Added `--profile <prefix>` flag to the CLI that enables ORT's built-in profiling via `SessionBuilder::with_profiling()`. Generates Chrome Trace Event JSON files with per-operator timing.

### Operator Breakdown

Profile of beat model on `short.wav` (1 chunk of 1500 frames):

```
Total operator time: 1.717s
Number of operator calls: 1737

Operator                        Time (ms)  Count  % Total
----------------------------------------------------------
MatMul                             1250.3     86    72.8%
Softmax                             208.1     12    12.1%
BiasGelu                             89.2     12     5.2%
Conv                                 41.2      4     2.4%
Transpose                            25.4    121     1.5%
Split                                19.9     37     1.2%
Mul                                  19.9    152     1.2%
ReduceL2                             18.5     25     1.1%
Add                                  10.2     75     0.6%
Gelu                                  8.6      4     0.5%
Concat                                8.5     88     0.5%
Expand                                7.8     29     0.5%
(everything else)                    ~10.0   ~800    ~0.6%
```

**MatMul (72.8%) + Softmax (12.1%) = 85% of total time.**

### Top MatMul Calls (Attention Operations)

The most expensive individual MatMul calls are the attention Q*K^T and Attn*V multiplications:

| Layer                                 | Shape                                   | Time (ms) |
| ------------------------------------- | --------------------------------------- | --------- |
| frontend/blocks.0 attn Q\*K^T         | [32, 1, 1500, 32] x [32, 1, 32, 1500]   | 66.6      |
| frontend/blocks.0 attn\*V             | [32, 1, 1500, 1500] x [32, 1, 1500, 32] | 67.6      |
| frontend/blocks.1 attn Q\*K^T         | [16, 2, 1500, 32] x [16, 2, 32, 1500]   | 67.6      |
| frontend/blocks.1 attn\*V             | [16, 2, 1500, 1500] x [16, 2, 1500, 32] | 48.7      |
| frontend/blocks.2 attn\*V             | [8, 4, 1500, 1500] x [8, 4, 1500, 32]   | 48.4      |
| frontend/blocks.2 attn Q\*K^T         | [8, 4, 1500, 32] x [8, 4, 32, 1500]     | 38.5      |
| transformer_blocks/layers.3.0 attn\*V | [1, 16, 1500, 1500] x [1, 16, 1500, 32] | 29.6      |

The frontend "partial attention" blocks (blocks.0-2) are particularly expensive with large batch dimensions (32, 16, 8 attention heads).

### Top Softmax Calls

| Layer                                 | Shape               | Time (ms) |
| ------------------------------------- | ------------------- | --------- |
| frontend/blocks.0 softmax             | [32, 1, 1500, 1500] | 76.9      |
| frontend/blocks.1 softmax             | [16, 2, 1500, 1500] | 26.1      |
| frontend/blocks.2 softmax             | [8, 4, 1500, 1500]  | 25.7      |
| transformer_blocks/layers.1.0 softmax | [1, 16, 1500, 1500] | 13.5      |

### Feed-Forward Network MatMuls

The transformer FFN layers also contribute significantly:

- Each FFN has two MatMuls: `[1, 1500, 512]` and `[1, 1500, 2048]` (~27ms each)
- 6 transformer layers × 2 MatMuls = ~320ms total

### Model Architecture Summary

The model has:

- **3 frontend "partial attention" blocks** with 32, 16, 8 heads respectively
- **6 transformer blocks** with 16 heads each
- **12 attention operations total** (Q*K^T + Softmax + Attn*V each)
- All operating on 1500-frame chunks (30 seconds at 50 fps)

## 3. Thread Count Tuning (Solution)

The default `intra_threads` was set to 1, meaning ORT ran all MatMul/Softmax operations single-threaded. Changing to `0` (auto) lets ORT use all available cores and parallelizes the batched MatMul operations.

Added `--threads` CLI flag and changed default from 1 to 0.

### Thread Count Benchmark — short.wav (9.3s audio)

| Threads  | Beat Inference | Speedup vs t=1 |
| -------- | -------------- | -------------- |
| 1        | 1.72s          | 1.0x           |
| 2        | 0.98s          | 1.8x           |
| 4        | 0.57s          | 3.0x           |
| 6        | 0.47s          | 3.7x           |
| 8        | 0.59s          | 2.9x           |
| **10**   | **0.35s**      | **4.9x**       |
| 0 (auto) | 0.40s          | 4.3x           |
| 12       | 0.52s          | 3.3x           |
| 16       | 0.64s          | 2.7x           |

Sweet spot is ~10 threads on M1 (8 perf + 2 efficiency cores). Auto (0) is close and portable.

### Thread Count Benchmark — Test1.mp3 (4.5 min audio)

| Threads      | Beat Inference | Total    | Speedup  |
| ------------ | -------------- | -------- | -------- |
| 1            | 15.7s          | 16.8s    | 1.0x     |
| 10           | 3.9s           | 4.9s     | 4.0x     |
| **0 (auto)** | **3.4s**       | **4.4s** | **4.6x** |

### Result

With `intra_threads: 0`, Rust now matches Python performance:

| Runtime                               | Beat Inference (4.5-min track) |
| ------------------------------------- | ------------------------------ |
| Python PyTorch CPU                    | ~5s                            |
| **Rust ORT CPU (threads=0)**          | **3.4s**                       |
| Rust ORT CPU (threads=1, old default) | 15.7s                          |

**Rust is now ~1.5x faster than Python** for beat inference, not 3x slower.

## 4. Root Cause Analysis

The original 3x performance gap was primarily due to **single-threaded execution**. PyTorch defaults to using all cores, while ORT with `intra_threads: 1` serialized all operations.

The profiling data explains why threading helps so much: MatMul (73% of time) operates on batched attention matrices with large batch dimensions (32, 16, 8 heads), which parallelize efficiently across cores.

## 5. ORT Transformer Optimizer (No Effect)

Ran `python -m onnxruntime.transformers.optimizer` via:

```bash
uvx --from onnxruntime --with onnx --with torch python -m onnxruntime.transformers.optimizer \
  --input models/beat_this.onnx --output models/beat_this_optimized.onnx \
  --num_heads 16 --hidden_size 512 --only_onnxruntime --verbose
```

**Result: No fusions achieved.** The model's custom attention pattern (uses Einsum, Split, Concat instead of standard Q/K/V projections) doesn't match the BERT/GPT patterns the optimizer recognizes. All fused operator counts remained at 0:

```
Optimized operators: Attention: 0, MultiHeadAttention: 0, Gelu: 0,
BiasGelu: 0, LayerNormalization: 0, SkipLayerNormalization: 0
```

## 6. FP16 Conversion (No Speed Improvement)

Converted the model to float16 using ORT's `convert_float_to_float16` with problematic ops (Range, CumSum, ScatterND, Cast, Einsum) kept in FP32:

```bash
uvx --from onnxruntime --with onnx --with torch python -c "
from onnxruntime.transformers.float16 import convert_float_to_float16
import onnx
model = onnx.load('models/beat_this.onnx')
model_fp16 = convert_float_to_float16(model, keep_io_types=True,
    op_block_list=['Range', 'CumSum', 'ScatterND', 'Cast', 'Einsum'])
onnx.save(model_fp16, 'models/beat_this_fp16.onnx')
"
```

Note: A naive FP16 conversion (without the block list) fails because the `Range` operator doesn't support float16 inputs.

### FP16 Benchmark — short.wav (9.3s audio, auto threads)

| Model           | Size | Beat Inference | Total |
| --------------- | ---- | -------------- | ----- |
| Standard (FP32) | 79MB | 0.52s          | 0.89s |
| FP16            | 41MB | 0.56s          | 1.22s |

**No speed improvement.** On CPU, FP16 tensors are cast back to FP32 for computation since x86/ARM SIMD doesn't natively compute in FP16. The model is just smaller on disk (41MB vs 79MB). Model loading is slightly slower due to cast overhead.

## 7. Small Model Comparison

| Model               | Size  | Beat Inference | Beats Found       |
| ------------------- | ----- | -------------- | ----------------- |
| Standard            | 79MB  | 0.52s          | 16 (13 downbeats) |
| Small (small1.onnx) | ~10MB | 0.33s          | 16 (12 downbeats) |

The small model is **37% faster** with nearly identical beat detection (1 fewer downbeat). Could be offered as a "fast mode" for real-time use cases.

## 8. Summary

| Optimization                 | Impact                                     |
| ---------------------------- | ------------------------------------------ |
| **Thread tuning (1 → auto)** | **4.6x speedup (the fix)**                 |
| ORT Transformer Optimizer    | No effect (non-standard attention pattern) |
| FP16 conversion              | No speed gain (CPU casts back to FP32)     |
| Small model                  | 37% faster, slightly less accurate         |
| CoreML                       | 5x slower (dead end)                       |

## Files Changed

- `src/runtime/ort.rs` — Added `profiling_path` option, `end_profiling()`, changed default `intra_threads` from 1 to 0
- `src/main.rs` — Added `--profile` and `--threads` CLI flags, profiling lifecycle management
- `src/mel.rs` — Added `session_mut()` accessor
- `src/inference.rs` — Added `session_mut()` accessor
