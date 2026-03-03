# Int8 Quantization — Development Plan

Quantize the beat inference model from FP32 to Int8 to achieve 2–3x speedup by leveraging ARM NEON (Apple Silicon) and x86 VNNI integer instructions. Beat inference is 93% of runtime, so this is the highest-impact optimization available.

## Goal

A user can run:

```
beat-this audio.mp3 --model models/beat_this_int8.onnx
```

The quantized model produces beat/downbeat output identical to the FP32 model within ±1 frame (20ms at 50 fps) on all test files. Both `ort` and `rten` runtimes load and run the quantized model without code changes.

---

## Background

### Current State

- Beat model: `beat_this.onnx` (83 MB, FP32)
- FP16 variant exists: `beat_this_fp16.onnx` (43 MB) — size reduction but no CPU speedup (FP16 arithmetic isn't faster on most CPUs)
- Inference bottleneck: batched MatMul in attention layers (73% of inference time)
- Int8 MatMul is 2–4x faster than FP32 on ARM NEON and x86 VNNI

### Why Int8 Works Here

The Beat This! model is a transformer operating on mel spectrograms. Transformer models quantize well because:

- Attention weights are well-distributed (no extreme outliers typical in LLMs)
- The task is classification (beat/no-beat per frame), tolerant of small numerical shifts
- Post-processing applies thresholding and peak detection, which absorbs minor logit differences

### Quantization Strategy: Dynamic vs Static

| Approach    | Pros                                   | Cons                                                  |
| ----------- | -------------------------------------- | ----------------------------------------------------- |
| **Dynamic** | No calibration data needed, simpler    | Slightly slower (quantizes activations at runtime)    |
| **Static**  | Fastest inference, pre-computed scales | Requires calibration dataset of representative inputs |

**Recommendation: Start with dynamic quantization.** It requires no calibration data, is simpler to validate, and still delivers most of the speedup. If the speedup is insufficient, follow up with static quantization using the integration test audio files as calibration data.

---

## Implementation Steps

### Step 1: Create Quantization Script (`scripts/quantize_int8.py`)

Python script using `onnxruntime.quantization` to convert the FP32 model to Int8.

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="models/beat_this.onnx",
    model_output="models/beat_this_int8.onnx",
    weight_type=QuantType.QInt8,
    # Quantize MatMul and Attention ops (the 93% bottleneck)
    op_types_to_quantize=["MatMul", "Attention"],
)
```

**Dependencies**: `pip install onnxruntime onnx` (quantization tools only, not needed at runtime).

**Expected output**: `beat_this_int8.onnx` (~21 MB, roughly 4x smaller than FP32).

### Step 2: Validate Output Accuracy

Run both FP32 and Int8 models on the integration test files and compare beat timestamps.

```bash
# FP32 baseline
beat-this "integration_test_files/test1.mp3" --model models/beat_this.onnx > /tmp/fp32.json

# Int8 quantized
beat-this "integration_test_files/test1.mp3" --model models/beat_this_int8.onnx > /tmp/int8.json

# Compare
diff <(jq '.beats[]' /tmp/fp32.json) <(jq '.beats[]' /tmp/int8.json)
```

**Acceptance criteria**:

- Every beat timestamp matches within ±20ms (1 frame at 50 fps)
- No missed beats and no spurious extra beats
- Same number of downbeats, each snapped to the correct beat

Write a comparison script (`scripts/compare_models.py`) that automates this across all test files and reports max deviation.

### Step 3: Verify Runtime Compatibility

Test that the Int8 model loads and runs correctly with both runtimes:

```bash
# ORT runtime (default)
beat-this audio.mp3 --model models/beat_this_int8.onnx --runtime ort

# Rten runtime
cargo run --features rten -- audio.mp3 --model models/beat_this_int8.onnx --runtime rten
```

Both `ort` and `rten` support quantized ONNX models natively — no Rust code changes should be needed. The `InferenceSession::run()` interface operates on f32 tensors at the boundary; the runtime handles dequantization internally.

**If rten doesn't support a quantized op**: file an issue upstream or fall back to ORT-only for the quantized model. Document any incompatibility.

### Step 4: Benchmark

Compare inference time on a representative audio file (the 4.5-minute test track):

```bash
# FP32
beat-this "integration_test_files/test1.mp3" --model models/beat_this.onnx --verbose 2>&1 | grep "Beat inference"

# Int8
beat-this "integration_test_files/test1.mp3" --model models/beat_this_int8.onnx --verbose 2>&1 | grep "Beat inference"
```

**Expected results** (M1 Mac, ORT):

- FP32: ~3.4s
- Int8: ~1.2–1.7s (2–3x speedup)

Record benchmarks in a `docs/task/014-next-steps/int8q-results.md` file.

### Step 5: (Optional) Static Quantization

If dynamic quantization speedup is less than 2x, follow up with static quantization:

```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader

class MelCalibrationReader(CalibrationDataReader):
    """Feeds mel spectrograms from test audio files."""
    def __init__(self, mel_files):
        self.data = iter(mel_files)

    def get_next(self):
        return next(self.data, None)

quantize_static(
    model_input="models/beat_this.onnx",
    model_output="models/beat_this_int8_static.onnx",
    calibration_data_reader=MelCalibrationReader(calibration_data),
    quant_format=QuantFormat.QDQ,  # Quantize-Dequantize format (best compatibility)
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
)
```

This requires generating calibration data by running the mel spectrogram model on 5–10 representative audio files and saving the intermediate tensors.

### Step 6: Update Documentation

- Add `--model models/beat_this_int8.onnx` usage example to README
- Note performance comparison (FP32 vs Int8)
- Document how to re-quantize if the base model changes

---

## File Changes

| File                         | Action                               |
| ---------------------------- | ------------------------------------ |
| `scripts/quantize_int8.py`   | **New** — quantization script        |
| `scripts/compare_models.py`  | **New** — accuracy validation script |
| `models/beat_this_int8.onnx` | **New** — quantized model artifact   |
| `README.md`                  | Update with Int8 model usage         |

**No Rust code changes required.** The quantized model is a drop-in replacement loaded via the same `--model` flag. Both runtimes handle Int8 ops internally.

---

## Risk Assessment

### Low Risk

- **Quantization tooling is mature**: ONNX Runtime's quantization APIs are production-grade and widely used for transformer models
- **No code changes needed**: the runtime trait boundary deals in f32 tensors; quantization is internal to the ONNX model
- **Easy rollback**: users just switch back to `--model models/beat_this.onnx`

### Medium Risk

- **Rten Int8 op coverage**: rten may not support all quantized operators that ORT produces. Test early. Worst case: Int8 model is ORT-only
- **Accuracy on edge cases**: some audio files with unusual dynamics (very quiet, tempo changes) might show larger deviations. Test on diverse material

### Low Probability, High Impact

- **Attention layer quantization**: if the attention mechanism has activation outliers, Int8 might produce noticeably different beat positions. Mitigation: exclude LayerNorm from quantization via `nodes_to_exclude`

---

## Success Criteria

1. `beat_this_int8.onnx` exists and is <25 MB
2. All test files produce beats within ±1 frame of FP32 output
3. Beat inference is at least 1.5x faster than FP32 on the primary test machine
4. Model loads and runs on at least the ORT runtime without code changes
