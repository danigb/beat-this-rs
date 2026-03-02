# ONNX Inference Runtimes for Rust

## Summary

| Crate | Loading | GPU | Transformer Support | CPU Perf vs ort | Pure Rust | Status |
|-------|---------|-----|---------------------|-----------------|-----------|--------|
| **ort** | Runtime | CUDA, CoreML, TensorRT, DirectML | Full | Baseline | No (C++) | Active, RC |
| **rten** | Runtime | None (Metal planned) | Excellent (Whisper, Llama 3, BERT) | ~80-100% | Yes | Active |
| **tract-onnx** | Runtime | None (CPU only) | Good (170+ ops, opset 9-18) | Slower on x86 | Yes | Active |
| **candle-onnx** | Runtime | CUDA, Metal | **Missing LayerNorm** | Unknown | Mostly | Partial |
| **burn-onnx** | Compile-time | CUDA, wgpu, Metal | Missing RotaryEmbed/RMSNorm | Unknown | Yes | Active |
| **wonnx** | Runtime | WebGPU | Limited | Unknown | Yes | **Archived** |

---

## Viable Options (3)

### 1. ort (primary choice)

- **Crate**: `ort` v2.0.0-rc.11 (wraps ONNX Runtime 1.23)
- **Loading**: Runtime `.onnx` file loading
- **Operator coverage**: Complete — wraps the official Microsoft ONNX Runtime
- **GPU**: CUDA, CoreML (macOS), TensorRT, DirectML, OpenVINO, QNN
- **Dynamic shapes**: Full support
- **Performance**: Best overall. 3-5x faster than Python. GPU acceleration available.
- **Platform**: macOS (x86 + Apple Silicon), Linux (x86 + ARM), Windows
- **Production users**: HuggingFace TEI, Google Magika, rust-bert, Remixatron
- **Cons**: Not pure Rust (C++ ONNX Runtime dylib ~20 MB), RC version (pin exact)

### 2. rten (best pure-Rust alternative)

- **Crate**: `rten` v0.23+
- **Loading**: Runtime `.onnx` file loading (since v0.23, previously required custom format)
- **Operator coverage**: Excellent for transformers — tested on Whisper, Llama 3, GPT-2, ModernBERT, CLIP, DistilViT. 18 new operators added in 2025. Supports quantized models (MatMulNBits).
- **GPU**: CPU-only. Metal is the top priority for upcoming releases.
- **Dynamic shapes**: Yes, via symbolic shape inference
- **Performance**: On M3 Pro CPU, "in the same ballpark as ONNX Runtime." Whisper benchmark: slightly faster than whisper.cpp. ~80-100% of ort on CPU depending on model.
- **Platform**: Anywhere Rust compiles (pure Rust)
- **Cons**: CPU-only (no GPU yet), solo maintainer
- **Notable**: Pure Rust with no C dependencies. Best option for WASM or embedded.

### 3. tract-onnx (solid pure-Rust fallback)

- **Crate**: `tract-onnx` v0.22.1 (by Sonos)
- **Loading**: Runtime `.onnx` file loading
- **Operator coverage**: 170+ operators, opset 9-18. Passes ~85% of ONNX backend tests. Has `tract-transformers` sub-crate. Less proven with large transformer models than rten.
- **GPU**: None. Maintainer explicitly stated GPU is not on the roadmap. Focus is ARM edge devices.
- **Dynamic shapes**: Yes
- **Performance**: Excellent on ARM (3x faster than TF Lite on Raspberry Pi). Slower than ort on desktop x86 for large models.
- **Platform**: Anywhere Rust compiles. Strong ARM/NEON optimization.
- **Production users**: Sonos devices (edge inference)
- **Cons**: CPU-only, no GPU planned, less validated for transformer-class models

---

## Not Viable

### candle-onnx
- v0.9.2 by Hugging Face. 82 operators.
- **Dealbreaker**: LayerNormalization not supported (issue open since March 2025, still unmerged). This op is almost certainly used in our transformer model.

### burn-onnx
- v0.21 by Tracel AI.
- **Dealbreaker**: Compile-time code generation only — no runtime `.onnx` loading. Model structure baked into binary at build time. No dynamic input shapes. Requires opset 16+ (our model is opset 14). An ~83 MB model would generate enormous Rust code with very long compile times.

### wonnx
- v0.5.1. WebGPU-based.
- **Dealbreaker**: Repository archived May 2025. Project is dead. Missing transformer operators. MatMul dimensions must be divisible by 2.

### onnxruntime-rs
- Legacy bindings pinned to ONNX Runtime 1.8.
- **Dealbreaker**: Abandoned. `ort` is the direct successor.

---

## Recommendation for Runtime Trait

Design the trait to work with all three viable runtimes:

```
ort    → full-featured, GPU, production default
rten   → pure Rust, near-ort CPU perf, WASM-ready
tract  → pure Rust, ARM-optimized, proven at Sonos
```

All three support: runtime `.onnx` loading, dynamic input shapes, named input/output tensors, f32 inference. The trait abstraction in `runtime.rs` (simple `Tensor` + `InferenceSession` + `InferenceRuntime`) covers the common surface cleanly.

Benchmarking plan: implement `ort` first, then `rten` as second backend, measure CPU inference time on the same audio file across both. `tract` as third if needed.
