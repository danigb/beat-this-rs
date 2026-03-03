# Next Steps

Possible future directions for beat-this-rs, grouped by expected return on investment.

## High ROI

### Int8 Quantization

Beat inference is 93% of runtime. Int8 quantization should yield 2–3x speedup by leveraging ARM NEON / x86 VNNI integer instructions. Both ort and rten support quantized ONNX models. Steps: quantize the model with ONNX Runtime tools, validate output stability within ±1 frame on test files.

### Python Bindings (PyO3)

The MIR research community lives in Python. A `pip install beat-this-rs` package with a simple API (`beat_this(path) → list[Beat]`) would give them a faster drop-in replacement with zero PyTorch dependency. The existing `BeatThis` public API maps almost directly. Use maturin for packaging.

### WASM Target

The rten backend is pure Rust and compiles to WASM. A browser-based beat tracker would be unique and immediately useful for web audio apps (DAWs, DJ tools, music education). Would need WASM-compatible audio decoding or accept pre-decoded Float32Array samples. Build with wasm-pack.

### Batch / Directory Processing

`beat-this ./album/` — process all audio files in a directory. Model loads once, amortizing startup cost. Could parallelize across files with rayon. Trivial to implement, high practical value for anyone processing music libraries.

### JSON Output

`--output-json` with timestamps, BPM, confidence scores, downbeat positions. Makes the output trivially consumable by scripts, web services, and other tools. Almost zero effort given the existing `BeatResult` struct.

## Medium ROI

### Time Signature & Meter Estimation

Beats and downbeats are already detected. Computing beats-per-bar (3/4, 4/4, 6/8) is straightforward pattern analysis on the existing output. A natural extension that leverages work already done.

### MIDI Output

Export beats as MIDI note events. Near-universal interoperability with DAWs (Ableton, Logic, Reaper). A small crate like `midly` handles the writing. Low effort, high integration value for music producers.

### Pre-built Binaries & Releases

GitHub Actions CI building for macOS (ARM + x86), Linux, Windows. Tools like cargo-dist make this straightforward. The rten feature flag enables a fully self-contained binary with no C++ dylib dependency, which simplifies distribution.

### Streaming / Real-time API

Process audio in chunks as it arrives, emitting beats with some latency. The model expects full context, but a sliding-window approach over the chunked inference could work. Opens up live performance, DJ, and broadcast monitoring use cases.

### crates.io Publication

Clean up the public API, add rustdoc documentation, publish. Other Rust audio projects could depend on it. The trait-based runtime abstraction is already well-designed for library consumers.

## Low ROI

### GPU Inference

Complex to set up (CUDA/Metal providers for ORT). CPU performance with int8 quantization would likely be sufficient for most use cases. Only worth it for batch-processing thousands of files.

### Confidence Scores in Output

Expose raw sigmoid probabilities alongside beat times. Useful for research and downstream algorithms that want to weight beats by confidence. Small change to `BeatResult` and output formats, but niche demand.

### Alternative Models / Ensemble

Support loading different beat tracking models or training checkpoints. The runtime trait already supports this. Low demand unless targeting niche genres where the default model underperforms.

### Audio Fingerprinting / Result Caching

Hash audio files, cache beat results, skip re-processing unchanged files. Only useful for repeated batch processing of the same library.

### Memory-Mapped Model Loading

Already measured at 20ms for rten. Not worth optimizing unless used as a hot-path library where microseconds matter.
