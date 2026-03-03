# Task 015: Default Runtime — Research

## Current State

**Runtime architecture**: Two backends behind a trait abstraction (`InferenceRuntime` / `InferenceSession`), dispatched via a `match` in `main.rs`:

- **ort** (always compiled): C/C++ ONNX Runtime wrapper with `load-dynamic` feature — the ORT shared library (.dylib/.so/.dll) is loaded at runtime via dlopen, not linked into the binary.
- **rten** (feature-gated): Pure Rust inference engine, only available when compiled with `--features rten`.

**Default**: ort (via `default_value = "ort"` in CLI).

**Feature flags**:
```toml
default = []
rten = ["dep:rten", "dep:rten-tensor"]
ort = { version = "=2.0.0-rc.10", features = ["ndarray", "load-dynamic", "coreml"] }  # unconditional
```

## Goal 1: Make rten the default

**Justification** (from task 013 benchmarks on Apple M1):

| Metric | ort | rten | Winner |
|--------|-----|------|--------|
| Model loading | 360ms | 20ms | rten (17x) |
| Mel spectrogram | 150ms | 70ms | rten (2x) |
| Beat inference (4.5 min) | 3.37s | 3.78s | ort (12%) |
| **Total wall time** (4.5 min) | 4.30s | **4.25s** | **rten** |
| Output correctness | — | — | Byte-identical |

rten wins on total wall time for all tested files despite slightly slower beat inference, thanks to 17x faster model loading and 2x faster mel computation. It's also pure Rust (no dylib management, no SIGABRT on exit, cross-platform).

**What needs to change**: Flip the default in CLI from `"ort"` to `"rten"`.

## Goal 2: Support multiple runtimes at runtime (not compilation)

The user wants: rten by default, ort available via `--runtime ort`, **without requiring recompilation**.

### Current limitation

rten is behind `#[cfg(feature = "rten")]`. If compiled without `--features rten`, the `Rten` enum variant doesn't exist and rten can't be selected at runtime. Today, to use both runtimes you must compile with `cargo build --features rten`.

### Key insight: ort already uses dynamic loading

The `ort` crate is compiled with `load-dynamic`, meaning it does NOT statically link the ONNX Runtime C++ library. Instead, it loads `libonnxruntime.dylib` (or equivalent) at runtime when creating a session. If the library isn't found, it fails with an error at that point, not at compile time.

This means **ort's Rust wrapper code is always compiled**, but the actual C++ runtime is only needed if the user selects `--runtime ort`.

### Approach: Make rten unconditional, keep ort unconditional

Remove the feature gate on rten and compile both runtimes always:

```toml
[features]
default = []
# No rten feature flag needed anymore

[dependencies]
ort = { version = "=2.0.0-rc.10", features = ["ndarray", "load-dynamic", "coreml"] }
rten = { version = "0.24", features = ["fft"] }
rten-tensor = { version = "0.24" }
```

In `src/runtime.rs`, remove the `#[cfg(feature = "rten")]`:
```rust
pub mod ort;
pub mod rten;  // always available
```

In `main.rs`, remove `#[cfg(feature = "rten")]` from the `Runtime` enum variant, and change the default:
```rust
#[derive(Clone, clap::ValueEnum)]
enum Runtime {
    Ort,
    Rten,  // no cfg gate
}

#[arg(long = "runtime", value_enum, default_value = "rten")]  // rten default
runtime: Runtime,
```

**Result**: A single binary supports both runtimes, selectable at runtime via `--runtime`. rten is the default and always works. ort is available if the user has the ORT shared library installed.

### Do we need `Box<dyn InferenceSession>`?

No. The current `match`-based static dispatch is sufficient and better:

```rust
match cli.runtime {
    Runtime::Rten => { /* monomorphized for RtenSession */ }
    Runtime::Ort =>  { /* monomorphized for OrtSession */ }
}
```

The `InferenceSession` trait IS object-safe, so `Box<dyn InferenceSession>` would work — but it offers no benefit here:

| | Static dispatch (match) | Dynamic dispatch (Box\<dyn\>) |
|---|---|---|
| Performance | Zero overhead | Vtable indirection per inference call |
| Code | Small duplication in match arms | Unified pipeline, but needs wrapper impl |
| Binary size | Slightly larger (two monomorphized copies) | Slightly smaller |
| Complexity | Simple, current pattern | Requires blanket impl or wrapper type |

The match duplication is ~10 lines. The vtable overhead is negligible for inference (called a few times per file, not millions). Either works, but static dispatch is simpler and matches the existing pattern. No reason to change.

### What about making ort optional instead?

An alternative: flip which runtime is optional:

```toml
[features]
default = []
ort = ["dep:ort", "dep:ndarray"]

[dependencies]
ort = { version = "=2.0.0-rc.10", optional = true, features = ["ndarray", "load-dynamic", "coreml"] }
ndarray = { version = "0.16", optional = true }
rten = { version = "0.24", features = ["fft"] }
rten-tensor = { version = "0.24" }
```

**Pros**: Smaller binary if ort isn't needed; no ort compilation overhead for pure-rten builds.
**Cons**: Users who want ort need `--features ort` at compile time — this contradicts the "both at runtime" goal. Also, ort's `load-dynamic` already means the heavy C++ library isn't in the binary — only the thin Rust wrapper is compiled.

**Verdict**: Not worth it. The ort Rust wrapper is small. Keeping both unconditional is simpler and achieves the "runtime selection without recompilation" goal.

### Error handling when ORT dylib is missing

If a user selects `--runtime ort` but doesn't have the ORT shared library installed, the `load_model()` call will fail with an ort error about the missing library. We should catch this and print a helpful message:

```
Error: ONNX Runtime library not found.
Install it via: brew install onnxruntime (macOS) or set ORT_DYLIB_PATH.
Use --runtime rten (default) for a pure-Rust runtime with no dependencies.
```

## Summary of changes

1. **Cargo.toml**: Make `rten` and `rten-tensor` unconditional dependencies. Remove the `rten` feature.
2. **src/runtime.rs**: Remove `#[cfg(feature = "rten")]` from `pub mod rten`.
3. **src/main.rs**:
   - Remove `#[cfg(feature = "rten")]` from `Runtime::Rten` variant.
   - Change default from `"ort"` to `"rten"`.
   - Add a helpful error message if ort dylib is missing.
4. **Tests**: Remove any `#[cfg(feature = "rten")]` guards on rten-related tests.

## Pros / Cons summary

**Pros of making rten default + both always available**:
- Faster total wall time (17x model loading, 2x mel)
- Zero external dependencies for default runtime (pure Rust)
- Single binary supports both runtimes via `--runtime` flag
- No feature-flag complexity for end users
- Cross-platform without dylib management

**Cons**:
- Beat inference 10-12% slower with rten (offset by loading/mel gains)
- Slightly larger binary (both runtime wrappers compiled)
- ort profiling (`--profile`) only works with `--runtime ort`
- Less battle-tested than ORT (though byte-identical results on all test files)
