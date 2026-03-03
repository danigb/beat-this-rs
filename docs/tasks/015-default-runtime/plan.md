# Task 015: Default Runtime — Implementation Plan

## Overview

Two changes: (1) make rten the default runtime, (2) make both runtimes always available without recompilation. Keep static dispatch (no `Box<dyn>`).

## Step 1: Make rten an unconditional dependency

**File**: `Cargo.toml`

- Remove the `rten` feature from `[features]`
- Make `rten` and `rten-tensor` non-optional dependencies

```toml
[features]
default = []
# rten feature removed

[dependencies]
rten = { version = "0.24", features = ["fft"] }
rten-tensor = { version = "0.24" }
```

## Step 2: Remove cfg gates from library code

**File**: `src/runtime.rs`

Remove `#[cfg(feature = "rten")]` from the module declaration:

```rust
pub mod ort;
pub mod rten;
```

## Step 3: Update CLI to default to rten

**File**: `src/main.rs`

Three changes:

1. Remove `#[cfg(feature = "rten")]` from `Runtime::Rten` enum variant
2. Change default from `"ort"` to `"rten"`
3. Remove `#[cfg(feature = "rten")]` from the match arm

```rust
#[derive(Clone, clap::ValueEnum)]
enum Runtime {
    Ort,
    Rten,
}

#[arg(long = "runtime", value_enum, default_value = "rten")]
runtime: Runtime,
```

## Step 4: Improve error message when ORT dylib is missing

**File**: `src/main.rs`

Wrap the ort match arm's model loading in a context that produces a helpful error if the ORT shared library isn't found:

```rust
Runtime::Ort => {
    let runtime = beat_this::runtime::ort::OrtRuntime { ... };
    let mut bt = beat_this::BeatThis::new(&runtime, &mel_path, &beat_path)
        .context("Failed to load models with ort runtime. \
            Is the ONNX Runtime library installed? \
            Install via: brew install onnxruntime (macOS) or set ORT_DYLIB_PATH. \
            Or use --runtime rten (default) for a pure-Rust runtime.")?;
    // ...
}
```

## Step 5: Remove cfg gates from tests

**File**: `tests/rten_integration.rs` — remove `#![cfg(feature = "rten")]` at line 1
**File**: `tests/cross_runtime.rs` — remove `#![cfg(feature = "rten")]` at line 1

These tests should now always run since rten is always available.

## Step 6: Build and test

```bash
cargo build --release
cargo test
```

Verify:
- `cargo run --release -- --help` shows `--runtime` with `rten` as default and both `ort`/`rten` as options
- `cargo run --release -- audio.mp3` uses rten (no `--features` needed)
- `cargo run --release -- audio.mp3 --runtime ort` uses ort (if ORT dylib installed)
- `cargo run --release -- audio.mp3 --runtime rten` uses rten explicitly
- All tests pass without `--features rten`

## Files changed

| File | Change |
|------|--------|
| `Cargo.toml` | Remove `rten` feature, make deps unconditional |
| `src/runtime.rs` | Remove `#[cfg(feature = "rten")]` |
| `src/main.rs` | Remove cfg gates, change default to `"rten"`, add ort error context |
| `tests/rten_integration.rs` | Remove `#![cfg(feature = "rten")]` |
| `tests/cross_runtime.rs` | Remove `#![cfg(feature = "rten")]` |
