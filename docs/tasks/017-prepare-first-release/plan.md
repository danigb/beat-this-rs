# Plan: Prepare First Release (v0.1.0)

## Steps

### 1. Add LICENSE file

Create `LICENSE` with MIT text (Copyright 2025 danigb).

### 2. Complete Cargo.toml metadata

Add to `[package]`:

- `description`
- `license = "MIT"`
- `repository`
- `readme = "README.md"`
- `keywords`
- `categories`

Add `[profile.release]` with `lto = true`, `strip = true`.

Bump `ort` to `=2.0.0-rc.11` and ensure it uses `load-dynamic` only (no static linking). Both runtimes stay unconditionally compiled — `load-dynamic` means ort compiles without needing the ONNX Runtime library installed, it loads via dlopen at runtime.

### 3. Add CHANGELOG.md

Create `CHANGELOG.md` with initial v0.1.0 entry listing all features.

### 5. Add GitHub Actions CI

Create `.github/workflows/ci.yml`:

- Trigger: push to main, pull requests
- Steps: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`
- Matrix: default features only (rten)

### 6. Fix `partial_cmp().unwrap()` (2 locations)

Replace with `f32::total_cmp` in:

- `src/output.rs` — `calculate_bpm()`
- `src/postprocessing.rs` — `snap_downbeats_to_beats()`

### 7. Run `cargo fmt` and `cargo clippy`

Fix any warnings.

### 8. Update README

- Add license badge
- Add install-from-source instructions

## Files to create

- `LICENSE`
- `CHANGELOG.md`
- `.github/workflows/ci.yml`

## Files to modify

- `Cargo.toml` — metadata, bump ort to rc.11, release profile
- `src/output.rs` — fix `partial_cmp().unwrap()`
- `src/postprocessing.rs` — fix `partial_cmp().unwrap()`
- `README.md` — badges, install instructions

## Verification

1. `cargo build` — builds without onnxruntime installed (load-dynamic)
2. `cargo test` — all unit tests pass
3. `cargo clippy -- -D warnings` — no warnings
4. `cargo fmt --check` — formatted
5. `cargo run -- --help` — CLI works
