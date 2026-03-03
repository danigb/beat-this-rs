# Research: Prepare First Release (v0.1.0)

## Current State

The codebase is feature-complete and well-structured: ~2,100 lines of production code across 10 source files, 38 unit tests (all passing), 27+ integration tests, and 168 doc comments on public APIs. The project provides both a CLI binary (`beat-this`) and a Rust library (`beat_this`).

---

## Items Ranked by Priority

### REQUIRED (Blockers)

#### 1. Add LICENSE file

No LICENSE file exists in the repository root. This is the single most critical blocker. The reference implementations use MIT:

- `beat_this` (Python, JKU Linz) — MIT
- `beat_this_cpp` (Masaki Ono / Melissa Audio) — MIT

**Action:** Add MIT LICENSE file to root. Add `license = "MIT"` to Cargo.toml.

#### 2. Complete Cargo.toml package metadata

Current Cargo.toml is minimal. Missing fields required for a proper release (and for crates.io if ever published):

```toml
[package]
name = "beat-this"
version = "0.1.0"
edition = "2021"
description = "Rust implementation of Beat This! — AI-powered beat and downbeat tracking"
license = "MIT"
repository = "https://github.com/danigb/beat-this-rs"
readme = "README.md"
keywords = ["beat-tracking", "music", "audio", "mir", "onnx"]
categories = ["multimedia::audio", "science"]
```

#### 3. Model distribution strategy

ONNX models (~83 MB for standard, ~10 MB for small) are git-ignored and not in the repo. Users need models to run the tool. Currently there's no documented way to obtain them besides running the Python conversion scripts.

**Action:** Document how to obtain models in README (download link, or script instructions). Consider hosting models as GitHub Release assets.

---

### HIGH PRIORITY (Strongly Recommended)

#### 4. Add CHANGELOG.md

No changelog exists. Create one following [Keep a Changelog](https://keepachangelog.com/) format with an initial v0.1.0 entry summarizing features.

#### 5. Review `ort` RC dependency

`ort` is pinned to `=2.0.0-rc.10` (a release candidate). This is acceptable for v0.1.0 since ort is optional (rten is the default runtime), but should be noted. Check if a stable ort 2.x has been released.

#### 6. Consider feature-gating runtimes

Both `ort` and `rten` are unconditionally compiled. This means:
- Larger binary size
- `ort` pulls in `load-dynamic` + `coreml` features even when unused
- Build complexity (ort requires ONNX Runtime shared library)

Consider making `ort` an optional feature:
```toml
[features]
default = ["rten"]
rten = ["dep:rten", "dep:rten-tensor"]
ort = ["dep:ort"]
```

This would simplify the default build and avoid the ort RC dependency for most users.

#### 7. Set up CI (GitHub Actions)

No CI/CD pipeline exists. Minimum for release:
- `cargo fmt --check`
- `cargo clippy -- -D warnings`
- `cargo test` (unit tests)
- `cargo build --release`

#### 8. Create GitHub Release with binaries

Set up a release workflow that:
- Builds binaries for macOS (arm64, x86_64) and Linux (x86_64)
- Attaches binaries + models as release assets
- Creates the GitHub Release from a git tag

---

### NICE TO HAVE (Post-release OK)

#### 9. Replace `partial_cmp().unwrap()` with safer alternatives

Two places use `.partial_cmp(b).unwrap()` for f32 sorting which would panic on NaN:
- `src/output.rs:152` — in `calculate_bpm()`
- `src/postprocessing.rs:156` — in `snap_downbeats_to_beats()`

Low risk in practice (NaN unlikely from real audio), but for correctness replace with `a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)` or use `f32::total_cmp`.

#### 10. Add `rust-version` (MSRV) to Cargo.toml

Document the minimum supported Rust version so users know what toolchain they need.

#### 11. Add a `[profile.release]` section

Consider adding optimizations for release builds:
```toml
[profile.release]
lto = true
strip = true
```

#### 12. Improve README for release

README is already comprehensive (~8.4 KB), but could benefit from:
- Installation instructions (from release binaries, from source, via cargo install)
- Model download instructions
- Badges (CI status, version, license)

#### 13. Add CONTRIBUTING.md

Brief guide for contributors: how to build, run tests, project structure.

#### 14. Run `cargo clippy` and `cargo fmt`

Verify no warnings or formatting issues. No suppressions (`#[allow(clippy::...)]`) exist currently, which is good.

#### 15. Consider `exclude` in Cargo.toml

If publishing to crates.io, exclude large/unnecessary files:
```toml
exclude = ["docs/", "scripts/", "references/", "integration_test_files/", "test_files/", "Dockerfile"]
```

#### 16. Add `examples/` directory

A simple example showing library usage would help adoption. Currently usage is documented in README but there's no runnable example.

---

## Code Quality Summary

| Area | Status | Notes |
|------|--------|-------|
| TODO/FIXME comments | Clean | None found |
| Error handling | Good | Consistent `anyhow::Result` + `?` + `.context()` |
| Doc comments | Good | 168 across all modules |
| Clippy suppressions | Clean | None |
| Deprecated APIs | Clean | None |
| Hardcoded values | Acceptable | Model paths are CLI defaults with overrides |
| Unsafe code | Clean | None |
| Test coverage | Good | 38 unit + 27 integration tests |

## Dependency Summary

| Dependency | Version | Notes |
|------------|---------|-------|
| ort | =2.0.0-rc.10 | RC pinned, consider feature-gating |
| rten | 0.24 | Pure Rust, default runtime |
| symphonia | 0.5 | Audio decoding (all formats) |
| ndarray | 0.16 | Array operations |
| rubato | 0.14 | Resampling |
| hound | 3.5 | WAV writing |
| clap | 4 | CLI |
| serde/serde_json | 1 | Serialization |
| anyhow | 1 | Error handling |
