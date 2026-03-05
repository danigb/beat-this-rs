# Summary: Expose Mel Spectrogram (Task 020)

Replaced the old `process_audio`/`process_file` API with `analyze_audio`/`analyze_file`, returning
a richer `BeatAnalysis` struct that includes the mel spectrogram and raw logits alongside beat and
downbeat timestamps. Added a `--mel` CLI flag to export the mel spectrogram as a numpy `.npy` file.

## Changes by file

| File                    | Change                                                                                                                                                         |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/lib.rs`            | Added `BeatAnalysis` struct; added `analyze_audio`/`analyze_file` methods; removed `process_audio`/`process_file` and `BeatResult` re-export                   |
| `src/postprocessing.rs` | Removed `BeatResult` struct; `PostProcessor::process` now returns `(Vec<f32>, Vec<f32>)`                                                                       |
| `src/runtime.rs`        | Added `serde::Serialize`/`serde::Deserialize` derives to `Tensor` (unconditional, not feature-gated)                                                           |
| `src/output.rs`         | Updated all functions to accept `&BeatAnalysis`; added `write_mel_npy` with npy v1.0 format writer                                                             |
| `src/main.rs`           | Added `--mel` CLI flag and `mel` field to `OutputFlags`; updated `write_outputs`, `process_single_file`, `run_pipeline`, and `run_batch` to use `BeatAnalysis` |
| `Cargo.toml`            | Version bumped to 0.2.0                                                                                                                                        |
| `CHANGELOG.md`          | Added 0.2.0 entry                                                                                                                                              |
| `tests/*.rs`            | Updated all integration tests: `process_file` → `analyze_file`, `process_audio` → `analyze_audio`, `BeatResult` field access → tuple destructuring             |

## Other changes

- **Serde on Tensor**: the plan suggested a `serde` feature flag (`#[cfg_attr(...)]`). Instead,
  serde derives were added unconditionally since the crate already depends on serde non-optionally.
- **No backward-compat wrappers**: the plan's research suggested keeping `process_audio`/`process_file`
  as thin wrappers. Instead, they were removed entirely (breaking change, reflected in the 0.2.0 bump).
- **`write_mel_npy` signature**: the plan had `write_mel_npy(path, &Tensor)`, the implementation
  takes `write_mel_npy(path, &BeatAnalysis)` for consistency with other output functions.

## Tests

- Unit test `test_write_mel_npy` verifies numpy magic bytes, version, header alignment (multiple
  of 64), data size, and spot-checks float values.
- All 66 existing tests pass after the API migration.
