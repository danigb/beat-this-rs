# Research: Public API Surface Audit

## Current public API (via `cargo doc`)

Everything below appears in the generated docs. Items are grouped by origin.

### From `lib.rs` directly

| Item | Kind | Used by main.rs | Used by tests | Verdict |
|------|------|:---:|:---:|---------|
| `BeatThis<S>` | struct | yes | yes | **keep** -- primary entry point |
| `BeatThis::new` | fn | yes | yes | **keep** |
| `BeatThis::analyze_file` | fn | no (uses manual pipeline) | yes | **keep** |
| `BeatThis::analyze_audio` | fn | no | no | **keep** -- needed for raw samples API |
| `BeatAnalysis` | struct | yes | yes | **keep** |
| `BeatThis.mel` (pub field) | field | yes | no | **make private** -- exposes `MelProcessor` |
| `BeatThis.inference` (pub field) | field | yes | no | **make private** -- exposes `BeatInference` |
| `BeatThis.post` (pub field) | field | yes | no | **make private** -- exposes `PostProcessor` |

### Re-exported from `audio`

| Item | Kind | Used by main.rs | Used by tests | Verdict |
|------|------|:---:|:---:|---------|
| `load_audio` | fn | yes | yes | **keep** -- needed for `--mix` (re-loads at 44100) |
| `AudioData` | struct | no | no | **keep** -- return type of `load_audio` |

### Re-exported from `mel`

| Item | Kind | Used by main.rs | Used by tests | Verdict |
|------|------|:---:|:---:|---------|
| `MelProcessor<S>` | struct | yes (constructs directly) | yes | **make private** -- only needed because main.rs builds `BeatThis` manually for ort profiling; can be solved with a builder |
| `num_mel_frames` | fn | no | yes | **make private** -- tests can use `tensor.shape[1]` directly |

### Re-exported from `inference`

| Item | Kind | Used by main.rs | Used by tests | Verdict |
|------|------|:---:|:---:|---------|
| `BeatInference<S>` | struct | yes (constructs directly) | yes | **make private** -- same reason as `MelProcessor` |

### Re-exported from `postprocessing`

| Item | Kind | Used by main.rs | Used by tests | Verdict |
|------|------|:---:|:---:|---------|
| `PostProcessor` | struct | yes (constructs directly) | yes | **make private** -- same reason |

### Re-exported from `runtime`

| Item | Kind | Used by main.rs | Used by tests | Verdict |
|------|------|:---:|:---:|---------|
| `InferenceRuntime` | trait | yes | yes | **make private** -- only needed if constructing `BeatThis` manually; `BeatThis::new` takes `impl InferenceRuntime` internally |
| `InferenceSession` | trait | yes (as bound) | yes | **make private** -- only appears as generic bound on `BeatThis<S>` |
| `Tensor` | struct | no | yes | **keep** -- part of `BeatAnalysis.mel` |
| `OrtRuntime` | struct | yes | yes | **keep** -- passed to `BeatThis::new` |
| `RtenRuntime` | struct | yes | yes | **keep** -- passed to `BeatThis::new` |
| `OrtSession` | struct | no | no | already private (not re-exported) |
| `RtenSession` | struct | no | no | already private (not re-exported) |

### Re-exported from `output` (via `pub use output::*`)

| Item | Kind | Used by main.rs | Used by tests | Verdict |
|------|------|:---:|:---:|---------|
| `write_json_file` | fn | yes | no | **make private** -- CLI-only |
| `write_beats_file` | fn | yes | yes | **make private** -- CLI-only |
| `write_click_track` | fn | yes | yes | **make private** -- CLI-only |
| `write_mixed_audio` | fn | yes | no | **make private** -- CLI-only |
| `write_mel_npy` | fn | yes | no | **make private** -- CLI-only |
| `write_batch_json` | fn | yes | no | **make private** -- CLI-only |
| `build_json_output` | fn | yes | no | **make private** -- CLI-only |
| `print_json_stdout` | fn | yes | no | **make private** -- CLI-only |
| `calculate_bpm` | fn | no | yes | **keep** -- useful for library consumers |
| `beat_counts` | fn | no | yes | **keep** -- useful for library consumers |
| `JsonOutput` | struct | no | no | **make private** -- CLI serialization detail |
| `BeatEntry` | struct | no | no | **make private** -- CLI serialization detail |
| `BatchFileEntry` | struct | yes | no | **make private** -- CLI batch detail |
| `BatchSummary` | struct | yes | no | **make private** -- CLI batch detail |
| `BatchSummaryOutput` | struct | yes | no | **make private** -- CLI batch detail |

## Problem: main.rs accesses internals

The biggest blocker is that `main.rs` builds `BeatThis` manually (not via `BeatThis::new`) in the
ort branch to support profiling with a separate runtime for the beat model:

```rust
let beat_runtime = OrtRuntime { profiling_path: Some(...), ..Default::default() };
let mel_session = runtime.load_model(&mel_path)?;
let beat_session = beat_runtime.load_model(&beat_path)?;
let mut bt = beat_this::BeatThis {
    mel: beat_this::MelProcessor::new(mel_session),
    inference: beat_this::BeatInference::new(beat_session),
    post: beat_this::PostProcessor::default(),
};
```

This forces `MelProcessor`, `BeatInference`, `PostProcessor`, `InferenceRuntime`, and
`InferenceSession` to all be public, plus the struct fields on `BeatThis`.

### Solution: builder or second constructor

Add a constructor that accepts separate sessions:

```rust
impl<S: InferenceSession> BeatThis<S> {
    pub fn from_sessions(mel_session: S, beat_session: S) -> Self { ... }
}
```

Then main.rs becomes:

```rust
let mel_session = runtime.load_model(&mel_path)?;
let beat_session = beat_runtime.load_model(&beat_path)?;
let mut bt = BeatThis::from_sessions(mel_session, beat_session);
```

This still requires `InferenceSession` to be public (as a trait bound), but removes the need for
`MelProcessor`, `BeatInference`, `PostProcessor`, and the pub fields.

`InferenceRuntime` is still needed for `load_model`. The alternative is to have `BeatThis` take
two `Path` arguments and two runtimes, but the `from_sessions` approach is simpler and more
flexible.

## Problem: output module is CLI-only

The `output` module contains file-writing functions and serialization structs that only the CLI
binary needs. Library consumers don't need `write_json_file` or `BatchSummaryOutput`.

### Solution: move output to binary crate

Since `main.rs` is a separate binary crate, the output module can live alongside it instead of
in the library. Options:

1. **Move `output.rs` out of the lib** -- Create `src/bin/` or keep `main.rs` and add
   `src/cli/output.rs` as a module of the binary crate. This requires restructuring to use
   `src/main.rs` as a binary that has its own module tree.

2. **Keep in lib but don't re-export** -- Simpler: remove `pub use output::*` from `lib.rs`.
   The output module stays where it is but is only accessible internally. main.rs would need to
   duplicate or inline the output logic. This is worse because it means code duplication.

3. **Use `#[doc(hidden)]`** -- Keep the re-exports but hide from docs. Items are still technically
   public but won't appear in `cargo doc`. This is a half-measure but pragmatic.

4. **Selective re-export** -- Only re-export `calculate_bpm` and `beat_counts` (the useful ones).
   Move the rest to the binary crate.

**Recommendation**: Option 4. Re-export only `calculate_bpm` and `beat_counts`. Move file-writing
functions and batch types to `main.rs` (or a `cli` module next to it).

## Proposed minimal public API

```
// Primary API
beat_this::BeatThis             struct  -- entry point
beat_this::BeatThis::new        fn      -- load from runtime + paths
beat_this::BeatThis::from_sessions fn   -- load from pre-built sessions (new)
beat_this::BeatThis::analyze_file  fn   -- process audio file
beat_this::BeatThis::analyze_audio fn   -- process raw samples
beat_this::BeatAnalysis         struct  -- result type

// Runtime backends
beat_this::OrtRuntime           struct  -- ONNX Runtime backend
beat_this::RtenRuntime          struct  -- pure-Rust backend

// Types needed by the API
beat_this::Tensor               struct  -- part of BeatAnalysis
beat_this::InferenceRuntime     trait   -- needed for load_model (used by from_sessions callers)
beat_this::InferenceSession     trait   -- generic bound on BeatThis<S>

// Utilities
beat_this::load_audio           fn      -- load audio files
beat_this::AudioData            struct  -- return type of load_audio
beat_this::calculate_bpm        fn      -- BPM from beat timestamps
beat_this::beat_counts          fn      -- beat numbering within measures
```

That's 12 items vs the current 28. The removed items are all either CLI implementation details
(`write_*`, `Batch*`, `JsonOutput`) or pipeline internals (`MelProcessor`, `BeatInference`,
`PostProcessor`, `num_mel_frames`).

## Implementation plan

1. Add `BeatThis::from_sessions(mel_session, beat_session)` constructor
2. Make `BeatThis` fields private; update main.rs to use `from_sessions`
3. Stop re-exporting `MelProcessor`, `BeatInference`, `PostProcessor`, `num_mel_frames`
4. Replace `pub use output::*` with `pub use output::{calculate_bpm, beat_counts}`
5. Move output file-writing functions and batch types into a `cli` module owned by the binary crate
6. Update tests that use pipeline internals to use `BeatThis::analyze_*` instead
7. Stop re-exporting `InferenceRuntime` if possible (may still be needed for `from_sessions` callers to call `load_model`)
