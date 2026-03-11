# Plan: Finish API Cleanup (Task 021)

## Completed

All renames from `research-naming.md` are done:

| Old name | New name | Status |
|----------|----------|--------|
| `InferenceSession` (trait) | `Model` | done |
| `InferenceRuntime` (trait) | `Runtime` | done |
| `BeatInference` (struct) | `BeatPredictor` | done |
| `MelProcessor` (struct) | `MelExtractor` | done |
| `PostProcessor` (struct) | `PeakPicker` | done |
| `.process()` methods | `.extract()` / `.predict()` / `.decode()` | done |
| `inference` field | `predictor` | done |
| `post` field | `peak_picker` | done |
| `pub mod` → `mod` + flat re-exports | — | done |

All modules are private. All re-exports are from the crate root. main.rs, all tests,
and README are updated with new names.

**Leftover**: Internal variable/field names still use "session" in several places.
These should be renamed to "model" for consistency with the new trait name:

| Location | Current | New |
|----------|---------|-----|
| `lib.rs:59-64` | `mel_session`, `beat_session` | `mel_model`, `beat_model` |
| `main.rs:545-554` | `mel_session`, `beat_session` | `mel_model`, `beat_model` |
| `inference.rs:19,24-25,30,57` | `session: M` field, `session` param | `model: M`, `model` |
| `mel.rs:10,15-16,21,37` | `session: M` field, `session` param | `model: M`, `model` |
| `inference.rs:29` | `model_mut()` accessor | keep (already correct) |
| `mel.rs:20` | `model_mut()` accessor | keep (already correct) |
| `runtime/ort.rs` | `OrtSession` struct | `OrtModel` |
| `runtime/rten.rs` | `RtenSession` struct | `RtenModel` |

## Remaining: Reduce Public API Surface

The library currently exports 28 items. Many are CLI-only or pipeline internals that
shouldn't be in the public API. Target: 16 items.

### Items to remove from public API

**Pipeline internals** (only needed because main.rs + tests build the pipeline manually):
- `MelExtractor` — internal pipeline stage
- `BeatPredictor` — internal pipeline stage
- `PeakPicker` — internal pipeline stage
- `num_mel_frames` — helper, tests can use `tensor.shape[1]`

**CLI-only output** (currently `pub use output::*` exports everything):
- `write_json_file`, `write_beats_file`, `write_click_track`, `write_mixed_audio`
- `write_mel_npy`, `write_batch_json`, `build_json_output`, `print_json_stdout`
- `JsonOutput`, `BeatEntry`, `BatchFileEntry`, `BatchSummary`, `BatchSummaryOutput`

### Items to keep

```
beat_this::BeatThis              struct  — entry point
beat_this::BeatAnalysis          struct  — result type
beat_this::OrtRuntime            struct  — ONNX Runtime backend
beat_this::RtenRuntime           struct  — pure-Rust backend
beat_this::Runtime               trait   — load models
beat_this::Model                 trait   — generic bound on BeatThis<M>
beat_this::Tensor                struct  — part of BeatAnalysis
beat_this::load_audio            fn      — load audio files
beat_this::AudioData             struct  — return type of load_audio
beat_this::calculate_bpm         fn      — BPM from beat timestamps
beat_this::beat_counts           fn      — beat numbering within measures
```

Plus new methods on BeatThis:
- `BeatThis::new` — load from runtime + paths (existing)
- `BeatThis::from_models` — load from pre-built models (new)
- `BeatThis::analyze_file` — process audio file (existing)
- `BeatThis::analyze_audio` — process raw samples (existing)
- `BeatThis::beat_model_mut` — access beat model for profiling (new)

## Steps

### Step 0: Rename remaining "session" references to "model"

Internal fields and variables still use "session" from the old `InferenceSession` name.
Rename for consistency:

- `inference.rs`: field `session: M` → `model: M`, update `new()`, `model_mut()`, `predict()`
- `mel.rs`: field `session: M` → `model: M`, update `new()`, `model_mut()`, `extract()`
- `lib.rs`: locals `mel_session`/`beat_session` → `mel_model`/`beat_model`
- `main.rs`: locals `mel_session`/`beat_session` → `mel_model`/`beat_model`
- `runtime/ort.rs`: `OrtSession` → `OrtModel` (the internal `session: Session` field
  keeps its name since it wraps `ort::Session`)
- `runtime/rten.rs`: `RtenSession` → `RtenModel`

### Step 1: Add `from_models` constructor and `beat_model_mut` accessor

```rust
impl<M: Model> BeatThis<M> {
    pub fn from_models(mel_model: M, beat_model: M) -> Self {
        Self {
            mel: MelExtractor::new(mel_model),
            predictor: BeatPredictor::new(beat_model),
            peak_picker: PeakPicker::default(),
        }
    }

    pub fn beat_model_mut(&mut self) -> &mut M {
        self.predictor.model_mut()
    }
}
```

### Step 2: Make `BeatThis` fields private

Change `pub mel`, `pub predictor`, `pub peak_picker` to private.

**Blocker**: main.rs accesses fields in two places:

1. **Struct literal construction** (ort profiling branch, line 552):
   ```rust
   let mut bt = beat_this::BeatThis {
       mel: beat_this::MelExtractor::new(mel_session),
       predictor: beat_this::BeatPredictor::new(beat_session),
       peak_picker: beat_this::PeakPicker::default(),
   };
   ```
   → Replace with `BeatThis::from_models(mel_model, beat_model)`

2. **`process_single_file` verbose timing** (lines 330-346):
   ```rust
   let mel = bt.mel.extract(&audio.samples)?;
   let (beat_logits, downbeat_logits) = bt.predictor.predict(&mel)?;
   let (beats, downbeats) = bt.peak_picker.decode(&beat_logits, &downbeat_logits)?;
   ```
   This manually runs each pipeline stage for per-stage timing.

   **Solution**: Add `BeatThis::analyze_audio_verbose` or restructure `process_single_file`
   to use `analyze_audio` and move timing into the `BeatThis` methods. Or simpler: add a
   `BeatThis::analyze_audio_timed` that returns per-stage durations:

   ```rust
   pub struct TimedAnalysis {
       pub analysis: BeatAnalysis,
       pub mel_time: Duration,
       pub predict_time: Duration,
       pub decode_time: Duration,
   }

   pub fn analyze_audio_timed(&mut self, samples: &[f32], sample_rate: u32)
       -> Result<TimedAnalysis>
   ```

   Or even simpler: accept an optional callback / just add timing inside `analyze_audio`
   behind a flag. But the cleanest approach is to just add a timed variant.

3. **Profiling end** (line 571):
   ```rust
   bt.predictor.model_mut().end_profiling()
   ```
   → Replace with `bt.beat_model_mut().end_profiling()`

### Step 3: Remove pipeline internal re-exports

In `lib.rs`, remove:
```rust
pub use inference::BeatPredictor;
pub use mel::{num_frames as num_mel_frames, MelExtractor};
pub use postprocessing::PeakPicker;
```

After Step 2, main.rs no longer needs these. Tests need updating (see Step 5).

### Step 4: Move CLI output code out of lib

Replace `pub use output::*` with:
```rust
pub use output::{calculate_bpm, beat_counts};
```

main.rs is a **separate binary crate** — it imports via `use beat_this::`, not `use crate::`.
So CLI-only output functions must move out of the library.

**Approach**: Move write functions and batch/json types from `src/output.rs` into main.rs.
The output code is ~250 lines. main.rs is ~600 lines. Total ~850 lines in one file is
manageable, or split into `src/bin/beat-this/{main.rs, output.rs}`.

Items to move:
- `write_json_file`, `write_beats_file`, `write_click_track`, `write_mixed_audio`
- `write_mel_npy`, `write_batch_json`, `build_json_output`, `print_json_stdout`
- `JsonOutput`, `BeatEntry`, `BatchFileEntry`, `BatchSummary`, `BatchSummaryOutput`

Items staying in `src/output.rs`:
- `calculate_bpm`, `beat_counts`

### Step 5: Update tests

| Test file | Currently imports | Change to |
|-----------|-------------------|-----------|
| `mel_integration.rs` | `MelExtractor, num_mel_frames, Runtime` | `BeatThis::analyze_*`, check `analysis.mel.shape` |
| `inference_integration.rs` | `BeatPredictor, MelExtractor, num_mel_frames, Runtime, Tensor` | `BeatThis::analyze_*`, check logits |
| `postprocessing_integration.rs` | `BeatPredictor, MelExtractor, PeakPicker, num_mel_frames, Runtime` | `BeatThis::analyze_*` |
| `output_integration.rs` | `write_beats_file, write_click_track, beat_counts, calculate_bpm` | Keep `beat_counts`, `calculate_bpm`. Move `write_*` tests to unit tests in `output.rs` |

### Step 6: Update docs

- Rewrite `summary.md` to reflect final state
- Verify README Rust API example
- Run `cargo doc --no-deps` and check public API is clean

## Suggested commit sequence

1. Rename internal "session" variables/fields to "model"
2. Add `from_models` + `beat_model_mut` + `analyze_audio_timed`; make fields private; update main.rs
3. Remove pipeline internal re-exports; update tests
4. Move CLI output code from lib to binary; selective re-exports
5. Update summary.md

## Notes

- **Breaking change** for anyone using pipeline internals or output functions. Fine since
  the crate is pre-1.0.
- **main.rs will grow** ~250 lines. Can split into `src/bin/` later if needed.
- Step 4 (moving output) is the biggest change. Steps 1-2 can land independently.
