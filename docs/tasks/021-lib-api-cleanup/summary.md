# Summary: Library API Cleanup (Task 021)

## Goal

Reduce the public API surface, make all internal modules private, and move CLI-only
output code out of the library. The library now exposes a minimal, flat public API
focused on the core analysis pipeline.

## Changes

### Modules made private

All six modules changed from `pub mod` to `mod` in `src/lib.rs`:
`audio`, `mel`, `inference`, `postprocessing`, `output`, `runtime`.

### Renames

- `InferenceRuntime` → `Runtime` (trait for loading models)
- `InferenceSession` → `Model` (trait for running inference)
- `MelProcessor` → `MelExtractor` (internal, no longer public)
- `BeatInference` → `BeatPredictor` (internal, no longer public)
- `PostProcessor` → `PeakPicker` (internal, no longer public)
- `OrtSession` → `OrtModel` (internal impl detail)
- `RtenSession` → `RtenModel` (internal impl detail)
- Internal `session` fields/params → `model` throughout

### Types removed from public API

These types are no longer re-exported from the library:

- `MelExtractor` (was `MelProcessor`) — internal pipeline stage
- `BeatPredictor` (was `BeatInference`) — internal pipeline stage
- `PeakPicker` (was `PostProcessor`) — internal pipeline stage
- `num_mel_frames` — helper removed (use `analysis.mel.shape` directly)

### Output code moved to binary

All CLI-only output functions and types moved from `src/output.rs` to `src/main.rs`:

- `write_json_file`, `write_beats_file`, `write_click_track`, `write_mixed_audio`
- `write_mel_npy`, `write_batch_json`, `build_json_output`, `print_json_stdout`
- `JsonOutput`, `BeatEntry`, `BatchFileEntry`, `BatchSummary`, `BatchSummaryOutput`

The library retains only `beat_counts()` and `calculate_bpm()` in `output.rs`.

### New API additions

- `BeatThis::from_models(mel_model, beat_model)` — construct from pre-built models
- `BeatThis::beat_model_mut()` — mutable access to the beat model (e.g. for ORT profiling)
- `BeatThis::analyze_audio_timed()` — returns `TimedAnalysis` with per-stage `Duration`
- `AnalysisTiming` struct — mel, predict, decode durations
- `TimedAnalysis` struct — analysis + timing

### `BeatThis` fields made private

The `mel`, `predictor`, and `peak_picker` fields are no longer `pub`. Use the
provided methods (`analyze_audio`, `analyze_file`, `from_models`, `beat_model_mut`)
instead.

### Tests updated

All integration tests rewritten to use the public API only:
- `mel_integration.rs` — uses `BeatThis::analyze_audio()`, checks `result.mel.shape`
- `inference_integration.rs` — uses `BeatThis::analyze_audio()`, checks logits
- `postprocessing_integration.rs` — uses `BeatThis::analyze_audio()`, checks beats/downbeats
- `output_integration.rs` — tests only `beat_counts()` and `calculate_bpm()`
- `cross_runtime.rs`, `ort_integration.rs`, `rten_integration.rs` — variable renames

## Public API surface (16 items)

```
beat_this::BeatThis          -- high-level pipeline struct (generic over Model)
beat_this::BeatAnalysis      -- analysis result (beats, downbeats, mel, logits)
beat_this::AnalysisTiming    -- per-stage timing (mel, predict, decode)
beat_this::TimedAnalysis     -- analysis + timing
beat_this::Runtime           -- trait for loading models from ONNX files
beat_this::Model             -- trait for running inference
beat_this::Tensor            -- simple f32 tensor with shape
beat_this::RtenRuntime       -- pure-Rust ONNX backend
beat_this::OrtRuntime        -- ONNX Runtime backend
beat_this::load_audio        -- load and resample audio files
beat_this::AudioData         -- loaded audio samples + sample rate
beat_this::calculate_bpm     -- estimate BPM from beat times
beat_this::beat_counts       -- assign beat numbers within measures
```
