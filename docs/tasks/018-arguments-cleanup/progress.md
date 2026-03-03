# 018 - Arguments Cleanup

## Summary

Removed redundant CLI arguments and unified output flags into a consistent interface.

## Phase 1: Remove redundant args

- **Removed `--model-variant`**: The `--model` path argument is sufficient to select any model. Removed `ModelVariant` enum, `resolve_beat_model_path()` function, and the CLI arg.
- **Removed `--bpm`**: Redundant since the default JSON output already includes `bpm`. Removed `show_bpm` field and the print logic in `run_pipeline`.
- **Removed `--threads`**: Internal ORT tuning knob that doesn't need to be exposed. `OrtRuntime::default()` already uses `0` (auto), the optimal setting.

## Phase 2: Unified output flags

Replaced `--output-beats` (bool), `--output-click <PATH>`, `--output-mixed <PATH>` with four symmetric flags:

```
--json[=FILE]       Write JSON output         (.json)
--beats[=FILE]      Write beats text file     (.beats)
--click[=FILE]      Write click-track WAV     (.click.wav)
--mix[=FILE]        Write mixed audio WAV     (.mix.wav)
--overwrite         Overwrite existing files
```

Syntax: `--json` auto-names from input, `--json=path.json` uses explicit path (requires `=`).

Behavior:
- No flags: JSON to stdout (unchanged default)
- Any flag: write files, summary to stderr, nothing to stdout
- Optional path: if omitted, derived from input filename
- Skip existing files unless `--overwrite`
- Batch mode: per-file outputs derived from each audio filename

Changes:
- **src/main.rs**: New CLI args with `num_args = 0..=1, require_equals = true` and `Option<String>` (clap can't parse empty `default_missing_value` into `PathBuf`). Added `resolve_output_path`, `has_output_flags`, `write_if_needed`, `write_outputs` helpers. Updated `run_pipeline` and `run_batch`. Removed `print_beats_stdout`.
- **src/output.rs**: Added `write_json_file` function.
- **README.md**: Updated examples, CLI options table, output format section headers.
- **scripts/integration-test.py**: Updated to use `--beats=<path>` syntax.

## Status

Done.
