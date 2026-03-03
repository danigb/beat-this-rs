# Output Arguments Plan

## New CLI args

```
--json [FILE]       Write JSON output         (default ext: .json)
--beats [FILE]      Write beats text file     (default ext: .beats)
--click [FILE]      Write click-track WAV     (default ext: .click.wav)
--mix [FILE]        Write mixed audio WAV     (default ext: .mix.wav)
--overwrite         Overwrite existing files  (default: skip)
```

## Behavior

### No output flags → JSON to stdout (current default, unchanged)

```bash
beat-this input.wav
# → JSON to stdout, nothing written to disk
```

This keeps pipe-friendly usage working: `beat-this input.wav | jq .bpm`

### Any output flag → write files, summary to stderr, nothing to stdout

```bash
beat-this input.wav --json --beats
# stderr: "Wrote input.json, input.beats (12 beats, 120.0 BPM)"
# stdout: (empty)
```

### Optional path argument

```bash
beat-this input.wav --json                  # → input.json
beat-this input.wav --json results.json     # → results.json
```

### Skip existing unless --overwrite

```bash
beat-this input.wav --json
# stderr: "Skipped input.json (already exists, use --overwrite)"

beat-this input.wav --json --overwrite
# stderr: "Wrote input.json (12 beats, 120.0 BPM)"
```

### Batch mode

File path argument is ignored. Output derived from each input filename.

```bash
beat-this ./music/ -r --json --beats
# stderr:
#   [1/3] song1.wav — 48 beats, 128.0 BPM (0.42s) → song1.json, song1.beats
#   [2/3] song2.wav — 36 beats, 95.2 BPM (0.38s) → song2.json, song2.beats
#   [3/3] song3.wav — 52 beats, 140.0 BPM (0.45s) → song3.json, song3.beats
```

Without any output flags, batch mode still writes `beat-this.json` in the directory (current behavior).

## Removed args

- `--output-beats` (bool) → replaced by `--beats`
- `--output-click <PATH>` → replaced by `--click`
- `--output-mixed <PATH>` → replaced by `--mix`

## Implementation steps

1. **Replace CLI args**: Remove `output_beats`, `output_click`, `output_mixed`. Add `json`, `beats`, `click`, `mix` as `Option<Option<PathBuf>>` (clap `num_args = 0..=1`). Add `overwrite` as bool.

2. **Add path resolution helper**: `resolve_output_path(input: &Path, flag: &Option<Option<PathBuf>>, ext: &str) -> Option<PathBuf>` — returns `None` if flag not set, `Some(explicit_path)` if path given, `Some(input.with_extension(ext))` if flag present without path.

3. **Add write-with-skip helper**: `write_if_needed(path: &Path, overwrite: bool, write_fn: impl FnOnce(&Path) -> Result<()>) -> Result<WriteStatus>` where `WriteStatus` is `Wrote | Skipped`.

4. **Update `run_pipeline`**: Compute resolved paths for all four outputs. If none are set, print JSON to stdout (current behavior). If any are set, write files and print summary to stderr.

5. **Update `run_batch`**: For each file, compute per-file output paths (ignore explicit path arg). Include written files in per-file stderr line. Batch summary JSON (`beat-this.json`) is still written when no output flags are given.

6. **Update README and tests**.
