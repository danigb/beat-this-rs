# 019 - Batch Mode Rework

## Goal

Change batch mode so that:
1. Per-file `<file>.json` is the **default** output (no flags needed)
2. `beat_this.json` becomes a **process summary** only (no beat data)
3. Output flags (`--json`, `--beats`, `--click`, `--mix`) control which per-file outputs are produced
4. Input accepts **glob patterns** (e.g. `folder/**/*.mp3`) in addition to directories

## Current Behavior

### Batch with no flags: `beat_this folder/`
- Writes `folder/beat-this.json` containing **all beat data** for every file + a summary section
- Uses `BatchOutput { files: Vec<BatchFileOutput>, summary: BatchSummary }` where each `BatchFileOutput` has flattened `JsonOutput` (beats, downbeats, bpm)

### Batch with flags: `beat_this folder/ --json --beats`
- Writes per-file outputs (e.g. `song.json`, `song.beats`)
- Prints summary to stderr
- Does **not** write `beat_this.json`

### Single-file with no flags: `beat_this song.wav`
- Prints JSON to stdout (unchanged, keep as-is)

### Single-file with flags: `beat_this song.wav --json --beats`
- Writes `song.json`, `song.beats`

## Desired Behavior

### Batch with no flags: `beat_this folder/`
- Writes per-file `<file>.json` for each audio file (same as current `--json` behavior)
- Writes `beat_this.json` as a **process summary** (no beat data)

### Batch with flags: `beat_this folder/ --json --beats --mix`
- Writes per-file outputs: `<file>.json`, `<file>.beats`, `<file>.mix.wav`
- Writes `beat_this.json` as a process summary listing which files were produced

### Single-file behavior: unchanged
- No flags → JSON to stdout
- With flags → write files, summary to stderr

## New `beat_this.json` Format (Process Summary)

```json
{
  "files": [
    {
      "input": "song1.mp3",
      "duration_secs": 120.0,
      "processing_time_secs": 1.5,
      "outputs": ["song1.json", "song1.beats", "song1.mix.wav"]
    },
    {
      "input": "song2.wav",
      "duration_secs": 60.0,
      "processing_time_secs": 0.8,
      "outputs": ["song2.json", "song2.beats", "song2.mix.wav"]
    }
  ],
  "summary": {
    "total_files": 2,
    "failed_files": 0,
    "total_duration_secs": 180.0,
    "total_processing_time_secs": 2.3,
    "model_loading_time_secs": 0.04,
    "realtime_factor": 78.3
  }
}
```

## Changes Required

### src/output.rs

1. **Replace `BatchFileOutput`** — remove flattened `JsonOutput`, add `outputs` list:
   ```rust
   #[derive(Serialize)]
   pub struct BatchFileEntry {
       pub input: String,
       pub duration_secs: f32,
       pub processing_time_secs: f32,
       pub outputs: Vec<String>,
   }
   ```

2. **Rename `BatchOutput`** to reflect it's a summary:
   ```rust
   #[derive(Serialize)]
   pub struct BatchSummaryOutput {
       pub files: Vec<BatchFileEntry>,
       pub summary: BatchSummary,
   }
   ```
   (`BatchSummary` stays the same.)

3. **Update `write_batch_json`** to use the new type.

4. **Remove** the old `BatchFileOutput` struct (which embedded `JsonOutput`).

### src/main.rs — `run_batch()`

1. **Default per-file JSON**: When no output flags are set, behave as if `--json` was passed. In code:
   ```rust
   // Always write per-file outputs in batch mode
   // Default to --json if no flags specified
   let effective_cli = if !has_output_flags(cli) {
       // treat as --json (auto-named from input)
       // ... set json = Some("".to_string())
   } else {
       cli
   };
   ```
   Cleanest approach: add a helper that returns effective output flags for batch mode.

2. **Always write `beat_this.json`** summary at the end of batch processing.

3. **Collect written file paths** per input file into `BatchFileEntry::outputs`.

4. **Remove** the `json_out` / `build_json_output` call that was only used for the old batch JSON (the per-file `write_outputs` already builds JSON internally).

### Key implementation detail

`write_outputs()` already returns `Vec<String>` of written paths — use that directly as `BatchFileEntry::outputs`.

For the default case (no flags), we need to ensure `write_outputs` gets called with `json = Some("")`. Two options:

- **Option A**: Mutate/clone the CLI to set `json = Some("")` before the loop. Simple but slightly ugly.
- **Option B**: Pass resolved flags to `write_outputs` instead of the whole `Cli`. Cleaner separation.

**Recommendation**: Option A is simplest. Create a local `let json_flag = cli.json.clone().or(Some("".to_string()))` and pass it through. Or just check `has_output_flags` and default:

```rust
let default_json = if has_output_flags(cli) { cli.json.clone() } else { Some("".to_string()) };
```

Then use `default_json` in place of `cli.json` when resolving output paths.

### Tests to update

- `test_write_batch_json` in `src/output.rs` — update to new struct shape
- Integration test in `scripts/integration-test.py` — verify new `beat_this.json` format

### README.md

Update the batch mode documentation to describe the new behavior.

## Glob Input Support

### Motivation

Currently batch mode only accepts a directory. Users should also be able to pass glob patterns to select specific files:

```
beat_this "folder/**/*.mp3"            # all mp3s recursively
beat_this "albums/**/track01.wav"      # first track of each album
beat_this folder/                      # directory (existing behavior)
beat_this song.wav                     # single file (existing behavior)
```

### Crate: `glob` (recommended)

- **Zero dependencies**, 365M+ downloads, maintained by `rust-lang`
- Single function: `glob::glob(pattern)` → iterator of `Result<PathBuf>`
- Supports `**` recursive matching
- Add to Cargo.toml: `glob = "0.3"`

### Input Resolution Logic

The `input` argument is resolved in order:

1. **Existing file** → single-file mode (unchanged)
2. **Existing directory** → find audio files in directory (existing behavior, uses `find_audio_files`)
3. **Contains glob characters** (`*`, `?`, `[`) → expand with `glob::glob()`, filter to audio extensions
4. **None of the above** → error: "Input not found"

```rust
fn resolve_input_files(input: &Path, recursive: bool) -> anyhow::Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    if input.is_dir() {
        return find_audio_files(input, recursive);
    }
    // Try as glob pattern
    let pattern = input.to_string_lossy();
    if pattern.contains('*') || pattern.contains('?') || pattern.contains('[') {
        let mut files: Vec<PathBuf> = glob::glob(&pattern)?
            .filter_map(|e| e.ok())
            .filter(|p| p.is_file() && is_audio_extension(p))
            .collect();
        files.sort();
        anyhow::ensure!(!files.is_empty(), "No audio files matched pattern: {}", pattern);
        return Ok(files);
    }
    anyhow::bail!("Input not found: {}", input.display());
}
```

### Shell Quoting

Glob patterns must be quoted to prevent shell expansion: `beat_this "folder/**/*.mp3"`. This is standard practice (same as `find`, `rg`, etc.). Document in README.

### Impact on `--recursive`

- Directory mode: `--recursive` controls whether to recurse into subdirectories (unchanged)
- Glob mode: `**` in the pattern handles recursion; `--recursive` is ignored
- Single-file mode: `--recursive` is ignored (unchanged)

### Summary output path for glob mode

When input is a glob pattern (not a directory), `beat_this.json` is written to the **current working directory** since there's no single parent directory.

## Summary of Behavior Matrix

| Input | No flags | With flags |
|-------|----------|------------|
| Single file | JSON to stdout | Write specified files |
| Directory | Write `<file>.json` + summary `beat_this.json` in dir | Write specified files + summary `beat_this.json` in dir |
| Glob pattern | Write `<file>.json` + summary `beat_this.json` in cwd | Write specified files + summary `beat_this.json` in cwd |
