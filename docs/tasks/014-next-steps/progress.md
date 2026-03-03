# JSON Output

## Changes

### Default output is now JSON

Running `beat-this audio.wav` prints structured JSON to stdout:

```json
{
  "beats": [
    { "time": 0.5, "beat": 1, "downbeat": true },
    { "time": 1.0, "beat": 2, "downbeat": false },
    { "time": 1.5, "beat": 3, "downbeat": false },
    { "time": 2.0, "beat": 1, "downbeat": true }
  ],
  "downbeats": [0.5, 2.0],
  "bpm": 120.0
}
```

Fields:
- `beats` — array of beat entries with `time` (seconds), `beat` (count within measure, 1 = downbeat), and `downbeat` (boolean)
- `downbeats` — array of downbeat timestamps in seconds
- `bpm` — estimated BPM (null if too few beats)

### `--output-beats` flag

Use `--output-beats` to get the previous plain text format (tab-separated time and beat count):

```
0.500	1
1.000	2
1.500	3
2.000	1
```

Note: `--output-beats` changed from `Option<PathBuf>` (file output) to a boolean flag (stdout).

---

# Batch / Directory Processing

## Changes

### Directory input triggers batch mode

`beat-this ./album/` processes all audio files (`.wav`, `.mp3`, `.flac`, `.ogg`) in the directory. Models load once and are reused for all files.

### `beat-this.json` output

Written to the input directory with per-file results and aggregate metrics:

```json
{
  "files": [
    {
      "file": "01-intro.mp3",
      "beats": [...],
      "downbeats": [0.5],
      "bpm": 120.0,
      "duration_secs": 245.3,
      "processing_time_secs": 3.41
    }
  ],
  "summary": {
    "total_files": 12,
    "failed_files": 0,
    "total_duration_secs": 2847.5,
    "total_processing_time_secs": 41.2,
    "model_loading_time_secs": 0.04,
    "realtime_factor": 69.1
  }
}
```

### `--recursive` flag

`-r` / `--recursive` walks subdirectories. Without it, only top-level files are processed.

### `--output-beats` in batch mode

Writes a `.beats` file next to each audio file. The `beat-this.json` summary is always written.

### Error resilience

Failed files are logged to stderr and skipped. The `failed_files` count appears in the summary.

### Refactored pipeline

Extracted `process_single_file()` to share code between single-file and batch modes. The positional argument was renamed from `audio_file` to `input` (accepts file or directory).

## Files modified

- `Cargo.toml` — added `serde` and `serde_json` dependencies
- `src/output.rs` — added `BeatEntry`, `JsonOutput`, `BatchFileOutput`, `BatchSummary`, `BatchOutput` structs; `build_json_output()`, `print_json_stdout()`, `write_batch_json()`; tests for JSON and batch output
- `src/main.rs` — renamed `audio_file` to `input`; added `--recursive` flag; added `find_audio_files()`, `process_single_file()`, `run_batch()`; `main()` dispatches based on file vs directory
