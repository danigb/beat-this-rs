# Batch / Directory Processing

## Goal

`beat-this ./album/` processes all audio files in a directory. Models load once. At the end, writes `beat-this.json` with per-file results, summary, and performance metrics.

## CLI changes

The `audio_file` positional argument already accepts a `PathBuf`. If the path is a directory, enter batch mode. No new flags needed for basic operation.

```
# Single file (unchanged)
beat-this song.mp3

# Directory (new)
beat-this ./album/
beat-this ./album/ --output-beats
```

In batch mode:
- Walk the directory (non-recursive by default)
- Filter for supported audio extensions: `.wav`, `.mp3`, `.flac`, `.ogg`
- Sort files alphabetically for deterministic order
- Process each file through the existing pipeline
- Write `beat-this.json` into the target directory at the end

Add `--recursive` flag to walk subdirectories.

## Output: `beat-this.json`

Written to the input directory (e.g. `./album/beat-this.json`). Contains per-file results plus aggregate metrics:

```json
{
  "files": [
    {
      "file": "01-intro.mp3",
      "beats": [
        { "time": 0.5, "beat": 1, "downbeat": true },
        { "time": 1.0, "beat": 2, "downbeat": false }
      ],
      "downbeats": [0.5],
      "bpm": 120.0,
      "duration_secs": 245.3,
      "processing_time_secs": 3.41
    },
    {
      "file": "02-verse.flac",
      "beats": [ ... ],
      "downbeats": [ ... ],
      "bpm": 98.5,
      "duration_secs": 312.1,
      "processing_time_secs": 4.56
    }
  ],
  "summary": {
    "total_files": 12,
    "total_duration_secs": 2847.5,
    "total_processing_time_secs": 41.2,
    "model_loading_time_secs": 0.04,
    "realtime_factor": 69.1
  }
}
```

Fields:
- `files[].file` — filename relative to the input directory
- `files[].beats`, `files[].downbeats`, `files[].bpm` — same as single-file JSON output (reuse `JsonOutput`)
- `files[].duration_secs` — audio duration in seconds
- `files[].processing_time_secs` — wall time for that file (audio load + mel + inference + post)
- `summary.total_files` — number of files processed
- `summary.total_duration_secs` — sum of audio durations
- `summary.total_processing_time_secs` — sum of per-file processing times
- `summary.model_loading_time_secs` — one-time model load cost
- `summary.realtime_factor` — `total_duration_secs / total_processing_time_secs`

### `--output-beats` in batch mode

When `--output-beats` is passed in batch mode, write a `.beats` file next to each audio file (e.g. `01-intro.beats`) instead of including beat data in the JSON. The `beat-this.json` summary is still written.

## stdout behavior

In batch mode, progress goes to stderr (already the case for single-file mode). The `beat-this.json` content is NOT printed to stdout — it's written to disk. This keeps batch mode pipe-friendly. Print the output path at the end:

```
Processing ./album/...
  [1/12] 01-intro.mp3 — 187 beats, 120.0 BPM (3.41s)
  [2/12] 02-verse.flac — 243 beats, 98.5 BPM (4.56s)
  ...
Wrote ./album/beat-this.json (12 files, 41.2s total)
```

All of this goes to stderr.

## Implementation

### Files to modify

- **`src/main.rs`** — detect directory vs file, dispatch to batch or single-file path
- **`src/output.rs`** — add `BatchFileOutput`, `BatchSummary`, `BatchOutput` structs and `write_batch_json()`

### New code in `src/main.rs`

Sketch of the batch path inside `run_pipeline`:

```rust
fn run_batch<S: InferenceSession>(
    bt: &mut BeatThis<S>,
    dir: &Path,
    cli: &Cli,
) -> Result<()> {
    let files = find_audio_files(dir, cli.recursive)?;
    eprintln!("Processing {}... ({} files)", dir.display(), files.len());

    let mut file_outputs = Vec::new();
    let mut total_duration = 0.0f64;
    let mut total_processing = 0.0f64;

    for (i, path) in files.iter().enumerate() {
        let t = Instant::now();
        let audio = load_audio(path, 22050)?;
        let duration = audio.samples.len() as f64 / audio.sample_rate as f64;
        let result = bt.process_audio(&audio.samples, audio.sample_rate)?;
        let elapsed = t.elapsed().as_secs_f64();

        let json_out = output::build_json_output(&result);
        let filename = path.file_name().unwrap().to_string_lossy().to_string();

        eprintln!(
            "  [{}/{}] {} — {} beats, {:.1} BPM ({:.2}s)",
            i + 1, files.len(), filename,
            result.beats.len(),
            json_out.bpm.unwrap_or(0.0),
            elapsed
        );

        // Optionally write .beats file
        if cli.output_beats {
            let beats_path = path.with_extension("beats");
            output::write_beats_file(&beats_path, &result)?;
        }

        file_outputs.push(output::BatchFileOutput {
            file: filename,
            json: json_out,
            duration_secs: duration as f32,
            processing_time_secs: elapsed as f32,
        });

        total_duration += duration;
        total_processing += elapsed;
    }

    // Write beat-this.json
    let batch = output::BatchOutput { files: file_outputs, summary: ... };
    let out_path = dir.join("beat-this.json");
    output::write_batch_json(&out_path, &batch)?;
    eprintln!("Wrote {} ({} files, {:.1}s total)", out_path.display(), files.len(), total_processing);

    Ok(())
}

fn find_audio_files(dir: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    // Walk dir, filter by extension, sort alphabetically
    let extensions = ["wav", "mp3", "flac", "ogg"];
    // ...
}
```

### New structs in `src/output.rs`

```rust
#[derive(Serialize)]
pub struct BatchFileOutput {
    pub file: String,
    #[serde(flatten)]
    pub json: JsonOutput,       // reuse existing beats/downbeats/bpm
    pub duration_secs: f32,
    pub processing_time_secs: f32,
}

#[derive(Serialize)]
pub struct BatchSummary {
    pub total_files: usize,
    pub total_duration_secs: f32,
    pub total_processing_time_secs: f32,
    pub model_loading_time_secs: f32,
    pub realtime_factor: f32,
}

#[derive(Serialize)]
pub struct BatchOutput {
    pub files: Vec<BatchFileOutput>,
    pub summary: BatchSummary,
}

pub fn write_batch_json(path: &Path, output: &BatchOutput) -> Result<()> {
    let file = std::fs::File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, output)?;
    Ok(())
}
```

### Error handling

If a file fails to process, log the error to stderr and continue with the next file. Add an `errors` field to the summary:

```json
"summary": {
  "total_files": 12,
  "failed_files": 1,
  ...
}
```

### Steps

1. Add `--recursive` flag to CLI
2. Add `find_audio_files()` helper in `main.rs`
3. Add `BatchFileOutput`, `BatchSummary`, `BatchOutput` structs and `write_batch_json()` to `output.rs`
4. Add `run_batch()` function in `main.rs`
5. Update `main()` to detect directory and dispatch
6. Add tests for `find_audio_files`, `write_batch_json`, batch output serialization
7. Test with a real directory of audio files

### Not in scope (future)

- Parallelization with rayon (model session may not be `Send`/`Sync` — needs investigation)
- Custom output path for `beat-this.json` (could add `--output-json <path>` later)
- Recursive by default (start conservative, add `--recursive`)
