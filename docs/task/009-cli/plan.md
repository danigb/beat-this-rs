# Task 009: CLI — Development Plan

Implements **Step 8** from [plan.md](../001-research/plan.md): a clap-based command-line interface that ties together the full beat-tracking pipeline — audio loading, inference, and output generation.

## Goal

Replace the placeholder `main.rs` with a fully functional CLI binary. The CLI accepts an audio file, locates ONNX models (with sensible defaults and overrides), runs the beat-tracking pipeline, and produces the requested output: `.beats` file, click track WAV, mixed audio WAV, BPM printout, or any combination. The binary should provide clear progress feedback and helpful error messages.

---

## CLI Interface

```
beat-this [OPTIONS] <AUDIO_FILE>

Arguments:
  <AUDIO_FILE>  Path to the input audio file (WAV, MP3, FLAC, OGG)

Options:
      --model <PATH>          Path to the beat model ONNX file
                              [default: models/beat_this.onnx]
      --mel-model <PATH>      Path to the mel spectrogram ONNX file
                              [default: models/mel_spectrogram.onnx]
      --model-variant <VARIANT>
                              Model variant to use [default: standard]
                              [possible values: standard, small]
  -o, --output-beats <FILE>   Write beat timestamps to a .beats file
      --output-click <FILE>   Write a click-track WAV file
      --output-mixed <FILE>   Write a mixed audio WAV file (original + clicks)
      --bpm                   Print estimated BPM to stdout
  -h, --help                  Print help
  -V, --version               Print version
```

### Usage examples

```sh
# Minimal: just print beats to stdout
beat-this song.mp3

# Save .beats file
beat-this song.mp3 -o song.beats

# Generate click track + show BPM
beat-this song.mp3 --output-click clicks.wav --bpm

# All outputs at once
beat-this song.mp3 -o song.beats --output-click clicks.wav --output-mixed mixed.wav --bpm

# Use the small model variant
beat-this song.mp3 --model-variant small -o song.beats

# Explicit model paths (overrides --model-variant)
beat-this song.mp3 --model custom/beat.onnx --mel-model custom/mel.onnx -o out.beats
```

### Default behavior

When no output flags are provided, print beat timestamps to stdout (one per line: `time\tbeat_count`), so the tool is useful in pipelines and for quick inspection.

---

## Implementation Steps

### 1. Add `clap` dependency

Add to `Cargo.toml`:

```toml
clap = { version = "4", features = ["derive"] }
```

### 2. Define CLI args struct

In `src/main.rs`, define the argument struct using clap's derive API:

```rust
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "beat-this", version, about = "Beat and downbeat tracking using Beat This! models")]
struct Cli {
    /// Path to the input audio file (WAV, MP3, FLAC, OGG)
    audio_file: PathBuf,

    /// Path to the beat model ONNX file
    #[arg(long = "model", default_value = "models/beat_this.onnx")]
    model_path: PathBuf,

    /// Path to the mel spectrogram ONNX file
    #[arg(long = "mel-model", default_value = "models/mel_spectrogram.onnx")]
    mel_model_path: PathBuf,

    /// Model variant to use (standard or small)
    #[arg(long = "model-variant", value_enum, default_value_t = ModelVariant::Standard)]
    model_variant: ModelVariant,

    /// Write beat timestamps to a .beats file
    #[arg(short = 'o', long = "output-beats")]
    output_beats: Option<PathBuf>,

    /// Write a click-track WAV file
    #[arg(long = "output-click")]
    output_click: Option<PathBuf>,

    /// Write a mixed audio WAV file (original + clicks)
    #[arg(long = "output-mixed")]
    output_mixed: Option<PathBuf>,

    /// Print estimated BPM to stdout
    #[arg(long = "bpm")]
    show_bpm: bool,
}

#[derive(Clone, clap::ValueEnum)]
enum ModelVariant {
    Standard,
    Small,
}
```

### 3. Implement model path resolution

Resolve the final model paths with these rules:

1. If `--model` is explicitly provided, use it as-is (ignore `--model-variant`).
2. Otherwise, apply `--model-variant` to select the default path:
   - `standard` → `models/beat_this.onnx`
   - `small` → `models/beat_this_small.onnx`
3. The mel model path is always `--mel-model` (it's the same for both variants).

```rust
fn resolve_model_paths(cli: &Cli) -> (PathBuf, PathBuf) {
    let model_was_explicit = /* check if --model was provided by the user */;

    let beat_model = if model_was_explicit {
        cli.model_path.clone()
    } else {
        match cli.model_variant {
            ModelVariant::Standard => PathBuf::from("models/beat_this.onnx"),
            ModelVariant::Small => PathBuf::from("models/beat_this_small.onnx"),
        }
    };

    (cli.mel_model_path.clone(), beat_model)
}
```

To detect whether `--model` was explicitly provided vs. using the default, check `cli.model_path != PathBuf::from("models/beat_this.onnx")` or use clap's `value_source()` API by deriving `Args` and checking at parse time. The simplest approach: just check if the path differs from the default value.

### 4. Implement main function

The `main` function orchestrates the pipeline:

```rust
fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // 1. Validate input file exists
    anyhow::ensure!(
        cli.audio_file.exists(),
        "Audio file not found: {}",
        cli.audio_file.display()
    );

    // 2. Resolve model paths and validate
    let (mel_path, beat_path) = resolve_model_paths(&cli);
    anyhow::ensure!(
        mel_path.exists(),
        "Mel model not found: {}\nDownload models or use --mel-model to specify the path.",
        mel_path.display()
    );
    anyhow::ensure!(
        beat_path.exists(),
        "Beat model not found: {}\nDownload models or use --model to specify the path.",
        beat_path.display()
    );

    // 3. Initialize runtime and load models
    eprintln!("Loading models...");
    let runtime = beat_this::runtime::ort::OrtRuntime::default();
    let mut bt = beat_this::BeatThis::new(&runtime, &mel_path, &beat_path)?;

    // 4. Run pipeline
    eprintln!("Processing {}...", cli.audio_file.display());
    let result = bt.process_file(&cli.audio_file)?;
    eprintln!(
        "Found {} beats ({} downbeats)",
        result.beats.len(),
        result.downbeats.len()
    );

    // 5. Determine if any output flag was given
    let has_output_flag = cli.output_beats.is_some()
        || cli.output_click.is_some()
        || cli.output_mixed.is_some()
        || cli.show_bpm;

    // 6. Default behavior: print beats to stdout
    if !has_output_flag {
        print_beats_stdout(&result);
    }

    // 7. Write requested outputs
    if let Some(ref path) = cli.output_beats {
        beat_this::output::write_beats_file(path, &result)?;
        eprintln!("Wrote beats to {}", path.display());
    }

    if let Some(ref path) = cli.output_click {
        beat_this::output::write_click_track(path, &result)?;
        eprintln!("Wrote click track to {}", path.display());
    }

    if let Some(ref path) = cli.output_mixed {
        let audio = beat_this::load_audio(&cli.audio_file, 44100)?;
        beat_this::output::write_mixed_audio(path, &result, &audio.samples, audio.sample_rate)?;
        eprintln!("Wrote mixed audio to {}", path.display());
    }

    if cli.show_bpm {
        match beat_this::output::calculate_bpm(&result) {
            Some(bpm) => println!("{:.1} BPM", bpm),
            None => eprintln!("Could not calculate BPM (too few beats)"),
        }
    }

    Ok(())
}
```

### 5. Implement `print_beats_stdout`

Default output when no flags are given — same format as `.beats` file but to stdout:

```rust
fn print_beats_stdout(result: &BeatResult) {
    let counts = beat_this::output::beat_counts(result);
    for (&time, &count) in result.beats.iter().zip(counts.iter()) {
        println!("{:.3}\t{}", time, count);
    }
}
```

### 6. Progress messages

All progress/status messages go to `stderr` (via `eprintln!`) so they don't interfere with stdout output. This matters when piping:

```sh
beat-this song.mp3 | head -5        # only beat lines, no noise
beat-this song.mp3 -o out.beats     # progress on stderr, file written silently
```

---

## File Changes

| File | Action |
|------|--------|
| `Cargo.toml` | Add `clap = { version = "4", features = ["derive"] }` dependency |
| `src/main.rs` | **Rewrite** — replace placeholder with full CLI implementation |

---

## Design Decisions

1. **clap derive API**: uses `#[derive(Parser)]` for a declarative, type-safe argument definition. This is the idiomatic modern Rust approach and produces auto-generated `--help` text. No need for the builder API's flexibility here.

2. **Default behavior prints to stdout**: when no output flags are given, beats are printed to stdout in the same `.beats` format. This makes the tool immediately useful for quick inspection (`beat-this song.mp3`) and composable in shell pipelines (`beat-this song.mp3 | wc -l` to count beats). This follows the Unix convention of producing useful output by default.

3. **Progress to stderr**: all progress messages (`Loading models...`, `Processing...`, `Found N beats`) go to stderr, keeping stdout clean for data output. This is critical for piping and scripting.

4. **`--model-variant` vs `--model`**: the variant flag provides a convenient way to switch between `standard` and `small` models without remembering file names. Explicit `--model` overrides the variant, giving power users full control. This two-level approach avoids a confusing interaction between the flags.

5. **`anyhow::Result` from `main`**: returning `Result` from `main` gives clean error output on failure (prints the error chain). Combined with early validation of file paths, the user gets clear messages like `"Beat model not found: models/beat_this.onnx"` instead of cryptic ONNX loading errors.

6. **Reload audio for mixed output**: when `--output-mixed` is requested, the original audio is reloaded at 44100 Hz (not the 22050 Hz used by the pipeline). This is because `process_file` internally resamples to 22050 Hz and discards the original samples. Reloading at 44100 Hz preserves audio quality in the mixed output. This is a one-time cost and simpler than threading the original audio through the pipeline.

7. **No `--output` default path magic**: each output flag requires an explicit file path. This avoids surprising file creation and is explicit about what gets written where. The user is in full control.

8. **Two files only**: the entire CLI is implemented in `src/main.rs` plus the `Cargo.toml` change. No new modules needed — the CLI is a thin orchestration layer over the existing library API.

---

## Validation

1. **`cargo build` succeeds**: the binary compiles without warnings.
2. **`beat-this --help`**: prints well-formatted help text with all arguments and options.
3. **`beat-this --version`**: prints the version from `Cargo.toml`.
4. **Missing audio file**: prints a clear error message and exits with non-zero status.
5. **Missing model files**: prints a clear error message pointing to `--model` / `--mel-model` flags.
6. **Default output**: `beat-this song.mp3` prints beat timestamps to stdout.
7. **File outputs**: `-o song.beats`, `--output-click clicks.wav`, `--output-mixed mixed.wav` all produce valid output files.
8. **`--bpm` flag**: prints BPM to stdout.
9. **Combined flags**: `beat-this song.mp3 -o out.beats --output-click clicks.wav --bpm` produces all three outputs.
10. **`--model-variant small`**: uses `models/beat_this_small.onnx` correctly.
11. **Pipe-friendly**: `beat-this song.mp3 2>/dev/null` outputs only beat data; progress messages go to stderr.
