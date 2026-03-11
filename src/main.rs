use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context};
use clap::Parser;

use beat_this::{
    build_json_output, print_json_stdout, write_batch_json, write_beats_file, write_click_track,
    write_json_file, write_mel_npy, write_mixed_audio, BatchFileEntry, BatchSummary,
    BatchSummaryOutput, BeatAnalysis, Model, OrtRuntime, RtenRuntime,
    Runtime as _,
};

const DEFAULT_MODEL_PATH: &str = "models/beat_this.onnx";
const DEFAULT_MEL_MODEL_PATH: &str = "models/mel_spectrogram.onnx";

#[derive(Parser)]
#[command(
    name = "beat-this",
    version,
    about = "Beat and downbeat tracking using Beat This! models"
)]
struct Cli {
    /// Path to an audio file, directory, or glob pattern (e.g. "folder/**/*.mp3")
    input: String,

    /// Path to the beat model ONNX file
    #[arg(long = "model", default_value = DEFAULT_MODEL_PATH)]
    model_path: PathBuf,

    /// Path to the mel spectrogram ONNX file
    #[arg(long = "mel-model", default_value = DEFAULT_MEL_MODEL_PATH)]
    mel_model_path: PathBuf,

    /// Inference runtime to use
    #[arg(long = "runtime", value_enum, default_value = "rten")]
    runtime: RuntimeChoice,

    /// Write JSON output [=FILE]
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "")]
    json: Option<String>,

    /// Write beats text file [=FILE]
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "")]
    beats: Option<String>,

    /// Write click-track WAV [=FILE]
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "")]
    click: Option<String>,

    /// Write mixed audio WAV [=FILE]
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "")]
    mix: Option<String>,

    /// Write mel spectrogram as numpy .npy file [=FILE]
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "")]
    mel: Option<String>,

    /// Overwrite existing output files
    #[arg(long)]
    overwrite: bool,

    /// Recurse into subdirectories (batch mode only)
    #[arg(short = 'r', long = "recursive")]
    recursive: bool,

    /// Print timing for each processing stage
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,

    /// Enable ORT profiling and write trace JSON to this prefix
    #[arg(long = "profile")]
    profile: Option<String>,
}

#[derive(Clone, clap::ValueEnum)]
enum RuntimeChoice {
    Ort,
    Rten,
}

// --- Input resolution ---

/// Resolved input: either a single file or a batch of files.
enum InputMode {
    SingleFile(PathBuf),
    Batch {
        files: Vec<PathBuf>,
        summary_dir: PathBuf,
    },
}

const AUDIO_EXTENSIONS: &[&str] = &["wav", "mp3", "flac", "ogg"];

/// Check if a path has an audio file extension.
fn is_audio_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| AUDIO_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
}

/// Resolve the input argument into a single file or batch of files.
fn resolve_input(input: &str, recursive: bool) -> anyhow::Result<InputMode> {
    let path = Path::new(input);

    // 1. Existing file
    if path.is_file() {
        return Ok(InputMode::SingleFile(path.to_path_buf()));
    }

    // 2. Existing directory
    if path.is_dir() {
        let files = find_audio_files(path, recursive)?;
        ensure!(
            !files.is_empty(),
            "No audio files found in {}",
            path.display()
        );
        return Ok(InputMode::Batch {
            files,
            summary_dir: path.to_path_buf(),
        });
    }

    // 3. Glob pattern
    if input.contains('*') || input.contains('?') || input.contains('[') {
        let mut files: Vec<PathBuf> = glob::glob(input)
            .with_context(|| format!("Invalid glob pattern: {}", input))?
            .filter_map(|e| e.ok())
            .filter(|p| p.is_file() && is_audio_extension(p))
            .collect();
        files.sort();
        ensure!(
            !files.is_empty(),
            "No audio files matched pattern: {}",
            input
        );
        let summary_dir = std::env::current_dir()?;
        return Ok(InputMode::Batch { files, summary_dir });
    }

    // 4. Nothing matched
    bail!("Input not found: {}", input);
}

/// Find audio files in a directory, optionally recursing into subdirectories.
fn find_audio_files(dir: &Path, recursive: bool) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_audio_files(dir, recursive, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_audio_files(dir: &Path, recursive: bool, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("Cannot read directory: {}", dir.display()))?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() && recursive {
            collect_audio_files(&path, true, out)?;
        } else if path.is_file() && is_audio_extension(&path) {
            out.push(path);
        }
    }
    Ok(())
}

// --- Output flags ---

/// Decoupled output flags for write_outputs (so batch mode can override defaults).
struct OutputFlags {
    json: Option<String>,
    beats: Option<String>,
    click: Option<String>,
    mix: Option<String>,
    mel: Option<String>,
    overwrite: bool,
}

impl OutputFlags {
    /// Build from CLI args directly.
    fn from_cli(cli: &Cli) -> Self {
        Self {
            json: cli.json.clone(),
            beats: cli.beats.clone(),
            click: cli.click.clone(),
            mix: cli.mix.clone(),
            mel: cli.mel.clone(),
            overwrite: cli.overwrite,
        }
    }

    /// Build for batch mode: if no flags set, default to --json (auto-named).
    fn for_batch(cli: &Cli) -> Self {
        if cli.json.is_some()
            || cli.beats.is_some()
            || cli.click.is_some()
            || cli.mix.is_some()
            || cli.mel.is_some()
        {
            Self::from_cli(cli)
        } else {
            Self {
                json: Some(String::new()),
                beats: None,
                click: None,
                mix: None,
                mel: None,
                overwrite: cli.overwrite,
            }
        }
    }

    fn has_flags(&self) -> bool {
        self.json.is_some()
            || self.beats.is_some()
            || self.click.is_some()
            || self.mix.is_some()
            || self.mel.is_some()
    }
}

/// Resolve an output file path from a flag value and input file path.
///
/// - Flag not set (`None`) → `None`
/// - Flag set without value (`Some("")`) → derive from input path with given extension
/// - Flag set with value (`Some("file.json")`) → use as-is
fn resolve_output_path(input: &Path, flag: &Option<String>, ext: &str) -> Option<PathBuf> {
    let value = flag.as_ref()?;
    if value.is_empty() {
        Some(input.with_extension(ext))
    } else {
        Some(PathBuf::from(value))
    }
}

/// Write a file if it doesn't already exist (or --overwrite is set).
/// Returns `true` if the file was written, `false` if skipped.
fn write_if_needed(
    path: &Path,
    overwrite: bool,
    write_fn: impl FnOnce(&Path) -> anyhow::Result<()>,
) -> anyhow::Result<bool> {
    if path.exists() && !overwrite {
        eprintln!(
            "Skipped {} (already exists, use --overwrite)",
            path.display()
        );
        return Ok(false);
    }
    write_fn(path)?;
    Ok(true)
}

/// Write all requested outputs for a single file, returning list of written file names.
fn write_outputs(
    input: &Path,
    analysis: &BeatAnalysis,
    flags: &OutputFlags,
) -> anyhow::Result<Vec<String>> {
    let mut written = Vec::new();

    if let Some(path) = resolve_output_path(input, &flags.json, "json") {
        if write_if_needed(&path, flags.overwrite, |p| write_json_file(p, analysis))? {
            written.push(path.display().to_string());
        }
    }

    if let Some(path) = resolve_output_path(input, &flags.beats, "beats") {
        if write_if_needed(&path, flags.overwrite, |p| write_beats_file(p, analysis))? {
            written.push(path.display().to_string());
        }
    }

    if let Some(path) = resolve_output_path(input, &flags.click, "click.wav") {
        if write_if_needed(&path, flags.overwrite, |p| write_click_track(p, analysis))? {
            written.push(path.display().to_string());
        }
    }

    if let Some(path) = resolve_output_path(input, &flags.mix, "mix.wav") {
        let write_mix = |p: &Path| -> anyhow::Result<()> {
            let audio = beat_this::load_audio(input, 44100)?;
            write_mixed_audio(p, analysis, &audio.samples, audio.sample_rate)?;
            Ok(())
        };
        if write_if_needed(&path, flags.overwrite, write_mix)? {
            written.push(path.display().to_string());
        }
    }

    if let Some(path) = resolve_output_path(input, &flags.mel, "mel.npy") {
        if write_if_needed(&path, flags.overwrite, |p| write_mel_npy(p, analysis))? {
            written.push(path.display().to_string());
        }
    }

    Ok(written)
}

// --- Processing ---

/// Result of processing a single file (analysis + audio duration).
struct FileResult {
    analysis: BeatAnalysis,
    duration_secs: f32,
}

/// Process a single audio file through the pipeline, returning analysis and duration.
fn process_single_file<S: Model>(
    bt: &mut beat_this::BeatThis<S>,
    path: &Path,
    verbose: bool,
) -> anyhow::Result<FileResult> {
    let t = Instant::now();
    let audio = beat_this::load_audio(path, 22050)?;
    let duration_secs = audio.samples.len() as f32 / audio.sample_rate as f32;
    if verbose {
        eprintln!(
            "[timing] Audio loading: {:.3}s ({} samples, {:.1}s duration)",
            t.elapsed().as_secs_f64(),
            audio.samples.len(),
            duration_secs
        );
    }

    let t = Instant::now();
    let mel = bt.mel.extract(&audio.samples)?;
    if verbose {
        eprintln!(
            "[timing] Mel spectrogram: {:.3}s ({} frames)",
            t.elapsed().as_secs_f64(),
            mel.shape[1]
        );
    }

    let t = Instant::now();
    let (beat_logits, downbeat_logits) = bt.predictor.predict(&mel)?;
    if verbose {
        eprintln!("[timing] Beat prediction: {:.3}s", t.elapsed().as_secs_f64());
    }

    let t = Instant::now();
    let (beats, downbeats) = bt.peak_picker.decode(&beat_logits, &downbeat_logits)?;
    if verbose {
        eprintln!(
            "[timing] Post-processing: {:.3}s",
            t.elapsed().as_secs_f64()
        );
    }

    Ok(FileResult {
        analysis: BeatAnalysis {
            beats,
            downbeats,
            mel,
            beat_logits,
            downbeat_logits,
        },
        duration_secs,
    })
}

/// Run the full single-file pipeline (audio → mel → inference → postprocessing → output).
fn run_pipeline<S: Model>(
    bt: &mut beat_this::BeatThis<S>,
    cli: &Cli,
    input_path: &Path,
) -> anyhow::Result<()> {
    eprintln!("Processing {}...", input_path.display());

    let file_result = process_single_file(bt, input_path, cli.verbose)?;
    let analysis = &file_result.analysis;

    let json_out = build_json_output(analysis);
    eprintln!(
        "Found {} beats ({} downbeats, {:.1} BPM)",
        analysis.beats.len(),
        analysis.downbeats.len(),
        json_out.bpm.unwrap_or(0.0),
    );

    let flags = OutputFlags::from_cli(cli);
    if !flags.has_flags() {
        // Default: JSON to stdout
        print_json_stdout(analysis)?;
    } else {
        let written = write_outputs(input_path, analysis, &flags)?;
        if !written.is_empty() {
            eprintln!("Wrote {}", written.join(", "));
        }
    }

    Ok(())
}

/// Run batch processing over a list of audio files.
fn run_batch<S: Model>(
    bt: &mut beat_this::BeatThis<S>,
    files: &[PathBuf],
    summary_dir: &Path,
    cli: &Cli,
    model_loading_secs: f32,
) -> anyhow::Result<()> {
    eprintln!("Processing {} files...", files.len());

    let flags = OutputFlags::for_batch(cli);
    let mut file_entries = Vec::new();
    let mut total_duration = 0.0f64;
    let mut total_processing = 0.0f64;
    let mut failed = 0usize;

    for (i, path) in files.iter().enumerate() {
        let filename = path.to_string_lossy().to_string();

        let t = Instant::now();
        let result = match process_single_file(bt, path, cli.verbose) {
            Ok(r) => r,
            Err(e) => {
                failed += 1;
                eprintln!("  [{}/{}] {} — ERROR: {}", i + 1, files.len(), filename, e);
                continue;
            }
        };
        let elapsed = t.elapsed().as_secs_f64();

        let json_out = build_json_output(&result.analysis);

        let written = write_outputs(path, &result.analysis, &flags)?;

        if written.is_empty() {
            eprintln!(
                "  [{}/{}] {} — {} beats, {:.1} BPM ({:.2}s)",
                i + 1,
                files.len(),
                filename,
                result.analysis.beats.len(),
                json_out.bpm.unwrap_or(0.0),
                elapsed
            );
        } else {
            eprintln!(
                "  [{}/{}] {} — {} beats, {:.1} BPM ({:.2}s) → {}",
                i + 1,
                files.len(),
                filename,
                result.analysis.beats.len(),
                json_out.bpm.unwrap_or(0.0),
                elapsed,
                written.join(", ")
            );
        }

        file_entries.push(BatchFileEntry {
            input: filename,
            duration_secs: result.duration_secs,
            processing_time_secs: elapsed as f32,
            outputs: written,
        });

        total_duration += result.duration_secs as f64;
        total_processing += elapsed;
    }

    let realtime_factor = if total_processing > 0.0 {
        total_duration / total_processing
    } else {
        0.0
    };

    // Always write batch summary
    let batch = BatchSummaryOutput {
        files: file_entries,
        summary: BatchSummary {
            total_files: files.len(),
            failed_files: failed,
            total_duration_secs: total_duration as f32,
            total_processing_time_secs: total_processing as f32,
            model_loading_time_secs: model_loading_secs,
            realtime_factor: realtime_factor as f32,
        },
    };

    let out_path = summary_dir.join("beat_this.json");
    write_batch_json(&out_path, &batch)?;
    eprintln!(
        "Wrote {} ({} files, {:.1}s total)",
        out_path.display(),
        files.len(),
        total_processing
    );

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Resolve input into single file or batch
    let input_mode = resolve_input(&cli.input, cli.recursive)?;

    // Resolve model paths and validate
    let mel_path = cli.mel_model_path.clone();
    let beat_path = cli.model_path.clone();

    ensure!(
        mel_path.exists(),
        "Mel model not found: {}\nDownload models or use --mel-model to specify the path.",
        mel_path.display()
    );
    ensure!(
        beat_path.exists(),
        "Beat model not found: {}\nDownload models or use --model to specify the path.",
        beat_path.display()
    );

    let total_start = Instant::now();

    eprintln!("Loading models...");
    let t = Instant::now();

    match cli.runtime {
        RuntimeChoice::Ort => {
            let runtime = OrtRuntime::default();
            if cli.verbose {
                let coreml = if runtime.is_coreml_available() {
                    "yes"
                } else {
                    "no"
                };
                eprintln!("[info] Runtime: ort");
                eprintln!("[info] CoreML available: {}", coreml);
            }
            // Use a separate runtime for the beat model when profiling
            let beat_runtime = if let Some(ref prefix) = cli.profile {
                OrtRuntime {
                    profiling_path: Some(std::path::PathBuf::from(prefix)),
                    ..Default::default()
                }
            } else {
                OrtRuntime::default()
            };
            let mel_session = runtime.load_model(&mel_path)
                .context("Failed to initialize ort runtime. Is the ONNX Runtime library installed?\n  \
                    macOS: brew install onnxruntime\n  \
                    Or use --runtime rten (default) for a pure-Rust runtime with no external dependencies.")?;
            let beat_session = beat_runtime
                .load_model(&beat_path)
                .context("Failed to load beat model with ort runtime.")?;
            let mut bt = beat_this::BeatThis {
                mel: beat_this::MelExtractor::new(mel_session),
                predictor: beat_this::BeatPredictor::new(beat_session),
                peak_picker: beat_this::PeakPicker::default(),
            };
            let model_loading_secs = t.elapsed().as_secs_f64() as f32;
            if cli.verbose {
                eprintln!("[timing] Model loading: {:.3}s", model_loading_secs);
            }

            match &input_mode {
                InputMode::SingleFile(path) => run_pipeline(&mut bt, &cli, path)?,
                InputMode::Batch { files, summary_dir } => {
                    run_batch(&mut bt, files, summary_dir, &cli, model_loading_secs)?;
                }
            }

            // End ORT profiling
            if cli.profile.is_some() {
                if let Ok(path) = bt.predictor.model_mut().end_profiling() {
                    eprintln!("[profile] Beat model trace written to: {}", path);
                }
            }
        }

        RuntimeChoice::Rten => {
            if cli.verbose {
                eprintln!("[info] Runtime: rten (pure Rust)");
            }
            if cli.profile.is_some() {
                eprintln!(
                    "[warn] Profiling is only supported with the ort runtime, ignoring --profile"
                );
            }
            let runtime = RtenRuntime;
            let mut bt = beat_this::BeatThis::new(&runtime, &mel_path, &beat_path)?;
            let model_loading_secs = t.elapsed().as_secs_f64() as f32;
            if cli.verbose {
                eprintln!("[timing] Model loading: {:.3}s", model_loading_secs);
            }

            match &input_mode {
                InputMode::SingleFile(path) => run_pipeline(&mut bt, &cli, path)?,
                InputMode::Batch { files, summary_dir } => {
                    run_batch(&mut bt, files, summary_dir, &cli, model_loading_secs)?;
                }
            }
        }
    }

    if cli.verbose {
        eprintln!(
            "[timing] Total: {:.3}s",
            total_start.elapsed().as_secs_f64()
        );
    }

    Ok(())
}
