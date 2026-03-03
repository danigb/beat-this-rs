use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{ensure, Context};
use clap::Parser;

use beat_this::output;
use beat_this::postprocessing::BeatResult;
use beat_this::runtime::InferenceRuntime;
use beat_this::InferenceSession;

const DEFAULT_MODEL_PATH: &str = "models/beat_this.onnx";
const DEFAULT_MEL_MODEL_PATH: &str = "models/mel_spectrogram.onnx";

#[derive(Parser)]
#[command(name = "beat-this", version, about = "Beat and downbeat tracking using Beat This! models")]
struct Cli {
    /// Path to an audio file or directory of audio files
    input: PathBuf,

    /// Path to the beat model ONNX file
    #[arg(long = "model", default_value = DEFAULT_MODEL_PATH)]
    model_path: PathBuf,

    /// Path to the mel spectrogram ONNX file
    #[arg(long = "mel-model", default_value = DEFAULT_MEL_MODEL_PATH)]
    mel_model_path: PathBuf,

    /// Model variant to use (standard or small)
    #[arg(long = "model-variant", value_enum, default_value_t = ModelVariant::Standard)]
    model_variant: ModelVariant,

    /// Inference runtime to use
    #[arg(long = "runtime", value_enum, default_value = "ort")]
    runtime: Runtime,

    /// Print beats as plain text (tab-separated time and count) instead of JSON
    #[arg(long = "output-beats")]
    output_beats: bool,

    /// Recurse into subdirectories (batch mode only)
    #[arg(short = 'r', long = "recursive")]
    recursive: bool,

    /// Write a click-track WAV file
    #[arg(long = "output-click")]
    output_click: Option<PathBuf>,

    /// Write a mixed audio WAV file (original + clicks)
    #[arg(long = "output-mixed")]
    output_mixed: Option<PathBuf>,

    /// Print estimated BPM to stdout
    #[arg(long = "bpm")]
    show_bpm: bool,

    /// Print timing for each processing stage
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,

    /// Enable ORT profiling and write trace JSON to this prefix
    #[arg(long = "profile")]
    profile: Option<String>,

    /// Number of intra-op threads for ORT (0 = auto)
    #[arg(long = "threads", default_value_t = 0)]
    threads: usize,
}

#[derive(Clone, clap::ValueEnum)]
enum ModelVariant {
    Standard,
    Small,
}

#[derive(Clone, clap::ValueEnum)]
enum Runtime {
    Ort,
    #[cfg(feature = "rten")]
    Rten,
}

/// Resolve beat model path: explicit --model wins, otherwise --model-variant selects the default.
fn resolve_beat_model_path(cli: &Cli) -> PathBuf {
    let model_was_explicit = cli.model_path != PathBuf::from(DEFAULT_MODEL_PATH);
    if model_was_explicit {
        cli.model_path.clone()
    } else {
        match cli.model_variant {
            ModelVariant::Standard => PathBuf::from("models/beat_this.onnx"),
            ModelVariant::Small => PathBuf::from("models/beat_this_small.onnx"),
        }
    }
}

fn print_beats_stdout(result: &BeatResult) {
    let counts = output::beat_counts(result);
    for (&time, &count) in result.beats.iter().zip(counts.iter()) {
        println!("{:.3}\t{}", time, count);
    }
}

const AUDIO_EXTENSIONS: &[&str] = &["wav", "mp3", "flac", "ogg"];

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
        } else if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if AUDIO_EXTENSIONS.contains(&ext.to_lowercase().as_str()) {
                    out.push(path);
                }
            }
        }
    }
    Ok(())
}

/// Run batch processing over a directory of audio files.
fn run_batch<S: InferenceSession>(
    bt: &mut beat_this::BeatThis<S>,
    dir: &Path,
    cli: &Cli,
    model_loading_secs: f32,
) -> anyhow::Result<()> {
    let files = find_audio_files(dir, cli.recursive)?;
    ensure!(!files.is_empty(), "No audio files found in {}", dir.display());

    eprintln!("Processing {}... ({} files)", dir.display(), files.len());

    let mut file_outputs = Vec::new();
    let mut total_duration = 0.0f64;
    let mut total_processing = 0.0f64;
    let mut failed = 0usize;

    for (i, path) in files.iter().enumerate() {
        let filename = path
            .strip_prefix(dir)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        let t = Instant::now();
        let result = match process_single_file(bt, path, cli) {
            Ok(r) => r,
            Err(e) => {
                failed += 1;
                eprintln!("  [{}/{}] {} — ERROR: {}", i + 1, files.len(), filename, e);
                continue;
            }
        };
        let elapsed = t.elapsed().as_secs_f64();

        let json_out = output::build_json_output(&result.beat_result);

        eprintln!(
            "  [{}/{}] {} — {} beats, {:.1} BPM ({:.2}s)",
            i + 1,
            files.len(),
            filename,
            result.beat_result.beats.len(),
            json_out.bpm.unwrap_or(0.0),
            elapsed
        );

        if cli.output_beats {
            let beats_path = path.with_extension("beats");
            output::write_beats_file(&beats_path, &result.beat_result)?;
        }

        file_outputs.push(output::BatchFileOutput {
            file: filename,
            json: json_out,
            duration_secs: result.duration_secs,
            processing_time_secs: elapsed as f32,
        });

        total_duration += result.duration_secs as f64;
        total_processing += elapsed;
    }

    let realtime_factor = if total_processing > 0.0 {
        total_duration / total_processing
    } else {
        0.0
    };

    let batch = output::BatchOutput {
        files: file_outputs,
        summary: output::BatchSummary {
            total_files: files.len(),
            failed_files: failed,
            total_duration_secs: total_duration as f32,
            total_processing_time_secs: total_processing as f32,
            model_loading_time_secs: model_loading_secs,
            realtime_factor: realtime_factor as f32,
        },
    };

    let out_path = dir.join("beat-this.json");
    output::write_batch_json(&out_path, &batch)?;
    eprintln!(
        "Wrote {} ({} files, {:.1}s total)",
        out_path.display(),
        files.len(),
        total_processing
    );

    Ok(())
}

/// Result of processing a single file (beat result + audio duration).
struct FileResult {
    beat_result: BeatResult,
    duration_secs: f32,
}

/// Process a single audio file through the pipeline, returning beats and duration.
fn process_single_file<S: InferenceSession>(
    bt: &mut beat_this::BeatThis<S>,
    path: &Path,
    cli: &Cli,
) -> anyhow::Result<FileResult> {
    let t = Instant::now();
    let audio = beat_this::load_audio(path, 22050)?;
    let duration_secs = audio.samples.len() as f32 / audio.sample_rate as f32;
    if cli.verbose {
        eprintln!(
            "[timing] Audio loading: {:.3}s ({} samples, {:.1}s duration)",
            t.elapsed().as_secs_f64(),
            audio.samples.len(),
            duration_secs
        );
    }

    let t = Instant::now();
    let mel = bt.mel.process(&audio.samples)?;
    if cli.verbose {
        eprintln!(
            "[timing] Mel spectrogram: {:.3}s ({} frames)",
            t.elapsed().as_secs_f64(),
            mel.shape[1]
        );
    }

    let t = Instant::now();
    let (beat_logits, downbeat_logits) = bt.inference.process(&mel)?;
    if cli.verbose {
        eprintln!(
            "[timing] Beat inference: {:.3}s",
            t.elapsed().as_secs_f64()
        );
    }

    let t = Instant::now();
    let beat_result = bt.post.process(&beat_logits, &downbeat_logits)?;
    if cli.verbose {
        eprintln!(
            "[timing] Post-processing: {:.3}s",
            t.elapsed().as_secs_f64()
        );
    }

    Ok(FileResult {
        beat_result,
        duration_secs,
    })
}

/// Run the full single-file pipeline (audio → mel → inference → postprocessing → output).
///
/// Generic over the inference session — works with any backend.
fn run_pipeline<S: InferenceSession>(bt: &mut beat_this::BeatThis<S>, cli: &Cli) -> anyhow::Result<()> {
    eprintln!("Processing {}...", cli.input.display());

    let file_result = process_single_file(bt, &cli.input, cli)?;
    let result = &file_result.beat_result;

    eprintln!(
        "Found {} beats ({} downbeats)",
        result.beats.len(),
        result.downbeats.len()
    );

    // stdout output: --output-beats prints plain text, otherwise JSON (default)
    if cli.output_beats {
        print_beats_stdout(result);
    } else {
        output::print_json_stdout(result)?;
    }

    if let Some(ref path) = cli.output_click {
        output::write_click_track(path, result)?;
        eprintln!("Wrote click track to {}", path.display());
    }

    if let Some(ref path) = cli.output_mixed {
        let audio = beat_this::load_audio(&cli.input, 44100)?;
        output::write_mixed_audio(path, result, &audio.samples, audio.sample_rate)?;
        eprintln!("Wrote mixed audio to {}", path.display());
    }

    if cli.show_bpm {
        match output::calculate_bpm(result) {
            Some(bpm) => println!("{:.1} BPM", bpm),
            None => eprintln!("Could not calculate BPM (too few beats)"),
        }
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let is_batch = cli.input.is_dir();

    // Validate input exists
    ensure!(
        cli.input.exists(),
        "Input not found: {}",
        cli.input.display()
    );

    // Resolve model paths and validate
    let mel_path = cli.mel_model_path.clone();
    let beat_path = resolve_beat_model_path(&cli);

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
        Runtime::Ort => {
            let runtime = beat_this::runtime::ort::OrtRuntime {
                intra_threads: cli.threads,
                ..Default::default()
            };
            if cli.verbose {
                let coreml = if runtime.is_coreml_available() { "yes" } else { "no" };
                eprintln!("[info] Runtime: ort");
                eprintln!("[info] CoreML available: {}", coreml);
                eprintln!("[info] Intra-op threads: {}", cli.threads);
            }
            // Use a separate runtime for the beat model when profiling
            let beat_runtime = if let Some(ref prefix) = cli.profile {
                beat_this::runtime::ort::OrtRuntime {
                    intra_threads: cli.threads,
                    profiling_path: Some(std::path::PathBuf::from(prefix)),
                    ..Default::default()
                }
            } else {
                beat_this::runtime::ort::OrtRuntime {
                    intra_threads: cli.threads,
                    ..Default::default()
                }
            };
            let mel_session = runtime.load_model(&mel_path)?;
            let beat_session = beat_runtime.load_model(&beat_path)?;
            let mut bt = beat_this::BeatThis {
                mel: beat_this::MelProcessor::new(mel_session),
                inference: beat_this::BeatInference::new(beat_session),
                post: beat_this::PostProcessor::default(),
            };
            let model_loading_secs = t.elapsed().as_secs_f64() as f32;
            if cli.verbose {
                eprintln!("[timing] Model loading: {:.3}s", model_loading_secs);
            }

            if is_batch {
                run_batch(&mut bt, &cli.input, &cli, model_loading_secs)?;
            } else {
                run_pipeline(&mut bt, &cli)?;
            }

            // End ORT profiling
            if cli.profile.is_some() {
                if let Ok(path) = bt.inference.session_mut().end_profiling() {
                    eprintln!("[profile] Beat model trace written to: {}", path);
                }
            }
        }

        #[cfg(feature = "rten")]
        Runtime::Rten => {
            if cli.verbose {
                eprintln!("[info] Runtime: rten (pure Rust)");
            }
            if cli.profile.is_some() {
                eprintln!("[warn] Profiling is only supported with the ort runtime, ignoring --profile");
            }
            let runtime = beat_this::runtime::rten::RtenRuntime;
            let mut bt = beat_this::BeatThis::new(&runtime, &mel_path, &beat_path)?;
            let model_loading_secs = t.elapsed().as_secs_f64() as f32;
            if cli.verbose {
                eprintln!("[timing] Model loading: {:.3}s", model_loading_secs);
            }

            if is_batch {
                run_batch(&mut bt, &cli.input, &cli, model_loading_secs)?;
            } else {
                run_pipeline(&mut bt, &cli)?;
            }
        }
    }

    if cli.verbose {
        eprintln!("[timing] Total: {:.3}s", total_start.elapsed().as_secs_f64());
    }

    Ok(())
}
