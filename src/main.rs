use std::path::PathBuf;
use std::time::Instant;

use anyhow::ensure;
use clap::Parser;

use beat_this::output;
use beat_this::postprocessing::BeatResult;
use beat_this::runtime::InferenceRuntime;

const DEFAULT_MODEL_PATH: &str = "models/beat_this.onnx";
const DEFAULT_MEL_MODEL_PATH: &str = "models/mel_spectrogram.onnx";

#[derive(Parser)]
#[command(name = "beat-this", version, about = "Beat and downbeat tracking using Beat This! models")]
struct Cli {
    /// Path to the input audio file (WAV, MP3, FLAC, OGG)
    audio_file: PathBuf,

    /// Path to the beat model ONNX file
    #[arg(long = "model", default_value = DEFAULT_MODEL_PATH)]
    model_path: PathBuf,

    /// Path to the mel spectrogram ONNX file
    #[arg(long = "mel-model", default_value = DEFAULT_MEL_MODEL_PATH)]
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

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Validate input file exists
    ensure!(
        cli.audio_file.exists(),
        "Audio file not found: {}",
        cli.audio_file.display()
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

    // Initialize runtime and load models
    eprintln!("Loading models...");
    let t = Instant::now();
    let runtime = beat_this::runtime::ort::OrtRuntime {
        intra_threads: cli.threads,
        ..Default::default()
    };
    if cli.verbose {
        let coreml = if runtime.is_coreml_available() { "yes" } else { "no" };
        eprintln!("[info] CoreML available: {}", coreml);
        eprintln!("[info] Intra-op threads: {}", cli.threads);
    }
    // Use a separate runtime for the beat model when profiling, so traces don't collide
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
    if cli.verbose {
        eprintln!("[timing] Model loading: {:.3}s", t.elapsed().as_secs_f64());
    }

    // Load audio
    eprintln!("Processing {}...", cli.audio_file.display());
    let t = Instant::now();
    let audio = beat_this::load_audio(&cli.audio_file, 22050)?;
    if cli.verbose {
        eprintln!(
            "[timing] Audio loading: {:.3}s ({} samples, {:.1}s duration)",
            t.elapsed().as_secs_f64(),
            audio.samples.len(),
            audio.samples.len() as f64 / audio.sample_rate as f64
        );
    }

    // Mel spectrogram
    let t = Instant::now();
    let mel = bt.mel.process(&audio.samples)?;
    if cli.verbose {
        eprintln!(
            "[timing] Mel spectrogram: {:.3}s ({} frames)",
            t.elapsed().as_secs_f64(),
            mel.shape[1]
        );
    }

    // Beat inference
    let t = Instant::now();
    let (beat_logits, downbeat_logits) = bt.inference.process(&mel)?;
    if cli.verbose {
        eprintln!(
            "[timing] Beat inference: {:.3}s",
            t.elapsed().as_secs_f64()
        );
    }

    // Post-processing
    let t = Instant::now();
    let result = bt.post.process(&beat_logits, &downbeat_logits)?;
    if cli.verbose {
        eprintln!(
            "[timing] Post-processing: {:.3}s",
            t.elapsed().as_secs_f64()
        );
    }

    // End profiling if enabled
    if cli.profile.is_some() {
        if let Ok(path) = bt.inference.session_mut().end_profiling() {
            eprintln!("[profile] Beat model trace written to: {}", path);
        }
    }

    eprintln!(
        "Found {} beats ({} downbeats)",
        result.beats.len(),
        result.downbeats.len()
    );
    if cli.verbose {
        eprintln!("[timing] Total: {:.3}s", total_start.elapsed().as_secs_f64());
    }

    // If no output flag was given, print beats to stdout
    let has_output_flag = cli.output_beats.is_some()
        || cli.output_click.is_some()
        || cli.output_mixed.is_some()
        || cli.show_bpm;

    if !has_output_flag {
        print_beats_stdout(&result);
    }

    // Write requested outputs
    if let Some(ref path) = cli.output_beats {
        output::write_beats_file(path, &result)?;
        eprintln!("Wrote beats to {}", path.display());
    }

    if let Some(ref path) = cli.output_click {
        output::write_click_track(path, &result)?;
        eprintln!("Wrote click track to {}", path.display());
    }

    if let Some(ref path) = cli.output_mixed {
        let audio = beat_this::load_audio(&cli.audio_file, 44100)?;
        output::write_mixed_audio(path, &result, &audio.samples, audio.sample_rate)?;
        eprintln!("Wrote mixed audio to {}", path.display());
    }

    if cli.show_bpm {
        match output::calculate_bpm(&result) {
            Some(bpm) => println!("{:.1} BPM", bpm),
            None => eprintln!("Could not calculate BPM (too few beats)"),
        }
    }

    Ok(())
}
