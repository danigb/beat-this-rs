use std::path::Path;

use beat_this::runtime::rten::RtenRuntime;
use beat_this::{InferenceRuntime, InferenceSession, Tensor};

const MEL_MODEL_PATH: &str = "references/remixatron_rust/MelSpectrogram_Ultimate.onnx";
const BEAT_MODEL_PATH: &str = "references/remixatron_rust/BeatThis_small0.onnx";

#[test]
fn test_rten_load_mel_model() {
    let model_path = Path::new(MEL_MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {MEL_MODEL_PATH}");
        return;
    }

    let runtime = RtenRuntime;
    let _session = runtime
        .load_model(model_path)
        .expect("Failed to load mel model with rten");
}

#[test]
fn test_rten_mel_inference() {
    let model_path = Path::new(MEL_MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {MEL_MODEL_PATH}");
        return;
    }

    let runtime = RtenRuntime;
    let mut session = runtime
        .load_model(model_path)
        .expect("Failed to load mel model with rten");

    // Input: 1 second of silence at 22050 Hz → shape [1, 22050]
    let num_samples = 22050;
    let input = Tensor {
        shape: vec![1, num_samples],
        data: vec![0.0; num_samples],
    };

    let outputs = session
        .run(&[("audio_pcm", &input)])
        .expect("Mel inference failed with rten");

    let mel = outputs
        .get("mel_spectrogram")
        .expect("Missing mel_spectrogram output");
    assert_eq!(mel.shape.len(), 3, "Expected 3D output [batch, time, mels]");
    assert_eq!(mel.shape[0], 1, "Batch size should be 1");
    assert_eq!(mel.shape[2], 128, "Should have 128 mel bins");
    eprintln!("rten mel output shape: {:?}", mel.shape);
}

#[test]
fn test_rten_load_beat_model() {
    let model_path = Path::new(BEAT_MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {BEAT_MODEL_PATH}");
        return;
    }

    let runtime = RtenRuntime;
    let _session = runtime
        .load_model(model_path)
        .expect("Failed to load beat model with rten");
}

#[test]
fn test_rten_beat_inference() {
    let model_path = Path::new(BEAT_MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {BEAT_MODEL_PATH}");
        return;
    }

    let runtime = RtenRuntime;
    let mut session = runtime
        .load_model(model_path)
        .expect("Failed to load beat model with rten");

    // Input: fake mel spectrogram — shape [1, 100, 128] (100 time frames)
    let time_frames = 100;
    let n_mels = 128;
    let input = Tensor {
        shape: vec![1, time_frames, n_mels],
        data: vec![0.0; time_frames * n_mels],
    };

    let outputs = session
        .run(&[("mel_spectrogram", &input)])
        .expect("Beat inference failed with rten");

    assert!(
        outputs.contains_key("beat") || outputs.contains_key("beat_logits"),
        "Missing beat output. Available keys: {:?}",
        outputs.keys().collect::<Vec<_>>()
    );
    eprintln!(
        "rten beat model output keys: {:?}",
        outputs.keys().collect::<Vec<_>>()
    );
}

#[test]
fn test_rten_full_pipeline() {
    let mel_path = Path::new("models/mel_spectrogram.onnx");
    let beat_path = Path::new("models/beat_this.onnx");
    let audio_path = Path::new("tests/fixtures/test.mp3");

    if !mel_path.exists() || !beat_path.exists() || !audio_path.exists() {
        eprintln!("Skipping test: model or audio files not found");
        return;
    }

    let runtime = RtenRuntime;
    let mut bt = beat_this::BeatThis::new(&runtime, mel_path, beat_path)
        .expect("Failed to create BeatThis with rten");
    let result = bt
        .process_file(audio_path)
        .expect("Full pipeline failed with rten");

    assert!(!result.beats.is_empty(), "should detect beats");
    assert!(!result.downbeats.is_empty(), "should detect downbeats");
    eprintln!(
        "rten pipeline: {} beats, {} downbeats",
        result.beats.len(),
        result.downbeats.len()
    );
}
