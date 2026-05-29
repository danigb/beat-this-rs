#![cfg(feature = "ort")]
//! Requires the `ort` feature (and libonnxruntime at runtime).

use std::panic::AssertUnwindSafe;
use std::path::Path;

use beat_this::{Model, OrtRuntime, Runtime, Tensor};

/// Check if the ORT dynamic library is available at runtime.
/// ort with `load-dynamic` panics if the dylib isn't found, so we use catch_unwind.
fn ort_is_available() -> bool {
    std::panic::catch_unwind(AssertUnwindSafe(|| {
        let rt = OrtRuntime::default();
        let _ = rt.is_coreml_available();
    }))
    .is_ok()
}

/// ONNX models for testing (both committed to the repo).
const MEL_MODEL_PATH: &str = "models/mel_spectrogram.onnx";
const BEAT_MODEL_PATH: &str = "models/beat_this_small.onnx";

#[test]
fn test_load_mel_model() {
    if !ort_is_available() {
        eprintln!("Skipping test: ORT runtime not available");
        return;
    }
    let model_path = Path::new(MEL_MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {MEL_MODEL_PATH}");
        return;
    }

    let runtime = OrtRuntime::default();
    let _model = runtime
        .load_model(model_path)
        .expect("Failed to load mel model");
}

#[test]
fn test_mel_inference() {
    if !ort_is_available() {
        eprintln!("Skipping test: ORT runtime not available");
        return;
    }
    let model_path = Path::new(MEL_MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {MEL_MODEL_PATH}");
        return;
    }

    let runtime = OrtRuntime::default();
    let mut model = runtime
        .load_model(model_path)
        .expect("Failed to load mel model");

    // Input: 1 second of silence at 22050 Hz → shape [1, 22050]
    let num_samples = 22050;
    let input = Tensor {
        shape: vec![1, num_samples],
        data: vec![0.0; num_samples],
    };

    let outputs = model
        .run(&[("audio_pcm", &input)])
        .expect("Mel inference failed");

    let mel = outputs
        .get("mel_spectrogram")
        .expect("Missing mel_spectrogram output");
    assert_eq!(mel.shape.len(), 3, "Expected 3D output [batch, time, mels]");
    assert_eq!(mel.shape[0], 1, "Batch size should be 1");
    assert_eq!(mel.shape[2], 128, "Should have 128 mel bins");
    eprintln!("Mel output shape: {:?}", mel.shape);
}

#[test]
fn test_load_beat_model() {
    if !ort_is_available() {
        eprintln!("Skipping test: ORT runtime not available");
        return;
    }
    let model_path = Path::new(BEAT_MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {BEAT_MODEL_PATH}");
        return;
    }

    let runtime = OrtRuntime::default();
    let _model = runtime
        .load_model(model_path)
        .expect("Failed to load beat model");
}

#[test]
fn test_beat_inference() {
    if !ort_is_available() {
        eprintln!("Skipping test: ORT runtime not available");
        return;
    }
    let model_path = Path::new(BEAT_MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {BEAT_MODEL_PATH}");
        return;
    }

    let runtime = OrtRuntime::default();
    let mut model = runtime
        .load_model(model_path)
        .expect("Failed to load beat model");

    // Input: fake mel spectrogram — shape [1, 100, 128] (100 time frames)
    let time_frames = 100;
    let n_mels = 128;
    let input = Tensor {
        shape: vec![1, time_frames, n_mels],
        data: vec![0.0; time_frames * n_mels],
    };

    let outputs = model
        .run(&[("spectrogram", &input)])
        .expect("Beat inference failed");

    // The model should produce beat and downbeat logit outputs
    assert!(
        outputs.contains_key("beat") || outputs.contains_key("beat_logits"),
        "Missing beat output. Available keys: {:?}",
        outputs.keys().collect::<Vec<_>>()
    );
    eprintln!(
        "Beat model output keys: {:?}",
        outputs.keys().collect::<Vec<_>>()
    );
}
