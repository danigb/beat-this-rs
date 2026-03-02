#![cfg(feature = "rten")]

use std::path::Path;

use beat_this::runtime::ort::OrtRuntime;
use beat_this::runtime::rten::RtenRuntime;
use beat_this::{InferenceRuntime, InferenceSession, Tensor};

const MEL_MODEL_PATH: &str = "references/remixatron_rust/MelSpectrogram_Ultimate.onnx";
const BEAT_MODEL_PATH: &str = "references/remixatron_rust/BeatThis_small0.onnx";

/// Run mel inference through both backends and compare output shapes and values.
#[test]
fn test_cross_runtime_mel() {
    let model_path = Path::new(MEL_MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {MEL_MODEL_PATH}");
        return;
    }

    let num_samples = 22050;
    let input = Tensor {
        shape: vec![1, num_samples],
        data: vec![0.0; num_samples],
    };

    // ort
    let ort_runtime = OrtRuntime::default();
    let mut ort_session = ort_runtime.load_model(model_path).unwrap();
    let ort_outputs = ort_session.run(&[("audio_pcm", &input)]).unwrap();
    let ort_mel = ort_outputs.get("mel_spectrogram").unwrap();

    // rten
    let rten_runtime = RtenRuntime;
    let mut rten_session = rten_runtime.load_model(model_path).unwrap();
    let rten_outputs = rten_session.run(&[("audio_pcm", &input)]).unwrap();
    let rten_mel = rten_outputs.get("mel_spectrogram").unwrap();

    // Shapes must match exactly
    assert_eq!(ort_mel.shape, rten_mel.shape, "Mel output shapes differ");

    // Values should be close (tolerance for different float implementations)
    let max_diff: f32 = ort_mel
        .data
        .iter()
        .zip(rten_mel.data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!("Cross-runtime mel max abs diff: {:.6}", max_diff);
    assert!(
        max_diff < 0.01,
        "Mel outputs differ too much: max_diff={:.6}",
        max_diff
    );
}

/// Run beat inference through both backends and compare outputs.
#[test]
fn test_cross_runtime_beat() {
    let model_path = Path::new(BEAT_MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {BEAT_MODEL_PATH}");
        return;
    }

    let time_frames = 100;
    let n_mels = 128;
    let input = Tensor {
        shape: vec![1, time_frames, n_mels],
        data: vec![0.0; time_frames * n_mels],
    };

    // ort
    let ort_runtime = OrtRuntime::default();
    let mut ort_session = ort_runtime.load_model(model_path).unwrap();
    let ort_outputs = ort_session.run(&[("mel_spectrogram", &input)]).unwrap();

    // rten
    let rten_runtime = RtenRuntime;
    let mut rten_session = rten_runtime.load_model(model_path).unwrap();
    let rten_outputs = rten_session.run(&[("mel_spectrogram", &input)]).unwrap();

    // Both should have the same output keys
    let mut ort_keys: Vec<_> = ort_outputs.keys().collect();
    let mut rten_keys: Vec<_> = rten_outputs.keys().collect();
    ort_keys.sort();
    rten_keys.sort();
    assert_eq!(ort_keys, rten_keys, "Output keys differ between runtimes");

    // Compare each output tensor
    for key in ort_keys {
        let ort_tensor = &ort_outputs[key];
        let rten_tensor = &rten_outputs[key];

        assert_eq!(
            ort_tensor.shape, rten_tensor.shape,
            "Shape mismatch for output '{}'",
            key
        );

        let max_diff: f32 = ort_tensor
            .data
            .iter()
            .zip(rten_tensor.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        eprintln!("Cross-runtime '{}' max abs diff: {:.6}", key, max_diff);
        assert!(
            max_diff < 0.1,
            "Output '{}' differs too much: max_diff={:.6}",
            key,
            max_diff
        );
    }
}
