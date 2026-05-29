#![cfg(feature = "ort")]
//! Requires the `ort` feature (and libonnxruntime at runtime).

use std::panic::AssertUnwindSafe;
use std::path::Path;

use beat_this::{Model, OrtRuntime, RtenRuntime, Runtime, Tensor};

/// Check if the ORT dynamic library is available at runtime.
/// ort with `load-dynamic` panics if the dylib isn't found, so we use catch_unwind.
fn ort_is_available() -> bool {
    std::panic::catch_unwind(AssertUnwindSafe(|| {
        let rt = OrtRuntime::default();
        let _ = rt.is_coreml_available();
    }))
    .is_ok()
}

const MEL_MODEL_PATH: &str = "models/mel_spectrogram.onnx";
const BEAT_MODEL_PATH: &str = "models/beat_this_small.onnx";

/// Run mel inference through both backends and compare output shapes and values.
#[test]
fn test_cross_runtime_mel() {
    if !ort_is_available() {
        eprintln!("Skipping test: ORT runtime not available");
        return;
    }
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
    let mut ort_model = ort_runtime.load_model(model_path).unwrap();
    let ort_outputs = ort_model.run(&[("audio_pcm", &input)]).unwrap();
    let ort_mel = ort_outputs.get("mel_spectrogram").unwrap();

    // rten
    let rten_runtime = RtenRuntime;
    let mut rten_model = rten_runtime.load_model(model_path).unwrap();
    let rten_outputs = rten_model.run(&[("audio_pcm", &input)]).unwrap();
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
    if !ort_is_available() {
        eprintln!("Skipping test: ORT runtime not available");
        return;
    }
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
    let mut ort_model = ort_runtime.load_model(model_path).unwrap();
    let ort_outputs = ort_model.run(&[("spectrogram", &input)]).unwrap();

    // rten
    let rten_runtime = RtenRuntime;
    let mut rten_model = rten_runtime.load_model(model_path).unwrap();
    let rten_outputs = rten_model.run(&[("spectrogram", &input)]).unwrap();

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
