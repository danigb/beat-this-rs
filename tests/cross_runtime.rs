use std::panic::AssertUnwindSafe;
use std::path::Path;

use beat_this::{BeatThis, Model, OrtRuntime, RtenRuntime, Runtime, Tensor};

mod common;
use common::f_measure;

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
const TEST_AUDIO_PATH: &str = "test_files/It Don't Mean A Thing - Kings of Swing.mp3";

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

/// rten and ort must agree on real-signal beat/downbeat timestamps — the enforced
/// cross-runtime parity invariant — not just on all-zero tensors.
///
/// Agreement is scored with the standard +/-70 ms MIR F-measure (with ort as the
/// reference) rather than exact equality: on the small structural model a few
/// logit peaks sit right at the `logit > 0` threshold where the two float backends
/// can differ by an epsilon and tip a peak in/out (the same sub-MIR effect the
/// Python parity test documents). The exact counts and max timestamp diff are
/// logged for regression visibility.
#[test]
fn test_cross_runtime_real_signal() {
    if !ort_is_available()
        || !Path::new(MEL_MODEL_PATH).exists()
        || !Path::new(BEAT_MODEL_PATH).exists()
        || !Path::new(TEST_AUDIO_PATH).exists()
    {
        eprintln!("Skipping test: required files / ORT not available");
        return;
    }

    let rten = RtenRuntime;
    let mut bt_rten = BeatThis::new(&rten, Path::new(MEL_MODEL_PATH), Path::new(BEAT_MODEL_PATH))
        .expect("Failed to build rten pipeline");
    let r = bt_rten
        .analyze_file(Path::new(TEST_AUDIO_PATH))
        .expect("rten analyze_file failed");

    let ort = OrtRuntime::default();
    let mut bt_ort = BeatThis::new(&ort, Path::new(MEL_MODEL_PATH), Path::new(BEAT_MODEL_PATH))
        .expect("Failed to build ort pipeline");
    let o = bt_ort
        .analyze_file(Path::new(TEST_AUDIO_PATH))
        .expect("ort analyze_file failed");

    let beats = f_measure(&o.beats, &r.beats, 0.070);
    let downbeats = f_measure(&o.downbeats, &r.downbeats, 0.070);
    eprintln!(
        "cross-runtime real signal beats: F={:.4} rten {} vs ort {} matched {} max_diff={:.1}ms",
        beats.f_measure,
        r.beats.len(),
        o.beats.len(),
        beats.matched,
        beats.max_matched_diff * 1000.0,
    );
    eprintln!(
        "cross-runtime real signal downbeats: F={:.4} rten {} vs ort {} matched {} max_diff={:.1}ms",
        downbeats.f_measure, r.downbeats.len(), o.downbeats.len(), downbeats.matched,
        downbeats.max_matched_diff * 1000.0,
    );

    assert!(
        beats.f_measure >= 0.99,
        "rten vs ort beat F-measure {:.4} < 0.99",
        beats.f_measure
    );
    assert!(
        downbeats.f_measure >= 0.99,
        "rten vs ort downbeat F-measure {:.4} < 0.99",
        downbeats.f_measure
    );
}
