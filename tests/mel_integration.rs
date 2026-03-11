use std::panic::AssertUnwindSafe;
use std::path::Path;

use beat_this::{BeatThis, OrtRuntime};

const MEL_MODEL_PATH: &str = "references/remixatron_rust/MelSpectrogram_Ultimate.onnx";
const BEAT_MODEL_PATH: &str = "references/remixatron_rust/BeatThis_small0.onnx";
const TEST_AUDIO_PATH: &str = "test_files/It Don't Mean A Thing - Kings of Swing.mp3";

/// Check if the ORT dynamic library is available at runtime.
fn ort_is_available() -> bool {
    std::panic::catch_unwind(AssertUnwindSafe(|| {
        let rt = OrtRuntime::default();
        let _ = rt.is_coreml_available();
    }))
    .is_ok()
}

fn skip_if_missing() -> bool {
    !ort_is_available()
        || !Path::new(MEL_MODEL_PATH).exists()
        || !Path::new(BEAT_MODEL_PATH).exists()
}

#[test]
fn test_mel_output_shape_silence() {
    if skip_if_missing() {
        eprintln!("Skipping test: required files not found");
        return;
    }

    let runtime = OrtRuntime::default();
    let mut bt = BeatThis::new(
        &runtime,
        Path::new(MEL_MODEL_PATH),
        Path::new(BEAT_MODEL_PATH),
    )
    .expect("Failed to create BeatThis");

    // 1 second of silence at 22050 Hz
    let samples = vec![0.0f32; 22050];
    let result = bt
        .analyze_audio(&samples, 22050)
        .expect("analyze_audio failed");

    let mel = &result.mel;
    assert_eq!(mel.shape.len(), 3, "Expected 3D output");
    assert_eq!(mel.shape[0], 1, "Batch size should be 1");
    assert_eq!(mel.shape[2], 128, "Should have 128 mel bins");

    // ~50 frames for 1 second (22050 / 441 ≈ 50), allow ±2 for padding
    let frames = mel.shape[1];
    assert!(
        (48..=52).contains(&frames),
        "Expected ~50 frames for 1s, got {frames}"
    );

    assert!(
        mel.data.iter().all(|v| v.is_finite()),
        "Mel output contains NaN or Inf"
    );

    eprintln!("Mel output shape: {:?} ({frames} frames)", mel.shape);
}

#[test]
fn test_mel_duration_scaling() {
    if skip_if_missing() {
        eprintln!("Skipping test: required files not found");
        return;
    }

    let runtime = OrtRuntime::default();
    let mut bt = BeatThis::new(
        &runtime,
        Path::new(MEL_MODEL_PATH),
        Path::new(BEAT_MODEL_PATH),
    )
    .expect("Failed to create BeatThis");

    // 2 seconds
    let samples_2s = vec![0.0f32; 22050 * 2];
    let result_2s = bt.analyze_audio(&samples_2s, 22050).expect("2s failed");
    let frames_2s = result_2s.mel.shape[1];

    // 5 seconds
    let samples_5s = vec![0.0f32; 22050 * 5];
    let result_5s = bt.analyze_audio(&samples_5s, 22050).expect("5s failed");
    let frames_5s = result_5s.mel.shape[1];

    // Frame count should scale linearly: ratio ≈ 2.5
    let ratio = frames_5s as f64 / frames_2s as f64;
    assert!(
        (2.3..=2.7).contains(&ratio),
        "Frame count should scale ~2.5x, got {ratio:.2} ({frames_2s} vs {frames_5s})"
    );

    eprintln!("2s: {frames_2s} frames, 5s: {frames_5s} frames, ratio: {ratio:.2}");
}

#[test]
fn test_mel_with_real_audio() {
    if skip_if_missing() || !Path::new(TEST_AUDIO_PATH).exists() {
        eprintln!("Skipping test: required files not found");
        return;
    }

    let runtime = OrtRuntime::default();
    let mut bt = BeatThis::new(
        &runtime,
        Path::new(MEL_MODEL_PATH),
        Path::new(BEAT_MODEL_PATH),
    )
    .expect("Failed to create BeatThis");

    let result = bt
        .analyze_file(Path::new(TEST_AUDIO_PATH))
        .expect("analyze_file failed");

    let mel = &result.mel;
    assert_eq!(mel.shape[0], 1);
    assert_eq!(mel.shape[2], 128);

    // Real audio should produce non-trivial mel values
    let max_val = mel.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(max_val > 0.0, "Mel output is all zeros for real audio");

    assert!(
        mel.data.iter().all(|v| v.is_finite()),
        "Mel output contains NaN or Inf"
    );

    let frames = mel.shape[1];
    eprintln!(
        "Real audio mel: {:?} ({frames} frames, max={max_val:.2})",
        mel.shape
    );
}
