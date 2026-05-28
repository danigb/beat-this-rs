use std::panic::AssertUnwindSafe;
use std::path::Path;

use beat_this::{BeatThis, OrtRuntime};

/// Check if the ORT dynamic library is available at runtime.
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

fn skip_if_missing() -> bool {
    !ort_is_available()
        || !Path::new(MEL_MODEL_PATH).exists()
        || !Path::new(BEAT_MODEL_PATH).exists()
}

#[test]
fn test_beat_prediction_short() {
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

    // 2 seconds of silence — produces a short mel spectrogram (~100 frames)
    let samples = vec![0.0f32; 22050 * 2];
    let result = bt
        .analyze_audio(&samples, 22050)
        .expect("analyze_audio failed");

    // Logits should match mel frame count
    assert_eq!(result.beat_logits.len(), result.mel.shape[1]);
    assert_eq!(result.downbeat_logits.len(), result.mel.shape[1]);

    // All values should be finite
    assert!(
        result.beat_logits.iter().all(|v| v.is_finite()),
        "Beat logits contain NaN or Inf"
    );
    assert!(
        result.downbeat_logits.iter().all(|v| v.is_finite()),
        "Downbeat logits contain NaN or Inf"
    );

    // Sentinels should have been overwritten
    assert!(
        result.beat_logits.iter().all(|&v| v != -1000.0),
        "Beat logits still contain sentinel values"
    );

    eprintln!(
        "Short prediction: {} frames → beat range [{:.2}, {:.2}]",
        result.mel.shape[1],
        result
            .beat_logits
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min),
        result
            .beat_logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
    );
}

#[test]
fn test_beat_prediction_long() {
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

    // 60 seconds of silence — produces ~3000 frames (needs multiple chunks)
    let samples = vec![0.0f32; 22050 * 60];
    let result = bt
        .analyze_audio(&samples, 22050)
        .expect("analyze_audio failed");

    let frames = result.mel.shape[1];
    assert_eq!(result.beat_logits.len(), frames);
    assert_eq!(result.downbeat_logits.len(), frames);

    assert!(
        result.beat_logits.iter().all(|v| v.is_finite()),
        "Beat logits contain NaN or Inf"
    );

    // No sentinels should remain
    let sentinel_count = result.beat_logits.iter().filter(|&&v| v == -1000.0).count();
    assert_eq!(
        sentinel_count, 0,
        "{sentinel_count} frames still have sentinel value (-1000.0)"
    );

    eprintln!(
        "Long prediction: {frames} frames → beat range [{:.2}, {:.2}]",
        result
            .beat_logits
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min),
        result
            .beat_logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
    );
}

#[test]
fn test_beat_prediction_with_real_audio() {
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

    let mel_frames = result.mel.shape[1];
    assert_eq!(result.beat_logits.len(), mel_frames);
    assert_eq!(result.downbeat_logits.len(), mel_frames);

    assert!(
        result.beat_logits.iter().all(|v| v.is_finite()),
        "Beat logits contain NaN or Inf"
    );

    // Real music should trigger some positive beat logits
    let positive_beats = result.beat_logits.iter().filter(|&&v| v > 0.0).count();
    assert!(
        positive_beats > 0,
        "No positive beat logits found in real music"
    );

    let positive_downbeats = result.downbeat_logits.iter().filter(|&&v| v > 0.0).count();
    assert!(
        positive_downbeats > 0,
        "No positive downbeat logits found in real music"
    );

    // No sentinels should remain
    assert!(
        result.beat_logits.iter().all(|&v| v != -1000.0),
        "Beat logits still contain sentinel values"
    );

    eprintln!(
        "Real audio: {mel_frames} frames → {positive_beats} beats, {positive_downbeats} downbeats (positive logits)",
    );
}
