use std::panic::AssertUnwindSafe;
use std::path::Path;

use beat_this::mel::{self, MelProcessor};
use beat_this::runtime::ort::OrtRuntime;
use beat_this::InferenceRuntime;

const MEL_MODEL_PATH: &str = "references/remixatron_rust/MelSpectrogram_Ultimate.onnx";
const TEST_AUDIO_PATH: &str = "test_files/It Don't Mean A Thing - Kings of Swing.mp3";

/// Check if the ORT dynamic library is available at runtime.
/// ort with `load-dynamic` panics if the dylib isn't found, so we use catch_unwind.
fn ort_is_available() -> bool {
    std::panic::catch_unwind(AssertUnwindSafe(|| {
        let rt = OrtRuntime::default();
        let _ = rt.is_coreml_available();
    }))
    .is_ok()
}

fn load_mel_session() -> Option<impl beat_this::InferenceSession> {
    if !ort_is_available() {
        eprintln!("Skipping test: ORT runtime not available");
        return None;
    }
    let model_path = Path::new(MEL_MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {MEL_MODEL_PATH}");
        return None;
    }
    let runtime = OrtRuntime::default();
    Some(runtime.load_model(model_path).expect("Failed to load mel model"))
}

#[test]
fn test_mel_processor_basic() {
    let session = match load_mel_session() {
        Some(s) => s,
        None => return,
    };

    let mut processor = MelProcessor::new(session);

    // 1 second of silence at 22050 Hz
    let samples = vec![0.0f32; 22050];
    let mel = processor.process(&samples).expect("Mel processing failed");

    assert_eq!(mel.shape.len(), 3, "Expected 3D output");
    assert_eq!(mel.shape[0], 1, "Batch size should be 1");
    assert_eq!(mel.shape[2], 128, "Should have 128 mel bins");

    // ~50 frames for 1 second (22050 / 441 ≈ 50), allow ±2 for padding
    let frames = mel::num_frames(&mel);
    assert!(
        (48..=52).contains(&frames),
        "Expected ~50 frames for 1s, got {frames}"
    );

    // All values should be finite
    assert!(
        mel.data.iter().all(|v| v.is_finite()),
        "Mel output contains NaN or Inf"
    );

    eprintln!("Mel output shape: {:?} ({frames} frames)", mel.shape);
}

#[test]
fn test_mel_processor_duration_scaling() {
    let session = match load_mel_session() {
        Some(s) => s,
        None => return,
    };

    let mut processor = MelProcessor::new(session);

    // 2 seconds
    let samples_2s = vec![0.0f32; 22050 * 2];
    let mel_2s = processor.process(&samples_2s).expect("2s mel failed");
    let frames_2s = mel::num_frames(&mel_2s);

    // 5 seconds
    let samples_5s = vec![0.0f32; 22050 * 5];
    let mel_5s = processor.process(&samples_5s).expect("5s mel failed");
    let frames_5s = mel::num_frames(&mel_5s);

    // Frame count should scale linearly: ratio ≈ 2.5
    let ratio = frames_5s as f64 / frames_2s as f64;
    assert!(
        (2.3..=2.7).contains(&ratio),
        "Frame count should scale ~2.5x, got {ratio:.2} ({frames_2s} vs {frames_5s})"
    );

    eprintln!("2s: {frames_2s} frames, 5s: {frames_5s} frames, ratio: {ratio:.2}");
}

#[test]
fn test_mel_processor_with_audio() {
    let session = match load_mel_session() {
        Some(s) => s,
        None => return,
    };

    let audio_path = Path::new(TEST_AUDIO_PATH);
    if !audio_path.exists() {
        eprintln!("Skipping test: audio not found at {TEST_AUDIO_PATH}");
        return;
    }

    let audio = beat_this::load_audio(audio_path, 22050).expect("Failed to load audio");
    let mut processor = MelProcessor::new(session);
    let mel = processor
        .process(&audio.samples)
        .expect("Mel processing failed");

    assert_eq!(mel.shape[0], 1);
    assert_eq!(mel.shape[2], 128);

    let frames = mel::num_frames(&mel);
    let expected_frames = audio.samples.len() / 441;
    let tolerance = 5;
    assert!(
        frames.abs_diff(expected_frames) <= tolerance,
        "Expected ~{expected_frames} frames, got {frames}"
    );

    // Real audio should produce non-trivial mel values (not all zeros)
    let max_val = mel.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(max_val > 0.0, "Mel output is all zeros for real audio");

    assert!(
        mel.data.iter().all(|v| v.is_finite()),
        "Mel output contains NaN or Inf"
    );

    eprintln!(
        "Audio: {} samples ({:.1}s) → mel: {:?} ({frames} frames, max={max_val:.2})",
        audio.samples.len(),
        audio.samples.len() as f64 / 22050.0,
        mel.shape,
    );
}
