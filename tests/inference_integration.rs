use std::panic::AssertUnwindSafe;
use std::path::Path;

use beat_this::inference::BeatInference;
use beat_this::mel::{self, MelProcessor};
use beat_this::runtime::ort::OrtRuntime;
use beat_this::{InferenceRuntime, Tensor};

/// Check if the ORT dynamic library is available at runtime.
/// ort with `load-dynamic` panics if the dylib isn't found, so we use catch_unwind.
fn ort_is_available() -> bool {
    std::panic::catch_unwind(AssertUnwindSafe(|| {
        let rt = OrtRuntime::default();
        let _ = rt.is_coreml_available();
    }))
    .is_ok()
}

const MEL_MODEL_PATH: &str = "references/remixatron_rust/MelSpectrogram_Ultimate.onnx";
const BEAT_MODEL_PATH: &str = "references/remixatron_rust/BeatThis_small0.onnx";
const TEST_AUDIO_PATH: &str = "test_files/It Don't Mean A Thing - Kings of Swing.mp3";

fn load_beat_session() -> Option<impl beat_this::InferenceSession> {
    if !ort_is_available() {
        eprintln!("Skipping test: ORT runtime not available");
        return None;
    }
    let model_path = Path::new(BEAT_MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {BEAT_MODEL_PATH}");
        return None;
    }
    let runtime = OrtRuntime::default();
    Some(runtime.load_model(model_path).expect("Failed to load beat model"))
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
fn test_beat_inference_short() {
    let session = match load_beat_session() {
        Some(s) => s,
        None => return,
    };

    let mut processor = BeatInference::new(session);

    // Short spectrogram: 100 frames of zeros (fits in a single chunk).
    let time_frames = 100;
    let n_mels = 128;
    let mel = Tensor {
        shape: vec![1, time_frames, n_mels],
        data: vec![0.0; time_frames * n_mels],
    };

    let (beat_logits, downbeat_logits) = processor.process(&mel).expect("Inference failed");

    assert_eq!(beat_logits.len(), time_frames, "Beat logits length mismatch");
    assert_eq!(
        downbeat_logits.len(),
        time_frames,
        "Downbeat logits length mismatch"
    );

    // All values should be finite.
    assert!(
        beat_logits.iter().all(|v| v.is_finite()),
        "Beat logits contain NaN or Inf"
    );
    assert!(
        downbeat_logits.iter().all(|v| v.is_finite()),
        "Downbeat logits contain NaN or Inf"
    );

    // Sentinels should have been overwritten.
    assert!(
        beat_logits.iter().all(|&v| v != -1000.0),
        "Beat logits still contain sentinel values"
    );

    eprintln!(
        "Short inference: {time_frames} frames → beat range [{:.2}, {:.2}], downbeat range [{:.2}, {:.2}]",
        beat_logits.iter().cloned().fold(f32::INFINITY, f32::min),
        beat_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        downbeat_logits.iter().cloned().fold(f32::INFINITY, f32::min),
        downbeat_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );
}

#[test]
fn test_beat_inference_long() {
    let session = match load_beat_session() {
        Some(s) => s,
        None => return,
    };

    let mut processor = BeatInference::new(session);

    // Long spectrogram: 3000 frames (needs multiple chunks).
    let time_frames = 3000;
    let n_mels = 128;
    let mel = Tensor {
        shape: vec![1, time_frames, n_mels],
        data: vec![0.0; time_frames * n_mels],
    };

    let (beat_logits, downbeat_logits) = processor.process(&mel).expect("Inference failed");

    assert_eq!(beat_logits.len(), time_frames);
    assert_eq!(downbeat_logits.len(), time_frames);

    assert!(
        beat_logits.iter().all(|v| v.is_finite()),
        "Beat logits contain NaN or Inf"
    );

    // No sentinels should remain — every frame must be covered.
    let sentinel_count = beat_logits.iter().filter(|&&v| v == -1000.0).count();
    assert_eq!(
        sentinel_count, 0,
        "{sentinel_count} frames still have sentinel value (-1000.0)"
    );

    eprintln!(
        "Long inference: {time_frames} frames → beat range [{:.2}, {:.2}]",
        beat_logits.iter().cloned().fold(f32::INFINITY, f32::min),
        beat_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );
}

#[test]
fn test_beat_inference_with_real_audio() {
    let mel_session = match load_mel_session() {
        Some(s) => s,
        None => return,
    };
    let beat_session = match load_beat_session() {
        Some(s) => s,
        None => return,
    };

    let audio_path = Path::new(TEST_AUDIO_PATH);
    if !audio_path.exists() {
        eprintln!("Skipping test: audio not found at {TEST_AUDIO_PATH}");
        return;
    }

    // Load audio → mel spectrogram → beat inference.
    let audio = beat_this::load_audio(audio_path, 22050).expect("Failed to load audio");
    let mut mel_proc = MelProcessor::new(mel_session);
    let mel = mel_proc.process(&audio.samples).expect("Mel processing failed");
    let mel_frames = mel::num_frames(&mel);

    let mut beat_proc = BeatInference::new(beat_session);
    let (beat_logits, downbeat_logits) = beat_proc.process(&mel).expect("Beat inference failed");

    assert_eq!(beat_logits.len(), mel_frames);
    assert_eq!(downbeat_logits.len(), mel_frames);

    assert!(
        beat_logits.iter().all(|v| v.is_finite()),
        "Beat logits contain NaN or Inf"
    );

    // Real music should trigger some positive beat logits.
    let positive_beats = beat_logits.iter().filter(|&&v| v > 0.0).count();
    assert!(
        positive_beats > 0,
        "No positive beat logits found in real music"
    );

    let positive_downbeats = downbeat_logits.iter().filter(|&&v| v > 0.0).count();
    assert!(
        positive_downbeats > 0,
        "No positive downbeat logits found in real music"
    );

    // No sentinels should remain.
    assert!(
        beat_logits.iter().all(|&v| v != -1000.0),
        "Beat logits still contain sentinel values"
    );

    eprintln!(
        "Real audio: {:.1}s → {mel_frames} frames → {positive_beats} beats, {positive_downbeats} downbeats (positive logits)",
        audio.samples.len() as f64 / 22050.0,
    );
}
