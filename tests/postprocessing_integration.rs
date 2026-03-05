use std::panic::AssertUnwindSafe;
use std::path::Path;

use beat_this::inference::BeatInference;
use beat_this::mel::{self, MelProcessor};
use beat_this::postprocessing::PostProcessor;
use beat_this::runtime::ort::OrtRuntime;
use beat_this::InferenceRuntime;

const MEL_MODEL_PATH: &str = "references/remixatron_rust/MelSpectrogram_Ultimate.onnx";
const BEAT_MODEL_PATH: &str = "references/remixatron_rust/BeatThis_small0.onnx";
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

#[test]
fn test_postprocessing_with_real_inference() {
    if !ort_is_available() {
        eprintln!("Skipping test: ORT runtime not available");
        return;
    }

    let mel_path = Path::new(MEL_MODEL_PATH);
    let beat_path = Path::new(BEAT_MODEL_PATH);
    let audio_path = Path::new(TEST_AUDIO_PATH);

    if !mel_path.exists() || !beat_path.exists() || !audio_path.exists() {
        eprintln!("Skipping test: required files not found");
        return;
    }

    let runtime = OrtRuntime::default();

    // Full pipeline: audio → mel → inference → post-processing.
    let audio = beat_this::load_audio(audio_path, 22050).expect("Failed to load audio");
    let duration = audio.samples.len() as f32 / 22050.0;

    let mel_session = runtime
        .load_model(mel_path)
        .expect("Failed to load mel model");
    let mut mel_proc = MelProcessor::new(mel_session);
    let mel = mel_proc.process(&audio.samples).expect("Mel failed");
    let mel_frames = mel::num_frames(&mel);

    let beat_session = runtime
        .load_model(beat_path)
        .expect("Failed to load beat model");
    let mut beat_proc = BeatInference::new(beat_session);
    let (beat_logits, downbeat_logits) = beat_proc.process(&mel).expect("Inference failed");

    assert_eq!(beat_logits.len(), mel_frames);

    // Post-process.
    let pp = PostProcessor::default();
    let (beats, downbeats) = pp
        .process(&beat_logits, &downbeat_logits)
        .expect("Post-processing failed");

    // Beats and downbeats should be non-empty for real music.
    assert!(!beats.is_empty(), "No beats detected in real music");
    assert!(!downbeats.is_empty(), "No downbeats detected in real music");

    // All times should be non-negative and within audio duration.
    for &t in &beats {
        assert!(t >= 0.0, "Negative beat time: {t}");
        assert!(
            t <= duration + 0.1,
            "Beat time {t} exceeds duration {duration}"
        );
    }
    for &t in &downbeats {
        assert!(t >= 0.0, "Negative downbeat time: {t}");
        assert!(
            t <= duration + 0.1,
            "Downbeat time {t} exceeds duration {duration}"
        );
    }

    // Times should be sorted.
    assert!(
        beats.windows(2).all(|w| w[0] <= w[1]),
        "Beat times are not sorted"
    );
    assert!(
        downbeats.windows(2).all(|w| w[0] <= w[1]),
        "Downbeat times are not sorted"
    );

    // Every downbeat should appear in the beats vector (snapping invariant).
    for &d in &downbeats {
        assert!(beats.contains(&d), "Downbeat time {d} not found in beats");
    }

    // Beat intervals should be musically plausible (0.2s–2.0s for most music).
    if beats.len() >= 2 {
        let intervals: Vec<f32> = beats.windows(2).map(|w| w[1] - w[0]).collect();
        let median = {
            let mut sorted = intervals.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        assert!(
            median > 0.2 && median < 2.0,
            "Median beat interval {median:.3}s outside plausible range [0.2, 2.0]"
        );
    }

    eprintln!(
        "Post-processing: {:.1}s audio → {} beats, {} downbeats",
        duration,
        beats.len(),
        downbeats.len(),
    );
    if beats.len() >= 2 {
        let intervals: Vec<f32> = beats.windows(2).map(|w| w[1] - w[0]).collect();
        let median = {
            let mut sorted = intervals.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let bpm = 60.0 / median;
        eprintln!("  Median beat interval: {median:.3}s ({bpm:.1} BPM)");
        eprintln!("  First 5 beats: {:?}", &beats[..beats.len().min(5)]);
        eprintln!(
            "  First 5 downbeats: {:?}",
            &downbeats[..downbeats.len().min(5)]
        );
    }
}
