use std::panic::AssertUnwindSafe;
use std::path::Path;

use beat_this::{BeatThis, OrtRuntime};

const MEL_MODEL_PATH: &str = "models/mel_spectrogram.onnx";
const BEAT_MODEL_PATH: &str = "models/beat_this_small.onnx";
const TEST_AUDIO_PATH: &str = "test_files/It Don't Mean A Thing - Kings of Swing.mp3";

/// Check if the ORT dynamic library is available at runtime.
fn ort_is_available() -> bool {
    std::panic::catch_unwind(AssertUnwindSafe(|| {
        let rt = OrtRuntime::default();
        let _ = rt.is_coreml_available();
    }))
    .is_ok()
}

#[test]
fn test_full_pipeline_peak_picking() {
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
    let mut bt = BeatThis::new(&runtime, mel_path, beat_path).expect("Failed to create BeatThis");

    let audio = beat_this::load_audio(audio_path, 22050).expect("Failed to load audio");
    let duration = audio.samples.len() as f32 / 22050.0;

    let result = bt
        .analyze_audio(&audio.samples, audio.sample_rate)
        .expect("analyze_audio failed");

    let beats = &result.beats;
    let downbeats = &result.downbeats;

    // Beats and downbeats should be non-empty for real music.
    assert!(!beats.is_empty(), "No beats detected in real music");
    assert!(!downbeats.is_empty(), "No downbeats detected in real music");

    // All times should be non-negative and within audio duration.
    for &t in beats {
        assert!(t >= 0.0, "Negative beat time: {t}");
        assert!(
            t <= duration + 0.1,
            "Beat time {t} exceeds duration {duration}"
        );
    }
    for &t in downbeats {
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
    for &d in downbeats {
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

        let bpm = 60.0 / median;
        eprintln!(
            "Peak picking: {:.1}s audio → {} beats, {} downbeats",
            duration,
            beats.len(),
            downbeats.len(),
        );
        eprintln!("  Median beat interval: {median:.3}s ({bpm:.1} BPM)");
        eprintln!("  First 5 beats: {:?}", &beats[..beats.len().min(5)]);
        eprintln!(
            "  First 5 downbeats: {:?}",
            &downbeats[..downbeats.len().min(5)]
        );
    }
}
