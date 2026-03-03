use std::panic::AssertUnwindSafe;
use std::path::Path;

use beat_this::runtime::ort::OrtRuntime;
use beat_this::BeatThis;

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

fn skip_if_missing() -> bool {
    !ort_is_available()
        || !Path::new(MEL_MODEL_PATH).exists()
        || !Path::new(BEAT_MODEL_PATH).exists()
        || !Path::new(TEST_AUDIO_PATH).exists()
}

#[test]
fn test_process_file() {
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

    let result = bt
        .process_file(Path::new(TEST_AUDIO_PATH))
        .expect("process_file failed");

    // Beats and downbeats should be non-empty for real music.
    assert!(!result.beats.is_empty(), "No beats detected");
    assert!(!result.downbeats.is_empty(), "No downbeats detected");

    // Times should be sorted.
    assert!(
        result.beats.windows(2).all(|w| w[0] <= w[1]),
        "Beat times are not sorted"
    );
    assert!(
        result.downbeats.windows(2).all(|w| w[0] <= w[1]),
        "Downbeat times are not sorted"
    );

    // All times should be non-negative.
    assert!(
        result.beats.iter().all(|&t| t >= 0.0),
        "Negative beat time found"
    );

    // Every downbeat should appear in the beats vector (snapping invariant).
    for &d in &result.downbeats {
        assert!(
            result.beats.contains(&d),
            "Downbeat time {d} not found in beats"
        );
    }

    // Beat intervals should be musically plausible.
    if result.beats.len() >= 2 {
        let mut intervals: Vec<f32> = result.beats.windows(2).map(|w| w[1] - w[0]).collect();
        intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = intervals[intervals.len() / 2];
        assert!(
            median > 0.2 && median < 2.0,
            "Median beat interval {median:.3}s outside plausible range"
        );
        let bpm = 60.0 / median;
        eprintln!(
            "process_file: {} beats, {} downbeats, {bpm:.1} BPM",
            result.beats.len(),
            result.downbeats.len()
        );
    }
}

#[test]
fn test_process_audio() {
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

    // Load audio manually, then pass to process_audio.
    let audio = beat_this::load_audio(Path::new(TEST_AUDIO_PATH), 22050).expect("Failed to load audio");
    let result = bt
        .process_audio(&audio.samples, audio.sample_rate)
        .expect("process_audio failed");

    assert!(!result.beats.is_empty(), "No beats detected");
    assert!(!result.downbeats.is_empty(), "No downbeats detected");

    // Compare with process_file: results should be identical since
    // both paths go through the same pipeline with the same sample rate.
    let result2 = bt
        .process_file(Path::new(TEST_AUDIO_PATH))
        .expect("process_file failed");

    assert_eq!(
        result.beats, result2.beats,
        "process_audio and process_file produced different beats"
    );
    assert_eq!(
        result.downbeats, result2.downbeats,
        "process_audio and process_file produced different downbeats"
    );

    eprintln!(
        "process_audio: {} beats, {} downbeats (matches process_file)",
        result.beats.len(),
        result.downbeats.len()
    );
}

#[test]
fn test_process_audio_different_sample_rate() {
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

    // Load audio at 44100 Hz (different from model's 22050 Hz).
    let audio = beat_this::load_audio(Path::new(TEST_AUDIO_PATH), 44100).expect("Failed to load audio");
    assert_eq!(audio.sample_rate, 44100);

    let result = bt
        .process_audio(&audio.samples, audio.sample_rate)
        .expect("process_audio with 44100 Hz failed");

    // Pipeline should still produce valid results after internal resampling.
    assert!(!result.beats.is_empty(), "No beats detected at 44100 Hz");
    assert!(
        !result.downbeats.is_empty(),
        "No downbeats detected at 44100 Hz"
    );

    if result.beats.len() >= 2 {
        let mut intervals: Vec<f32> = result.beats.windows(2).map(|w| w[1] - w[0]).collect();
        intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = intervals[intervals.len() / 2];
        assert!(
            median > 0.2 && median < 2.0,
            "Median beat interval {median:.3}s outside plausible range at 44100 Hz"
        );
        let bpm = 60.0 / median;
        eprintln!(
            "process_audio (44100 Hz): {} beats, {} downbeats, {bpm:.1} BPM",
            result.beats.len(),
            result.downbeats.len()
        );
    }
}
