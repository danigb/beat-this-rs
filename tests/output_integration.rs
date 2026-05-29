use std::path::Path;

use beat_this::{beat_counts, calculate_bpm, BeatThis, RtenRuntime};

const MEL_MODEL_PATH: &str = "models/mel_spectrogram.onnx";
const BEAT_MODEL_PATH: &str = "models/beat_this_small.onnx";
const TEST_AUDIO_PATH: &str = "test_files/It Don't Mean A Thing - Kings of Swing.mp3";

fn skip_if_missing() -> bool {
    !Path::new(MEL_MODEL_PATH).exists()
        || !Path::new(BEAT_MODEL_PATH).exists()
        || !Path::new(TEST_AUDIO_PATH).exists()
}

#[test]
fn test_full_pipeline_beat_counts() {
    if skip_if_missing() {
        eprintln!("Skipping test: required files not found");
        return;
    }

    let runtime = RtenRuntime;
    let mut bt = BeatThis::new(
        &runtime,
        Path::new(MEL_MODEL_PATH),
        Path::new(BEAT_MODEL_PATH),
    )
    .expect("Failed to create BeatThis");

    let result = bt
        .analyze_file(Path::new(TEST_AUDIO_PATH))
        .expect("analyze_file failed");

    let counts = beat_counts(&result);
    assert_eq!(counts.len(), result.beats.len());

    // All counts should be >= 1
    assert!(counts.iter().all(|&c| c >= 1), "Beat count should be >= 1");

    // First downbeat should have count 1
    let first_downbeat_idx = result
        .beats
        .iter()
        .position(|&t| result.downbeats.contains(&t));
    if let Some(idx) = first_downbeat_idx {
        assert_eq!(counts[idx], 1, "Downbeat should have count 1");
    }

    eprintln!("Beat counts: {} beats", counts.len());
}

#[test]
fn test_full_pipeline_bpm() {
    if skip_if_missing() {
        eprintln!("Skipping test: required files not found");
        return;
    }

    let runtime = RtenRuntime;
    let mut bt = BeatThis::new(
        &runtime,
        Path::new(MEL_MODEL_PATH),
        Path::new(BEAT_MODEL_PATH),
    )
    .expect("Failed to create BeatThis");

    let result = bt
        .analyze_file(Path::new(TEST_AUDIO_PATH))
        .expect("analyze_file failed");

    let bpm = calculate_bpm(&result);
    assert!(bpm.is_some(), "BPM calculation should succeed");

    let bpm = bpm.unwrap();
    assert!(
        bpm > 40.0 && bpm < 300.0,
        "BPM {:.1} outside plausible range for music",
        bpm
    );

    eprintln!("Estimated BPM: {:.1}", bpm);
}
