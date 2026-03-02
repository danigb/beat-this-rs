use std::path::Path;

use beat_this::output;
use beat_this::runtime::ort::OrtRuntime;
use beat_this::BeatThis;

const MEL_MODEL_PATH: &str = "references/remixatron_rust/MelSpectrogram_Ultimate.onnx";
const BEAT_MODEL_PATH: &str = "references/remixatron_rust/BeatThis_small0.onnx";
const TEST_AUDIO_PATH: &str = "test_files/It Don't Mean A Thing - Kings of Swing.mp3";

fn skip_if_missing() -> bool {
    !Path::new(MEL_MODEL_PATH).exists()
        || !Path::new(BEAT_MODEL_PATH).exists()
        || !Path::new(TEST_AUDIO_PATH).exists()
}

#[test]
fn test_full_pipeline_to_beats_file() {
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

    let dir = tempfile::tempdir().unwrap();
    let beats_path = dir.path().join("output.beats");

    output::write_beats_file(&beats_path, &result).expect("write_beats_file failed");

    let content = std::fs::read_to_string(&beats_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();

    // Should have one line per beat.
    assert_eq!(lines.len(), result.beats.len());

    // Each line should be tab-separated with time and count.
    for line in &lines {
        let parts: Vec<&str> = line.split('\t').collect();
        assert_eq!(parts.len(), 2, "Line not tab-separated: {}", line);
        let _time: f32 = parts[0].parse().expect("Invalid time");
        let count: i32 = parts[1].parse().expect("Invalid beat count");
        assert!(count >= 1, "Beat count should be >= 1");
    }

    // First downbeat should have count 1.
    let counts = output::beat_counts(&result);
    let first_downbeat_idx = result
        .beats
        .iter()
        .position(|&t| result.downbeats.contains(&t));
    if let Some(idx) = first_downbeat_idx {
        assert_eq!(counts[idx], 1, "Downbeat should have count 1");
    }

    eprintln!(
        "Wrote {} beats to .beats file",
        lines.len()
    );
}

#[test]
fn test_full_pipeline_to_click_track() {
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

    let dir = tempfile::tempdir().unwrap();
    let wav_path = dir.path().join("clicks.wav");

    output::write_click_track(&wav_path, &result).expect("write_click_track failed");

    // Verify WAV is valid and has expected format.
    let reader = hound::WavReader::open(&wav_path).unwrap();
    let spec = reader.spec();
    assert_eq!(spec.channels, 1);
    assert_eq!(spec.sample_rate, 44100);
    assert_eq!(spec.sample_format, hound::SampleFormat::Float);

    eprintln!(
        "Generated click track: {} samples at {} Hz",
        reader.len(),
        spec.sample_rate
    );
}

#[test]
fn test_full_pipeline_bpm() {
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

    let bpm = output::calculate_bpm(&result);
    assert!(bpm.is_some(), "BPM calculation should succeed");

    let bpm = bpm.unwrap();
    assert!(
        bpm > 40.0 && bpm < 300.0,
        "BPM {:.1} outside plausible range for music",
        bpm
    );

    eprintln!("Estimated BPM: {:.1}", bpm);
}
