use beat_this::load_audio;
use std::path::Path;

/// Reference MP3 file for integration tests.
const TEST_MP3: &str = "test_files/It Don't Mean A Thing - Kings of Swing.mp3";

/// Target sample rate for the beat-tracking pipeline.
const TARGET_SR: u32 = 22050;

fn skip_if_missing(path: &str) -> bool {
    if !Path::new(path).exists() {
        eprintln!("Skipping test: {} not found", path);
        true
    } else {
        false
    }
}

#[test]
fn test_load_mp3() {
    if skip_if_missing(TEST_MP3) {
        return;
    }
    let audio = load_audio(Path::new(TEST_MP3), TARGET_SR).unwrap();

    assert_eq!(audio.sample_rate, TARGET_SR);
    assert!(
        !audio.samples.is_empty(),
        "decoded samples should not be empty"
    );

    // All samples should be finite (no NaN or Inf)
    assert!(
        audio.samples.iter().all(|s| s.is_finite()),
        "all samples should be finite"
    );

    // Samples should not all be zero (source file has content)
    let max_abs = audio.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(max_abs > 0.001, "samples should contain audible content");
}

#[test]
fn test_sample_count_matches_duration() {
    if skip_if_missing(TEST_MP3) {
        return;
    }
    let audio = load_audio(Path::new(TEST_MP3), TARGET_SR).unwrap();

    // The MP3 is ~2:36 (156 seconds). Allow ±5% for MP3 padding/encoder delay.
    let duration_secs = audio.samples.len() as f64 / TARGET_SR as f64;
    assert!(
        duration_secs > 148.0 && duration_secs < 164.0,
        "expected ~156s, got {:.1}s",
        duration_secs
    );
}

#[test]
fn test_load_nonexistent_file() {
    let result = load_audio(Path::new("nonexistent.wav"), TARGET_SR);
    assert!(result.is_err());
}
