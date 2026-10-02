#![cfg(feature = "decode")]

//! The streaming API used as documented: decoded audio pushed in chunks, then `finish`.
//!
//! These tests guard the documented usage through the public API only; bit identity against 1.0.0
//! and against the whole-buffer path is checked in `tests/identity_v1_0_0.rs`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};
use beat_this::{BeatStream, BeatThis, Model, RtenRuntime, Runtime, Tensor};

const MEL_MODEL_PATH: &str = "models/mel_spectrogram.onnx";
const BEAT_MODEL_PATH: &str = "models/beat_this_small.onnx";
const TEST_AUDIO_PATH: &str = "test_files/It Don't Mean A Thing - Kings of Swing.mp3";

fn skip_if_missing() -> bool {
    !Path::new(MEL_MODEL_PATH).exists()
        || !Path::new(BEAT_MODEL_PATH).exists()
        || !Path::new(TEST_AUDIO_PATH).exists()
}

/// A decode loop handing over 8192-frame mono chunks, as the crate docs show.
#[test]
fn stream_chunks_match_analyze_file() {
    if skip_if_missing() {
        eprintln!("Skipping test: required files not found");
        return;
    }
    let mut bt = BeatThis::new(
        &RtenRuntime,
        Path::new(MEL_MODEL_PATH),
        Path::new(BEAT_MODEL_PATH),
    )
    .expect("Failed to create BeatThis");

    let expected = bt.analyze_file(Path::new(TEST_AUDIO_PATH)).unwrap();

    // The file's native rate (44.1 kHz), so `stream` does the resampling.
    let audio = beat_this::load_audio(Path::new(TEST_AUDIO_PATH), 44100).unwrap();
    let mut stream = bt.stream(audio.sample_rate).unwrap();
    for chunk in audio.samples.chunks(8192) {
        stream.push(chunk).unwrap();
    }
    let timed = stream.finish_timed().unwrap();
    let analysis = timed.analysis;

    // 44.1 kHz is an exact resampling ratio, so streaming the native samples gives the same bits
    // as `analyze_file` (which resamples the whole file in `load_audio`).
    assert_eq!(analysis.mel.shape, expected.mel.shape);
    assert!(!analysis.beats.is_empty() && !analysis.downbeats.is_empty());
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&analysis.mel.data), bits(&expected.mel.data), "mel");
    assert_eq!(
        bits(&analysis.beat_logits),
        bits(&expected.beat_logits),
        "beat logits"
    );
    assert_eq!(
        bits(&analysis.downbeat_logits),
        bits(&expected.downbeat_logits),
        "downbeat logits"
    );
    assert_eq!(bits(&analysis.beats), bits(&expected.beats), "beats");
    assert_eq!(
        bits(&analysis.downbeats),
        bits(&expected.downbeats),
        "downbeats"
    );
    assert!(timed.timing.mel > std::time::Duration::ZERO);
}

#[test]
fn stream_rejects_zero_rate_and_empty_input() {
    if skip_if_missing() {
        eprintln!("Skipping test: required files not found");
        return;
    }
    let mut bt = BeatThis::new(
        &RtenRuntime,
        Path::new(MEL_MODEL_PATH),
        Path::new(BEAT_MODEL_PATH),
    )
    .unwrap();
    assert!(bt.stream(0).is_err());
    assert!(bt.stream(48_000).unwrap().finish().is_err());
    assert!(bt.stream(22_050).unwrap().finish().is_err());
}

/// A model whose every run fails, to reach the stream's error paths without real models.
struct FailingModel;

impl Model for FailingModel {
    fn run(&mut self, _inputs: &[(&str, &Tensor)]) -> Result<HashMap<String, Tensor>> {
        bail!("model failure")
    }
}

/// An error in `push` is returned, and the stream refuses to continue afterwards rather than
/// produce an analysis with a hole in it.
#[test]
fn stream_is_unusable_after_an_error() {
    let mut bt = BeatThis::from_models(FailingModel, FailingModel);
    let mut stream = bt.stream(22_050).unwrap();
    // Short pushes only buffer; nothing has run yet.
    stream.push(&[0.0; 1000]).unwrap();
    // Two minutes complete the first mel window, whose graph run fails.
    let err = stream.push(&vec![0.0; 120 * 22_050]).unwrap_err();
    assert!(err.to_string().contains("model failure"), "{err}");
    assert!(stream.push(&[0.0; 10]).is_err());
    assert!(stream.finish().is_err());

    // A failure at the end of the stream is returned by `finish`.
    let mut stream = bt.stream(48_000).unwrap();
    stream.push(&[0.0; 1000]).unwrap();
    let err = stream.finish().unwrap_err();
    assert!(err.to_string().contains("model failure"), "{err}");
}

/// Compile-time check behind the `Send` claim in the `BeatStream` docs: with the rten model, a
/// stream (and the `BeatThis` it borrows) can be moved to another thread.
#[test]
fn beat_stream_is_send_with_rten() {
    fn assert_send<T: Send>() {}
    assert_send::<BeatStream<'static, <RtenRuntime as Runtime>::Model>>();
    assert_send::<BeatThis<<RtenRuntime as Runtime>::Model>>();
}
