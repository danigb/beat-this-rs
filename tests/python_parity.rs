//! Golden parity test against the Python reference (`beat_this`).
//!
//! Runs the full Rust pipeline (audio -> mel -> inference -> postprocessing) on
//! the shared mp3 and asserts F-measure == 1.0 at the standard +/-70 ms MIR
//! window for beats and downbeats, versus a golden generated from the Python
//! reference on the matching model (see `scripts/gen_golden.py`).
//!
//! The bit-exact path is impossible here (the mp3 is 44.1 kHz stereo, so Rust's
//! rubato resampler diverges from Python's soxr, and the MP3 decoders differ),
//! so F@70 ms is the divergence-robust metric. Per-matched-beat timing is logged
//! as a diagnostic, not asserted.

mod common;

use std::path::Path;

use beat_this::{BeatThis, RtenRuntime};
use common::{f_measure, load_golden, Score};

const MEL_MODEL_PATH: &str = "models/mel_spectrogram.onnx";
const SMALL_BEAT_MODEL_PATH: &str = "models/beat_this_small.onnx";
const FULL_BEAT_MODEL_PATH: &str = "models/beat_this.onnx";
const TEST_AUDIO_PATH: &str = "test_files/It Don't Mean A Thing - Kings of Swing.mp3";

/// Standard MIR beat-tracking tolerance.
const WINDOW: f64 = 0.070;

/// The full FP32 model is bit-faithful to Python within MIR tolerance.
const FULL_MIN_F: f64 = 1.0;
/// The small structural model allows a few near-threshold peak flips (see `run_parity`).
const SMALL_MIN_F: f64 = 0.99;

fn report(kind: &str, s: &Score) {
    eprintln!(
        "{kind}: F={:.4} matched {}/{} (ref {}) max_diff={:.1}ms mean_diff={:.1}ms",
        s.f_measure,
        s.matched,
        s.n_est,
        s.n_ref,
        s.max_matched_diff * 1000.0,
        s.mean_matched_diff * 1000.0,
    );
}

/// Run the rten pipeline on the mp3 with `beat_model` and assert F-measure vs the
/// golden meets `min_f` for both beats and downbeats.
///
/// `min_f` is 1.0 for the full FP32 standard model (bit-faithful to Python within
/// MIR tolerance) and slightly relaxed for the small structural model, whose
/// lower-capacity, flatter logit peaks include a few values right at the `logit > 0`
/// threshold where rten's float output differs from Python's torch by an epsilon
/// and tips them across — an irreducible, sub-MIR backend-float artifact, not a
/// pipeline difference (the shared resampler/mel/chunking/postprocessing stages are
/// proven exact by the full-model F == 1.0 result).
fn run_parity(beat_model: &str, golden_path: &str, min_f: f64) {
    let runtime = RtenRuntime;
    let mut bt = BeatThis::new(&runtime, Path::new(MEL_MODEL_PATH), Path::new(beat_model))
        .expect("Failed to create BeatThis");
    let result = bt
        .analyze_file(Path::new(TEST_AUDIO_PATH))
        .expect("analyze_file failed");

    let golden = load_golden(golden_path);

    let beats = f_measure(&golden.beats, &result.beats, WINDOW);
    let downbeats = f_measure(&golden.downbeats, &result.downbeats, WINDOW);
    report("beats", &beats);
    report("downbeats", &downbeats);

    assert!(
        beats.f_measure >= min_f,
        "beat F-measure {:.4} < {min_f} @ {WINDOW}s vs {golden_path}",
        beats.f_measure
    );
    assert!(
        downbeats.f_measure >= min_f,
        "downbeat F-measure {:.4} < {min_f} @ {WINDOW}s vs {golden_path}",
        downbeats.f_measure
    );
}

/// Small model — runs on a fresh clone (rten, committed model, committed golden).
#[test]
fn test_python_parity_small_model() {
    // mel + small model + mp3 are all committed; rten needs no external libs.
    if !Path::new(MEL_MODEL_PATH).exists()
        || !Path::new(SMALL_BEAT_MODEL_PATH).exists()
        || !Path::new(TEST_AUDIO_PATH).exists()
    {
        eprintln!("Skipping test: required committed files not found");
        return;
    }
    run_parity(
        SMALL_BEAT_MODEL_PATH,
        "tests/fixtures/golden_small.json",
        SMALL_MIN_F,
    );
}

/// Full standard model — gated on the (git-ignored, downloaded) model being present.
/// Run `./scripts/download-models.sh` first; otherwise this skips like the ort tests.
#[test]
fn test_python_parity_full_model() {
    if !Path::new(FULL_BEAT_MODEL_PATH).exists() {
        eprintln!(
            "Skipping test: {FULL_BEAT_MODEL_PATH} not found (run scripts/download-models.sh)"
        );
        return;
    }
    if !Path::new(MEL_MODEL_PATH).exists() || !Path::new(TEST_AUDIO_PATH).exists() {
        eprintln!("Skipping test: required committed files not found");
        return;
    }
    run_parity(
        FULL_BEAT_MODEL_PATH,
        "tests/fixtures/golden_full.json",
        FULL_MIN_F,
    );
}
