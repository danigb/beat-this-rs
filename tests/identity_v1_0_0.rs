//! Identity tests against a live, verbatim copy of the 1.0.0 pipeline.
//!
//! The reference lives in `tests/common/v1_0_0/`: byte-identical copies of the 1.0.0 `audio`,
//! `mel`, `inference`, `postprocessing` and `runtime/rten` sources, plus `pipeline.rs`, a
//! transcription of the 1.0.0 `BeatThis` (renamed types only, see its header). They are mounted
//! here so their own `use` paths resolve, and they are never edited to make a test pass.
//!
//! Every comparison is by `to_bits()`. Each identity test has a committed degradation
//! (`comparator_catches_one_ulp`) proving the comparator can fail.
//!
//! Long checks are `#[ignore]`:
//! `cargo test --release --test identity_v1_0_0 -- --ignored --nocapture`
// The verbatim 1.0.0 copies carry a `needless_range_loop` pattern that newer clippy flags in test
// builds; they must not be edited, so the lint is allowed for this crate.
#![allow(dead_code, clippy::needless_range_loop)]

mod common;

// Makes `crate::runtime::{Model, Runtime, Tensor}` (mel, inference) and `super::{..}` (rten)
// resolve for the verbatim copies. `rten.rs` is mounted at the root, where its `super::` is this
// file's imports, and re-exported as `runtime::rten` (a `#[path]` inside an inline `mod runtime`
// would be resolved against a `tests/runtime/` directory that does not exist).
#[allow(unused_imports)]
use beat_this::{Model, Runtime, Tensor};
#[path = "common/v1_0_0/rten.rs"]
pub mod rten_v1;
mod runtime {
    pub use super::rten_v1 as rten;
    pub use beat_this::{Model, Runtime, Tensor};
}
#[path = "common/v1_0_0/audio.rs"]
mod audio;
#[path = "common/v1_0_0/inference.rs"]
mod inference;
#[path = "common/v1_0_0/mel.rs"]
mod mel;
#[path = "common/v1_0_0/postprocessing.rs"]
mod postprocessing;
#[path = "common/v1_0_0/pipeline.rs"]
mod v1;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

use beat_this::__probe::Synth;
use beat_this::BeatAnalysis;
use common::bits::{assert_bits_eq, bit_diff};

const MEL_MODEL_PATH: &str = "models/mel_spectrogram.onnx";
const BEAT_MODEL_PATH: &str = "models/beat_this_small.onnx";
const TEST_AUDIO_PATH: &str = "test_files/It Don't Mean A Thing - Kings of Swing.mp3";

type NewBt = beat_this::BeatThis<<beat_this::RtenRuntime as Runtime>::Model>;
type OldBt = v1::V1BeatThis<<runtime::rten::RtenRuntime as Runtime>::Model>;

fn models_present() -> bool {
    Path::new(MEL_MODEL_PATH).exists() && Path::new(BEAT_MODEL_PATH).exists()
}

fn new_bt() -> NewBt {
    beat_this::BeatThis::new(
        &beat_this::RtenRuntime,
        Path::new(MEL_MODEL_PATH),
        Path::new(BEAT_MODEL_PATH),
    )
    .expect("failed to load models (current pipeline)")
}

fn v1_bt() -> OldBt {
    v1::V1BeatThis::new(
        &runtime::rten::RtenRuntime,
        Path::new(MEL_MODEL_PATH),
        Path::new(BEAT_MODEL_PATH),
    )
    .expect("failed to load models (1.0.0 reference)")
}

/// Skip with a message when the committed models are missing (same pattern as the other tests).
macro_rules! require_models {
    () => {
        if !models_present() {
            eprintln!("Skipping test: required models not found");
            return;
        }
    };
}

fn assert_same(what: &str, new: &BeatAnalysis, old: &v1::V1Analysis) {
    assert_eq!(new.mel.shape, old.mel.shape, "{what}: mel shape");
    assert_bits_eq(&format!("{what}: mel"), &new.mel.data, &old.mel.data);
    assert_bits_eq(
        &format!("{what}: beat_logits"),
        &new.beat_logits,
        &old.beat_logits,
    );
    assert_bits_eq(
        &format!("{what}: downbeat_logits"),
        &new.downbeat_logits,
        &old.downbeat_logits,
    );
    assert_bits_eq(&format!("{what}: beats"), &new.beats, &old.beats);
    assert_bits_eq(
        &format!("{what}: downbeats"),
        &new.downbeats,
        &old.downbeats,
    );
}

/// 40 s is 2000 mel frames, longer than one 1500-frame beat chunk.
fn forty_seconds(rate: u32) -> Vec<f32> {
    Synth::take(rate, 40 * rate as usize)
}

// --- the reference itself -------------------------------------------------------------------

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// If this fails, the 1.0.0 reference was edited. Restore it with `git show 089b509:...` (see the
/// header of `common/v1_0_0/pipeline.rs` for the transcribed file); never update these constants.
#[test]
fn reference_is_verbatim_v1_0_0() {
    let files: [(&str, &[u8], u64); 6] = [
        (
            "audio.rs",
            include_bytes!("common/v1_0_0/audio.rs"),
            0x658a_181f_f406_881f,
        ),
        (
            "mel.rs",
            include_bytes!("common/v1_0_0/mel.rs"),
            0x0934_4b66_cf55_2263,
        ),
        (
            "inference.rs",
            include_bytes!("common/v1_0_0/inference.rs"),
            0x16ce_bdfe_aae2_1024,
        ),
        (
            "postprocessing.rs",
            include_bytes!("common/v1_0_0/postprocessing.rs"),
            0x31db_633f_847b_878e,
        ),
        (
            "rten.rs",
            include_bytes!("common/v1_0_0/rten.rs"),
            0xbc6d_460c_550c_de73,
        ),
        (
            "pipeline.rs",
            include_bytes!("common/v1_0_0/pipeline.rs"),
            0xc0b9_076f_3837_6d02,
        ),
    ];
    for (name, bytes, expected) in files {
        assert_eq!(
            fnv1a64(bytes),
            expected,
            "tests/common/v1_0_0/{name} was edited; restore it from git"
        );
    }
}

// --- the comparator can fail ----------------------------------------------------------------

#[test]
fn comparator_catches_one_ulp() {
    require_models!();
    let mut new = new_bt();
    let mut v1 = v1_bt();
    let x = forty_seconds(22050);
    let current = new.analyze_audio(&x, 22050).unwrap();
    let old = v1.analyze_audio(&x, 22050).unwrap();
    assert_same("unperturbed", &current, &old);

    // One ULP on one mel value.
    let mut bad = old.clone();
    let k = bad
        .mel
        .data
        .iter()
        .position(|&v| v != 0.0)
        .expect("a non-zero mel value");
    bad.mel.data[k] = f32::from_bits(bad.mel.data[k].to_bits() + 1);
    let d = bit_diff(&current.mel.data, &bad.mel.data);
    assert_eq!((d.differing, d.first, d.max_ulp), (1, Some(k), 1));
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        assert_same("perturbed", &current, &bad)
    }));
    assert!(outcome.is_err(), "assert_same accepted a 1-ULP mel change");

    // One ULP on the first beat time.
    assert!(
        !old.beats.is_empty(),
        "the synthetic signal yields no beats"
    );
    let mut bad = old.clone();
    bad.beats[0] = f32::from_bits(bad.beats[0].to_bits() + 1);
    let d = bit_diff(&current.beats, &bad.beats);
    assert_eq!((d.differing, d.first, d.max_ulp), (1, Some(0), 1));
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        assert_same("perturbed", &current, &bad)
    }));
    assert!(outcome.is_err(), "assert_same accepted a 1-ULP beat change");

    // A length difference.
    let mut bad = old.clone();
    bad.beat_logits.pop();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        assert_same("perturbed", &current, &bad)
    }));
    assert!(outcome.is_err(), "assert_same accepted a length change");
}

#[test]
fn comparator_distinguishes_signed_zero() {
    let d = bit_diff(&[0.0, 1.0], &[-0.0, 1.0]);
    assert_eq!((d.differing, d.first, d.max_ulp), (1, Some(0), 1));
    // `==` would call these equal; the comparator must not.
    let (zero, neg_zero) = (std::hint::black_box(0.0f32), std::hint::black_box(-0.0f32));
    assert!(zero == neg_zero);
    let d = bit_diff(&[f32::NAN], &[f32::NAN]);
    assert_eq!(d.differing, 0);
}

// --- the current pipeline against 1.0.0 -----------------------------------------------------

fn check_analyze_audio(rate: u32) {
    require_models!();
    let x = forty_seconds(rate);
    let current = new_bt().analyze_audio(&x, rate).unwrap();
    let old = v1_bt().analyze_audio(&x, rate).unwrap();
    assert_same(&format!("analyze_audio @ {rate}"), &current, &old);
}

#[test]
fn analyze_audio_matches_v1_0_0_22050() {
    check_analyze_audio(22050);
}

#[test]
fn analyze_audio_matches_v1_0_0_44100() {
    check_analyze_audio(44100);
}

#[test]
fn analyze_audio_matches_v1_0_0_48000() {
    check_analyze_audio(48000);
}

#[test]
fn analyze_file_matches_v1_0_0() {
    require_models!();
    if !Path::new(TEST_AUDIO_PATH).exists() {
        eprintln!("Skipping test: test audio not found");
        return;
    }
    let current = new_bt().analyze_file(Path::new(TEST_AUDIO_PATH)).unwrap();
    let old = v1_bt().analyze_file(Path::new(TEST_AUDIO_PATH)).unwrap();
    assert_same("analyze_file", &current, &old);
}

#[test]
fn tiny_inputs_match_v1_0_0() {
    require_models!();
    let mut new = new_bt();
    let mut v1 = v1_bt();
    let lengths: Vec<usize> = (0..=8).chain([511, 512, 513, 1023, 1024, 1025]).collect();
    let (mut ok, mut err) = (0, 0);
    for rate in [22050u32, 48000] {
        for &n in &lengths {
            let x = Synth::take(rate, n);
            let current = new.analyze_audio(&x, rate);
            let old = v1.analyze_audio(&x, rate);
            let what = format!("tiny n={n} @ {rate}");
            assert_eq!(
                current.is_err(),
                old.is_err(),
                "{what}: error behaviour differs"
            );
            if let (Ok(c), Ok(o)) = (&current, &old) {
                assert_same(&what, c, o);
                ok += 1;
            } else {
                err += 1;
            }
        }
    }
    // Both outcomes must occur, or the comparison above is vacuous.
    assert!(ok > 0 && err > 0, "tiny inputs: {ok} ok, {err} err");
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn long_inputs_match_v1_0_0() {
    require_models!();
    let mut new = new_bt();
    let mut v1 = v1_bt();
    for rate in [48000u32, 44100] {
        for minutes in [5usize, 20] {
            let x = Synth::take(rate, minutes * 60 * rate as usize);
            let current = new.analyze_audio(&x, rate).unwrap();
            let old = v1.analyze_audio(&x, rate).unwrap();
            assert_same(&format!("{minutes} min @ {rate}"), &current, &old);
            eprintln!("{minutes} min @ {rate}: identical");
        }
    }
}
