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

use beat_this::__probe::{self, StreamResampler, Synth};
use beat_this::BeatAnalysis;
use common::bits::{assert_bits_eq, bit_diff, BitDiff};

const MEL_MODEL_PATH: &str = "models/mel_spectrogram.onnx";
const BEAT_MODEL_PATH: &str = "models/beat_this_small.onnx";

/// The beat model: `BEAT_THIS_MODEL` if set (e.g. `models/beat_this.onnx` for the long and drift
/// checks), else the committed small model.
fn beat_model_path() -> String {
    std::env::var("BEAT_THIS_MODEL").unwrap_or_else(|_| BEAT_MODEL_PATH.to_string())
}
const TEST_AUDIO_PATH: &str = "test_files/It Don't Mean A Thing - Kings of Swing.mp3";

type NewBt = beat_this::BeatThis<<beat_this::RtenRuntime as Runtime>::Model>;
type OldBt = v1::V1BeatThis<<runtime::rten::RtenRuntime as Runtime>::Model>;

fn models_present() -> bool {
    Path::new(MEL_MODEL_PATH).exists() && Path::new(&beat_model_path()).exists()
}

fn new_bt() -> NewBt {
    beat_this::BeatThis::new(
        &beat_this::RtenRuntime,
        Path::new(MEL_MODEL_PATH),
        Path::new(&beat_model_path()),
    )
    .expect("failed to load models (current pipeline)")
}

fn v1_bt() -> OldBt {
    v1::V1BeatThis::new(
        &runtime::rten::RtenRuntime,
        Path::new(MEL_MODEL_PATH),
        Path::new(&beat_model_path()),
    )
    .expect("failed to load models (1.0.0 reference)")
}

/// Skip with a message when the committed models are missing (same pattern as the other tests).
macro_rules! require_models {
    () => {
        if let Ok(path) = std::env::var("BEAT_THIS_MODEL") {
            assert!(
                Path::new(&path).exists(),
                "BEAT_THIS_MODEL is set to {path}, which does not exist"
            );
        }
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

/// Drift bounds for sources whose chunked resampling is not bit-identical to 1.0.0 (decision D1,
/// E1 answered 2026-10-02). The largest drifts recorded in `worklog/03-chunked-resampler.md` over
/// the in-repo inputs, the 5/20-minute synth at 16/32/48/96 kHz and the 23-file corpus are PCM
/// 1.57e-3 (32 kHz, 20 min), mel 3.47e-4 and logits 2.10e-4; the bounds add margin.
///
/// **Valid only for the inputs they are applied to: at most 20 minutes** (40 s in the default
/// tests, 5 and 20 min in the ignored long ones). They are not a general bound. The drift grows with
/// the length, because 1.0.0's one-shot resampler loses read-position precision on long inputs
/// (`resampler_accuracy_long`): at 48 kHz the beat-logit drift is 3.1e-4 at 40 min, 4.35e-3 at
/// 50 min and 1.63e-2 at 60 min, where beats move (`long_drift_downstream_48000`).
const PCM_DRIFT_BOUND: f32 = 2e-3;
const MEL_DRIFT_BOUND: f32 = 5e-4;
const LOGIT_DRIFT_BOUND: f32 = 5e-4;

/// True where the current pipeline must still be bit-identical to 1.0.0 at this input rate.
fn rate_is_exact(rate: u32) -> bool {
    rate == 22050 || __probe::chunking_is_exact(rate)
}

/// D1: same shapes and lengths, beats and downbeats bit-identical, mel and logits within the
/// drift bounds (valid up to 20 min, see `MEL_DRIFT_BOUND`). Prints the bit-level differences.
fn assert_drift(what: &str, new: &BeatAnalysis, old: &v1::V1Analysis) {
    assert_eq!(new.mel.shape, old.mel.shape, "{what}: mel shape");
    let mel = bit_diff(&new.mel.data, &old.mel.data);
    let beat = bit_diff(&new.beat_logits, &old.beat_logits);
    let down = bit_diff(&new.downbeat_logits, &old.downbeat_logits);
    eprintln!("{what}: drift: mel {mel:?}; beat logits {beat:?}; downbeat logits {down:?}");
    assert_eq!(beat.len_a, beat.len_b, "{what}: beat logits length");
    assert_eq!(down.len_a, down.len_b, "{what}: downbeat logits length");
    assert!(
        mel.max_abs <= MEL_DRIFT_BOUND,
        "{what}: mel drift too large: {mel:?}"
    );
    assert!(
        beat.max_abs <= LOGIT_DRIFT_BOUND && down.max_abs <= LOGIT_DRIFT_BOUND,
        "{what}: logit drift too large: {beat:?} {down:?}"
    );
    assert_bits_eq(&format!("{what}: beats"), &new.beats, &old.beats);
    assert_bits_eq(
        &format!("{what}: downbeats"),
        &new.downbeats,
        &old.downbeats,
    );
}

/// Bit identity at exact rates, the D1 drift check elsewhere.
fn assert_same_or_drift(what: &str, rate: u32, new: &BeatAnalysis, old: &v1::V1Analysis) {
    if rate_is_exact(rate) {
        assert_same(what, new, old);
    } else {
        assert_drift(what, new, old);
    }
}

/// D1 on resampled PCM: same length, within the PCM drift bound (valid up to 20 min, see
/// `MEL_DRIFT_BOUND`). Prints the difference.
fn assert_pcm_drift(what: &str, got: &[f32], reference: &[f32]) {
    let d = bit_diff(got, reference);
    if d.differing > 0 {
        eprintln!("{what}: pcm drift {d:?}");
    }
    assert_eq!(d.len_a, d.len_b, "{what}: length");
    assert!(
        d.max_abs <= PCM_DRIFT_BOUND,
        "{what}: pcm drift too large: {d:?}"
    );
}

/// Degradations for the D1 drift checks: each must reject a change larger than its bound, a
/// moved beat, and a length change, and accept the real 48 kHz drift.
#[test]
fn drift_checks_can_fail() {
    require_models!();
    let x = forty_seconds(48000);
    let current = new_bt().analyze_audio(&x, 48000).unwrap();
    let old = v1_bt().analyze_audio(&x, 48000).unwrap();
    assert_drift("48 kHz, unperturbed", &current, &old);
    let rejects = |what: &str, bad: &v1::V1Analysis| {
        let outcome = catch_unwind(AssertUnwindSafe(|| assert_drift(what, &current, bad)));
        assert!(outcome.is_err(), "assert_drift accepted: {what}");
    };
    let mut bad = old.clone();
    bad.mel.data[1000] += 2.0 * MEL_DRIFT_BOUND;
    rejects("mel change above the bound", &bad);
    let mut bad = old.clone();
    bad.beat_logits[100] += 2.0 * LOGIT_DRIFT_BOUND;
    rejects("beat logit change above the bound", &bad);
    let mut bad = old.clone();
    bad.downbeat_logits[100] += 2.0 * LOGIT_DRIFT_BOUND;
    rejects("downbeat logit change above the bound", &bad);
    assert!(!old.beats.is_empty() && !old.downbeats.is_empty());
    let mut bad = old.clone();
    bad.beats[0] = f32::from_bits(bad.beats[0].to_bits() + 1);
    rejects("a beat moved by 1 ULP", &bad);
    let mut bad = old.clone();
    bad.downbeats[0] = f32::from_bits(bad.downbeats[0].to_bits() + 1);
    rejects("a downbeat moved by 1 ULP", &bad);
    let mut bad = old.clone();
    bad.beat_logits.pop();
    rejects("a length change", &bad);

    let reference = reference_resample(&x, 48000);
    let got = __probe::resample(x.clone(), 48000).unwrap();
    assert_pcm_drift("48 kHz pcm, unperturbed", &got, &reference);
    let mut bad = got.clone();
    bad[5000] += 2.0 * PCM_DRIFT_BOUND;
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        assert_pcm_drift("pcm above the bound", &bad, &reference)
    }));
    assert!(
        outcome.is_err(),
        "assert_pcm_drift accepted a change above the bound"
    );
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        assert_pcm_drift("pcm length", &got[1..], &reference)
    }));
    assert!(
        outcome.is_err(),
        "assert_pcm_drift accepted a length change"
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
    assert_same_or_drift(&format!("analyze_audio @ {rate}"), rate, &current, &old);
}

#[test]
fn analyze_audio_matches_v1_0_0_22050() {
    check_analyze_audio(22050);
}

#[test]
fn analyze_audio_matches_v1_0_0_44100() {
    check_analyze_audio(44100);
}

/// 48 kHz drifts from 1.0.0 under D1: checked with `assert_drift`.
#[test]
fn analyze_audio_matches_v1_0_0_48000() {
    check_analyze_audio(48000);
}

fn check_analyze_owned(rate: u32) {
    require_models!();
    let x = forty_seconds(rate);
    let current = new_bt().analyze_owned(x.clone(), rate).unwrap();
    let old = v1_bt().analyze_audio(&x, rate).unwrap();
    assert_same_or_drift(&format!("analyze_owned @ {rate}"), rate, &current, &old);
}

#[test]
fn analyze_owned_matches_v1_0_0_22050() {
    check_analyze_owned(22050);
}

#[test]
fn analyze_owned_matches_v1_0_0_44100() {
    check_analyze_owned(44100);
}

/// 48 kHz drifts from 1.0.0 under D1: checked with `assert_drift`.
#[test]
fn analyze_owned_matches_v1_0_0_48000() {
    check_analyze_owned(48000);
}

/// `load_audio(mp3, 44100)` returns the native 44.1 kHz mono unchanged (source == target).
#[test]
fn analyze_owned_matches_v1_0_0_mp3_native() {
    require_models!();
    if !Path::new(TEST_AUDIO_PATH).exists() {
        eprintln!("Skipping test: test audio not found");
        return;
    }
    let audio = beat_this::load_audio(Path::new(TEST_AUDIO_PATH), 44100).unwrap();
    assert_eq!(
        audio.sample_rate, 44100,
        "fixture is expected to be 44.1 kHz"
    );
    let current = new_bt()
        .analyze_owned(audio.samples.clone(), audio.sample_rate)
        .unwrap();
    let old = v1_bt()
        .analyze_audio(&audio.samples, audio.sample_rate)
        .unwrap();
    assert_same("analyze_owned @ mp3 native", &current, &old);
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
    for rate in [22050u32, 44100, 48000] {
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
                assert_same_or_drift(&what, rate, c, o);
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
            assert_same_or_drift(&format!("{minutes} min @ {rate}"), rate, &current, &old);
            drop(current);
            let owned = new.analyze_owned(x, rate).unwrap();
            assert_same_or_drift(
                &format!("{minutes} min @ {rate} (owned)"),
                rate,
                &owned,
                &old,
            );
            let how = if rate_is_exact(rate) {
                "identical"
            } else {
                "within drift bounds, beats identical"
            };
            eprintln!("{minutes} min @ {rate}: {how}");
        }
    }
}

// --- ticket 03: the chunked resampler ---------------------------------------------------------

/// The 1.0.0 one-shot resampler, verbatim, to 22 050 Hz.
fn reference_resample(x: &[f32], sr: u32) -> Vec<f32> {
    audio::resample(x.to_vec(), sr, 22050).expect("1.0.0 resample")
}

/// The chunked resampler, fed `push_size` samples at a time.
fn stream_resample(x: &[f32], sr: u32, push_size: usize) -> Vec<f32> {
    let mut r = StreamResampler::new(sr, 22050).unwrap();
    let mut out = Vec::new();
    for piece in x.chunks(push_size) {
        r.push(piece, &mut out).unwrap();
    }
    r.finish(&mut out).unwrap();
    out
}

fn ten_seconds(rate: u32) -> Vec<f32> {
    Synth::take(rate, 10 * rate as usize)
}

#[test]
fn chunking_predicate() {
    for sr in [11025, 44100, 88200, 176400] {
        assert!(__probe::chunking_is_exact(sr), "{sr} should be exact");
    }
    for sr in [8000, 16000, 32000, 48000, 96000] {
        assert!(!__probe::chunking_is_exact(sr), "{sr} should not be exact");
    }
}

#[test]
fn resampler_is_identical_for_exact_ratios() {
    for sr in [11025u32, 44100, 88200] {
        let x = ten_seconds(sr);
        let reference = reference_resample(&x, sr);
        assert_eq!(reference.len(), __probe::one_shot_len(x.len(), sr));
        for push in [1usize, 7, 4096, 8192, 8193, x.len()] {
            let got = stream_resample(&x, sr, push);
            assert_bits_eq(&format!("resample @ {sr}, push {push}"), &got, &reference);
        }
    }
}

#[test]
fn output_length_matches_one_shot_for_many_input_lengths() {
    for sr in [44100u32, 48000] {
        let big = Synth::take(sr, 30_000);
        for n in (260..1200)
            .step_by(37)
            .chain([8191, 8192, 8193, 16384, 16385, 24576, 30_000])
        {
            let x = &big[..n];
            let Ok(reference) = audio::resample(x.to_vec(), sr, 22050) else {
                continue;
            };
            let got = stream_resample(x, sr, 1000);
            assert_eq!(got.len(), reference.len(), "length @ {sr}, n={n}");
        }
    }
}

#[test]
fn resampler_drift_report_48000() {
    let sr = 48000u32;
    let x = ten_seconds(sr);
    let reference = reference_resample(&x, sr);
    let first = stream_resample(&x, sr, 8192);
    assert_eq!(first.len(), reference.len(), "length");
    let head = __probe::one_shot_len(8192, sr);
    assert_bits_eq("first chunk", &first[..head], &reference[..head]);
    for push in [7usize, x.len()] {
        let other = stream_resample(&x, sr, push);
        assert_bits_eq(&format!("push {push} vs push 8192"), &other, &first);
    }
    let d = bit_diff(&first, &reference);
    eprintln!("48 kHz, 10 s drift vs 1.0.0: {d:?}");
    // A regression guard for this 10 s input only: the drift grows with the length (4.2e-5 at
    // 5 min, 1.8e-2 at 60 min, `resampler_drift_long_48000`).
    assert!(d.max_abs <= 1e-6, "drift grew: {d:?}");
    // Degradation: at 48 kHz chunking does change the output, so a comparison that saw nothing
    // here would not be comparing the chunked path.
    assert!(d.differing > 0, "48 kHz chunking shows no drift: {d:?}");
}

#[test]
fn push_size_invariance_all_rates() {
    for sr in [44100u32, 48000] {
        let x = ten_seconds(sr);
        let whole = stream_resample(&x, sr, 8192);
        for push in [1usize, 7, x.len()] {
            let got = stream_resample(&x, sr, push);
            assert_bits_eq(&format!("push {push} @ {sr}"), &got, &whole);
        }
    }
}

/// Degradations: the exact-ratio comparison must notice a tiny input change in the first chunk
/// and a dropped sample at a chunk boundary.
#[test]
fn resampler_comparison_degradations() {
    let sr = 44100u32;
    let x = ten_seconds(sr);
    let reference = reference_resample(&x, sr);

    // A tiny perturbation (1e-6, a few ULP) of a mid-chunk sample. A single ULP is not enough on
    // every CPU: on x86 the filter's f32 accumulation can absorb a 1-ULP input change entirely.
    let mut bumped = x.clone();
    bumped[4000] += 1e-6;
    let d = bit_diff(&stream_resample(&bumped, sr, 8192), &reference);
    assert!(d.differing > 0, "a 1e-6 input change went unnoticed: {d:?}");

    let mut dropped = x.clone();
    dropped.remove(8192);
    let got = stream_resample(&dropped, sr, 8192);
    let d = bit_diff(&got, &reference);
    assert!(
        d.len_a != d.len_b || d.differing > 0,
        "a dropped sample went unnoticed: {d:?}"
    );
}

#[test]
fn resampler_rejects_empty_input() {
    let mut r = StreamResampler::new(48000, 22050).unwrap();
    let mut out = Vec::new();
    r.push(&[], &mut out).unwrap();
    assert!(r.finish(&mut out).is_err());
    assert!(audio::resample(Vec::new(), 48000, 22050).is_err());
}

/// A sample rate of 0 is an error at every length, never a panic. 1.0.0 returned an error for
/// up to 2 samples (an empty resample, then an empty mel) and panicked inside rubato from 3 samples
/// on; the error behaviour must match where 1.0.0 had one.
#[test]
fn rate_zero_is_an_error() {
    assert!(StreamResampler::new(0, 22050).is_err());
    assert!(StreamResampler::new(22050, 0).is_err());
    assert!(!__probe::chunking_is_exact(0));
    for n in 0..=5usize {
        let x = Synth::take(22050, n);
        assert!(__probe::resample(x, 0).is_err(), "resample n={n} @ 0 Hz");
    }
    require_models!();
    let mut new = new_bt();
    let mut v1 = v1_bt();
    for n in 0..=5usize {
        let x = Synth::take(22050, n);
        let current = new.analyze_audio(&x, 0);
        assert!(
            current.is_err(),
            "analyze_audio n={n} @ 0 Hz must be an error"
        );
        let old = catch_unwind(AssertUnwindSafe(|| v1.analyze_audio(&x, 0).is_err()));
        if n <= 2 {
            assert_eq!(old.ok(), Some(true), "1.0.0 errors for n={n} @ 0 Hz");
        } else {
            assert!(old.is_err(), "1.0.0 panics for n={n} @ 0 Hz");
        }
    }
}

/// An extreme downsampling ratio (each 8192-frame chunk yields well under one output frame) still
/// terminates with the one-shot output length.
#[test]
fn resampler_extreme_ratio_terminates() {
    let sr = 4_000_000_000u32;
    let x = Synth::take(22050, 600_000);
    let got = stream_resample(&x, sr, 8192);
    assert_eq!(got.len(), __probe::one_shot_len(x.len(), sr));
}

fn print_drift_header() {
    eprintln!("| rate | min | differing | max_abs | max_ulp | len |");
    eprintln!("|---|---|---|---|---|---|");
}

fn drift_row(sr: u32, minutes: usize) -> BitDiff {
    let x = Synth::take(sr, minutes * 60 * sr as usize);
    let reference = reference_resample(&x, sr);
    let got = stream_resample(&x, sr, 8192);
    let d = bit_diff(&got, &reference);
    eprintln!(
        "| {sr} | {minutes} | {} | {:e} | {} | {} |",
        d.differing, d.max_abs, d.max_ulp, d.len_a
    );
    d
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn resampler_drift_long() {
    print_drift_header();
    for minutes in [5usize, 20] {
        for sr in [44100u32, 88200] {
            let d = drift_row(sr, minutes);
            assert!(d.is_identical(), "{sr} must be identical: {d:?}");
        }
        for sr in [48000u32, 32000, 16000, 96000] {
            let d = drift_row(sr, minutes);
            assert_eq!(d.len_a, d.len_b, "length @ {sr}");
        }
    }
}

// --- downstream effect of the 48 kHz-class resampler drift --------------------------------------

struct DriftReport {
    resampled: BitDiff,
    mel: BitDiff,
    beat_logits: BitDiff,
    downbeat_logits: BitDiff,
    min_abs_logit: f32,
    beats: usize,
    downbeats: usize,
}

/// Analyse `x` (48 kHz-class input) through 1.0.0 (one-shot resample) and through the current
/// default path (`analyze_audio`, which resamples in chunks under D1), and assert that beats and
/// downbeats are bit-identical.
fn compare_downstream(
    what: &str,
    x: &[f32],
    sr: u32,
    new: &mut NewBt,
    v1: &mut OldBt,
) -> DriftReport {
    let reference = reference_resample(x, sr);
    let routed = __probe::resample(x.to_vec(), sr).unwrap();
    let resampled = bit_diff(&routed, &reference);
    drop((reference, routed));
    let old = v1.analyze_audio(x, sr).unwrap();
    let current = new.analyze_audio(x, sr).unwrap();
    let min_abs_logit = old
        .beat_logits
        .iter()
        .chain(&old.downbeat_logits)
        .chain(&current.beat_logits)
        .chain(&current.downbeat_logits)
        .map(|v| v.abs())
        .fold(f32::INFINITY, f32::min);
    let report = DriftReport {
        resampled,
        mel: bit_diff(&current.mel.data, &old.mel.data),
        beat_logits: bit_diff(&current.beat_logits, &old.beat_logits),
        downbeat_logits: bit_diff(&current.downbeat_logits, &old.downbeat_logits),
        min_abs_logit,
        beats: old.beats.len(),
        downbeats: old.downbeats.len(),
    };
    assert_bits_eq(&format!("{what}: beats"), &current.beats, &old.beats);
    assert_bits_eq(
        &format!("{what}: downbeats"),
        &current.downbeats,
        &old.downbeats,
    );
    report
}

fn print_report(what: &str, r: &DriftReport) {
    eprintln!(
        "{what}: pcm differing {} (max_abs {:e}); mel differing {} (max_abs {:e}); beat logits max_abs {:e}; downbeat logits max_abs {:e}; min |logit| {:e}; {} beats, {} downbeats (identical)",
        r.resampled.differing,
        r.resampled.max_abs,
        r.mel.differing,
        r.mel.max_abs,
        r.beat_logits.max_abs,
        r.downbeat_logits.max_abs,
        r.min_abs_logit,
        r.beats,
        r.downbeats
    );
}

#[test]
#[ignore = "long: run with --release -- --ignored; BEAT_THIS_MODEL selects the beat model"]
fn resampler_drift_beats() {
    require_models!();
    let mut new = new_bt();
    let mut v1 = v1_bt();
    eprintln!("beat model: {}", beat_model_path());
    if Path::new(TEST_AUDIO_PATH).exists() {
        let audio = beat_this::load_audio(Path::new(TEST_AUDIO_PATH), 48000).unwrap();
        assert_eq!(audio.sample_rate, 48000);
        let r = compare_downstream(
            "mp3 upsampled to 48 kHz",
            &audio.samples,
            48000,
            &mut new,
            &mut v1,
        );
        print_report("mp3 upsampled to 48 kHz", &r);
    }
    let x = Synth::take(48000, 20 * 60 * 48000);
    let r = compare_downstream("20 min synth @ 48 kHz", &x, 48000, &mut new, &mut v1);
    print_report("20 min synth @ 48 kHz", &r);
}

/// Every audio file of the corpus, as 48 kHz input. `BEAT_THIS_DRIFT_CORPUS` is a directory
/// (searched recursively) or a text file listing one path per line.
#[test]
#[ignore = "long: set BEAT_THIS_DRIFT_CORPUS; run with --release -- --ignored"]
fn resampler_drift_corpus() {
    let Ok(root) = std::env::var("BEAT_THIS_DRIFT_CORPUS") else {
        eprintln!("Skipping: BEAT_THIS_DRIFT_CORPUS is not set");
        return;
    };
    require_models!();
    let files = corpus_files(Path::new(&root));
    assert!(!files.is_empty(), "no files found under {root}");
    let mut new = new_bt();
    let mut v1 = v1_bt();
    eprintln!("beat model: {}", beat_model_path());
    let mut compared = 0;
    for path in &files {
        let audio = match beat_this::load_audio(path, 48000) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("skip {}: {e}", path.display());
                continue;
            }
        };
        assert_eq!(audio.sample_rate, 48000);
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let r = compare_downstream(&name, &audio.samples, 48000, &mut new, &mut v1);
        print_report(&name, &r);
        compared += 1;
    }
    eprintln!("corpus: {compared} of {} files compared", files.len());
}

fn corpus_files(root: &Path) -> Vec<std::path::PathBuf> {
    if root.is_file() {
        let list = std::fs::read_to_string(root).unwrap();
        return list
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(std::path::PathBuf::from)
            .collect();
    }
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("mp3" | "wav" | "flac" | "ogg" | "m4a")
            ) {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// The routed `resample` (always chunked, D1) against the 1.0.0 one-shot call, including tiny
/// inputs and the error behaviour: bit-identical at exact rates, within the drift bound elsewhere.
#[test]
fn routed_resample_matches_v1_0_0() {
    for sr in [11025u32, 44100, 88200, 48000, 32000] {
        let big = Synth::take(sr, 20_000);
        let lengths = (0..=40usize)
            .chain((41..2000).step_by(61))
            .chain([8191, 8192, 8193, 16384, 20_000]);
        for n in lengths {
            let x = &big[..n];
            let old = audio::resample(x.to_vec(), sr, 22050);
            let new = __probe::resample(x.to_vec(), sr);
            assert_eq!(old.is_err(), new.is_err(), "error behaviour @ {sr}, n={n}");
            if let (Ok(o), Ok(c)) = (old, new) {
                let what = format!("routed resample @ {sr}, n={n}");
                if __probe::chunking_is_exact(sr) {
                    assert_bits_eq(&what, &c, &o);
                } else {
                    assert_pcm_drift(&what, &c, &o);
                }
            }
        }
    }
}

// --- long inputs: accuracy against an analytic signal ------------------------------------------
//
// At rates whose ratio is not a short dyadic fraction, the chunked resampler and 1.0.0's one-shot
// call differ, and the difference grows with the input length. These tests decide which side is
// right by resampling a pure tone whose ideal resampled values are known in closed form.

/// The analytic test tone: 1 kHz, amplitude 0.5. 1 kHz is deep in the pass band of every
/// conversion used here (rubato's cutoff is 0.95 of the lower Nyquist: 10.5 kHz at 22 050 Hz).
const TONE_HZ: u64 = 1000;
const TONE_AMP: f64 = 0.5;

/// `n` samples of the tone at `sr`. The phase of sample `j` is `(j * 1000 mod sr) / sr` cycles,
/// computed in integers, so it is exact for any `j`.
fn tone(sr: u32, n: usize) -> Vec<f32> {
    (0..n as u64)
        .map(|j| {
            let cycles = ((j * TONE_HZ) % sr as u64) as f64 / sr as f64;
            (TONE_AMP * (std::f64::consts::TAU * cycles).sin()) as f32
        })
        .collect()
}

/// The ideal value of output frame `k` when the tone is resampled from `sr` to `target` with this
/// crate's rubato 3.0.0 settings (sinc_len 256, oversampling 256, linear interpolation between sinc
/// points), in f64.
///
/// Alignment, derived from rubato 3.0.0's source and shared by both paths, which use the same
/// parameters: the read position starts at `last_index = -(256 - 1)` (`asynchro_sinc.rs`
/// `init_last_index`) and advances by `t = sr / target` before each output frame, so frame `k` is
/// read at `idx_k = -255 + (k + 1) t`. The input sits at offset `2 * 256` of rubato's buffer, and
/// sinc table `s` holds the kernel at offsets `p + 1 - (s + 1) / 256 - 128` (`sinc.rs` `make_sincs`),
/// so with linear interpolation between tables `s` and `s + 1` frame `k` is the band-limited input
/// at input position `tau_k = idx_k + 127 + 1/256 = (k + 1) t - 128 + 1/256`. Neither path trims
/// that leading delay. The tone's phase there is `1000 (k + 1) / target - 1000 (128 - 1/256) / sr`
/// cycles; the first term is reduced modulo 1 in integers, so the expected value carries no position
/// error at any `k`. (rubato's own `t` is `1 / (target / sr)` in f64, at most 1 ULP from the exact
/// ratio: under 2e-8 input samples of position after 80 M output frames, which is negligible here.)
/// The kernel's gain at 1 kHz is taken as 1; its deviation is part of the error floor measured.
fn tone_expected(k: usize, sr: u32, target: u32) -> f64 {
    TONE_AMP * tone_phase(k, sr, target).sin()
}

/// The tone's phase in radians at output frame `k` (see [`tone_expected`]).
fn tone_phase(k: usize, sr: u32, target: u32) -> f64 {
    let whole = ((k as u64 + 1) * TONE_HZ % target as u64) as f64 / target as f64;
    let delay = TONE_HZ as f64 * (128.0 - 1.0 / 256.0) / sr as f64;
    std::f64::consts::TAU * (whole - delay)
}

/// Error of one resampled segment against [`tone_expected`].
#[derive(Clone, Copy, Debug)]
struct ToneError {
    rms: f64,
    max_abs: f64,
    /// Mean position error over the segment in input samples, from the phase of the output against
    /// the expected tone (positive: the output reads the input late).
    shift: f64,
}

/// Errors per segment of `seg` output frames. The first and last `edge` frames are left out: the
/// tone starts and stops abruptly there, so the band-limited ideal does not apply.
fn tone_errors(y: &[f32], sr: u32, target: u32, seg: usize, edge: usize) -> Vec<ToneError> {
    let end = y.len().saturating_sub(edge);
    let mut out = Vec::new();
    let mut start = 0;
    while start < end {
        let (lo, hi) = (start.max(edge), (start + seg).min(end));
        let (mut sq, mut max_abs) = (0.0f64, 0.0f64);
        // Least-squares fit of v = a sin(phase) + b cos(phase) over the segment.
        let (mut ss, mut sc, mut cc, mut vs, mut vc) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (k, &v) in y.iter().enumerate().take(hi).skip(lo) {
            let e = tone_expected(k, sr, target);
            let d = v as f64 - e;
            sq += d * d;
            max_abs = max_abs.max(d.abs());
            let (s, c) = tone_phase(k, sr, target).sin_cos();
            ss += s * s;
            sc += s * c;
            cc += c * c;
            vs += v as f64 * s;
            vc += v as f64 * c;
        }
        let n = (hi - lo).max(1) as f64;
        let det = ss * cc - sc * sc;
        let a = (vs * cc - vc * sc) / det;
        let b = (vc * ss - vs * sc) / det;
        // v = A sin(phase + phi) = A cos(phi) sin(phase) + A sin(phi) cos(phase), so phi = atan2(b, a);
        // a read position late by d input samples is phi = -2 pi 1000 d / sr.
        let phi = b.atan2(a);
        out.push(ToneError {
            rms: (sq / n).sqrt(),
            max_abs,
            shift: -phi * sr as f64 / (std::f64::consts::TAU * TONE_HZ as f64),
        });
        start += seg;
    }
    out
}

/// The chunked resampler from `sr` to `target`, fed in 8192-frame pushes.
fn stream_resample_to(x: &[f32], sr: u32, target: u32) -> Vec<f32> {
    let mut r = StreamResampler::new(sr, target).unwrap();
    let mut out = Vec::new();
    for piece in x.chunks(8192) {
        r.push(piece, &mut out).unwrap();
    }
    r.finish(&mut out).unwrap();
    out
}

/// Per-minute tone errors of the chunked path (this crate) and of 1.0.0's one-shot call, on
/// `minutes` of the tone at `sr`, resampled to `target`. Prints a table.
fn tone_accuracy(sr: u32, target: u32, minutes: usize) -> (Vec<ToneError>, Vec<ToneError>) {
    let n = minutes * 60 * sr as usize;
    let seg = 60 * target as usize;
    let edge = target as usize;
    let chunked = {
        let y = stream_resample_to(&tone(sr, n), sr, target);
        tone_errors(&y, sr, target, seg, edge)
    };
    let one_shot = {
        let y = audio::resample(tone(sr, n), sr, target).expect("1.0.0 resample");
        tone_errors(&y, sr, target, seg, edge)
    };
    eprintln!(
        "tone {sr} -> {target} Hz, {minutes} min (per minute of output; shift in input samples)"
    );
    eprintln!("| minute | chunked rms | chunked max | chunked shift | 1.0.0 rms | 1.0.0 max | 1.0.0 shift |");
    eprintln!("|---|---|---|---|---|---|---|");
    for (m, (c, o)) in chunked.iter().zip(&one_shot).enumerate() {
        eprintln!(
            "| {}-{} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {:.2e} |",
            m,
            m + 1,
            c.rms,
            c.max_abs,
            c.shift,
            o.rms,
            o.max_abs,
            o.shift
        );
    }
    (chunked, one_shot)
}

/// Which resampler is accurate on long inputs: the chunked path (this crate) or 1.0.0's one-shot
/// call. Both are compared with the analytic tone, minute by minute, over an hour.
///
/// Recorded run (arm64, `worklog/03-chunked-resampler.md`, "Long inputs"): the chunked path's
/// per-minute rms error is 3.24e-6 to 3.30e-6 at 48 kHz, 6.89e-6 at 96 kHz and 2.90e-6 to 3.09e-6
/// for 44.1 -> 48 kHz, over the whole hour; its position error grows linearly but stays under
/// 2.2e-5 input samples. 1.0.0's error grows with the length (48 kHz: rms 3.65e-6 in minute 1,
/// 1.19e-4 in minute 20, 2.80e-4 in minute 46, 1.14e-2 in minute 60). rubato accumulates the read
/// position by repeated f64 addition over the whole input, so each addition rounds at the spacing
/// of the position's binade; past 2^27 input frames (46.6 min at 48 kHz) that spacing doubles and,
/// for this ratio's bits, the per-frame rounding error jumps. At 96 kHz the step is exactly twice
/// the 48 kHz one, so the jump lands at 2^28 frames, the same 46.6 min (measured). The chunked path
/// subtracts each chunk's length from the position, so it stays below 2^14 and its rounding never
/// grows.
///
/// Asserted, with margin over the recorded run: the chunked path's rms <= 2e-5, max <= 5e-5 and
/// |position error| <= 1e-4 input samples at every minute, and no minute's rms above 1.5 times the
/// first; 1.0.0's worst minute above 10 times its first minute and 10 times the chunked path's worst
/// minute. For the 22 050 Hz target, 1.0.0's last minute is also above 10 times its minute 46
/// (the jump past 46.6 min).
#[test]
#[ignore = "long (3 x 60 min, ~5.5 GB rss): run with --release -- --ignored --nocapture"]
fn resampler_accuracy_long() {
    for (sr, target, minutes) in [
        (48000u32, 22050u32, 60usize),
        (96000, 22050, 60),
        (44100, 48000, 60),
    ] {
        let (chunked, one_shot) = tone_accuracy(sr, target, minutes);
        let what = format!("{sr} -> {target} Hz, {minutes} min");
        let first = chunked[0].rms;
        for (m, c) in chunked.iter().enumerate() {
            assert!(
                c.rms <= 2e-5 && c.max_abs <= 5e-5 && c.shift.abs() <= 1e-4,
                "{what}: chunked error at minute {m}: {c:?}"
            );
            assert!(
                c.rms <= 1.5 * first,
                "{what}: chunked error grew at minute {m}: {c:?} (first minute rms {first:e})"
            );
        }
        let worst = |v: &[ToneError]| v.iter().map(|e| e.rms).fold(0.0f64, f64::max);
        let (c_worst, o_worst, o_first) = (worst(&chunked), worst(&one_shot), one_shot[0].rms);
        assert!(
            o_worst > 10.0 * o_first && o_worst > 10.0 * c_worst,
            "{what}: 1.0.0's error did not grow: first minute rms {o_first:e}, worst {o_worst:e}, chunked worst {c_worst:e}"
        );
        if target == 22050 {
            let (m46, last) = (one_shot[45].rms, one_shot.last().unwrap().rms);
            assert!(
                last > 10.0 * m46,
                "{what}: no jump past 2^27 frames: minute 46 rms {m46:e}, last {last:e}"
            );
        }
        eprintln!(
            "{what}: worst minute rms: chunked {c_worst:.2e}, 1.0.0 {o_worst:.2e} ({:.0}x)",
            o_worst / c_worst
        );
    }
}

// --- long inputs: drift against 1.0.0 at 48 kHz ---------------------------------------------------

/// Largest |difference| per bin of `bin` frames (lengths must match).
fn max_abs_per_bin(a: &[f32], b: &[f32], bin: usize) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.chunks(bin)
        .zip(b.chunks(bin))
        .map(|(x, y)| {
            x.iter()
                .zip(y)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max)
        })
        .collect()
}

/// Resampled PCM, chunked (this crate) against 1.0.0's one-shot call, on the 48 kHz synthetic signal
/// at 5 to 60 minutes. Documents the drift; asserts only equal lengths and the shape recorded in
/// `worklog/03-chunked-resampler.md` ("Long inputs"): the drift grows with the length, and once the
/// input passes 2^27 frames (46.6 min), where 1.0.0's accumulated read position loses a further
/// bit, 1.0.0's position error reverses direction and grows much faster: the difference dips
/// towards zero just after 2796 s of output, then rises above anything before it within about a
/// minute (see `resampler_accuracy_long` for which side is accurate).
#[test]
#[ignore = "long (up to 60 min at 48 kHz, ~3 GB): run with --release -- --ignored --nocapture"]
fn resampler_drift_long_48000() {
    let sr = 48000u32;
    eprintln!("| min | differing | max_abs | len |");
    eprintln!("|---|---|---|---|");
    let mut rows = Vec::new();
    for minutes in [5usize, 10, 20, 40, 50, 60] {
        let x = Synth::take(sr, minutes * 60 * sr as usize);
        let got = stream_resample(&x, sr, 8192);
        let reference = reference_resample(&x, sr);
        let d = bit_diff(&got, &reference);
        assert_eq!(d.len_a, d.len_b, "length at {minutes} min");
        eprintln!(
            "| {minutes} | {} | {:.3e} | {} |",
            d.differing, d.max_abs, d.len_a
        );
        rows.push((minutes, d.max_abs));
        if minutes == 60 {
            // Where the jump is: the largest |difference| per 10 s of output, around 2^27 input
            // frames (2^27 / 48 000 = 2796.2 s; output frame k reads input near k * 48000 / 22050).
            let bins = max_abs_per_bin(&got, &reference, 10 * 22050);
            let before = bins[..270].iter().fold(0.0f32, |m, &v| m.max(v));
            eprintln!("60 min: max |diff| over the first 45 min of output: {before:.3e}");
            eprintln!("| output s | max |diff| |");
            eprintln!("|---|---|");
            for (b, v) in bins.iter().enumerate().take(290).skip(274) {
                eprintln!("| {}-{} | {:.3e} |", b * 10, b * 10 + 10, v);
            }
            // 1.0.0's position error reverses direction there: the difference first dips towards
            // zero, then grows far faster than before.
            let dip = (274..290)
                .min_by(|&a, &b| bins[a].total_cmp(&bins[b]))
                .unwrap();
            let jump = bins
                .iter()
                .position(|&v| v > 2.0 * before)
                .expect("the drift jumps past 2^27 input frames");
            eprintln!(
                "60 min: smallest 10 s bin from 2740 s at {} s; first bin above twice the first-45-min maximum at {} s",
                dip * 10,
                jump * 10
            );
            assert!(
                (2790..2830).contains(&(dip * 10)) && (2800..2900).contains(&(jump * 10)),
                "the change is expected just past 2796.2 s: dip at {} s, jump at {} s",
                dip * 10,
                jump * 10
            );
        }
    }
    for w in rows.windows(2) {
        assert!(w[1].1 > w[0].1, "drift did not grow: {rows:?}");
    }
}

/// Beat or downbeat times whose bits differ, as `(index, 1.0.0, current)`.
type Moved = Vec<(usize, f32, f32)>;

/// [`Moved`] times, when the counts match.
fn moved_times(old: &[f32], new: &[f32]) -> Moved {
    old.iter()
        .zip(new)
        .enumerate()
        .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
        .map(|(i, (&a, &b))| (i, a, b))
        .collect()
}

/// `analyze_audio` (chunked resampler) against 1.0.0 on `minutes` of the 48 kHz synthetic signal,
/// printed per 5 minutes. Returns the moved beats and downbeats (counts are asserted equal).
fn long_downstream(minutes: usize, new: &mut NewBt, v1: &mut OldBt) -> (Moved, Moved) {
    let x = Synth::take(48000, minutes * 60 * 48000);
    let old = v1.analyze_audio(&x, 48000).unwrap();
    let cur = new.analyze_audio(&x, 48000).unwrap();
    drop(x);
    let what = format!("{minutes} min @ 48000");
    assert_eq!(cur.mel.shape, old.mel.shape, "{what}: mel shape");
    assert_eq!(cur.beats.len(), old.beats.len(), "{what}: beat count");
    assert_eq!(
        cur.downbeats.len(),
        old.downbeats.len(),
        "{what}: downbeat count"
    );
    let mel = bit_diff(&cur.mel.data, &old.mel.data);
    let beat = bit_diff(&cur.beat_logits, &old.beat_logits);
    let down = bit_diff(&cur.downbeat_logits, &old.downbeat_logits);
    eprintln!(
        "{what}: mel max_abs {:.3e}; beat logits max_abs {:.3e}; downbeat logits max_abs {:.3e}; {} beats, {} downbeats",
        mel.max_abs,
        beat.max_abs,
        down.max_abs,
        old.beats.len(),
        old.downbeats.len()
    );
    let seg = 5 * 60 * 50; // frames per 5 minutes
    let mel_bins = max_abs_per_bin(&cur.mel.data, &old.mel.data, seg * 128);
    let beat_bins = max_abs_per_bin(&cur.beat_logits, &old.beat_logits, seg);
    let down_bins = max_abs_per_bin(&cur.downbeat_logits, &old.downbeat_logits, seg);
    eprintln!("| minutes | mel max_abs | beat logit max_abs | downbeat logit max_abs |");
    eprintln!("|---|---|---|---|");
    for (i, ((m, b), d)) in mel_bins.iter().zip(&beat_bins).zip(&down_bins).enumerate() {
        eprintln!("| {}-{} | {m:.3e} | {b:.3e} | {d:.3e} |", i * 5, i * 5 + 5);
    }
    let over = cur
        .beat_logits
        .iter()
        .zip(&old.beat_logits)
        .position(|(a, b)| (a - b).abs() > LOGIT_DRIFT_BOUND);
    if let Some(f) = over {
        eprintln!(
            "{what}: first beat-logit drift above {LOGIT_DRIFT_BOUND:e} at frame {f} ({:.2} s)",
            f as f32 / 50.0
        );
    }
    let print_moved = |kind: &str, moved: &[(usize, f32, f32)], old_l: &[f32], new_l: &[f32]| {
        for &(i, a, b) in moved {
            let f = (a * 50.0).round() as usize;
            let lo = f.saturating_sub(1);
            let hi = (f + 3).min(old_l.len());
            eprintln!(
                "{what}: {kind} {i} moved {a} s -> {b} s ({:+.3} frames); logits frames {lo}..{hi}: 1.0.0 {:?}, current {:?}",
                (b - a) * 50.0,
                &old_l[lo..hi],
                &new_l[lo..hi]
            );
        }
    };
    let beats = moved_times(&old.beats, &cur.beats);
    let downbeats = moved_times(&old.downbeats, &cur.downbeats);
    print_moved("beat", &beats, &old.beat_logits, &cur.beat_logits);
    print_moved(
        "downbeat",
        &downbeats,
        &old.downbeat_logits,
        &cur.downbeat_logits,
    );
    eprintln!(
        "{what}: {} of {} beats and {} of {} downbeats moved",
        beats.len(),
        old.beats.len(),
        downbeats.len(),
        old.downbeats.len()
    );
    (beats, downbeats)
}

/// Downstream effect of the long-input drift, full model only (`BEAT_THIS_MODEL` must name
/// `beat_this.onnx`; the committed small model is not what users run). Documents, and asserts the
/// recorded shape (`worklog/03-chunked-resampler.md`, "Long inputs"): at 40 and 50 min beats and
/// downbeats are bit-identical to 1.0.0; at 60 min the beat and downbeat counts are equal and every
/// moved time moved by at most one frame (20 ms). Run alone: the 1.0.0 reference takes ~5.6 GB at
/// 60 min.
#[test]
#[ignore = "long (40/50/60 min at 48 kHz, full model, ~6 GB): run alone with --release -- --ignored --nocapture"]
fn long_drift_downstream_48000() {
    require_models!();
    let model = beat_model_path();
    eprintln!("beat model: {model}");
    if !model.ends_with("beat_this.onnx") {
        eprintln!("Skipping: set BEAT_THIS_MODEL=models/beat_this.onnx (the full model)");
        return;
    }
    let mut new = new_bt();
    let mut v1 = v1_bt();
    for minutes in [40usize, 50, 60] {
        let (beats, downbeats) = long_downstream(minutes, &mut new, &mut v1);
        if minutes < 60 {
            assert!(
                beats.is_empty() && downbeats.is_empty(),
                "{minutes} min: beats or downbeats moved"
            );
        } else {
            for &(i, a, b) in beats.iter().chain(&downbeats) {
                assert!(
                    (b - a).abs() <= 0.0201,
                    "60 min: time {i} moved by more than one frame: {a} -> {b}"
                );
            }
        }
    }
}

/// `load_audio` to targets other than 22 050 Hz, and 44.1 -> 48 kHz, against 1.0.0. Whether the
/// chunked resampler is bit-identical depends on the source/target ratio, not on the source alone:
/// it is when rubato's step `source / target` is a short dyadic fraction (44.1 kHz to 22 050 or
/// 88 200 Hz), and drifts otherwise. Documents the drift; asserts identity exactly where the ratio
/// predicts it and equal lengths everywhere.
#[test]
#[ignore = "long: run with --release -- --ignored --nocapture"]
fn load_audio_targets_report() {
    if !Path::new(TEST_AUDIO_PATH).exists() {
        eprintln!("Skipping test: test audio not found");
        return;
    }
    let path = Path::new(TEST_AUDIO_PATH);
    eprintln!("| load_audio(44.1 kHz mp3, target) | differing | max_abs | len |");
    eprintln!("|---|---|---|---|");
    for target in [22050u32, 88200, 48000, 32000, 16000] {
        let got = beat_this::load_audio(path, target).unwrap();
        let old = audio::load_audio(path, target).unwrap();
        let d = bit_diff(&got.samples, &old.samples);
        eprintln!(
            "| {target} | {} | {:.3e} | {} |",
            d.differing, d.max_abs, d.len_a
        );
        assert_eq!(d.len_a, d.len_b, "length at {target}");
        assert_eq!(
            d.is_identical(),
            __probe::chunking_is_exact_between(44100, target),
            "identity at {target}: {d:?}"
        );
    }
    eprintln!("| synth 44.1 -> 48 kHz, min | differing | max_abs | len |");
    eprintln!("|---|---|---|---|");
    for minutes in [20usize, 60] {
        let x = Synth::take(44100, minutes * 60 * 44100);
        let got = stream_resample_to(&x, 44100, 48000);
        let old = audio::resample(x, 44100, 48000).unwrap();
        let d = bit_diff(&got, &old);
        eprintln!(
            "| {minutes} | {} | {:.3e} | {} |",
            d.differing, d.max_abs, d.len_a
        );
        assert_eq!(d.len_a, d.len_b, "length at {minutes} min");
        assert!(d.differing > 0, "44.1 -> 48 kHz is not an exact ratio");
    }
}

// --- ticket 04: windowed mel ----------------------------------------------------------------

type RefMel = mel::MelExtractor<<runtime::rten::RtenRuntime as Runtime>::Model>;
type NewMelModel = <beat_this::RtenRuntime as Runtime>::Model;

/// The verbatim 1.0.0 whole-signal mel extractor.
fn ref_mel() -> RefMel {
    mel::MelExtractor::new(
        runtime::rten::RtenRuntime
            .load_model(Path::new(MEL_MODEL_PATH))
            .expect("mel model (1.0.0 reference)"),
    )
}

/// The current runtime's mel model, for the windowed paths.
fn new_mel_model() -> NewMelModel {
    beat_this::RtenRuntime
        .load_model(Path::new(MEL_MODEL_PATH))
        .expect("mel model (current)")
}

const HOP: usize = 441;
const STRIDES: [usize; 2] = [__probe::MEL_STRIDE, 64];

fn assert_mel_same(what: &str, got: &beat_this::Tensor, reference: &beat_this::Tensor) {
    assert_eq!(got.shape, reference.shape, "{what}: mel shape");
    assert_bits_eq(what, &got.data, &reference.data);
}

#[test]
fn mel_frame_count_formula() {
    if !Path::new(MEL_MODEL_PATH).exists() {
        eprintln!("Skipping test: mel model not found");
        return;
    }
    let mut reference = ref_mel();
    let x = Synth::take(22050, 10_000);
    let lengths = [
        1,
        2,
        100,
        440,
        441,
        442,
        511,
        512,
        513,
        1023,
        1024,
        1025,
        HOP * 7,
        HOP * 7 + 1,
        HOP * 7 + 440,
    ];
    for n in lengths {
        let mel = reference.extract(&x[..n]).unwrap();
        assert_eq!(mel.shape, vec![1, 1 + n / HOP, 128], "frames for n={n}");
    }
    assert!(reference.extract(&[]).is_err(), "n=0 must error");
    assert!(__probe::mel_windowed(&mut new_mel_model(), &[], 64).is_err());
}

fn lengths_under_test() -> Vec<usize> {
    let mut lengths = vec![
        1,
        2,
        100,
        440,
        441,
        442,
        511,
        512,
        513,
        1023,
        1024,
        1025,
        HOP * 7,
        HOP * 7 + 1,
        HOP * 7 + 440,
        HOP * 64 - 1,
        HOP * 64,
        HOP * 64 + 1,
        HOP * 128 - 1,
        HOP * 128,
        HOP * 128 + 1,
    ];
    // Sweeps T mod 64 over ~60-90 s inputs.
    lengths.extend((0..30).map(|i| HOP * (3072 + 37 * i) + (97 * i) % HOP));
    lengths
}

#[test]
fn windowed_mel_is_identical() {
    if !Path::new(MEL_MODEL_PATH).exists() {
        eprintln!("Skipping test: mel model not found");
        return;
    }
    let mut reference = ref_mel();
    let mut model = new_mel_model();

    // Three 10 s signals.
    let n = 10 * 22050;
    let mut impulse = vec![0.0f32; n];
    impulse[n / 2] = 1.0;
    let signals: [(&str, Vec<f32>); 3] = [
        ("synth", Synth::take(22050, n)),
        ("silence", vec![0.0; n]),
        ("impulse", impulse),
    ];
    for (name, x) in &signals {
        let expected = reference.extract(x).unwrap();
        for stride in STRIDES {
            let got = __probe::mel_windowed(&mut model, x, stride).unwrap();
            assert_mel_same(&format!("{name} 10 s, stride {stride}"), &got, &expected);
        }
    }

    // The synthetic signal at awkward lengths.
    let lengths = lengths_under_test();
    let x = Synth::take(22050, *lengths.iter().max().unwrap());
    for n in lengths {
        let expected = reference.extract(&x[..n]).unwrap();
        for stride in STRIDES {
            let got = __probe::mel_windowed(&mut model, &x[..n], stride).unwrap();
            assert_mel_same(&format!("synth n={n}, stride {stride}"), &got, &expected);
        }
    }
}

fn stream_mel(model: &mut NewMelModel, x: &[f32], stride: usize, push: usize) -> beat_this::Tensor {
    let mut stream = __probe::MelStream::new(stride);
    for piece in x.chunks(push) {
        stream.push(model, piece).unwrap();
    }
    stream.finish(model).unwrap()
}

#[test]
fn mel_stream_push_size_invariance() {
    if !Path::new(MEL_MODEL_PATH).exists() {
        eprintln!("Skipping test: mel model not found");
        return;
    }
    let mut reference = ref_mel();
    let mut model = new_mel_model();
    let x = Synth::take(22050, 100 * 22050);
    let expected = reference.extract(&x).unwrap();
    let windowed = __probe::mel_windowed(&mut model, &x, __probe::MEL_STRIDE).unwrap();
    assert_mel_same("extract_windowed", &windowed, &expected);
    for push in [7usize, 441, 8192, x.len()] {
        let got = stream_mel(&mut model, &x, __probe::MEL_STRIDE, push);
        assert_mel_same(&format!("stream, push {push}"), &got, &expected);
    }
    // A small stride exercises many window hand-overs.
    let got = stream_mel(&mut model, &x, 64, 8192);
    assert_mel_same("stream, stride 64", &got, &expected);
    // One sample at a time, on a shorter signal.
    let short = &x[..10 * 22050];
    let expected = reference.extract(short).unwrap();
    let got = stream_mel(&mut model, short, __probe::MEL_STRIDE, 1);
    assert_mel_same("stream, push 1", &got, &expected);
}

#[test]
fn mel_stream_edge_cases() {
    if !Path::new(MEL_MODEL_PATH).exists() {
        eprintln!("Skipping test: mel model not found");
        return;
    }
    let mut reference = ref_mel();
    let mut model = new_mel_model();
    let x = Synth::take(22050, 4000);
    for n in [1usize, 2, 440, 441, 442, 511, 512, 513, 1023, 1025, 3000] {
        let expected = reference.extract(&x[..n]).unwrap();
        let got = stream_mel(&mut model, &x[..n], 64, 100);
        assert_mel_same(&format!("stream n={n}"), &got, &expected);
    }
    let stream = __probe::MelStream::new(__probe::MEL_STRIDE);
    assert!(
        stream.finish(&mut model).is_err(),
        "empty stream must error"
    );
}

/// Degradations: the windowed comparison must see the overlap region and a one-sample shift.
#[test]
fn windowed_mel_degradations() {
    if !Path::new(MEL_MODEL_PATH).exists() {
        eprintln!("Skipping test: mel model not found");
        return;
    }
    let mut reference = ref_mel();
    let mut model = new_mel_model();
    let stride = __probe::MEL_STRIDE;
    let n = 100 * 22050;
    let x = Synth::take(22050, n);
    let expected = reference.extract(&x).unwrap();

    // (b) One ULP on a sample inside the first overlap (the first sample of window 1). The control
    // (same perturbed signal through the reference) must still match, so the difference seen is
    // the perturbation, carried through the windowed path.
    let at = HOP * (stride - 64);
    let mut bumped = x.clone();
    bumped[at] = f32::from_bits(bumped[at].to_bits() + 1);
    let windowed = __probe::mel_windowed(&mut model, &bumped, stride).unwrap();
    let control = reference.extract(&bumped).unwrap();
    assert_mel_same("perturbed control", &windowed, &control);
    let d = bit_diff(&windowed.data, &expected.data);
    assert!(
        d.differing > 0,
        "overlap perturbation went unnoticed: {d:?}"
    );

    // (c) Shifted by one sample against an unshifted reference of the same length.
    let shifted = __probe::mel_windowed(&mut model, &x[1..], stride).unwrap();
    let truncated = reference.extract(&x[..n - 1]).unwrap();
    let d = bit_diff(&shifted.data, &truncated.data);
    assert!(d.differing > 0, "a one-sample shift went unnoticed: {d:?}");
}

/// The window plan of `extract_windowed`, written out with a configurable halo: window `k` owns
/// frames `[k*stride, (k+1)*stride)` and reads `halo` frames of context on each side.
fn mel_with_halo(model: &mut NewMelModel, x: &[f32], stride: usize, halo: usize) -> Vec<f32> {
    let n = x.len();
    let total = 1 + n / HOP;
    let mut data = Vec::with_capacity(total * 128);
    let mut k = 0;
    loop {
        let o = (k * stride).saturating_sub(halo);
        let interior_end = HOP * ((k + 1) * stride + halo - 1);
        let (end, frames, keep_end) = if interior_end <= n {
            (
                interior_end,
                (k + 1) * stride + halo - o,
                (k + 1) * stride - o,
            )
        } else {
            (n, total - o, total - o)
        };
        let input = beat_this::Tensor {
            shape: vec![1, end - HOP * o],
            data: x[HOP * o..end].to_vec(),
        };
        let out = beat_this::Model::run(model, &[("audio_pcm", &input)]).unwrap();
        let mel = &out["mel_spectrogram"];
        assert_eq!(mel.shape[1], frames, "window {k} frames");
        data.extend_from_slice(&mel.data[(k * stride - o) * 128..keep_end * 128]);
        if interior_end > n {
            return data;
        }
        k += 1;
    }
}

/// Degradation that breaks the window plan rather than the input: with a 1-frame halo the first
/// owned frame of every window after the first reads the window's own reflect padding instead of
/// real samples. That is a structural difference, not rounding, so it must show on every CPU.
/// The control (the production halo through the same helper) must match.
#[test]
fn windowed_mel_degradation_short_halo() {
    if !Path::new(MEL_MODEL_PATH).exists() {
        eprintln!("Skipping test: mel model not found");
        return;
    }
    let mut reference = ref_mel();
    let mut model = new_mel_model();
    let x = Synth::take(22050, 10 * 22050);
    let expected = reference.extract(&x).unwrap();
    let control = mel_with_halo(&mut model, &x, 64, 64);
    assert_bits_eq("halo 64 (control)", &control, &expected.data);
    let broken = mel_with_halo(&mut model, &x, 64, 1);
    let d = bit_diff(&broken, &expected.data);
    assert!(d.differing > 0, "a 1-frame halo went unnoticed: {d:?}");
    // The damage is in the first owned frame of window 1 (global frame 64).
    let row = |v: &[f32]| {
        v[64 * 128..65 * 128]
            .iter()
            .map(|f| f.to_bits())
            .collect::<Vec<_>>()
    };
    assert_ne!(row(&broken), row(&expected.data), "frame 64 should differ");
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn windowed_mel_long() {
    if !Path::new(MEL_MODEL_PATH).exists() {
        eprintln!("Skipping test: mel model not found");
        return;
    }
    let mut reference = ref_mel();
    let mut model = new_mel_model();
    let x = Synth::take(22050, 20 * 60 * 22050);
    let expected = reference.extract(&x).unwrap();
    for stride in STRIDES {
        let got = __probe::mel_windowed(&mut model, &x, stride).unwrap();
        assert_mel_same(&format!("20 min, stride {stride}"), &got, &expected);
        eprintln!("20 min, stride {stride}: identical");
    }
}

/// Documents why the windows are 64-aligned: the ticket's naive scheme (a 2-frame halo, any width)
/// shows sporadic 1-ULP differences from rten's fused-GEMM partial tiles. Prints, asserts nothing:
/// the outcome depends on the CPU's kernels.
#[test]
#[ignore = "diagnostic: prints, asserts nothing"]
fn windowed_mel_naive_halo_diagnostic() {
    if !Path::new(MEL_MODEL_PATH).exists() {
        eprintln!("Skipping test: mel model not found");
        return;
    }
    let mut reference = ref_mel();
    let mut model = new_mel_model();
    let x = Synth::take(22050, 30 * 22050);
    let expected = reference.extract(&x).unwrap();
    let total = expected.shape[1];
    for width in [8usize, 500, 1000, 1500] {
        let mut data = Vec::new();
        let mut s = 0;
        while s < total {
            let lo = s.saturating_sub(2);
            let hi = (s + width + 2).min(total);
            let end = if hi == total { x.len() } else { HOP * (hi - 1) };
            let input = beat_this::Tensor {
                shape: vec![1, end - HOP * lo],
                data: x[HOP * lo..end].to_vec(),
            };
            let out = beat_this::Model::run(&mut model, &[("audio_pcm", &input)]).unwrap();
            let local = &out["mel_spectrogram"];
            let own = (s - lo)..(s - lo + width.min(total - s));
            data.extend_from_slice(&local.data[own.start * 128..own.end * 128]);
            s += width;
        }
        let d = bit_diff(&data, &expected.data);
        eprintln!(
            "naive halo 2, width {width}: {} of {} values differ (max_ulp {})",
            d.differing,
            expected.data.len(),
            d.max_ulp
        );
    }
}

// --- ticket 05: the streaming front end -----------------------------------------------------------

/// Panics unless two current-pipeline analyses are bit-identical everywhere.
fn assert_same_analysis(what: &str, a: &BeatAnalysis, b: &BeatAnalysis) {
    assert_eq!(a.mel.shape, b.mel.shape, "{what}: mel shape");
    assert_bits_eq(&format!("{what}: mel"), &a.mel.data, &b.mel.data);
    assert_bits_eq(
        &format!("{what}: beat_logits"),
        &a.beat_logits,
        &b.beat_logits,
    );
    assert_bits_eq(
        &format!("{what}: downbeat_logits"),
        &a.downbeat_logits,
        &b.downbeat_logits,
    );
    assert_bits_eq(&format!("{what}: beats"), &a.beats, &b.beats);
    assert_bits_eq(&format!("{what}: downbeats"), &a.downbeats, &b.downbeats);
}

/// `BeatThis::stream`, fed `push` samples at a time (`push >= 1`).
fn stream_analyze(
    bt: &mut NewBt,
    x: &[f32],
    rate: u32,
    push: usize,
) -> anyhow::Result<BeatAnalysis> {
    let mut stream = bt.stream(rate)?;
    for piece in x.chunks(push) {
        stream.push(piece)?;
    }
    stream.finish()
}

/// How a test-side composition of the stream's building blocks is wired. `Production` mirrors
/// `BeatStream` exactly (its output must equal `stream`'s, bit for bit); the others each break the
/// code path in one place, and must be seen by the identity comparisons.
#[derive(Clone, Copy, Debug)]
enum StreamPlan {
    Production,
    /// A fresh resampler for every push, flushed at the end of each: resampler state is lost at
    /// push boundaries.
    ResamplerPerPush,
    /// The resampler's end-of-stream flush is replaced by zeros of the same length.
    ZeroFlush,
    /// The first sample of every push after the first is lost and replaced by zero before the mel
    /// stage (an off-by-one at the push boundary; the length is unchanged).
    ZeroPushBoundary,
}

fn composed_stream(
    bt: &mut NewBt,
    x: &[f32],
    rate: u32,
    push: usize,
    plan: StreamPlan,
) -> anyhow::Result<BeatAnalysis> {
    let mut resampler = (rate != 22050).then(|| StreamResampler::new(rate, 22050).unwrap());
    let mut mel = __probe::MelStream::new(__probe::MEL_STRIDE);
    let mut scratch = Vec::new();
    for (i, piece) in x.chunks(push).enumerate() {
        let mut piece = piece.to_vec();
        if matches!(plan, StreamPlan::ZeroPushBoundary) && i > 0 {
            piece[0] = 0.0;
        }
        match plan {
            StreamPlan::ResamplerPerPush if rate != 22050 => {
                let mut r = StreamResampler::new(rate, 22050).unwrap();
                r.push(&piece, &mut scratch)?;
                r.finish(&mut scratch)?;
                mel.push(__probe::mel_model(bt), &scratch)?;
                scratch.clear();
            }
            _ => match &mut resampler {
                Some(r) => {
                    for sub in piece.chunks(__probe::RESAMPLE_CHUNK) {
                        r.push(sub, &mut scratch)?;
                        mel.push(__probe::mel_model(bt), &scratch)?;
                        scratch.clear();
                    }
                }
                None => mel.push(__probe::mel_model(bt), &piece)?,
            },
        }
    }
    if !matches!(plan, StreamPlan::ResamplerPerPush) || rate == 22050 {
        if let Some(r) = resampler {
            r.finish(&mut scratch)?;
            if matches!(plan, StreamPlan::ZeroFlush) {
                scratch.fill(0.0);
            }
            mel.push(__probe::mel_model(bt), &scratch)?;
        }
    }
    let mel = mel.finish(__probe::mel_model(bt))?;
    __probe::predict_decode(bt, mel)
}

fn assert_differs(what: &str, a: &BeatAnalysis, b: &BeatAnalysis) {
    let outcome = catch_unwind(AssertUnwindSafe(|| assert_same_analysis(what, a, b)));
    assert!(outcome.is_err(), "{what}: the comparison saw no difference");
}

#[test]
fn stream_push_size_invariance() {
    require_models!();
    let mut bt = new_bt();
    for rate in [22050u32, 44100, 48000] {
        let x = forty_seconds(rate);
        let reference = stream_analyze(&mut bt, &x, rate, 8192).unwrap();
        for push in [7usize, 8193, x.len()] {
            let got = stream_analyze(&mut bt, &x, rate, push).unwrap();
            assert_same_analysis(&format!("push {push} @ {rate}"), &got, &reference);
        }
        // Empty pushes between the pieces change nothing.
        let mut stream = bt.stream(rate).unwrap();
        stream.push(&[]).unwrap();
        for piece in x.chunks(10_000) {
            stream.push(piece).unwrap();
            stream.push(&[]).unwrap();
        }
        let got = stream.finish().unwrap();
        assert_same_analysis(&format!("empty pushes @ {rate}"), &got, &reference);

        // One sample at a time, on 5 s.
        let short = &x[..5 * rate as usize];
        let reference = stream_analyze(&mut bt, short, rate, 8192).unwrap();
        let got = stream_analyze(&mut bt, short, rate, 1).unwrap();
        assert_same_analysis(&format!("push 1 @ {rate}"), &got, &reference);
    }
}

#[test]
fn stream_matches_v1_0_0() {
    require_models!();
    let mut bt = new_bt();
    let mut v1 = v1_bt();
    for rate in [22050u32, 44100, 88200, 48000, 32000] {
        let x = forty_seconds(rate);
        let current = stream_analyze(&mut bt, &x, rate, 8192).unwrap();
        let old = v1.analyze_audio(&x, rate).unwrap();
        // Bit identity at 22 050 and 22 050 * 2^k, the D1 drift check (beats bit-identical) at the
        // 48 kHz class.
        assert_same_or_drift(&format!("stream @ {rate}"), rate, &current, &old);
    }
}

/// `stream` against the whole-buffer pipeline that `analyze_owned` ran before ticket 05 (whole
/// resample, windowed mel over the whole buffer), and against the routed 1.x methods.
#[test]
fn stream_matches_analyze_owned() {
    require_models!();
    let mut bt = new_bt();
    for rate in [22050u32, 44100, 88200, 48000, 32000] {
        let x = forty_seconds(rate);
        let whole = __probe::analyze_whole_buffer(&mut bt, x.clone(), rate).unwrap();
        // Push-size invariance is covered by `stream_push_size_invariance`; 8193 straddles the
        // resampler's chunk boundary.
        let got = stream_analyze(&mut bt, &x, rate, 8193).unwrap();
        assert_same_analysis(&format!("stream vs whole buffer @ {rate}"), &got, &whole);
        // `analyze_audio` shares `analyze_owned`'s route (one push of the whole signal).
        let owned = bt.analyze_owned(x, rate).unwrap();
        assert_same_analysis(&format!("analyze_owned @ {rate}"), &owned, &whole);
    }
}

/// Degradations that break the stream's code path rather than its input: the test-side
/// composition wired as production must match `stream` bit for bit, and each broken wiring must be
/// caught, both against the whole-buffer pipeline and as a push-size dependence.
#[test]
fn stream_degradations() {
    require_models!();
    let mut bt = new_bt();
    for rate in [22050u32, 44100, 48000] {
        let x = Synth::take(rate, 10 * rate as usize);
        let whole = __probe::analyze_whole_buffer(&mut bt, x.clone(), rate).unwrap();
        let real = stream_analyze(&mut bt, &x, rate, 8193).unwrap();
        let control = composed_stream(&mut bt, &x, rate, 8193, StreamPlan::Production).unwrap();
        assert_same_analysis(&format!("control @ {rate}"), &control, &real);
        assert_same_analysis(&format!("control vs whole @ {rate}"), &control, &whole);
        let broken: &[StreamPlan] = if rate == 22050 {
            &[StreamPlan::ZeroPushBoundary]
        } else {
            &[
                StreamPlan::ResamplerPerPush,
                StreamPlan::ZeroFlush,
                StreamPlan::ZeroPushBoundary,
            ]
        };
        for &plan in broken {
            let got = composed_stream(&mut bt, &x, rate, 8193, plan).unwrap();
            assert_differs(&format!("{plan:?} @ {rate} vs whole"), &got, &whole);
            if !matches!(plan, StreamPlan::ZeroFlush) {
                // Push-size dependent: the same wiring with a different push size differs too.
                let other = composed_stream(&mut bt, &x, rate, 4096, plan).unwrap();
                assert_differs(
                    &format!("{plan:?} @ {rate}, push 4096 vs 8193"),
                    &other,
                    &got,
                );
            }
        }
    }
}

#[test]
fn stream_edge_cases() {
    require_models!();
    let mut bt = new_bt();
    let mut v1 = v1_bt();

    // A rate of 0 is an error when the stream is created, as `analyze_audio` errs at that rate.
    assert!(bt.stream(0).is_err(), "stream(0) must be an error");
    assert!(bt.analyze_audio(&Synth::take(22050, 1000), 0).is_err());

    for rate in [22050u32, 48000] {
        // Nothing pushed, or only empty pushes: an error, like 1.0.0 on an empty signal.
        assert!(
            bt.stream(rate).unwrap().finish().is_err(),
            "no push @ {rate}"
        );
        let mut stream = bt.stream(rate).unwrap();
        stream.push(&[]).unwrap();
        stream.push(&[]).unwrap();
        assert!(stream.finish().is_err(), "empty pushes @ {rate}");
        assert!(v1.analyze_audio(&[], rate).is_err());

        let lengths: Vec<usize> = (0..=8).chain([440, 441, 442, 512, 513]).collect();
        let (mut ok, mut err) = (0, 0);
        for n in lengths {
            let x = Synth::take(rate, n);
            let old = v1.analyze_audio(&x, rate);
            let whole = __probe::analyze_whole_buffer(&mut bt, x.clone(), rate);
            for push in [1usize, n.max(1)] {
                let got = stream_analyze(&mut bt, &x, rate, push);
                let what = format!("stream n={n} push {push} @ {rate}");
                assert_eq!(got.is_err(), old.is_err(), "{what}: error vs 1.0.0");
                assert_eq!(got.is_err(), whole.is_err(), "{what}: error vs whole");
                if let (Ok(g), Ok(o), Ok(w)) = (&got, &old, &whole) {
                    assert_same_or_drift(&what, rate, g, o);
                    assert_same_analysis(&what, g, w);
                    ok += 1;
                } else {
                    err += 1;
                }
            }
        }
        assert!(ok > 0 && err > 0, "edge cases @ {rate}: {ok} ok, {err} err");
    }

    // An extreme downsampling ratio goes through the resampler's bounded flush and terminates with
    // the whole-buffer result.
    let sr = 4_000_000_000u32;
    let x = Synth::take(22050, 600_000);
    let got = stream_analyze(&mut bt, &x, sr, 8192);
    let whole = __probe::analyze_whole_buffer(&mut bt, x.clone(), sr);
    assert_eq!(got.is_err(), whole.is_err(), "4 GHz: error behaviour");
    if let (Ok(g), Ok(w)) = (&got, &whole) {
        assert_same_analysis("4 GHz", g, w);
    }
}

/// Memory held between pushes, outside the mel frames, is bounded by the chunk and window sizes,
/// whatever the input length and push size (no buffer whose capacity grows with the input).
#[test]
fn stream_retains_o_chunk() {
    require_models!();
    let mut bt = new_bt();
    // The mel window buffer (reserved once) plus a few resampler chunks.
    let bound = __probe::MEL_WINDOW_SAMPLES + 8 * __probe::RESAMPLE_CHUNK;
    for rate in [22050u32, 44100, 48000] {
        let n = 100 * rate as usize;
        let early = 50 * rate as usize;
        for push in [1000usize, 8192, 100_003, n] {
            let mut synth = Synth::new(rate);
            let mut piece = vec![0.0f32; push];
            let mut stream = bt.stream(rate).unwrap();
            let (mut max_early, mut max_all, mut fed) = (0usize, 0usize, 0usize);
            while fed < n {
                let take = push.min(n - fed);
                synth.fill(&mut piece[..take]);
                stream.push(&piece[..take]).unwrap();
                fed += take;
                let (pcm, frames) = __probe::stream_retained(&stream);
                assert!(
                    pcm <= bound,
                    "push {push} @ {rate}: {pcm} floats retained after {fed} samples (bound {bound})"
                );
                // The mel frames grow with the input, by doubling at worst.
                let emitted_max = (1 + fed * 22050 / rate as usize / 441) * 128;
                assert!(
                    frames <= 2 * emitted_max,
                    "push {push} @ {rate}: {frames} frame floats"
                );
                max_all = max_all.max(pcm);
                if fed <= early {
                    max_early = max_early.max(pcm);
                }
            }
            if push < early {
                assert!(
                    max_all <= max_early,
                    "push {push} @ {rate}: retained grew after 50 s ({max_early} -> {max_all})"
                );
            }
            eprintln!("push {push} @ {rate}: max retained {max_all} floats outside the mel frames");
            if push == 8192 {
                let analysis = stream.finish().unwrap();
                let pcm_len = if rate == 22050 {
                    n
                } else {
                    __probe::one_shot_len(n, rate)
                };
                assert_eq!(analysis.mel.shape[1], 1 + pcm_len / 441);
            }
        }
    }
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn stream_long() {
    require_models!();
    eprintln!("beat model: {}", beat_model_path());
    let mut bt = new_bt();
    let mut v1 = v1_bt();
    for rate in [48000u32, 44100] {
        let x = Synth::take(rate, 20 * 60 * rate as usize);
        let got = stream_analyze(&mut bt, &x, rate, 8192).unwrap();
        let whole = __probe::analyze_whole_buffer(&mut bt, x.clone(), rate).unwrap();
        assert_same_analysis(&format!("20 min @ {rate}: stream vs whole"), &got, &whole);
        drop(whole);
        let old = v1.analyze_audio(&x, rate).unwrap();
        assert_same_or_drift(
            &format!("20 min @ {rate}: stream vs 1.0.0"),
            rate,
            &got,
            &old,
        );
        let how = if rate_is_exact(rate) {
            "identical to 1.0.0"
        } else {
            "within drift bounds of 1.0.0, beats identical"
        };
        eprintln!("20 min @ {rate}: stream identical to the whole-buffer path; {how}");
    }
}
