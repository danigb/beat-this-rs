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

fn check_analyze_owned(rate: u32) {
    require_models!();
    let x = forty_seconds(rate);
    let current = new_bt().analyze_owned(x.clone(), rate).unwrap();
    let old = v1_bt().analyze_audio(&x, rate).unwrap();
    assert_same(&format!("analyze_owned @ {rate}"), &current, &old);
}

#[test]
fn analyze_owned_matches_v1_0_0_22050() {
    check_analyze_owned(22050);
}

#[test]
fn analyze_owned_matches_v1_0_0_44100() {
    check_analyze_owned(44100);
}

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
            drop(current);
            let owned = new.analyze_owned(x, rate).unwrap();
            assert_same(&format!("{minutes} min @ {rate} (owned)"), &owned, &old);
            eprintln!("{minutes} min @ {rate}: identical");
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
    assert!(d.max_abs <= 1e-6, "drift grew: {d:?}");
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

/// Analyse `x` (48 kHz-class input) through 1.0.0 (one-shot resample) and through the chunked
/// resampler, and assert that beats and downbeats are bit-identical.
fn compare_downstream(
    what: &str,
    x: &[f32],
    sr: u32,
    new: &mut NewBt,
    v1: &mut OldBt,
) -> DriftReport {
    let reference = reference_resample(x, sr);
    let streamed = stream_resample(x, sr, 8192);
    let resampled = bit_diff(&streamed, &reference);
    let old = v1.analyze_audio(&reference, 22050).unwrap();
    let current = new.analyze_audio(&streamed, 22050).unwrap();
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

/// The routed `resample` (chunked where exact, one-shot otherwise) against the 1.0.0 one-shot call,
/// including tiny inputs and the error behaviour.
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
            // Exact rates go through the chunked path and must match it; the others take the
            // one-shot path, which is the reference itself.
            if let (Ok(o), Ok(c)) = (old, new) {
                assert_bits_eq(&format!("routed resample @ {sr}, n={n}"), &c, &o);
            }
        }
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
