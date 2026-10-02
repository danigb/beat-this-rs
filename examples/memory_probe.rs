//! Stage-by-stage memory probe for the front end (resample -> mel -> beat model).
//!
//! ```text
//! cargo run --release --example memory_probe -- --minutes N --stage <stage> [--rate 48000] [--model PATH]
//! ```
//!
//! Stages run the chain up to and including themselves, in one process, so the single line printed
//! at the end is that stage's high-water mark. Every stage loads the models first, so each line
//! includes the fixed cost of the models:
//!
//! - `models`: load both models.
//! - `resample`: synthesise native-rate mono, then `to_vec` + `resample` as `analyze_audio` does.
//! - `resample-stream`: push the synthetic signal in 8192-frame pieces through `StreamResampler`,
//!   collecting the output; the native-rate input never exists whole.
//! - `mel-input`: ... then build the mel graph's input (the `Tensor` and rten's input value)
//!   without running the graph. Splits the PCM copies from the graph intermediates.
//! - `mel`: ... then run the mel graph.
//! - `full`: synthesise native-rate mono, then `analyze_audio` (the caller keeps its buffer).
//! - `full-owned`: as `full`, through `analyze_owned`; the caller does not keep the buffer.
//!
//! The caller's native-rate mono buffer stays alive through every stage except `full-owned`, as it
//! does for a caller of the borrowed `analyze_audio`. `mel-input` and `mel` take the resampled
//! samples by value, as the pipeline does.
//!
//! The probe reaches the crate's private pieces through the `#[doc(hidden)] pub mod __probe`
//! re-export. That keeps it measuring the real code after later tickets change it, which a
//! copy of `resample` in this file would not. `__probe` is unstable and not covered by semver.
//!
//! The input is synthesised (`__probe::Synth`), so no corpus is needed. Peak memory is
//! `ru_maxrss` from `getrusage(RUSAGE_SELF)`: bytes on macOS, KiB on Linux.

use std::hint::black_box;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use beat_this::__probe::{self, Synth};
use beat_this::{BeatThis, RtenRuntime};

const MEL_MODEL: &str = "models/mel_spectrogram.onnx";
const FULL_MODEL: &str = "models/beat_this.onnx";
const SMALL_MODEL: &str = "models/beat_this_small.onnx";

struct Args {
    minutes: f64,
    stage: String,
    rate: u32,
    model: Option<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut minutes = None;
    let mut stage = None;
    let mut rate = 48_000u32;
    let mut model = None;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |name: &str| it.next().with_context(|| format!("{name} needs a value"));
        match flag.as_str() {
            "--minutes" => minutes = Some(value("--minutes")?.parse()?),
            "--stage" => stage = Some(value("--stage")?),
            "--rate" => rate = value("--rate")?.parse()?,
            "--model" => model = Some(PathBuf::from(value("--model")?)),
            other => bail!("unknown argument '{other}'"),
        }
    }
    Ok(Args {
        minutes: minutes.context("--minutes is required")?,
        stage: stage.context("--stage is required")?,
        rate,
        model,
    })
}

/// Peak resident set size of this process so far, in MiB.
fn peak_rss_mib() -> f64 {
    // SAFETY: getrusage writes into a zeroed, properly sized rusage struct we own; RUSAGE_SELF is
    // always valid.
    let maxrss = unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        let rc = libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        assert_eq!(rc, 0, "getrusage failed");
        usage.ru_maxrss as f64
    };
    #[cfg(target_os = "macos")]
    let bytes = maxrss;
    #[cfg(not(target_os = "macos"))]
    let bytes = maxrss * 1024.0;
    bytes / (1024.0 * 1024.0)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let beat_model = match &args.model {
        Some(p) => p.clone(),
        None if Path::new(FULL_MODEL).exists() => PathBuf::from(FULL_MODEL),
        None => PathBuf::from(SMALL_MODEL),
    };
    let n = (args.minutes * 60.0 * args.rate as f64) as usize;

    let mut bt = BeatThis::new(&RtenRuntime, Path::new(MEL_MODEL), &beat_model)?;

    match args.stage.as_str() {
        "models" => {}
        "resample" | "mel-input" | "mel" => {
            let mono = Synth::take(args.rate, n);
            let pcm = __probe::resample(mono.to_vec(), args.rate)?;
            match args.stage.as_str() {
                "resample" => {
                    black_box(&pcm);
                }
                "mel-input" => {
                    black_box(__probe::mel_input(pcm)?);
                }
                _ => {
                    black_box(__probe::mel(&mut bt, pcm)?);
                }
            }
            black_box(&mono);
        }
        "resample-stream" => {
            // The synthetic input is generated in 8192-frame pieces and pushed through the
            // chunked resampler; the whole input never exists.
            let mut synth = Synth::new(args.rate);
            let mut resampler = __probe::StreamResampler::new(args.rate, 22050)?;
            let mut pcm = Vec::with_capacity(__probe::one_shot_len(n, args.rate));
            let mut piece = vec![0.0f32; 8192];
            let mut left = n;
            while left > 0 {
                let take = left.min(piece.len());
                synth.fill(&mut piece[..take]);
                resampler.push(&piece[..take], &mut pcm)?;
                left -= take;
            }
            resampler.finish(&mut pcm)?;
            black_box(&pcm);
        }
        "full" => {
            let mono = Synth::take(args.rate, n);
            black_box(bt.analyze_audio(&mono, args.rate)?);
            black_box(&mono);
        }
        "full-owned" => {
            // The caller hands the buffer over, so it is not alive alongside the pipeline.
            let mono = Synth::take(args.rate, n);
            black_box(bt.analyze_owned(mono, args.rate)?);
        }
        other => bail!("unknown stage '{other}' (models|resample|resample-stream|mel-input|mel|full|full-owned)"),
    }

    println!(
        "stage={} minutes={} rate={} model={} peak_rss_mib={:.1}",
        args.stage,
        args.minutes,
        args.rate,
        beat_model
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("?"),
        peak_rss_mib()
    );
    Ok(())
}
