//! Beat and downbeat tracking with the [Beat This!](https://github.com/CPJKU/beat_this) model.
//!
//! [`BeatThis`] runs the whole pipeline: resampling to 22 050 Hz mono, a log-mel spectrogram, the
//! beat model on 30-second chunks, and peak picking into beat and downbeat times in seconds.
//!
//! ```no_run
//! # #[cfg(feature = "decode")]
//! # fn main() -> anyhow::Result<()> {
//! use beat_this::{BeatThis, RtenRuntime};
//! use std::path::Path;
//!
//! let mut bt = BeatThis::new(
//!     &RtenRuntime,
//!     Path::new("models/mel_spectrogram.onnx"),
//!     Path::new("models/beat_this.onnx"),
//! )?;
//! let analysis = bt.analyze_file(Path::new("input.wav"))?;
//! println!("{} beats", analysis.beats.len());
//! # Ok(()) }
//! # #[cfg(not(feature = "decode"))]
//! # fn main() {}
//! ```
//!
//! # Long inputs
//!
//! [`BeatThis::analyze_audio`] needs the whole signal in memory. To analyse long audio while it
//! is decoded, push mono chunks into a [`BeatStream`] instead: peak memory is then O(chunk) plus
//! the mel spectrogram, which grows by about 92 MB per hour of audio. The result is bit-identical
//! to `analyze_audio` on the same samples, whatever the chunk sizes.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use beat_this::{BeatThis, RtenRuntime};
//! use std::path::Path;
//!
//! let mut bt = BeatThis::new(
//!     &RtenRuntime,
//!     Path::new("models/mel_spectrogram.onnx"),
//!     Path::new("models/beat_this.onnx"),
//! )?;
//! # let decoded_chunks: Vec<Vec<f32>> = Vec::new();
//! let mut stream = bt.stream(48_000)?;
//! for chunk in decoded_chunks {
//!     // Mono f32 at 48 kHz, any chunk size.
//!     stream.push(&chunk)?;
//! }
//! let analysis = stream.finish()?;
//! println!("{} beats, {} downbeats", analysis.beats.len(), analysis.downbeats.len());
//! # Ok(()) }
//! ```

#[doc(hidden)]
#[path = "probe.rs"]
pub mod __probe;
mod audio;
mod inference;
mod mel;
mod output;
mod postprocessing;
mod runtime;
mod stream;

use std::path::Path;
use std::time::Duration;

use anyhow::Result;

#[cfg(feature = "decode")]
pub use audio::{load_audio, AudioData};
pub use output::{beat_counts, calculate_bpm};
#[cfg(feature = "ort")]
pub use runtime::ort::OrtRuntime;
pub use runtime::{rten::RtenRuntime, Model, Runtime, Tensor};
pub use stream::BeatStream;

use inference::BeatPredictor;
use mel::MelExtractor;
use postprocessing::PeakPicker;

/// Target sample rate expected by the mel spectrogram model.
const TARGET_SAMPLE_RATE: u32 = 22050;

/// Full analysis result from the beat tracking pipeline.
#[derive(Debug, Clone)]
pub struct BeatAnalysis {
    /// Beat times in seconds (sorted, deduplicated).
    pub beats: Vec<f32>,
    /// Downbeat times in seconds (sorted, deduplicated, snapped to nearest beat).
    pub downbeats: Vec<f32>,
    /// Mel spectrogram tensor with shape `[1, T, 128]` at 50 fps.
    pub mel: Tensor,
    /// Raw beat logits, one per spectrogram frame.
    pub beat_logits: Vec<f32>,
    /// Raw downbeat logits, one per spectrogram frame.
    pub downbeat_logits: Vec<f32>,
}

/// High-level beat tracker composing the full pipeline.
///
/// Owns the mel spectrogram model, the beat prediction model, and the
/// peak picker. Generic over the model type, so it works
/// with any backend (ort, rten, tract).
pub struct BeatThis<M: Model> {
    mel: MelExtractor<M>,
    predictor: BeatPredictor<M>,
    peak_picker: PeakPicker,
}

/// Per-stage timing from [`BeatThis::analyze_audio_timed`] and [`BeatStream::finish_timed`].
/// `mel` does not include resampling.
#[derive(Debug, Clone)]
pub struct AnalysisTiming {
    pub mel: Duration,
    pub predict: Duration,
    pub decode: Duration,
}

/// Analysis result with optional per-stage timing.
#[derive(Debug, Clone)]
pub struct TimedAnalysis {
    pub analysis: BeatAnalysis,
    pub timing: AnalysisTiming,
}

impl<M: Model> BeatThis<M> {
    /// Create a new beat tracker by loading both ONNX models via the given runtime.
    ///
    /// - `runtime`: any `Runtime` (e.g. `OrtRuntime::default()`)
    /// - `mel_model_path`: path to the mel spectrogram ONNX model
    /// - `beat_model_path`: path to the beat tracking ONNX model
    pub fn new<R: Runtime<Model = M>>(
        runtime: &R,
        mel_model_path: &Path,
        beat_model_path: &Path,
    ) -> Result<Self> {
        let mel_model = runtime.load_model(mel_model_path)?;
        let beat_model = runtime.load_model(beat_model_path)?;

        Ok(Self {
            mel: MelExtractor::new(mel_model),
            predictor: BeatPredictor::new(beat_model),
            peak_picker: PeakPicker::default(),
        })
    }

    /// Create a beat tracker from pre-built models.
    ///
    /// Use this when you need separate runtimes for the mel and beat models
    /// (e.g. one with profiling enabled).
    pub fn from_models(mel_model: M, beat_model: M) -> Self {
        Self {
            mel: MelExtractor::new(mel_model),
            predictor: BeatPredictor::new(beat_model),
            peak_picker: PeakPicker::default(),
        }
    }

    /// Get a mutable reference to the beat prediction model.
    ///
    /// Useful for runtime-specific operations like ending ORT profiling.
    pub fn beat_model_mut(&mut self) -> &mut M {
        self.predictor.model_mut()
    }

    /// Run the full pipeline on raw audio samples.
    ///
    /// The samples are resampled to 22050 Hz if `sample_rate` differs.
    /// Input should be mono f32 PCM.
    ///
    /// The slice is not copied: it runs through [`stream`](Self::stream) as a single push, so the
    /// same result comes from feeding the signal in chunks. For long inputs that are decoded
    /// incrementally, use [`stream`](Self::stream) directly so the whole signal never has to be in
    /// memory.
    pub fn analyze_audio(&mut self, samples: &[f32], sample_rate: u32) -> Result<BeatAnalysis> {
        Ok(self.analyze_audio_timed(samples, sample_rate)?.analysis)
    }

    /// Run the full pipeline on raw audio samples, returning per-stage timing.
    pub fn analyze_audio_timed(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<TimedAnalysis> {
        let mut stream = self.stream(sample_rate)?;
        stream.push(samples)?;
        stream.finish_timed()
    }

    /// Run the full pipeline on owned mono f32 samples.
    ///
    /// Same output as [`analyze_audio`](Self::analyze_audio), bit for bit. The buffer is consumed
    /// and freed as soon as it has been fed through [`stream`](Self::stream), before the beat
    /// model runs.
    pub fn analyze_owned(&mut self, samples: Vec<f32>, sample_rate: u32) -> Result<BeatAnalysis> {
        Ok(self.analyze_owned_timed(samples, sample_rate)?.analysis)
    }

    /// Run the full pipeline on owned mono f32 samples, returning per-stage timing.
    pub fn analyze_owned_timed(
        &mut self,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> Result<TimedAnalysis> {
        let mut stream = self.stream(sample_rate)?;
        stream.push(&samples)?;
        drop(samples);
        stream.finish_timed()
    }

    /// Start a streaming analysis of mono f32 audio at `sample_rate`: push the signal in chunks of
    /// any size with [`BeatStream::push`], then call [`BeatStream::finish`].
    ///
    /// This is the way to analyse long inputs: peak memory is O(chunk) plus the mel spectrogram
    /// (about 92 MB per hour of audio), instead of the whole signal. The result is bit-identical
    /// to [`analyze_audio`](Self::analyze_audio) on the concatenated chunks. Errors if
    /// `sample_rate` is 0.
    ///
    /// The stream mutably borrows this `BeatThis` for its lifetime (so it cannot be stored beside
    /// it in one struct); with the rten model it is `Send`, so it can be moved to another thread.
    pub fn stream(&mut self, sample_rate: u32) -> Result<BeatStream<'_, M>> {
        BeatStream::new(self, sample_rate)
    }

    /// Beat model and peak picking on a whole mel spectrogram, timed as `analyze_audio_timed`
    /// reports them. `mel_time` is the time the mel stage took.
    fn predict_and_decode(&mut self, mel: Tensor, mel_time: Duration) -> Result<TimedAnalysis> {
        let t = std::time::Instant::now();
        let (beat_logits, downbeat_logits) = self.predictor.predict(&mel)?;
        let predict_time = t.elapsed();

        let t = std::time::Instant::now();
        let (beats, downbeats) = self.peak_picker.decode(&beat_logits, &downbeat_logits)?;
        let decode_time = t.elapsed();

        Ok(TimedAnalysis {
            analysis: BeatAnalysis {
                beats,
                downbeats,
                mel,
                beat_logits,
                downbeat_logits,
            },
            timing: AnalysisTiming {
                mel: mel_time,
                predict: predict_time,
                decode: decode_time,
            },
        })
    }

    /// Run the full pipeline on an audio file.
    ///
    /// Loads the file, resamples to 22050 Hz mono, computes mel spectrogram,
    /// runs beat prediction, and decodes into beat/downbeat timestamps.
    #[cfg(feature = "decode")]
    pub fn analyze_file(&mut self, path: &Path) -> Result<BeatAnalysis> {
        let audio = load_audio(path, TARGET_SAMPLE_RATE)?;
        self.analyze_owned(audio.samples, audio.sample_rate)
    }
}
