use anyhow::{anyhow, Result};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::fs::File;
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Mono audio data at a known sample rate.
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Load an audio file, convert to mono, and resample to `target_sr`.
///
/// Supports MP3, WAV, FLAC, OGG, and other formats via symphonia.
/// Uses high-quality sinc resampling via rubato when the source rate
/// differs from `target_sr`.
pub fn load_audio(path: &Path, target_sr: u32) -> Result<AudioData> {
    let (samples, source_sr, channels) = decode(path)?;
    let mono = to_mono(&samples, channels);
    let resampled = resample(mono, source_sr, target_sr)?;

    Ok(AudioData {
        samples: resampled,
        sample_rate: target_sr,
    })
}

/// Decode an audio file into interleaved f32 samples.
/// Returns (samples, sample_rate, channel_count).
fn decode(path: &Path) -> Result<(Vec<f32>, u32, usize)> {
    let src = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    // Provide file extension hint for better format detection
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("no supported audio tracks"))?;

    let track_id = track.id;
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut samples: Vec<f32> = Vec::new();
    let mut source_sr = 0u32;
    let mut channels = 0usize;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(_)) => break,
            Err(err) => return Err(anyhow!(err)),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                source_sr = spec.rate;
                channels = spec.channels.count();

                let mut sample_buf =
                    SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                sample_buf.copy_interleaved_ref(decoded);
                samples.extend_from_slice(sample_buf.samples());
            }
            Err(Error::IoError(_)) => break,
            Err(Error::DecodeError(_)) => (), // skip corrupted packets
            Err(err) => return Err(anyhow!(err)),
        }
    }

    if source_sr == 0 {
        return Err(anyhow!("failed to decode any audio packets"));
    }

    Ok((samples, source_sr, channels))
}

/// Convert interleaved multi-channel audio to mono by averaging channels.
fn to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }
    samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Resample mono audio from `source_sr` to `target_sr` using sinc interpolation.
/// Returns samples unchanged if rates already match.
fn resample(samples: Vec<f32>, source_sr: u32, target_sr: u32) -> Result<Vec<f32>> {
    if source_sr == target_sr {
        return Ok(samples);
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        target_sr as f64 / source_sr as f64,
        2.0,
        params,
        samples.len(),
        1, // mono
    )?;

    let waves_out = resampler.process(&[samples], None)?;
    Ok(waves_out.into_iter().next().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_mono_passthrough() {
        let samples = vec![1.0, 2.0, 3.0];
        let mono = to_mono(&samples, 1);
        assert_eq!(mono, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_to_mono_stereo() {
        // Two frames of stereo: (0.5, 1.5) and (1.0, 3.0)
        let samples = vec![0.5, 1.5, 1.0, 3.0];
        let mono = to_mono(&samples, 2);
        assert_eq!(mono, vec![1.0, 2.0]);
    }

    #[test]
    fn test_resample_identity() {
        // Same rate should return unchanged
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let result = resample(samples.clone(), 22050, 22050).unwrap();
        assert_eq!(result, samples);
    }
}
