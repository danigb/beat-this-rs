use anyhow::{anyhow, Result};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use std::fs::File;
use std::path::Path;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

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

    let mut format = symphonia::default::get_probe().probe(
        &hint,
        mss,
        FormatOptions::default(),
        MetadataOptions::default(),
    )?;

    let track = format
        .first_track_known_codec(TrackType::Audio)
        .ok_or_else(|| anyhow!("no supported audio tracks"))?;
    let track_id = track.id;
    let codec_params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| anyhow!("audio track missing codec parameters"))?;

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(codec_params, &AudioDecoderOptions::default())?;

    let mut samples: Vec<f32> = Vec::new();
    let mut scratch: Vec<f32> = Vec::new();
    let mut source_sr = 0u32;
    let mut channels = 0usize;

    // `next_packet` returns `Ok(None)` at end of stream in symphonia 0.6.
    while let Some(packet) = format.next_packet()? {
        if packet.track_id != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = decoded.spec();
                source_sr = spec.rate();
                channels = spec.channels().count();

                // `copy_to_vec_interleaved` resizes `scratch` to exactly this
                // packet's sample count, so we append it to the running buffer.
                decoded.copy_to_vec_interleaved(&mut scratch);
                samples.extend_from_slice(&scratch);
            }
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
pub fn resample(samples: Vec<f32>, source_sr: u32, target_sr: u32) -> Result<Vec<f32>> {
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

    // rubato 3.0: `SincFixedIn` became `Async` with `FixedAsync::Input`. We
    // process the whole buffer in a single call (chunk_size = input length),
    // matching the previous one-shot behavior.
    let frames = samples.len();
    let mut resampler = Async::<f32>::new_sinc(
        target_sr as f64 / source_sr as f64,
        2.0,
        &params,
        frames,
        1, // mono
        FixedAsync::Input,
    )?;

    // For mono, interleaved layout is just the flat sample slice.
    let input = InterleavedSlice::new(&samples, 1, frames)
        .map_err(|e| anyhow!("resampler input adapter: {e:?}"))?;
    let output = resampler.process(&input, 0, None)?;
    Ok(output.take_data())
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
