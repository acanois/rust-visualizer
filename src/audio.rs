use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Maximum number of mono samples to keep in the shared ring buffer.
/// Large enough to hold several FFT windows worth of data.
const MAX_BUFFER_SIZE: usize = 2048 * 4;

/// Shared ring buffer that the audio thread writes into and the render
/// loop reads from.
pub type SharedBuffer = Arc<Mutex<VecDeque<f32>>>;

/// Create a new shared buffer.
pub fn new_shared_buffer() -> SharedBuffer {
    Arc::new(Mutex::new(VecDeque::with_capacity(MAX_BUFFER_SIZE)))
}

// ---------------------------------------------------------------------------
// Device input (captures from default input device — e.g. BlackHole for
// Logic Pro routing, or any other virtual/hardware input)
// ---------------------------------------------------------------------------

/// Print available input devices to stdout so the user knows what's there.
pub fn list_input_devices() {
    let host = cpal::default_host();
    println!("Available input devices:");
    if let Ok(devices) = host.input_devices() {
        for (i, device) in devices.enumerate() {
            if let Ok(name) = device.name() {
                println!("  [{}] {}", i, name);
            }
        }
    }
    println!();
    println!("Using default input device.");
    println!("To visualize Logic Pro output, route it through a virtual audio");
    println!("device like BlackHole and set that as the default input.");
    println!();
    println!("Pass a .wav file path as an argument to visualize a file instead:");
    println!("  cargo run -- path/to/song.wav");
    println!();
}

/// Start capturing audio from the default system input device.
/// Returns a `cpal::Stream` that must be kept alive for the duration of capture.
pub fn start_input_capture(buffer: SharedBuffer) -> cpal::Stream {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("No input device available");

    println!("Capturing from: {}", device.name().unwrap_or_default());

    let supported = device
        .default_input_config()
        .expect("No default input config");
    let channels = supported.channels() as usize;
    let config: cpal::StreamConfig = supported.into();

    let stream = device
        .build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                push_samples(data, channels, &buffer);
            },
            |err| eprintln!("Audio input error: {err}"),
            None,
        )
        .expect("Failed to build input stream");

    stream.play().expect("Failed to start input stream");
    stream
}

// ---------------------------------------------------------------------------
// File playback (reads a WAV, plays through speakers, and feeds the
// visualizer simultaneously)
// ---------------------------------------------------------------------------

/// Load a WAV file, play it through the default output device, and
/// simultaneously feed samples into the shared buffer for visualization.
/// Returns a `cpal::Stream` that must be kept alive.
pub fn start_file_playback(path: &str, buffer: SharedBuffer) -> cpal::Stream {
    // ---- decode the WAV file ----
    let mut reader =
        hound::WavReader::open(path).unwrap_or_else(|e| panic!("Failed to open {path}: {e}"));
    let spec = reader.spec();
    println!(
        "Playing: {} ({}Hz, {} ch, {:?})",
        path, spec.sample_rate, spec.channels, spec.sample_format
    );

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .map(|s| s.unwrap() as f32 / i16::MAX as f32)
                .collect(),
            24 => reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / 8_388_607.0)
                .collect(),
            _ => reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / i32::MAX as f32)
                .collect(),
        },
    };

    let src_channels = spec.channels as usize;
    let samples = Arc::new(samples);
    let position = Arc::new(AtomicUsize::new(0));

    // ---- set up cpal output stream ----
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("No output device available");

    let out_supported = device
        .default_output_config()
        .expect("No default output config");
    let dst_channels = out_supported.channels() as usize;

    // Use the file's sample rate so pitch is correct.
    // Most devices accept 44100 / 48000 natively.
    let config = cpal::StreamConfig {
        channels: dst_channels as u16,
        sample_rate: cpal::SampleRate(spec.sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let samples_c = samples.clone();
    let position_c = position.clone();

    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut pos = position_c.load(Ordering::Relaxed);
                let total = samples_c.len();
                let frames_needed = data.len() / dst_channels;
                let mut mono_samples: Vec<f32> = Vec::with_capacity(frames_needed);

                for frame in 0..frames_needed {
                    // Wrap position back to start when we reach the end (loop)
                    if pos + src_channels > total {
                        pos = 0;
                    }

                    // Mix source channels → mono for visualization
                    let mono: f32 = (0..src_channels)
                        .map(|ch| samples_c[pos + ch])
                        .sum::<f32>()
                        / src_channels as f32;
                    mono_samples.push(mono);

                    // Write to output channels (duplicate / map as needed)
                    for ch in 0..dst_channels {
                        data[frame * dst_channels + ch] =
                            samples_c[pos + (ch % src_channels)];
                    }

                    pos += src_channels;
                }

                position_c.store(pos, Ordering::Relaxed);

                // Feed mono samples into the visualization buffer
                let mut buf = buffer.lock().unwrap();
                buf.extend(mono_samples);
                while buf.len() > MAX_BUFFER_SIZE {
                    buf.pop_front();
                }
            },
            |err| eprintln!("Audio output error: {err}"),
            None,
        )
        .expect("Failed to build output stream");

    stream.play().expect("Failed to start output stream");
    stream
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Push interleaved multi-channel samples into the shared ring buffer as mono.
fn push_samples(data: &[f32], channels: usize, buffer: &SharedBuffer) {
    let mut buf = buffer.lock().unwrap();
    if channels > 1 {
        for chunk in data.chunks(channels) {
            let mono: f32 = chunk.iter().sum::<f32>() / channels as f32;
            buf.push_back(mono);
        }
    } else {
        for &s in data {
            buf.push_back(s);
        }
    }
    while buf.len() > MAX_BUFFER_SIZE {
        buf.pop_front();
    }
}
