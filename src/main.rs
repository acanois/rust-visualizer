mod audio;
mod fft;
mod renderer;

use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

// ---- Tuning knobs (change these to taste) ----------------------------------

/// Number of samples fed into each FFT frame.
const FFT_SIZE: usize = 2048;
/// Number of bars drawn on screen.
const NUM_BARS: usize = 88;
/// Smoothing factor for bar decay (0 = instant, 1 = frozen). Higher = slower.
const DECAY: f32 = 0.88;
/// Gain applied to raw FFT magnitudes before display.
const GAIN: f32 = 6.0;
/// Maximum bar height in clip-space units (screen goes from -1 to +1).
const MAX_HEIGHT: f32 = 2.0;

// ----------------------------------------------------------------------------

enum AudioSource {
    /// Capture from the default system input device.
    Device,
    /// Play a WAV file and visualize it.
    File(String),
}

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<renderer::Renderer>,
    // Must keep the stream alive or audio stops
    _audio_stream: Option<cpal::Stream>,
    sample_buffer: audio::SharedBuffer,
    fft_processor: fft::FftProcessor,
    smoothed: Vec<f32>,
    audio_source: AudioSource,
}

impl App {
    fn new(audio_source: AudioSource) -> Self {
        Self {
            window: None,
            renderer: None,
            _audio_stream: None,
            sample_buffer: audio::new_shared_buffer(),
            fft_processor: fft::FftProcessor::new(FFT_SIZE, NUM_BARS),
            smoothed: vec![0.0; NUM_BARS],
            audio_source,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Only initialise once
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("Audio Visualizer")
            .with_inner_size(LogicalSize::new(1200, 600));

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("Failed to create window"),
        );

        let renderer = pollster::block_on(renderer::Renderer::new(window.clone(), NUM_BARS as u32));

        // Start the audio stream
        let stream = match &self.audio_source {
            AudioSource::Device => audio::start_input_capture(self.sample_buffer.clone()),
            AudioSource::File(path) => audio::start_file_playback(path, self.sample_buffer.clone()),
        };

        self._audio_stream = Some(stream);
        self.renderer = Some(renderer);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(size);
                }
            }

            WindowEvent::RedrawRequested => {
                // ---- grab the latest samples from the ring buffer ----
                let samples: Vec<f32> = {
                    let buf = self.sample_buffer.lock().unwrap();
                    if buf.len() < FFT_SIZE {
                        // Not enough data yet — render what we have (silence)
                        if let Some(r) = &mut self.renderer {
                            r.render(&self.smoothed);
                        }
                        return;
                    }
                    // Take the most recent FFT_SIZE samples
                    buf.iter().rev().take(FFT_SIZE).copied().collect::<Vec<_>>()
                };
                // Reverse because we collected in reverse order
                let samples: Vec<f32> = samples.into_iter().rev().collect();

                // ---- FFT → bar magnitudes ----
                let raw = self.fft_processor.process(&samples);

                // ---- smooth with exponential decay ----
                for (i, &mag) in raw.iter().enumerate() {
                    let scaled = (mag * GAIN).min(MAX_HEIGHT);
                    if scaled > self.smoothed[i] {
                        // Attack: jump up instantly
                        self.smoothed[i] = scaled;
                    } else {
                        // Decay: fade down smoothly
                        self.smoothed[i] *= DECAY;
                    }
                }

                // ---- render ----
                if let Some(r) = &mut self.renderer {
                    r.render(&self.smoothed);
                }
            }

            _ => {}
        }
    }

    /// Called after all pending events have been processed.
    /// We use this to request continuous redraws (~vsync rate).
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();

    // If a file is passed in the arguemnts, load it
    let audio_source = if args.len() > 1 {
        AudioSource::File(args[1].clone())
    } else {
        // Otherwise, load use the default system device
        audio::list_input_devices();
        AudioSource::Device
    };

    let event_loop = EventLoop::new().expect("Failed to create event loop");

    // Create a new app and pass in the audio source
    let mut app = App::new(audio_source);
    event_loop.run_app(&mut app).expect("Event loop error");
}
