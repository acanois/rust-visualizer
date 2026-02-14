use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::sync::Arc;

pub struct FftProcessor {
    fft: Arc<dyn rustfft::Fft<f32>>,
    size: usize,
    num_bars: usize,
    window: Vec<f32>,
    scratch: Vec<Complex<f32>>,
}

impl FftProcessor {
    pub fn new(size: usize, num_bars: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(size);
        let scratch_len = fft.get_inplace_scratch_len();

        // Hann window — reduces spectral leakage at chunk boundaries
        let window: Vec<f32> = (0..size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos())
            })
            .collect();

        Self {
            fft,
            size,
            num_bars,
            window,
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
        }
    }

    /// Process raw audio samples and return `num_bars` magnitude values.
    ///
    /// The returned values are in arbitrary units — the caller should scale
    /// and smooth them before sending to the GPU.
    pub fn process(&mut self, samples: &[f32]) -> Vec<f32> {
        // Apply Hann window and convert to complex
        let mut buffer: Vec<Complex<f32>> = samples
            .iter()
            .take(self.size)
            .zip(self.window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();

        // Zero-pad if we received fewer samples than FFT size
        buffer.resize(self.size, Complex::new(0.0, 0.0));

        // Forward FFT (in-place)
        self.fft
            .process_with_scratch(&mut buffer, &mut self.scratch);

        // Compute magnitude spectrum (only positive frequencies = first half)
        let spectrum: Vec<f32> = buffer[..self.size / 2]
            .iter()
            .map(|c| c.norm() / self.size as f32)
            .collect();

        // Group frequency bins into bars
        self.group_into_bars(&spectrum)
    }

    /// Group FFT bins into `num_bars` using a power-law (quasi-logarithmic)
    /// mapping so that low frequencies get more bars than high frequencies.
    /// This matches how humans perceive pitch.
    fn group_into_bars(&self, spectrum: &[f32]) -> Vec<f32> {
        let n = spectrum.len();
        let mut bars = vec![0.0f32; self.num_bars];

        for i in 0..self.num_bars {
            let t0 = i as f32 / self.num_bars as f32;
            let t1 = (i + 1) as f32 / self.num_bars as f32;

            // Power of 2 gives a nice logarithmic-ish spread
            let start = (t0.powf(2.0) * n as f32) as usize;
            let end = (t1.powf(2.0) * n as f32) as usize;

            let start = start.min(n - 1);
            let end = end.max(start + 1).min(n);

            // Average magnitude across the bin range
            let sum: f32 = spectrum[start..end].iter().sum();
            bars[i] = sum / (end - start) as f32;
        }

        bars
    }
}
