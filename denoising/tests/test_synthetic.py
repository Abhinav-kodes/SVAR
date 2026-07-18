import unittest
import numpy as np
from denoising.snr_calculator import calculate_snr
from denoising.clipping_detector import detect_clipping
from denoising.vad_basic import compute_vad
from denoising.silence_ratio import calculate_silence_ratio
from denoising.highpass_filter import highpass_filter
from denoising.notch_filter import notch_filter
from denoising.compressor import compress_dynamic_range
from denoising.declipper import declip_audio
from denoising.spectral_denoiser import wiener_denoise

class TestDenoisingComponents(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        self.duration = 2.0  # seconds
        self.t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        
    def test_snr_calculator_high_snr(self):
        # Pure sine wave (signal) for half the duration, silence for the rest
        signal = np.zeros_like(self.t)
        signal[:len(self.t)//2] = 0.5 * np.sin(2 * np.pi * 440 * self.t[:len(self.t)//2])
        # Very low noise everywhere
        noise = 0.001 * np.random.randn(len(self.t))
        audio = signal + noise
        
        snr = calculate_snr(audio, self.sr)
        self.assertGreater(snr, 30.0) # Expect high SNR
        
    def test_snr_calculator_low_snr(self):
        # Pure sine wave (signal) for half the duration, silence for the rest
        signal = np.zeros_like(self.t)
        signal[:len(self.t)//2] = 0.01 * np.sin(2 * np.pi * 440 * self.t[:len(self.t)//2])
        # High noise everywhere
        noise = 0.1 * np.random.randn(len(self.t))
        audio = signal + noise
        
        snr = calculate_snr(audio, self.sr)
        self.assertLess(snr, 10.0) # Expect low SNR
        
    def test_clipping_detector_no_clipping(self):
        signal = 0.5 * np.sin(2 * np.pi * 440 * self.t)
        clipping_ratio = detect_clipping(signal)
        self.assertEqual(clipping_ratio, 0.0)
        
    def test_clipping_detector_with_clipping(self):
        signal = 1.5 * np.sin(2 * np.pi * 440 * self.t)
        # Hard clip
        signal = np.clip(signal, -0.99, 0.99)
        clipping_ratio = detect_clipping(signal, threshold=0.98)
        self.assertGreater(clipping_ratio, 0.1)
        
    def test_vad_and_silence_ratio(self):
        # Create an audio with 1 second of silence, 1 second of speech (sine wave)
        silence = 0.001 * np.random.randn(self.sr)
        speech = 0.5 * np.sin(2 * np.pi * 440 * self.t[:self.sr]) + 0.001 * np.random.randn(self.sr)
        audio = np.concatenate([silence, speech])
        
        vad_mask = compute_vad(audio, self.sr, frame_duration_ms=25)
        
        # We expect roughly half of the frames to be silence and half speech
        total_frames = len(vad_mask)
        speech_frames = np.sum(vad_mask)
        
        self.assertAlmostEqual(speech_frames / total_frames, 0.5, delta=0.1)
        
        # Test silence ratio
        silence_ratio = calculate_silence_ratio(audio, self.sr, frame_duration_ms=25)
        self.assertAlmostEqual(silence_ratio, 0.5, delta=0.1)

    def test_highpass_filter(self):
        # 10 Hz (rumble) and 440 Hz (signal) mixed
        rumble = 0.5 * np.sin(2 * np.pi * 10 * self.t)
        signal = 0.5 * np.sin(2 * np.pi * 440 * self.t)
        audio = rumble + signal
        
        filtered = highpass_filter(audio, self.sr, cutoff_hz=80.0)
        
        # In frequency domain, 10Hz should be heavily attenuated
        # We can also check that the output has smaller variance/amplitude than the original
        # because the 10Hz component is removed.
        # Specifically, let's measure the correlation with the original rumble vs signal.
        # Or just compute FFT power at 10 Hz and 440 Hz.
        fft_orig = np.abs(np.fft.rfft(audio))
        fft_filt = np.abs(np.fft.rfft(filtered))
        
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)
        idx_10 = np.argmin(np.abs(freqs - 10))
        idx_440 = np.argmin(np.abs(freqs - 440))
        
        # 10 Hz power should drop significantly (by at least 80%)
        self.assertLess(fft_filt[idx_10], fft_orig[idx_10] * 0.2)
        # 440 Hz power should remain mostly intact
        self.assertGreater(fft_filt[idx_440], fft_orig[idx_440] * 0.9)

    def test_notch_filter(self):
        # 50 Hz (hum) and 440 Hz (signal) mixed
        hum = 0.5 * np.sin(2 * np.pi * 50 * self.t)
        signal = 0.5 * np.sin(2 * np.pi * 440 * self.t)
        audio = hum + signal
        
        filtered = notch_filter(audio, self.sr, notch_hz=50.0)
        
        fft_orig = np.abs(np.fft.rfft(audio))
        fft_filt = np.abs(np.fft.rfft(filtered))
        
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)
        idx_50 = np.argmin(np.abs(freqs - 50))
        idx_440 = np.argmin(np.abs(freqs - 440))
        
        # 50 Hz power should drop significantly
        self.assertLess(fft_filt[idx_50], fft_orig[idx_50] * 0.1)
        # 440 Hz power should remain mostly intact
        self.assertGreater(fft_filt[idx_440], fft_orig[idx_440] * 0.9)

    def test_compressor(self):
        # Create a signal that goes from soft to loud
        soft = 0.05 * np.sin(2 * np.pi * 440 * self.t[:len(self.t)//2])
        loud = 0.8 * np.sin(2 * np.pi * 440 * self.t[len(self.t)//2:])
        audio = np.concatenate([soft, loud])
        
        # Threshold at -20 dB (approx 0.1 amplitude)
        compressed = compress_dynamic_range(audio, self.sr, threshold_db=-20.0, ratio=4.0)
        
        # Soft part should be mostly unchanged (ratio 1:1 below threshold)
        soft_orig_rms = np.sqrt(np.mean(audio[:len(audio)//2]**2))
        soft_comp_rms = np.sqrt(np.mean(compressed[:len(compressed)//2]**2))
        self.assertAlmostEqual(soft_orig_rms, soft_comp_rms, delta=0.01)
        
        # Loud part should be significantly attenuated
        loud_orig_rms = np.sqrt(np.mean(audio[len(audio)//2:]**2))
        loud_comp_rms = np.sqrt(np.mean(compressed[len(compressed)//2:]**2))
        self.assertLess(loud_comp_rms, loud_orig_rms * 0.6)

    def test_declipper(self):
        # Create a clipped sine wave
        signal = 1.5 * np.sin(2 * np.pi * 440 * self.t)
        clipped = np.clip(signal, -0.99, 0.99)
        
        declipped = declip_audio(clipped, threshold=0.99)
        
        # Declipped wave peaks should be restored to exceed 0.99
        self.assertGreater(np.max(declipped), 0.99)
        self.assertLess(np.min(declipped), -0.99)

    def test_spectral_denoiser_reduction(self):
        # First 0.5s is silence (only noise), next 1.5s is sine wave (signal) + noise
        noise_std = 0.05
        noise = noise_std * np.random.randn(len(self.t))
        
        signal = np.zeros_like(self.t)
        signal[int(0.5 * self.sr):] = 0.5 * np.sin(2 * np.pi * 440 * self.t[int(0.5 * self.sr):])
        
        noisy_audio = signal + noise
        
        denoised = wiener_denoise(noisy_audio, self.sr, noise_duration_s=0.5)
        
        # Calculate SNR before and after denoising on the active segment
        active_noisy = noisy_audio[int(0.5 * self.sr):]
        active_denoised = denoised[int(0.5 * self.sr):]
        active_signal = signal[int(0.5 * self.sr):]
        
        noisy_noise = active_noisy - active_signal
        denoised_noise = active_denoised - active_signal
        
        p_signal = np.mean(active_signal ** 2)
        p_noisy_noise = np.mean(noisy_noise ** 2)
        p_denoised_noise = np.mean(denoised_noise ** 2)
        
        snr_before = 10 * np.log10(p_signal / p_noisy_noise)
        snr_after = 10 * np.log10(p_signal / p_denoised_noise)
        
        # Denoising should improve SNR by at least 3.0 dB
        self.assertGreater(snr_after, snr_before + 3.0)

if __name__ == '__main__':
    unittest.main()
