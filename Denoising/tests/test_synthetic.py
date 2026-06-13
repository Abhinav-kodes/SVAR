import unittest
import numpy as np
from Denoising.snr_calculator import calculate_snr
from Denoising.clipping_detector import detect_clipping
from Denoising.vad_basic import compute_vad
from Denoising.silence_ratio import calculate_silence_ratio

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

if __name__ == '__main__':
    unittest.main()
