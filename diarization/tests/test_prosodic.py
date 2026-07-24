import unittest
import numpy as np
from diarization.prosodic_extractor import (
    extract_pitch_autocorrelation,
    extract_prosodic_features
)


class TestProsodicExtractor(unittest.TestCase):

    def test_pitch_autocorrelation_sine(self):
        """Test F0 estimation on a pure synthetic 200 Hz sine wave."""
        sr = 16000
        target_f0 = 200.0
        duration_s = 0.05
        t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
        sine_wave = np.sin(2 * np.pi * target_f0 * t)

        estimated_f0 = extract_pitch_autocorrelation(sine_wave, sr, voicing_threshold=0.3)
        # Should estimate within +/- 5 Hz of 200 Hz
        self.assertAlmostEqual(estimated_f0, target_f0, delta=5.0)

    def test_pitch_autocorrelation_silence(self):
        """Test F0 estimation on pure silence."""
        sr = 16000
        silent_frame = np.zeros(480, dtype=np.float32)
        estimated_f0 = extract_pitch_autocorrelation(silent_frame, sr)
        self.assertEqual(estimated_f0, 0.0)

    def test_extract_prosodic_features_shape_and_keys(self):
        """Test extraction returns all expected dictionary keys and a 32-dim vector."""
        sr = 16000
        duration_s = 1.0
        t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 150.0 * t)

        results = extract_prosodic_features(audio, sr)

        expected_keys = [
            "pitch_mean", "pitch_std", "pitch_min", "pitch_max",
            "rms_mean", "rms_std", "rms_min", "rms_max",
            "speaking_rate_zcr", "jitter", "shimmer", "pause_ratio",
            "compression_ratio", "hif0_relative_pos", "hif0_section",
            "voiced_ratio", "vector"
        ]

        for key in expected_keys:
            self.assertIn(key, results)

        vector = results["vector"]
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape, (32,))
        self.assertFalse(np.isnan(vector).any(), "Vector contains NaN values")

    def test_extract_prosodic_features_empty(self):
        """Test extractor handles empty audio gracefully."""
        results = extract_prosodic_features(np.array([], dtype=np.float32), 16000)
        self.assertEqual(results["vector"].shape, (32,))
        self.assertEqual(results["pitch_mean"], 0.0)
        self.assertEqual(results["rms_mean"], 0.0)

    def test_hif0_section_categorization(self):
        """Test HiF0 section calculation for peak at beginning/middle/end."""
        sr = 16000
        duration_s = 0.9
        t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
        # Create audio with tone only in first third
        audio = np.zeros_like(t)
        tone = np.sin(2 * np.pi * 220.0 * t[:int(sr * 0.2)])
        audio[:len(tone)] = tone

        results = extract_prosodic_features(audio, sr)
        self.assertIn(results["hif0_section"], ["beginning", "middle", "end"])


if __name__ == "__main__":
    unittest.main()
