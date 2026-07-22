import unittest
import numpy as np
from diarization.lpc_formant_estimator import estimate_formants_lpc, extract_segment_formants


class TestLPCFormantEstimator(unittest.TestCase):

    def test_estimate_formants_synthetic_sine(self):
        """Test formant estimation output on synthetic tone."""
        sr = 16000
        t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
        # Sum of 2 tones: 500 Hz (F1) and 1500 Hz (F2)
        signal = 0.5 * np.sin(2 * np.pi * 500.0 * t) + 0.3 * np.sin(2 * np.pi * 1500.0 * t)

        f1, f2 = estimate_formants_lpc(signal, sr)

        self.assertIsInstance(f1, float)
        self.assertIsInstance(f2, float)
        self.assertGreaterEqual(f1, 0.0)
        self.assertGreaterEqual(f2, 0.0)

    def test_estimate_formants_silence(self):
        """Test formant estimation returns (0.0, 0.0) on silence."""
        sr = 16000
        silent_frame = np.zeros(480, dtype=np.float32)
        f1, f2 = estimate_formants_lpc(silent_frame, sr)

        self.assertEqual(f1, 0.0)
        self.assertEqual(f2, 0.0)

    def test_extract_segment_formants_keys(self):
        """Test segment formant extraction dictionary output."""
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 600.0 * t)

        formants = extract_segment_formants(audio, sr)

        expected_keys = ["f1_mean", "f1_std", "f2_mean", "f2_std"]
        for key in expected_keys:
            self.assertIn(key, formants)
            self.assertFalse(np.isnan(formants[key]))


if __name__ == "__main__":
    unittest.main()
