import unittest
import numpy as np
from diarization.speaker_fingerprinter import extract_speaker_fingerprint


class TestSpeakerFingerprinter(unittest.TestCase):

    def test_extract_speaker_fingerprint_shape_and_norm(self):
        """Test speaker fingerprint vector shape, non-NaN, and L2 normalization."""
        sr = 16000
        duration_s = 1.0
        t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 250.0 * t)

        fingerprint = extract_speaker_fingerprint(audio, sr)

        self.assertIsInstance(fingerprint, np.ndarray)
        self.assertEqual(fingerprint.shape, (32,))
        self.assertFalse(np.isnan(fingerprint).any(), "Fingerprint contains NaN values")

        # L2 norm should be 1.0
        norm = np.linalg.norm(fingerprint)
        self.assertAlmostEqual(norm, 1.0, delta=1e-5)

    def test_extract_speaker_fingerprint_empty(self):
        """Test empty audio produces 32-dim zero vector."""
        fingerprint = extract_speaker_fingerprint(np.array([], dtype=np.float32), 16000)
        self.assertEqual(fingerprint.shape, (32,))
        self.assertEqual(np.linalg.norm(fingerprint), 0.0)

    def test_fingerprint_difference_between_different_pitches(self):
        """Test that different pitch tones produce distinct fingerprint vectors."""
        sr = 16000
        t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
        audio_low = np.sin(2 * np.pi * 120.0 * t)
        audio_high = np.sin(2 * np.pi * 400.0 * t)

        fp_low = extract_speaker_fingerprint(audio_low, sr)
        fp_high = extract_speaker_fingerprint(audio_high, sr)

        cosine_sim = np.dot(fp_low, fp_high)
        # Cosine similarity should be distinct (< 0.99)
        self.assertLess(cosine_sim, 0.99)


if __name__ == "__main__":
    unittest.main()
