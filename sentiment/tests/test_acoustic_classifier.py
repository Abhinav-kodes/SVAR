import unittest
from sentiment.acoustic_emotion.delta_normalizer import compute_feature_deltas
from sentiment.acoustic_emotion.acoustic_emotion_classifier import (
    classify_acoustic_emotion,
    EMOTION_PROFILES
)


class TestAcousticEmotionClassifier(unittest.TestCase):

    def test_compute_feature_deltas(self):
        """Test standardized Z-score feature delta calculation."""
        current = {
            "pitch_mean": 220.0,
            "rms_mean": 0.05,
            "speaking_rate_zcr": 1200.0,
            "jitter": 0.02,
            "pause_ratio": 0.1
        }
        baseline = {
            "pitch_mean": 150.0,
            "rms_mean": 0.02,
            "speaking_rate_zcr": 800.0,
            "jitter": 0.01,
            "pause_ratio": 0.3
        }
        baseline_stds = {
            "pitch_mean": 35.0,
            "rms_mean": 0.01,
            "speaking_rate_zcr": 200.0,
            "jitter": 0.005,
            "pause_ratio": 0.1
        }

        deltas = compute_feature_deltas(current, baseline, baseline_stds)

        self.assertIn("pitch_d", deltas)
        self.assertIn("energy_d", deltas)
        self.assertIn("rate_d", deltas)
        self.assertIn("jitter_d", deltas)
        self.assertIn("pause_r", deltas)

        self.assertGreater(deltas["pitch_d"], 0.0)
        self.assertGreater(deltas["energy_d"], 0.0)
        self.assertLess(deltas["pause_r"], 0.0)

    def test_classify_anger_profile(self):
        """Test high pitch, high energy, fast rate delta maps to anger."""
        anger_deltas = {
            "pitch_d": 2.0,
            "energy_d": 2.0,
            "rate_d": 1.5,
            "jitter_d": 1.0,
            "pause_r": -2.0
        }

        result = classify_acoustic_emotion(anger_deltas, hif0_section="middle")

        self.assertEqual(result["emotion"], "anger")
        self.assertFalse(result["indeterminate"])
        self.assertGreater(result["confidence"], 0.15)

    def test_classify_sadness_profile(self):
        """Test low pitch, low energy, slow rate delta maps to sadness."""
        sadness_deltas = {
            "pitch_d": -2.0,
            "energy_d": -2.0,
            "rate_d": -2.0,
            "jitter_d": 0.5,
            "pause_r": 2.0
        }

        result = classify_acoustic_emotion(sadness_deltas, hif0_section="beginning")

        self.assertEqual(result["emotion"], "sadness")
        self.assertFalse(result["indeterminate"])


if __name__ == "__main__":
    unittest.main()
