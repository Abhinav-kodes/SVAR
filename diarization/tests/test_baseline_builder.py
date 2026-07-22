import unittest
import numpy as np
from diarization.baseline_builder import SpeakerBaselineTracker


class TestSpeakerBaselineTracker(unittest.TestCase):

    def test_set_and_get_baseline(self):
        """Test setting and retrieving a speaker baseline."""
        tracker = SpeakerBaselineTracker(vector_dim=32)
        fp = np.random.randn(32).astype(np.float32)
        fp = fp / np.linalg.norm(fp)

        tracker.set_baseline("agent", fp)
        self.assertTrue(tracker.has_baseline("agent"))
        self.assertFalse(tracker.has_baseline("customer"))

        retrieved = tracker.get_baseline("agent")
        self.assertAlmostEqual(np.linalg.norm(retrieved), 1.0, delta=1e-5)

    def test_update_baseline_ewa(self):
        """Test Exponential Weighted Average (EWA) baseline updates."""
        tracker = SpeakerBaselineTracker(vector_dim=32)
        fp1 = np.ones(32, dtype=np.float32) / np.sqrt(32)
        fp2 = np.zeros(32, dtype=np.float32)
        fp2[0] = 1.0

        tracker.set_baseline("agent", fp1)
        tracker.update_baseline("agent", fp2, alpha=0.5)

        updated = tracker.get_baseline("agent")
        norm = np.linalg.norm(updated)
        self.assertAlmostEqual(norm, 1.0, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
