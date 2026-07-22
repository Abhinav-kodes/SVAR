import unittest
import numpy as np
from diarization.speaker_assigner import SpeakerAssigner, cosine_similarity


class TestSpeakerAssigner(unittest.TestCase):

    def test_cosine_similarity(self):
        """Test cosine similarity function."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0, delta=1e-5)
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0, delta=1e-5)

    def test_assign_speakers_two_roles(self):
        """Test speaker assignment initializes Agent greeting and separates distinct Customer voice."""
        sr = 16000
        t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)

        # 3 segments: low pitch (agent greeting), high pitch (customer response), low pitch (agent reply)
        seg1 = {"audio": 0.5 * np.sin(2 * np.pi * 120.0 * t)}
        seg2 = {"audio": 0.5 * np.sin(2 * np.pi * 450.0 * t)}
        seg3 = {"audio": 0.5 * np.sin(2 * np.pi * 122.0 * t)}

        assigner = SpeakerAssigner(new_speaker_threshold=0.98)
        assigned = assigner.assign_speakers([seg1, seg2, seg3], sr)

        self.assertEqual(len(assigned), 3)
        self.assertEqual(assigned[0]["speaker"], "agent")
        self.assertEqual(assigned[1]["speaker"], "customer")
        self.assertEqual(assigned[2]["speaker"], "agent")


if __name__ == "__main__":
    unittest.main()
