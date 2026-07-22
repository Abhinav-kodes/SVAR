import unittest
import numpy as np
from diarization.pipeline import DiarizationPipeline


class TestDiarizationPipeline(unittest.TestCase):

    def test_diarization_pipeline_execution(self):
        """Test end-to-end execution of DiarizationPipeline on synthetic audio."""
        sr = 16000
        # 1s low pitch (agent) + 0.5s pause + 1s high pitch (customer)
        t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
        silence = np.zeros(int(sr * 0.5), dtype=np.float32)

        audio1 = 0.5 * np.sin(2 * np.pi * 120.0 * t)
        audio2 = 0.5 * np.sin(2 * np.pi * 450.0 * t)

        full_audio = np.concatenate([audio1, silence, audio2])

        pipeline = DiarizationPipeline(new_speaker_threshold=0.99)
        results = pipeline.process(full_audio, sr)

        self.assertIn("talk_ratio", results)
        self.assertIn("speakers", results)
        self.assertIn("segments", results)

        talk_ratio = results["talk_ratio"]
        self.assertGreater(talk_ratio["total_speech_s"], 0.0)
        self.assertEqual(len(results["segments"]), 2)

    def test_diarization_pipeline_empty(self):
        """Test pipeline gracefully handles empty audio input."""
        pipeline = DiarizationPipeline()
        results = pipeline.process(np.array([], dtype=np.float32), 16000)

        self.assertEqual(results["talk_ratio"]["total_speech_s"], 0.0)
        self.assertEqual(results["segments"], [])


if __name__ == "__main__":
    unittest.main()
