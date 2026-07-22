import unittest
import numpy as np
from diarization.pause_segmenter import segment_audio_by_pauses


class TestPauseSegmenter(unittest.TestCase):

    def test_segment_audio_by_pauses_synthetic(self):
        """Test pause segmentation on synthetic audio with silence gaps."""
        sr = 16000
        # Build audio: 1s tone + 0.6s silence + 1s tone
        t1 = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
        tone1 = np.sin(2 * np.pi * 220.0 * t1)
        silence = np.zeros(int(sr * 0.6), dtype=np.float32)
        t2 = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
        tone2 = np.sin(2 * np.pi * 440.0 * t2)

        audio = np.concatenate([tone1, silence, tone2])

        segments = segment_audio_by_pauses(audio, sr, min_pause_duration_s=0.4, min_segment_duration_s=0.5)
        # Should detect 2 speech segments
        self.assertEqual(len(segments), 2)
        self.assertGreater(segments[0]["duration_s"], 0.5)
        self.assertGreater(segments[1]["duration_s"], 0.5)

    def test_segment_audio_empty(self):
        """Test pause segmentation on empty input."""
        segments = segment_audio_by_pauses(np.array([], dtype=np.float32), 16000)
        self.assertEqual(segments, [])

    def test_segment_audio_continuous(self):
        """Test continuous audio without pauses returns a single segment."""
        sr = 16000
        t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
        tone = np.sin(2 * np.pi * 300.0 * t)
        segments = segment_audio_by_pauses(tone, sr, min_pause_duration_s=0.4)
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0]["duration_s"], 1.5, delta=0.2)


if __name__ == "__main__":
    unittest.main()
