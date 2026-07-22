import unittest
import numpy as np
from sentiment.stt.stt_transcriber import SpeechToTextTranscriber


class TestSpeechToTextTranscriber(unittest.TestCase):

    def test_transcribe_segment_empty(self):
        """Test STT handles empty audio input gracefully."""
        transcriber = SpeechToTextTranscriber()
        result = transcriber.transcribe_segment(np.array([], dtype=np.float32), 16000)

        self.assertEqual(result["text"], "")
        self.assertEqual(result["confidence"], 0.0)

    def test_transcribe_segment_synthetic_audio(self):
        """Test STT segment transcription output structure."""
        sr = 16000
        t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 300.0 * t)

        transcriber = SpeechToTextTranscriber()
        result = transcriber.transcribe_segment(audio, sr, language="hi")

        self.assertIn("text", result)
        self.assertIn("language", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["language"], "hi")
        self.assertIsInstance(result["text"], str)

    def test_transcribe_segments_list(self):
        """Test batch timeline segments transcription."""
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 200.0 * t)

        segments = [
            {"start_time_s": 0.0, "end_time_s": 0.5, "duration_s": 0.5, "audio": audio},
            {"start_time_s": 0.5, "end_time_s": 1.0, "duration_s": 0.5, "audio": audio}
        ]

        transcriber = SpeechToTextTranscriber()
        updated_segments = transcriber.transcribe_segments(segments, sr)

        self.assertEqual(len(updated_segments), 2)
        for seg in updated_segments:
            self.assertIn("transcript", seg)
            self.assertIn("stt_confidence", seg)


if __name__ == "__main__":
    unittest.main()
