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


def test_transcribe_words_returns_api_result(monkeypatch):
    from sentiment.stt.stt_transcriber import SpeechToTextTranscriber
    stub = SpeechToTextTranscriber()
    expected = [{"start": 0.1, "end": 0.4, "text": "namaste", "probability": 0.9}]
    monkeypatch.setattr(stub, "_transcribe_api", lambda audio, sr, language="hi": expected)
    import numpy as np
    assert stub.transcribe_words(np.zeros(16000, dtype=np.float32), 16000, "hi") == expected


def test_transcribe_words_propagates_none(monkeypatch):
    from sentiment.stt.stt_transcriber import SpeechToTextTranscriber
    stub = SpeechToTextTranscriber()
    monkeypatch.setattr(stub, "_transcribe_api", lambda audio, sr, language="hi": None)
    import numpy as np
    assert stub.transcribe_words(np.zeros(16000, dtype=np.float32), 16000, "hi") is None


def test_build_diarized_transcript_maps_words():
    from sentiment.stt.stt_transcriber import SpeechToTextTranscriber
    stub = SpeechToTextTranscriber()
    segments = [
        {"start_time_s": 0.0, "end_time_s": 1.0, "speaker": "spk_0", "text": ""},
        {"start_time_s": 1.6, "end_time_s": 2.6, "speaker": "spk_0", "text": ""},
    ]
    words = [
        {"start": 0.1, "end": 0.4, "text": "namaste", "probability": 0.9},
        {"start": 1.8, "end": 2.1, "text": "ji", "probability": 0.8},
    ]
    out = stub.build_diarized_transcript(segments, words)
    assert out[0]["text"] == "namaste"
    assert out[0]["words"][0]["word"] == "namaste"
    assert out[1]["text"] == "ji"
    assert out[1]["duration_s"] == 1.0


def test_build_diarized_transcript_merges_small_gaps():
    from sentiment.stt.stt_transcriber import SpeechToTextTranscriber
    stub = SpeechToTextTranscriber()
    segments = [
        {"start_time_s": 0.0, "end_time_s": 1.0, "speaker": "spk_0", "text": ""},
        {"start_time_s": 1.2, "end_time_s": 2.0, "speaker": "spk_0", "text": ""},
    ]
    words = [{"start": 0.1, "end": 0.4, "text": "namaste", "probability": 0.9}]
    out = stub.build_diarized_transcript(segments, words)
    assert len(out) == 1


def test_build_diarized_transcript_empty_words_fallback():
    from sentiment.stt.stt_transcriber import SpeechToTextTranscriber
    stub = SpeechToTextTranscriber()
    segments = [
        {"start_time_s": 0.0, "end_time_s": 1.0, "speaker": "spk_0", "text": ""},
    ]
    out = stub.build_diarized_transcript(segments, None)
    assert len(out) == 1
    assert out[0]["text"] == ""
    assert out[0]["words"] == []
