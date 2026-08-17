import numpy as np
import pytest

from pipeline.job_store import InMemoryJobStore
from pipeline.stages import JobContext, stage_stt


class FakeRoleResult:
    role_mapping = {}
    method = "heuristic"
    applied = False
    result = None


class FakeRoleEngine:
    def resolve(self, segments):
        return FakeRoleResult()

    def apply_mapping(self, segments, resolution):
        pass


class FakeGemini:
    def resolve(self, segments):
        return None


class FakeSTT:
    def __init__(self):
        self.calls = 0

    def transcribe_words(self, audio, sr, language="hi"):
        self.calls += 1
        return [{"start": 0.1, "end": 0.4, "text": "namaste", "probability": 0.9}]

    def build_diarized_transcript(self, segments, words):
        return [
            {**s, "text": "namaste", "words": [{"start": 0.1, "end": 0.4, "word": "namaste", "probability": 0.9}]}
            for s in segments
        ]


@pytest.fixture
def ctx(tmp_path):
    import scipy.io.wavfile as wavfile
    path = tmp_path / "synthetic.wav"
    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wavfile.write(str(path), sr, audio)
    cache = {
        "segments": [{"start_time_s": 0.0, "end_time_s": 1.0, "speaker": "spk_0", "text": ""}],
        "clean_audio": np.zeros(sr * 2, dtype=np.float32),
        "sr": sr,
    }
    return JobContext(filepath=str(path), filename="synthetic.wav", cache=cache, job_store=InMemoryJobStore())


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    from sentiment.stt import transcript_cache
    monkeypatch.setattr(transcript_cache, "CACHE_DIR", str(tmp_path / "cache"))
    stt = FakeSTT()
    monkeypatch.setattr("pipeline.stages._get_stt", lambda: stt)
    monkeypatch.setattr("pipeline.stages._get_role_engine", lambda: FakeRoleEngine())
    monkeypatch.setattr("sentiment.role_resolver_llm.GeminiRoleResolver", FakeGemini)
    return stt


def test_stage_stt_first_run_calls_api_and_writes_cache(ctx, fakes):
    stage_stt(ctx)
    assert fakes.calls == 1
    assert ctx.cache["transcribed"] is True
    assert ctx.cache["segments"][0]["text"] == "namaste"


def test_stage_stt_second_run_hits_cache(ctx, fakes, tmp_path):
    stage_stt(ctx)
    assert fakes.calls == 1

    ctx2 = JobContext(filepath=ctx.filepath, filename=ctx.filename,
                      cache={**ctx.cache, "transcribed": False}, job_store=InMemoryJobStore())
    stage_stt(ctx2)
    assert fakes.calls == 1
    assert ctx2.cache["segments"][0]["text"] == "namaste"
    import os
    cache_files = os.listdir(str(tmp_path / "cache"))
    assert any(f.endswith("-hi.json") for f in cache_files)