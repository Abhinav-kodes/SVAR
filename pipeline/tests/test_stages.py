import numpy as np
import pytest

from pipeline.job_store import InMemoryJobStore
from pipeline.stages import JobContext, stage_denoise


@pytest.fixture
def ctx(tmp_path):
    import scipy.io.wavfile as wavfile
    path = tmp_path / "synthetic.wav"
    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wavfile.write(str(path), sr, audio)
    return JobContext(filepath=str(path), filename="synthetic.wav", cache={}, job_store=InMemoryJobStore())


def test_stage_denoise_populates_cache(ctx):
    stage_denoise(ctx)
    assert "audio" in ctx.cache
    assert "clean_audio" in ctx.cache
    assert "denoise_metrics" in ctx.cache
    assert ctx.cache["sr"] == 16000
    assert ctx.cache["duration"] == pytest.approx(2.0, abs=0.1)
    assert len(ctx.cache["clean_audio"]) == len(ctx.cache["audio"])


def test_stage_denoise_idempotent(ctx):
    stage_denoise(ctx)
    first = ctx.cache["denoise_metrics"]
    stage_denoise(ctx)
    assert ctx.cache["denoise_metrics"] is first