import numpy as np
import pytest

from sentiment.stt.stt_transcriber import _build_vad_chunks

SR = 16000


def tone(duration_s, start_s=0.0, freq=440.0):
    n = int(round(duration_s * SR))
    t = np.linspace(0, duration_s, n, endpoint=False)
    audio = np.zeros(int(round(start_s * SR)) + n, dtype=np.float32)
    audio[int(round(start_s * SR)) :] = 0.1 * np.sin(2 * np.pi * freq * t)
    return audio


def test_vad_chunks_cover_only_speech_regions():
    audio = np.concatenate([
        np.zeros(int(1.0 * SR), dtype=np.float32),
        tone(1.0),
        np.zeros(int(1.0 * SR), dtype=np.float32),
        tone(1.0),
    ])
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    chunks = _build_vad_chunks(audio_int16, SR)

    assert len(chunks) == 2
    off1, off2 = chunks[0][1], chunks[1][1]
    assert off1 == pytest.approx(0.7, abs=0.05)
    assert off2 == pytest.approx(2.7, abs=0.05)
    assert len(chunks[0][0]) // 2 == pytest.approx(1.6 * SR, abs=0.05 * SR)
    assert len(chunks[1][0]) // 2 == pytest.approx(1.3 * SR, abs=0.05 * SR)


def test_vad_chunks_merge_small_gaps():
    audio = np.concatenate([
        tone(0.5),
        np.zeros(int(0.3 * SR), dtype=np.float32),
        tone(0.5),
    ])
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    chunks = _build_vad_chunks(audio_int16, SR)

    assert len(chunks) == 1
    assert chunks[0][1] == pytest.approx(0.0, abs=0.05)
    assert len(chunks[0][0]) // 2 == pytest.approx(1.3 * SR, abs=0.05 * SR)


def test_vad_chunks_split_long_region():
    audio = np.concatenate([
        np.zeros(int(8.0 * SR), dtype=np.float32),
        tone(52.0),
    ])
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    chunks = _build_vad_chunks(audio_int16, SR)

    assert len(chunks) == 2
    assert chunks[0][1] == pytest.approx(7.7, abs=0.05)
    assert chunks[1][1] == pytest.approx(57.7, abs=0.05)
    assert len(chunks[0][0]) // 2 == 50 * SR
    assert len(chunks[1][0]) // 2 == pytest.approx(2.3 * SR, abs=0.05 * SR)


def test_vad_chunks_drop_trailing_subchunk_under_1s():
    audio = np.concatenate([
        np.zeros(int(8.0 * SR), dtype=np.float32),
        tone(50.3),
    ])
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    chunks = _build_vad_chunks(audio_int16, SR)

    assert len(chunks) == 1
    assert chunks[0][1] == pytest.approx(7.7, abs=0.05)
    assert len(chunks[0][0]) // 2 == 50 * SR


def test_vad_chunks_all_silence():
    audio_int16 = np.zeros(int(4.0 * SR), dtype=np.int16)
    assert _build_vad_chunks(audio_int16, SR) == []