import json
import os
from datetime import datetime, timedelta

import pytest

from sentiment.stt import transcript_cache


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript_cache, "CACHE_DIR", str(tmp_path / "transcripts"))
    return transcript_cache.CACHE_DIR


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "sample.wav"
    path.write_bytes(b"hello world")
    return str(path)


def test_sha1_of_file_known_content(sample_file):
    assert transcript_cache.sha1_of_file(sample_file) == "2aae6c35c94fcfb415dbe95f408b9ce91ee846ed"


def test_roundtrip(cache_dir):
    words = [{"start": 0.1, "end": 0.4, "text": "namaste", "probability": 0.9}]
    transcript_cache.put_words("abc123", "hi", words)
    assert transcript_cache.get_words("abc123", "hi") == words
    assert os.path.exists(os.path.join(cache_dir, "abc123-hi.json"))


def test_language_separation(cache_dir):
    hi = [{"start": 0.1, "end": 0.4, "text": "namaste", "probability": 0.9}]
    en = [{"start": 0.2, "end": 0.5, "text": "hello", "probability": 0.8}]
    transcript_cache.put_words("abc123", "hi", hi)
    transcript_cache.put_words("abc123", "en", en)
    assert transcript_cache.get_words("abc123", "hi") == hi
    assert transcript_cache.get_words("abc123", "en") == en


def test_fingerprint_mismatch_is_miss(cache_dir):
    path = os.path.join(cache_dir, "abc123-hi.json")
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "config": {"model": "some_other_model", "language": "hi",
                       "vad_enabled": True, "VAD_FRAME_MS": 25, "VAD_PADDING_S": 0.3,
                       "VAD_GAP_TOLERANCE_S": 0.5, "CHUNK_SECONDS": 50},
            "created_at": datetime.utcnow().isoformat(),
            "words": [{"start": 0.1, "end": 0.4, "text": "namaste", "probability": 0.9}],
        }, f)
    assert transcript_cache.get_words("abc123", "hi") is None


def test_ttl_expiry_deletes_entry(cache_dir, monkeypatch):
    monkeypatch.setattr(transcript_cache, "CACHE_TTL_DAYS", 1)
    path = os.path.join(cache_dir, "abc123-hi.json")
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "config": {"model": "chirp_3", "language": "hi",
                       "vad_enabled": True, "VAD_FRAME_MS": 25, "VAD_PADDING_S": 0.3,
                       "VAD_GAP_TOLERANCE_S": 0.5, "CHUNK_SECONDS": 50},
            "created_at": (datetime.utcnow() - timedelta(days=2)).isoformat(),
            "words": [{"start": 0.1, "end": 0.4, "text": "namaste", "probability": 0.9}],
        }, f)
    assert transcript_cache.get_words("abc123", "hi") is None
    assert not os.path.exists(path)


def test_corrupt_file_is_miss(cache_dir):
    path = os.path.join(cache_dir, "abc123-hi.json")
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write("{not valid json!!!")
    assert transcript_cache.get_words("abc123", "hi") is None


def test_no_tmp_left_after_put(cache_dir):
    transcript_cache.put_words("abc123", "hi", [{"start": 0.1, "end": 0.4, "text": "namaste", "probability": 0.9}])
    leftovers = [p for p in os.listdir(cache_dir) if p.endswith(".tmp")]
    assert leftovers == []