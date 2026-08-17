# Disk Transcript Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache the raw Chirp 3 word stream on disk keyed by source-file sha1 so re-analysis after job-store loss skips the paid STT API.

**Architecture:** New pure module `sentiment/stt/transcript_cache.py` (sha1 → JSON word stream, TTL + config-fingerprint invalidation, atomic writes). `stage_stt` computes sha1, consults cache, and only calls the API on miss; the raw word stream is mapped to diarized segments locally either way.

**Tech Stack:** Python 3 stdlib only (hashlib, json, os, datetime) — no new dependencies, no google imports in the cache module.

## Global Constraints

- `sentiment/stt/transcript_cache.py` must NOT import google.cloud (pure module).
- Python: `venv/bin/python`; tests: `venv/bin/python -m pytest <file> -q`.
- Full suite outcome must remain: **66 existing tests pass, 3 skipped, 4 pre-existing failures** (diarization pipeline execution + 3 STT method-drift tests). No new failures allowed.
- Config fingerprint fields (exact names): `model`, `language`, `vad_enabled`, `VAD_FRAME_MS`, `VAD_PADDING_S`, `VAD_GAP_TOLERANCE_S`, `CHUNK_SECONDS`.
- Cache file layout: `data/transcripts/<sha1>-<lang>.json` = `{"config": {...}, "created_at": "<ISO>", "words": [...]}`.
- Env override: `SVAR_TRANSCRIPT_CACHE_DIR`. Module reads it ONCE at import into `CACHE_DIR`; tests monkeypatch `transcript_cache.CACHE_DIR` directly.
- No comments in code unless the existing file's style has them (stt_transcriber.py has docstrings — use docstrings, not inline comments).

---

### Task 1: `transcript_cache.py` module + unit tests

**Files:**
- Create: `sentiment/stt/transcript_cache.py`
- Create: `sentiment/tests/test_transcript_cache.py`

**Interfaces:**
- Consumes: `sentiment/stt/stt_transcriber.py` constants `VAD_FRAME_MS` (25), `VAD_PADDING_S` (0.3), `VAD_GAP_TOLERANCE_S` (0.5), `CHUNK_SECONDS` (50) — import at top of transcript_cache.py (module has no google imports; safe).
- Produces:
  - `sha1_of_file(path: str) -> str`
  - `get_words(sha1: str, lang: str) -> Optional[List[Dict[str, Any]]]`
  - `put_words(sha1: str, lang: str, words: List[Dict[str, Any]]) -> None`
  - `CACHE_DIR: str`, `CACHE_TTL_DAYS: int = 30`

- [ ] **Step 1: Write the failing tests**

Create `sentiment/tests/test_transcript_cache.py`:

```python
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
    path = os.path.join(cache_dir, "abc123-hi.json")
    with open(path, "w") as f:
        f.write("{not valid json!!!")
    assert transcript_cache.get_words("abc123", "hi") is None


def test_no_tmp_left_after_put(cache_dir):
    transcript_cache.put_words("abc123", "hi", [{"start": 0.1, "end": 0.4, "text": "namaste", "probability": 0.9}])
    leftovers = [p for p in os.listdir(cache_dir) if p.endswith(".tmp")]
    assert leftovers == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest sentiment/tests/test_transcript_cache.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentiment.stt.transcript_cache'`

- [ ] **Step 3: Write the minimal implementation**

Create `sentiment/stt/transcript_cache.py`:

```python
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sentiment.stt.stt_transcriber import (
    CHUNK_SECONDS,
    VAD_FRAME_MS,
    VAD_GAP_TOLERANCE_S,
    VAD_PADDING_S,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_DIR = os.environ.get(
    "SVAR_TRANSCRIPT_CACHE_DIR",
    os.path.join(_REPO_ROOT, "data", "transcripts"),
)
CACHE_TTL_DAYS = 30
_MODEL = "chirp_3"


def sha1_of_file(path: str) -> str:
    """Streamed sha1 hexdigest of a file's bytes."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_fingerprint(lang: str) -> Dict[str, Any]:
    vad_disabled = os.environ.get("STT_VAD_DISABLED", "").lower() in ("1", "true", "yes")
    return {
        "model": _MODEL,
        "language": lang,
        "vad_enabled": not vad_disabled,
        "VAD_FRAME_MS": VAD_FRAME_MS,
        "VAD_PADDING_S": VAD_PADDING_S,
        "VAD_GAP_TOLERANCE_S": VAD_GAP_TOLERANCE_S,
        "CHUNK_SECONDS": CHUNK_SECONDS,
    }


def _entry_path(sha1: str, lang: str) -> str:
    return os.path.join(CACHE_DIR, f"{sha1}-{lang}.json")


def get_words(sha1: str, lang: str) -> Optional[List[Dict[str, Any]]]:
    """Return the cached word stream, or None on miss/corrupt/expired/mismatch."""
    path = _entry_path(sha1, lang)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        created = datetime.fromisoformat(data["created_at"])
        if datetime.utcnow() - created > timedelta(days=CACHE_TTL_DAYS):
            os.remove(path)
            return None
        if data.get("config") != _config_fingerprint(lang):
            return None
        words = data.get("words")
        if not isinstance(words, list):
            return None
        return words
    except Exception as e:
        print(f"  [transcript-cache] read failed for {path}: {e}")
        return None


def put_words(sha1: str, lang: str, words: List[Dict[str, Any]]) -> None:
    """Atomically write the word stream (tmp + os.replace); failures never raise."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = _entry_path(sha1, lang)
        tmp_path = path + ".tmp"
        payload = {
            "config": _config_fingerprint(lang),
            "created_at": datetime.utcnow().isoformat(),
            "words": words,
        }
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"  [transcript-cache] write failed for {sha1}-{lang}: {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest sentiment/tests/test_transcript_cache.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add sentiment/stt/transcript_cache.py sentiment/tests/test_transcript_cache.py
git commit -m "feat: add disk transcript cache for STT word streams"
```

---

### Task 2: Refactor `stt_transcriber.py` — `transcribe_words` + `build_diarized_transcript`

**Files:**
- Modify: `sentiment/stt/stt_transcriber.py` (lines 223-299: `transcribe_diarized_segments`)
- Modify: `sentiment/tests/test_stt_transcriber.py` (append tests)

**Interfaces:**
- Consumes: existing `self._transcribe_api(audio, sr, language) -> Optional[List[Dict]]` (line 132), static `_merge_segments(segments, max_gap=0.5)` (line 301).
- Produces:
  - `transcribe_words(audio: np.ndarray, sr: int, language: str = "hi") -> Optional[List[Dict[str, Any]]]` — returns `_transcribe_api` result unchanged (may be None)
  - `build_diarized_transcript(segments: List[Dict[str, Any]], words: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]` — word→segment mapping (calls `self._merge_segments(segments, max_gap=0.5)` first; returns empty-segment fallback when `words` falsy)
  - `transcribe_diarized_segments` reimplemented as a thin convenience calling both (behavior identical to today)

- [ ] **Step 1: Write the failing tests**

Append to `sentiment/tests/test_stt_transcriber.py`:

```python
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
        {"start_time_s": 1.0, "end_time_s": 2.0, "speaker": "spk_0", "text": ""},
    ]
    words = [
        {"start": 0.1, "end": 0.4, "text": "namaste", "probability": 0.9},
        {"start": 1.2, "end": 1.5, "text": "ji", "probability": 0.8},
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest sentiment/tests/test_stt_transcriber.py -q -k "transcribe_words or build_diarized_transcript"`
Expected: FAIL — `AttributeError: 'SpeechToTextTranscriber' object has no attribute 'transcribe_words'`

- [ ] **Step 3: Implement the refactor**

In `sentiment/stt/stt_transcriber.py`, replace the body of `transcribe_diarized_segments` (current lines 230-299, everything after the docstring line 229) with:

```python
        merged = self._merge_segments(diarization_segments, max_gap=0.5)
        words = self._transcribe_api(full_audio, sr, language)
        return self.build_diarized_transcript(merged, words)

    def transcribe_words(
        self,
        audio: np.ndarray,
        sr: int,
        language: str = "hi",
    ) -> Optional[List[Dict[str, Any]]]:
        """Transcribe audio via Chirp 3 and return the raw word stream."""
        return self._transcribe_api(audio, sr, language)

    def build_diarized_transcript(
        self,
        segments: List[Dict[str, Any]],
        words: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Map a raw word stream onto diarized segments (local, free)."""
        merged = self._merge_segments(segments, max_gap=0.5)

        if not words:
            return [
                {
                    "start_time_s": d.get("start_time_s", 0),
                    "end_time_s": d.get("end_time_s", 0),
                    "text": "",
                    "speaker": d.get("speaker", "spk_0"),
                    "words": [],
                    "avg_logprob": 0,
                    "no_speech_prob": 0,
                }
                for d in merged
            ]

        word_to_seg = [None] * len(words)
        for wi, w in enumerate(words):
            best_overlap = 0
            best_idx = -1
            for di, dseg in enumerate(merged):
                d_start = dseg.get("start_time_s", 0)
                d_end = dseg.get("end_time_s", 0)
                overlap = max(0, min(w["end"], d_end) - max(w["start"], d_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = di
            if best_idx >= 0:
                word_to_seg[wi] = best_idx

        seg_word_lists = [[] for _ in merged]
        for wi, w in enumerate(words):
            idx = word_to_seg[wi]
            if idx is not None:
                seg_word_lists[idx].append(w)

        output = []
        for di, dseg in enumerate(merged):
            d_start = dseg.get("start_time_s", 0)
            d_end = dseg.get("end_time_s", 0)
            seg_words = seg_word_lists[di]
            text = " ".join(w.get("text", "") for w in seg_words).strip()

            entry = {
                "start_time_s": d_start,
                "end_time_s": d_end,
                "duration_s": round(d_end - d_start, 6),
                "text": text,
                "speaker": dseg.get("speaker", "spk_0"),
                "words": [
                    {
                        "start": round(w["start"], 3),
                        "end": round(w["end"], 3),
                        "word": w["text"],
                        "probability": w["probability"],
                    }
                    for w in seg_words
                ],
                "avg_logprob": round(
                    sum(w["probability"] for w in seg_words) / max(len(seg_words), 1), 3
                ),
                "no_speech_prob": 0,
            }
            for field in ("confidence", "uncertain", "sb_margin"):
                if field in dseg:
                    entry[field] = dseg[field]
            output.append(entry)

        return output
```

IMPORTANT: the `merged`-then-mapping logic moved into `build_diarized_transcript` is a VERBATIM extraction of the old code — do not change its behavior. The new `transcribe_diarized_segments` merges first (it must keep that behavior: it merges BEFORE transcribing and maps onto merged segments), then calls `build_diarized_transcript(merged, words)` which merges AGAIN (a no-op on already-merged input, preserving identical output).

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest sentiment/tests/test_stt_transcriber.py -q -k "transcribe_words or build_diarized_transcript"`
Expected: 5 passed

Also confirm no behavior regression in the wider suite (counts must match the 4 pre-existing failures):

Run: `venv/bin/python -m pytest -q`
Expected: 66 passed, 3 skipped, 4 failed (same failures as before — diarization pipeline execution + 3 STT method-drift)

- [ ] **Step 5: Commit**

```bash
git add sentiment/stt/stt_transcriber.py sentiment/tests/test_stt_transcriber.py
git commit -m "feat: extract transcribe_words and build_diarized_transcript from transcribe_diarized_segments"
```

---

### Task 3: Wire cache into `stage_stt` + pipeline stage test

**Files:**
- Modify: `pipeline/stages.py` — `stage_stt` (lines 118-127, before the role-resolution block)
- Create: `pipeline/tests/test_stage_stt_cache.py`

**Interfaces:**
- Consumes: `transcript_cache.sha1_of_file`, `transcript_cache.get_words`, `transcript_cache.put_words` (Task 1); `stt.transcribe_words`, `stt.build_diarized_transcript` (Task 2); `JobContext(filepath, filename, cache, job_store)` (stages.py:12-17).
- Produces: `stage_stt` cache-hit path (no API call, `c["segments"]` from cached words); cache-miss path (API + `put_words` only when words truthy).

- [ ] **Step 1: Write the failing test**

Create `pipeline/tests/test_stage_stt_cache.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest pipeline/tests/test_stage_stt_cache.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentiment.stt.transcript_cache'`

- [ ] **Step 3: Implement the wiring**

In `pipeline/stages.py`, replace lines 124-127 (the `stt = _get_stt()` through `c["transcribed"] = True` block):

```python
    stt = _get_stt()
    from sentiment.stt import transcript_cache
    sha1 = transcript_cache.sha1_of_file(ctx.filepath)
    words = transcript_cache.get_words(sha1, "hi")
    if words is None:
        words = stt.transcribe_words(c["clean_audio"], c["sr"], language="hi")
        if words:
            transcript_cache.put_words(sha1, "hi", words)
        log(f"  [stt] Chirp 3 API ({len(words) if words else 0} words)")
    else:
        log(f"  [stt] transcript cache hit ({len(words)} words)")
    c["segments"] = stt.build_diarized_transcript(c["segments"], words)
    c["transcribed"] = True
```

The role-resolution block below (lines 129-161) stays untouched.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest pipeline/tests/test_stage_stt_cache.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/stages.py pipeline/tests/test_stage_stt_cache.py
git commit -m "feat: serve stage_stt transcripts from disk cache with sha1 key"
```

---

### Task 4: Docs + .gitignore + full suite verification

**Files:**
- Modify: `Roadmap.md`
- Modify: `README.md`
- Modify: `.gitignore`

**Interfaces:** None (docs only).

- [ ] **Step 1: Update Roadmap.md**

In the STT Cost Optimization section, mark the disk transcript cache item done:
- Change `🔴 disk transcript cache` to `✅ disk transcript cache` (the summary-table row: `| STT Cost Optimization | 🟡 50% | ✅ VAD-gated chunks (STT_VAD_DISABLED=1 escape hatch), 🔴 disk transcript cache |` becomes `🟢 100%` / `✅ ... , ✅ disk transcript cache |`).
- Add a bullet under the section: `- ✅ Disk transcript cache: data/transcripts/<sha1>-<lang>.json keyed by source-file sha1, 30-day TTL, config fingerprint invalidation, atomic writes (SVAR_TRANSCRIPT_CACHE_DIR overrides location)`.
- In Remaining Work, mark the "disk transcript cache" item ✅ Complete; renumber remaining items down (Phase 2 → 3, Phase 3 → 4, Phase 4 → 5).

- [ ] **Step 2: Update README.md**

Add to the env-var table (after the `STT_VAD_DISABLED=1` row):

```
| SVAR_TRANSCRIPT_CACHE_DIR | Override the disk transcript cache location (default: data/transcripts/) |
```

- [ ] **Step 3: Update .gitignore**

Under the `# Data & Audio` block add:

```
data/transcripts/
```

- [ ] **Step 4: Run the full suite**

Run: `venv/bin/python -m pytest -q`
Expected: **80 passed, 3 skipped, 4 failed** (66 prior + 14 new: 7 transcript_cache + 5 stt refactor + 2 stage). The 4 failures must be exactly the pre-existing 4 (diarization pipeline execution + 3 STT method-drift) — no new failures.

- [ ] **Step 5: Commit**

```bash
git add Roadmap.md README.md .gitignore
git commit -m "docs: mark disk transcript cache complete"
```

---

## Self-Review

**Spec coverage:**
- `transcript_cache.py` module (sha1_of_file / get_words / put_words / CACHE_DIR / TTL) → Task 1 ✓
- Fingerprint invalidation + TTL deletion + corrupt→miss → Task 1 tests 4-6 ✓
- Atomic .tmp + os.replace, no .tmp left → Task 1 test 7 ✓
- `transcribe_words` + `build_diarized_transcript` + thin `transcribe_diarized_segments` → Task 2 ✓
- stage_stt wiring with miss/hit logging + not caching API failures → Task 3 ✓
- Stage-level first-run/second-run test with monkeypatched transcribe_words + tmp cache dir → Task 3 ✓
- Error handling (read→miss+log, write→catch+continue) → built into get_words/put_words (Task 1) ✓
- Docs (Roadmap ✅, README env row, .gitignore) → Task 4 ✓

**Placeholder scan:** No TBD/TODO; every step has concrete code and commands.

**Type consistency:** `sha1_of_file(path)->str`, `get_words(sha1, lang)->Optional[List[Dict]]`, `put_words(sha1, lang, words)->None`, `transcribe_words(audio, sr, language="hi")->Optional[List[Dict]]`, `build_diarized_transcript(segments, words)->List[Dict]` — identical names/signatures across Tasks 1-3. `CACHE_DIR`/`CACHE_TTL_DAYS` match spec. Language key is `"hi"` everywhere.
---

## Deviations recorded during execution

- **Task 2, `test_build_diarized_transcript_maps_words`:** initial expectation was wrong — segments `[0.0, 1.0]` and `[1.0, 2.0]` (gap 0.0 ≤ max_gap 0.5) are merged by `_merge_segments`, so both words landed in one segment (`"namaste ji"`). Fixed the test to use segments `[0.0, 1.0]` / `[1.6, 2.6]` (gap 0.6 > 0.5) with the second word at `[1.8, 2.1]` — consistent with the plan's own `merges_small_gaps` test. Implementation was faithful to the original code.
- **Task 4:** also flipped the Roadmap section header `## STT Cost Optimization 🟡 PARTIALLY DONE` → `🟢 COMPLETE` for consistency with the summary-table row.
