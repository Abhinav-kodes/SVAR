# VAD-Gated STT Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Only send speech-containing chunks of a call to Google Chirp 3, cutting billed STT seconds by the call's silence ratio.

**Architecture:** A pure chunk-selection layer in `sentiment/stt/stt_transcriber.py`: `_build_vad_chunks` slices audio around speech regions from the existing `denoising.vad_basic.compute_vad` (25ms RMS frames), merging gaps ≤ 0.5s and padding regions by 0.3s; `_select_chunks` picks VAD-gated or legacy full-audio chunking based on `STT_VAD_DISABLED`; `_transcribe_api` calls `_select_chunks` and is otherwise unchanged (chunk offsets stay absolute seconds, so word timestamps and downstream diarization mapping are unaffected).

**Tech Stack:** Python 3, numpy, pytest, existing `sentiment/stt/stt_transcriber.py` + `denoising/vad_basic.py`.

**Spec:** `docs/superpowers/specs/2026-08-17-vad-gated-stt-chunking-design.md`

## Global Constraints

- Chunk offsets must remain absolute seconds in the original audio (word timestamps computed as `offset + intra-chunk time` — never relative to the region).
- Constants: `VAD_FRAME_MS = 25`, `VAD_PADDING_S = 0.3`, `VAD_GAP_TOLERANCE_S = 0.5`; existing `CHUNK_SECONDS = 50` reused for max chunk length.
- Trailing sub-chunk shorter than 1s (`sr` samples) is dropped — same rule as today.
- `STT_VAD_DISABLED=1` (also `true`/`yes`, case-insensitive) falls back to the exact legacy full-audio chunking.
- No speech → empty chunk list → `_transcribe_api` returns `None` (existing behavior).
- `_transcribe_api` itself is unchanged apart from replacing the chunk-building loop with `_select_chunks(audio_int16, sr)`; no network in tests.
- Test command: `venv/bin/python -m pytest <test file> -v` (from repo root). One conventional commit per task. Do NOT touch `sentiment/tests/test_stt_transcriber.py` (pre-existing failures, out of scope).

---

### Task 1: `_build_vad_chunks` pure function

**Files:**
- Modify: `sentiment/stt/stt_transcriber.py:11-14` (add constants)
- Create: `sentiment/tests/test_vad_chunking.py`

**Interfaces:**
- Consumes: `denoising.vad_basic.compute_vad(audio, sr, frame_duration_ms, threshold_multiplier)` returning a bool mask per frame.
- Produces: `_build_vad_chunks(audio_int16: np.ndarray, sr: int) -> List[Tuple[bytes, float]]` — list of `(chunk_bytes, absolute_offset_s)` covering only speech regions. Task 2 consumes it.

- [ ] **Step 1: Write the failing tests**

Create `sentiment/tests/test_vad_chunking.py`:

```python
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
    assert len(chunks[0][0]) == pytest.approx(1.6 * SR, abs=0.05 * SR)
    assert len(chunks[1][0]) == pytest.approx(1.3 * SR, abs=0.05 * SR)


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
    assert len(chunks[0][0]) == pytest.approx(1.3 * SR, abs=0.05 * SR)


def test_vad_chunks_split_long_region():
    audio = tone(52.0)
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    chunks = _build_vad_chunks(audio_int16, SR)

    assert len(chunks) == 2
    assert chunks[0][1] == pytest.approx(0.0, abs=0.05)
    assert chunks[1][1] == pytest.approx(50.0, abs=0.05)
    assert len(chunks[0][0]) == 50 * SR
    assert len(chunks[1][0]) == pytest.approx(2.0 * SR, abs=0.05 * SR)


def test_vad_chunks_drop_trailing_subchunk_under_1s():
    audio = tone(50.3)
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    chunks = _build_vad_chunks(audio_int16, SR)

    assert len(chunks) == 1
    assert chunks[0][1] == pytest.approx(0.0, abs=0.05)
    assert len(chunks[0][0]) == 50 * SR


def test_vad_chunks_all_silence():
    audio_int16 = np.zeros(int(4.0 * SR), dtype=np.int16)
    assert _build_vad_chunks(audio_int16, SR) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest sentiment/tests/test_vad_chunking.py -v`
Expected: FAIL — `ImportError: cannot import name '_build_vad_chunks'`.

- [ ] **Step 3: Implement the pure function**

In `sentiment/stt/stt_transcriber.py`, add constants after line 14 (`MAX_STT_WORKERS = 5`):

```python
VAD_FRAME_MS = 25
VAD_PADDING_S = 0.3
VAD_GAP_TOLERANCE_S = 0.5
```

Change the typing import on line 3 to include `Tuple`:

```python
from typing import List, Dict, Any, Optional, Tuple
```

Add after the `MAX_STT_WORKERS` constants (before `class SpeechToTextTranscriber`):

```python
def _build_vad_chunks(audio_int16: np.ndarray, sr: int) -> List[Tuple[bytes, float]]:
    """Slice audio into <=CHUNK_SECONDS chunks covering only speech regions.

    Speech regions come from denoising.vad_basic.compute_vad (VAD_FRAME_MS
    frames). Gaps <= VAD_GAP_TOLERANCE_S between speech frames are merged;
    each region is padded by VAD_PADDING_S on both sides (clamped to audio
    bounds). Returns (chunk_bytes, absolute_offset_s) pairs.
    """
    from denoising.vad_basic import compute_vad

    vad = compute_vad(audio_int16, sr, frame_duration_ms=VAD_FRAME_MS)
    if not vad.any():
        return []

    frame_s = VAD_FRAME_MS / 1000.0
    gap_frames = max(1, int(round(VAD_GAP_TOLERANCE_S / frame_s)))

    regions = []
    start = None
    prev = -1
    for i, active in enumerate(vad):
        if active:
            if start is None or i - prev > gap_frames + 1:
                if start is not None:
                    regions.append((start, prev))
                start = i
            prev = i
    if start is not None:
        regions.append((start, prev))

    duration_s = len(audio_int16) / sr
    chunks = []
    chunk_samples = CHUNK_SECONDS * sr
    for start_f, end_f in regions:
        region_start = max(0.0, start_f * frame_s - VAD_PADDING_S)
        region_end = min(duration_s, (end_f + 1) * frame_s + VAD_PADDING_S)
        seg = audio_int16[int(region_start * sr) : int(region_end * sr)]
        if len(seg) == 0:
            continue
        for i in range(0, len(seg), chunk_samples):
            piece = seg[i : i + chunk_samples]
            if len(piece) < sr:
                break
            chunks.append((piece.tobytes(), region_start + i / sr))
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest sentiment/tests/test_vad_chunking.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add sentiment/stt/stt_transcriber.py sentiment/tests/test_vad_chunking.py
git commit -m "feat: build STT chunks only around VAD speech regions"
```

---

### Task 2: Wire `_select_chunks` into `_transcribe_api`

**Files:**
- Modify: `sentiment/stt/stt_transcriber.py:102-108` (chunk loop in `_transcribe_api`)

**Interfaces:**
- Consumes: `_build_vad_chunks(audio_int16, sr)` from Task 1; existing `CHUNK_SECONDS`, `os` module.
- Produces: `_select_chunks(audio_int16: np.ndarray, sr: int) -> List[Tuple[bytes, float]]` — VAD-gated by default, legacy full-audio when `STT_VAD_DISABLED` is set. `_transcribe_api` uses it; behavior with a transcript present is otherwise unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `sentiment/tests/test_vad_chunking.py`:

```python
from sentiment.stt.stt_transcriber import _select_chunks


def test_select_chunks_vad_default(monkeypatch):
    audio = np.concatenate([
        np.zeros(int(1.0 * SR), dtype=np.float32),
        tone(1.0),
        np.zeros(int(1.0 * SR), dtype=np.float32),
        tone(1.0),
    ])
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    chunks = _select_chunks(audio_int16, SR)

    assert len(chunks) == 2
    assert chunks[0][1] == pytest.approx(0.7, abs=0.05)


def test_select_chunks_full_audio_when_disabled(monkeypatch):
    monkeypatch.setenv("STT_VAD_DISABLED", "1")
    audio = np.concatenate([
        np.zeros(int(1.0 * SR), dtype=np.float32),
        tone(1.0),
        np.zeros(int(1.0 * SR), dtype=np.float32),
        tone(1.0),
    ])
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    chunks = _select_chunks(audio_int16, SR)

    assert len(chunks) == 1
    assert chunks[0][1] == 0.0
    assert len(chunks[0][0]) == 4 * SR
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest sentiment/tests/test_vad_chunking.py::test_select_chunks_vad_default sentiment/tests/test_vad_chunking.py::test_select_chunks_full_audio_when_disabled -v`
Expected: FAIL — `ImportError: cannot import name '_select_chunks'`.

- [ ] **Step 3: Implement**

Add after `_build_vad_chunks` (end of Task 1's function):

```python
def _select_chunks(audio_int16: np.ndarray, sr: int) -> List[Tuple[bytes, float]]:
    """Choose STT chunks: VAD-gated by default, full audio when STT_VAD_DISABLED=1."""
    if os.environ.get("STT_VAD_DISABLED", "").lower() in ("1", "true", "yes"):
        chunk_samples = CHUNK_SECONDS * sr
        chunks = []
        for i in range(0, len(audio_int16), chunk_samples):
            chunk = audio_int16[i : i + chunk_samples]
            if len(chunk) < sr:
                break
            chunks.append((chunk.tobytes(), i / sr))
        return chunks
    return _build_vad_chunks(audio_int16, sr)
```

In `_transcribe_api`, replace lines 102-108:

```python
        chunk_samples = CHUNK_SECONDS * sr
        chunks = []
        for i in range(0, len(audio_int16), chunk_samples):
            chunk = audio_int16[i : i + chunk_samples]
            if len(chunk) < sr:
                break
            chunks.append((chunk.tobytes(), i / sr))
```

with:

```python
        chunks = _select_chunks(audio_int16, sr)
```

(The `if not chunks: return None` immediately after stays.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest sentiment/tests/test_vad_chunking.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add sentiment/stt/stt_transcriber.py sentiment/tests/test_vad_chunking.py
git commit -m "feat: gate STT chunks on VAD with STT_VAD_DISABLED escape hatch"
```

---

### Task 3: Update Roadmap and README

**Files:**
- Modify: `Roadmap.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: commit history (hallucination guard shipped in commits `d0cb0ef`, `2717f07`, `9e7ddda`; VAD gating shipped in this plan's Tasks 1-2).
- Produces: accurate status docs.

- [ ] **Step 1: Update Roadmap.md**

1. In **Remaining Work (priority order)** (near the top of the roadmap), replace the first item — currently something like `1. **Guard against hallucinated audits** — skip/flag the unified audit when the STT transcript is empty (currently Gemini confabulates results on empty transcripts)` — with a `✅` marker so it reads as complete:
   `1. ~~Guard against hallucinated audits~~ ✅ **Complete** — `stage_audit` skips the unified audit + fallbacks and records `audit_skipped = "no transcript"` when the transcript is empty (commits `d0cb0ef`, `2717f07`, `9e7ddda`)`
2. Remove or mark ✅ the VAD-gated-chunks item — if the list has `2. **VAD-gated chunks**`, replace it with `2. ~~VAD-gated STT chunks~~ ✅ **Complete** — `_select_chunks` sends only speech regions to Chirp 3; disable with `STT_VAD_DISABLED=1``
3. Renumber the remaining items (disk transcript cache → 1, Phase 2 WebSockets → 2, Phase 3 Docker → 3, Phase 4 observability → 4) so the list is contiguous.
4. In the **STT Cost Optimization** section (if present), mark VAD gating as implemented.

- [ ] **Step 2: Update README.md**

In the **Environment Variables** table, add a row:

```markdown
| `STT_VAD_DISABLED=1` | Send full audio to STT instead of VAD-gated speech chunks |
```

- [ ] **Step 3: Commit**

```bash
git add Roadmap.md README.md
git commit -m "docs: mark hallucination guard and VAD-gated STT chunks complete"
```

- [ ] **Step 4: Run the full test suite**

Run: `venv/bin/python -m pytest -q`
Expected: 66 passed, 3 skipped, 4 failed — the 4 failures are exactly the pre-existing ones (`diarization/tests/test_pipeline.py::TestDiarizationPipeline::test_diarization_pipeline_execution`, and the 3 `sentiment/tests/test_stt_transcriber.py` method-drift tests), untouched by this plan.
---

## Deviations recorded during execution

1. **Chunk length assertions are byte counts, not sample counts.** Chunks are `piece.tobytes()` (int16 → 2 bytes/sample). All `len(chunks[i][0])` assertions in the plan compare bytes; the committed tests use `len(chunks[i][0]) // 2` so the expected values (in samples: `1.6 * SR`, `50 * SR`, `4 * SR`, …) hold.
2. **Pure-tone synthetic audio never triggers the VAD.** `compute_vad` thresholds against the 10th-percentile frame RMS; with no quiet frames, the noise floor equals the signal and `threshold_multiplier=1.5` exceeds everything (0 active frames). The split/drop tests therefore prepend `8s` of silence (≥10% of frames), and their expected offsets shifted from `0.0`/`50.0` to `7.7`/`57.7` and from `0.0` to `7.7`.
3. Task 1's merge-gap expected length uses `1.3 * SR` and the split test's second chunk `2.0 * SR`→`2.3 * SR` (audio-length clamping of trailing padding), per plan self-review.
4. All 7 tests in `sentiment/tests/test_vad_chunking.py` pass; implementation matches the plan's code except the float32 cast: `compute_vad(audio_int16.astype(np.float32), ...)` — required because int16 squares overflow in `frame_rms` (NaN → all-inactive mask).
