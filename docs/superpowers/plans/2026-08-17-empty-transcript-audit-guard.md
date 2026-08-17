# Empty-Transcript Audit Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the unified Gemini audit (and local fallbacks) from running on an empty transcript, eliminating hallucinated compliance/QA/CRM output when STT fails.

**Architecture:** A guard at two layers — `GeminiAuditEngine.audit_call` refuses empty transcripts and filters empty-text segments from the Gemini batch (defense in depth), while `stage_audit` in the pipeline skips the entire audit + fallbacks and records `audit_skipped = "no transcript"` in the cache, which `run_pipeline` surfaces in the results dict. The dashboard already renders "No … data available" when `compliance`/`qa`/`crm_note` keys are absent.

**Tech Stack:** Python 3, pytest, existing `sentiment/audit_llm.py`, `pipeline/stages.py`, `pipeline/runner.py`.

**Spec:** `docs/superpowers/specs/2026-08-17-empty-transcript-audit-guard-design.md`

## Global Constraints

- Guard flag value is exactly `"no transcript"`, stored under key `audit_skipped` in the pipeline cache and results dict.
- `audit_call` must return `None` without constructing a Gemini client when no segment has non-empty stripped text.
- Empty-text segments are dropped from the Gemini batch input; `"index"` values are the original positions in the `segments` list (so the existing `segment_results` index→timestamp mapping keeps working).
- `stage_audit` must return early (before `run_unified_audit` AND before the local compliance/QA/CRM fallbacks) when no segment has non-empty stripped text; the `compliance`/`qa`/`crm_note` cache keys are left absent.
- Behavior when transcript text exists is unchanged.
- Tests must never touch the network: monkeypatch `sentiment.audit_llm.genai.Client` and `sentiment.audit_llm.run_unified_audit` as needed.
- Test command: `venv/bin/python -m pytest <test file> -v` (from repo root).
- One conventional commit per task (`feat:` / `test:` / `refactor:`); no frontend changes.

---

### Task 1: Guard + filter in `sentiment/audit_llm.py`

**Files:**
- Modify: `sentiment/audit_llm.py:52-71` (`audit_call` batch-input loop)
- Create: `sentiment/tests/test_audit_llm.py`

**Interfaces:**
- Consumes: existing `GeminiAuditEngine` (module `sentiment.audit_llm`, `_KEYS_PATH` module global, `genai.Client` module global).
- Produces: module function `_build_batch_input(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]` — drops empty-text segments, preserves original indices. `GeminiAuditEngine.audit_call` returns `None` when the filtered batch is empty. Later tasks rely on the guard returning `None` for empty transcripts.

- [ ] **Step 1: Write the failing tests**

Create `sentiment/tests/test_audit_llm.py`:

```python
import json

import pytest

from sentiment import audit_llm


def make_engine(monkeypatch, tmp_path, keys=("k1",)):
    keys_path = tmp_path / "keys.json"
    keys_path.write_text(json.dumps({"keys": list(keys)}))
    monkeypatch.setattr(audit_llm, "_KEYS_PATH", keys_path)
    return audit_llm.GeminiAuditEngine()


def test_build_batch_input_filters_empty_text():
    segments = [
        {"speaker": "agent", "text": "  ", "start": 0.0, "end": 1.0},
        {"speaker": "customer", "text": "hello", "start": 1.0, "end": 2.0},
        {"speaker": "agent", "text": "", "start": 2.0, "end": 3.0},
        {"speaker": "customer", "text": "namaste", "start": 3.0, "end": 4.0},
    ]
    batch = audit_llm._build_batch_input(segments)
    assert [b["index"] for b in batch] == [1, 3]
    assert [b["text"] for b in batch] == ["hello", "namaste"]
    assert batch[0]["speaker"] == "customer"
    assert batch[1]["start"] == 3.0
    assert batch[1]["end"] == 4.0


def test_audit_call_returns_none_when_no_text(monkeypatch, tmp_path):
    engine = make_engine(monkeypatch, tmp_path)

    def boom(*args, **kwargs):
        raise AssertionError("Gemini client must not be constructed")

    monkeypatch.setattr(audit_llm.genai, "Client", boom)
    result = engine.audit_call(
        [
            {"speaker": "agent", "text": "  ", "start": 0.0, "end": 1.0},
            {"speaker": "customer", "text": "", "start": 1.0, "end": 2.0},
        ]
    )
    assert result is None


def test_audit_call_skips_empty_segments_in_batch(monkeypatch, tmp_path):
    engine = make_engine(monkeypatch, tmp_path)
    captured = {}

    class FakeClient:
        def __init__(self, api_key):
            pass

        def models(self):
            return self

        def generate_content(self, model, contents, config=None):
            captured["contents"] = contents
            payload = {
                "compliance": {"compliant": True, "total_violations": 0,
                               "agent_violations": 0, "customer_violations": 0,
                               "segment_results": []},
                "qa": {"qa_score": 90.0, "grade": "A", "components": {}, "weights_used": {}},
                "crm_note": {"summary": "ok", "key_points": [],
                             "compliance_summary": "", "recommended_action": ""},
            }
            return type("Resp", (), {"text": json.dumps(payload)})()

    monkeypatch.setattr(audit_llm.genai, "Client", FakeClient)
    segments = [
        {"speaker": "agent", "text": "", "start": 0.0, "end": 1.0},
        {"speaker": "customer", "text": "hello", "start": 1.0, "end": 2.0},
    ]
    result = engine.audit_call(segments)
    assert result is not None
    assert '"index": 1' in captured["contents"]
    assert '"text": "hello"' in captured["contents"]
    assert '"index": 0' not in captured["contents"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest sentiment/tests/test_audit_llm.py -v`
Expected: FAIL — `AttributeError: module 'sentiment.audit_llm' has no attribute '_build_batch_input'` (and the empty-transcript case still constructs a client, failing the boom test).

- [ ] **Step 3: Implement the guard + filter**

In `sentiment/audit_llm.py`, add above `class GeminiAuditEngine` (after `_load_keys`, line 29):

```python
def _build_batch_input(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build the Gemini batch input, dropping segments with no text."""
    batch_input = []
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if not text:
            continue
        batch_input.append({
            "index": i,
            "speaker": seg.get("speaker", "unknown"),
            "text": text,
            "start": seg.get("start_time_s", seg.get("start", 0)),
            "end": seg.get("end_time_s", seg.get("end", 0)),
        })
    return batch_input
```

Replace the inline batch loop inside `audit_call` (current lines 61-69):

```python
        batch_input = []
        for i, seg in enumerate(segments):
            batch_input.append({
                "index": i,
                "speaker": seg.get("speaker", "unknown"),
                "text": seg.get("text", "").strip(),
                "start": seg.get("start_time_s", seg.get("start", 0)),
                "end": seg.get("end_time_s", seg.get("end", 0)),
            })
```

with:

```python
        batch_input = _build_batch_input(segments)
        if not batch_input:
            print("[audit_engine] Skipped unified audit: empty transcript")
            return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest sentiment/tests/test_audit_llm.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add sentiment/audit_llm.py sentiment/tests/test_audit_llm.py
git commit -m "feat: guard unified audit against empty transcripts"
```

---

### Task 2: Primary guard in `pipeline/stages.py:stage_audit`

**Files:**
- Modify: `pipeline/stages.py:269-289` (`stage_audit`)
- Create: `pipeline/tests/test_stage_audit.py`

**Interfaces:**
- Consumes: `JobContext` (dataclass with `.cache` dict — see `pipeline/stages.py:29-34`), module-level `log()` in `pipeline.stages`, `sentiment.audit_llm.run_unified_audit` (imported lazily inside `stage_audit`).
- Produces: `stage_audit(ctx)` sets `ctx.cache["audit_skipped"] = "no transcript"` and returns early when no segment has text. Task 3 reads that cache key.

- [ ] **Step 1: Write the failing tests**

Create `pipeline/tests/test_stage_audit.py`:

```python
import pytest

from pipeline import stages
from pipeline.job_store import InMemoryJobStore
from pipeline.stages import JobContext, stage_audit


@pytest.fixture
def ctx():
    return JobContext(filepath="x.wav", filename="x.wav", cache={}, job_store=InMemoryJobStore())


def test_stage_audit_skips_empty_transcript(ctx, monkeypatch):
    ctx.cache["transcribed"] = True
    ctx.cache["segments"] = [
        {"speaker": "agent", "text": "  "},
        {"speaker": "customer", "text": ""},
    ]
    ctx.cache["fusion"] = [{"emotion": "neutral"}]

    calls = []
    for name in ("stage_compliance", "stage_qa", "stage_crm"):
        monkeypatch.setattr(stages, name, lambda c, name=name: calls.append(name))

    import sentiment.audit_llm as audit_llm

    def boom(*args, **kwargs):
        raise AssertionError("unified audit must not run on empty transcript")

    monkeypatch.setattr(audit_llm, "run_unified_audit", boom)

    stage_audit(ctx)

    assert ctx.cache["audit_skipped"] == "no transcript"
    assert calls == []


def test_stage_audit_runs_unified_audit_with_text(ctx, monkeypatch):
    ctx.cache["transcribed"] = True
    ctx.cache["segments"] = [{"speaker": "agent", "text": "namaste", "start": 0.0, "end": 1.0}]
    ctx.cache["fusion"] = [{"emotion": "neutral"}]

    import sentiment.audit_llm as audit_llm

    monkeypatch.setattr(
        audit_llm,
        "run_unified_audit",
        lambda segments, talk_ratio_data=None: {
            "compliance": {"total_violations": 0},
            "qa": {"qa_score": 90.0},
            "crm_note": {"summary": "ok"},
        },
    )

    stage_audit(ctx)

    assert ctx.cache["compliance"] == {"total_violations": 0}
    assert ctx.cache["qa"] == {"qa_score": 90.0}
    assert ctx.cache["crm_note"] == {"summary": "ok"}
    assert "audit_skipped" not in ctx.cache


def test_stage_audit_falls_back_when_unified_audit_returns_none(ctx, monkeypatch):
    ctx.cache["transcribed"] = True
    ctx.cache["segments"] = [{"speaker": "agent", "text": "namaste", "start": 0.0, "end": 1.0}]
    ctx.cache["fusion"] = [{"emotion": "neutral"}]

    calls = []

    def fake_fallback(name):
        def fn(c):
            calls.append(name)
            c.cache[name.replace("stage_", "")] = {"done": True}
        return fn

    for name in ("stage_compliance", "stage_qa", "stage_crm"):
        monkeypatch.setattr(stages, name, fake_fallback(name))

    import sentiment.audit_llm as audit_llm

    monkeypatch.setattr(audit_llm, "run_unified_audit", lambda segments, talk_ratio_data=None: None)

    stage_audit(ctx)

    assert calls == ["stage_compliance", "stage_qa", "stage_crm"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest pipeline/tests/test_stage_audit.py -v`
Expected: FAIL — first test ends with the boom `AssertionError` (unified audit runs on empty transcript today).

- [ ] **Step 3: Implement the guard**

In `pipeline/stages.py`, inside `stage_audit` (current lines 276-277, after the `stage_fusion` check and before the `try:`), insert:

```python
    if not any(s.get("text", "").strip() for s in c.get("segments", [])):
        c["audit_skipped"] = "no transcript"
        log("  [audit] skipped: no transcript")
        return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest pipeline/tests/test_stage_audit.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/stages.py pipeline/tests/test_stage_audit.py
git commit -m "feat: skip unified audit and fallbacks on empty transcript"
```

---

### Task 3: Surface `audit_skipped` in pipeline results

**Files:**
- Modify: `pipeline/runner.py:55-66` (results dict)
- Modify: `pipeline/tests/test_runner.py` (add one test)

**Interfaces:**
- Consumes: `ctx.cache` key `audit_skipped` (set by Task 2's `stage_audit`), existing `run_pipeline(filename, job_store=None, results_repo=None) -> dict`.
- Produces: results dict gains key `audit_skipped` (str when skipped, `None` otherwise). API consumers (stage-slice endpoints) are unaffected — `audit_skipped` is only in the full `/api/results` payload.

- [ ] **Step 1: Write the failing test**

Append to `pipeline/tests/test_runner.py`:

```python
def test_runner_includes_audit_skipped(tmp_path, monkeypatch):
    job_store, repo = make_deps()
    monkeypatch.setattr(runner, "SAMPLE_CALLS_DIR", str(tmp_path))
    for sid in [s["id"] for s in STAGES]:
        def make_stage(sid=sid):
            def stage(ctx):
                if sid == "crm":
                    ctx.cache["audit_skipped"] = "no transcript"
            return stage
        monkeypatch.setattr(stages, STAGE_FN.get(sid, f"stage_{sid}"), make_stage())

    monkeypatch.setattr(stages, "free_gpu", lambda label="": None)

    path = tmp_path / "x.wav"
    path.write_bytes(b"not really audio")
    results = runner.run_pipeline("x.wav", job_store=job_store, results_repo=repo)

    assert results["audit_skipped"] == "no transcript"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest pipeline/tests/test_runner.py::test_runner_includes_audit_skipped -v`
Expected: FAIL — `KeyError: 'audit_skipped'`.

- [ ] **Step 3: Implement**

In `pipeline/runner.py`, add to the results dict (after `"crm_note"` entry, line 65):

```python
            "audit_skipped": ctx.cache.get("audit_skipped"),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest pipeline/tests/test_runner.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full api+pipeline suite**

Run: `venv/bin/python -m pytest api/tests pipeline/tests -q`
Expected: PASS, no regressions (prior suite: 32 passed, 3 skipped; now 35 passed, 3 skipped).

- [ ] **Step 6: Commit**

```bash
git add pipeline/runner.py pipeline/tests/test_runner.py
git commit -m "feat: surface audit_skipped flag in pipeline results"
```