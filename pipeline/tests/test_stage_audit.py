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