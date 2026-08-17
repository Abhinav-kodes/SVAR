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
    constructed = []

    class BoomClient:
        def __init__(self, api_key):
            constructed.append(api_key)

        def models(self):
            raise AssertionError("Gemini client must not be used")

    monkeypatch.setattr(audit_llm.genai, "Client", BoomClient)
    result = engine.audit_call(
        [
            {"speaker": "agent", "text": "  ", "start": 0.0, "end": 1.0},
            {"speaker": "customer", "text": "", "start": 1.0, "end": 2.0},
        ]
    )
    assert result is None
    assert constructed == []


def test_audit_call_skips_empty_segments_in_batch(monkeypatch, tmp_path):
    engine = make_engine(monkeypatch, tmp_path)
    captured = {}

    class FakeClient:
        def __init__(self, api_key):
            pass

        @property
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