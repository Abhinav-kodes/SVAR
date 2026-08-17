import pytest

from pipeline.job_store import InMemoryJobStore, STAGES
from pipeline.results_repo import InMemoryResultsRepository
from pipeline import runner, stages
from pipeline.progress_pubsub import NoopProgressPublisher

STAGE_FN = {"text_emo": "stage_text_emotion", "compliance": "stage_audit"}


def make_deps():
    return InMemoryJobStore(), InMemoryResultsRepository()


def test_runner_skips_missing_file(tmp_path, monkeypatch):
    job_store, repo = make_deps()
    with pytest.raises(FileNotFoundError):
        runner.run_pipeline("nope.mp3", job_store=job_store, results_repo=repo, publisher=NoopProgressPublisher())


def test_runner_full_flow_with_mocked_stages(tmp_path, monkeypatch):
    job_store, repo = make_deps()
    monkeypatch.setattr(runner, "SAMPLE_CALLS_DIR", str(tmp_path))
    calls = []
    for sid in [s["id"] for s in STAGES]:
        def make_stage(sid=sid):
            def stage(ctx):
                calls.append(sid)
                ctx.cache[sid] = "done"
            return stage
        monkeypatch.setattr(stages, STAGE_FN.get(sid, f"stage_{sid}"), make_stage())

    monkeypatch.setattr(stages, "free_gpu", lambda label="": None)

    path = tmp_path / "x.wav"
    path.write_bytes(b"not really audio")
    results = runner.run_pipeline("x.wav", job_store=job_store, results_repo=repo, publisher=NoopProgressPublisher())

    assert calls == [s["id"] for s in STAGES]
    assert job_store.get("x.wav")["status"] == "completed"
    assert repo.get("x.wav") is not None
    assert results["processing_time_s"] >= 0


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
    results = runner.run_pipeline("x.wav", job_store=job_store, results_repo=repo, publisher=NoopProgressPublisher())

    assert results["audit_skipped"] == "no transcript"