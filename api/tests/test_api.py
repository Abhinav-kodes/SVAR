import os

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from pipeline.job_store import InMemoryJobStore
from pipeline.results_repo import InMemoryResultsRepository


@pytest.fixture
def client():
    job_store = InMemoryJobStore()
    repo = InMemoryResultsRepository()
    enqueued = []

    def fake_enqueue(filename):
        enqueued.append(filename)

    app = create_app(job_store=job_store, results_repo=repo, enqueue=fake_enqueue, job_alive=lambda f: True)
    app.state.enqueued = enqueued
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_sample_calls_returns_list(client):
    res = client.get("/api/sample_calls").json()
    assert isinstance(res, list)
    assert all(isinstance(f, str) for f in res)


def test_analyze_unknown_file_404(client):
    res = client.post("/api/analyze", json={"filename": "nope.mp3"})
    assert res.status_code == 404
    assert "error" in res.json()


def test_analyze_enqueues_and_queues(client, monkeypatch, tmp_path):
    os.makedirs(tmp_path / "data" / "sample_calls", exist_ok=True)
    monkeypatch.setattr("api.main.SAMPLE_CALLS_DIR", str(tmp_path / "data" / "sample_calls"))
    (tmp_path / "data" / "sample_calls" / "a.mp3").write_bytes(b"x")
    res = client.post("/api/analyze", json={"filename": "a.mp3"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "queued"
    assert client.app.state.enqueued == ["a.mp3"]
    assert client.get("/api/progress?file=a.mp3").json()["status"] == "queued"


def test_analyze_completed_if_results_exist(client, monkeypatch, tmp_path):
    os.makedirs(tmp_path / "data" / "sample_calls", exist_ok=True)
    monkeypatch.setattr("api.main.SAMPLE_CALLS_DIR", str(tmp_path / "data" / "sample_calls"))
    (tmp_path / "data" / "sample_calls" / "a.mp3").write_bytes(b"x")
    client.app.state.results_repo.save("a.mp3", {"qa": {"score": 99}})
    res = client.post("/api/analyze", json={"filename": "a.mp3"}).json()
    assert res["status"] == "completed"
    assert client.app.state.enqueued == []
    progress = client.get("/api/progress?file=a.mp3").json()
    assert progress["status"] == "completed"
    assert progress["percent"] == 100
    progress = client.get("/api/progress?file=a.mp3").json()
    assert progress["status"] == "completed"
    assert progress["percent"] == 100


def test_progress_falls_back_to_completed_when_results_exist(client):
    client.app.state.results_repo.save("a.mp3", {"qa": {"score": 99}})
    res = client.get("/api/progress?file=a.mp3").json()
    assert res["status"] == "completed"
    assert res["percent"] == 100


def test_analyze_running_returns_running(client, monkeypatch, tmp_path):
    os.makedirs(tmp_path / "data" / "sample_calls", exist_ok=True)
    monkeypatch.setattr("api.main.SAMPLE_CALLS_DIR", str(tmp_path / "data" / "sample_calls"))
    (tmp_path / "data" / "sample_calls" / "a.mp3").write_bytes(b"x")
    client.app.state.job_store.create("a.mp3")
    client.app.state.job_store.update_stage("a.mp3", "denoise", "running")
    res = client.post("/api/analyze", json={"filename": "a.mp3"}).json()
    assert res["status"] == "running"
    assert client.app.state.enqueued == []


def test_analyze_stale_running_requeues_when_job_gone(monkeypatch, tmp_path):
    os.makedirs(tmp_path / "data" / "sample_calls", exist_ok=True)
    monkeypatch.setattr("api.main.SAMPLE_CALLS_DIR", str(tmp_path / "data" / "sample_calls"))
    (tmp_path / "data" / "sample_calls" / "a.mp3").write_bytes(b"x")
    job_store = InMemoryJobStore()
    enqueued = []
    app = create_app(
        job_store=job_store,
        results_repo=InMemoryResultsRepository(),
        enqueue=enqueued.append,
        job_alive=lambda f: False,
    )
    client = TestClient(app)
    job_store.create("a.mp3")
    job_store.update_stage("a.mp3", "denoise", "running")
    res = client.post("/api/analyze", json={"filename": "a.mp3"}).json()
    assert res["status"] == "queued"
    assert enqueued == ["a.mp3"]
    assert job_store.get("a.mp3")["status"] == "queued"


def test_analyze_enqueue_failure_returns_503_and_cleans_up(monkeypatch, tmp_path):
    os.makedirs(tmp_path / "data" / "sample_calls", exist_ok=True)
    monkeypatch.setattr("api.main.SAMPLE_CALLS_DIR", str(tmp_path / "data" / "sample_calls"))
    (tmp_path / "data" / "sample_calls" / "a.mp3").write_bytes(b"x")
    job_store = InMemoryJobStore()

    def broken_enqueue(filename):
        raise RuntimeError("redis down")

    app = create_app(job_store=job_store, results_repo=InMemoryResultsRepository(), enqueue=broken_enqueue)
    client = TestClient(app)
    res = client.post("/api/analyze", json={"filename": "a.mp3"})
    assert res.status_code == 503
    assert "error" in res.json()
    assert job_store.get("a.mp3") is None


def test_progress_idle_default(client):
    res = client.get("/api/progress?file=zzz.mp3").json()
    assert res["status"] == "idle"
    assert res["percent"] == 0


def test_results_missing_404(client):
    res = client.post("/api/results", json={"filename": "zzz.mp3"})
    assert res.status_code == 404


def test_results_roundtrip(client):
    client.app.state.results_repo.save("a.mp3", {"qa": {"score": 88}, "crm_note": "n"})
    res = client.post("/api/results", json={"filename": "a.mp3"})
    assert res.json()["qa"]["score"] == 88


def test_stage_endpoints_slice(client):
    client.app.state.results_repo.save("a.mp3", {"compliance": {"flags": []}, "crm_note": "n", "qa": {"score": 10}})
    assert client.post("/api/compliance", json={"filename": "a.mp3"}).json() == {
        "compliance": {"flags": []}, "crm_note": "n"
    }
    assert client.post("/api/qa-score", json={"filename": "a.mp3"}).json()["qa"]["score"] == 10
    assert client.post("/api/denoise", json={"filename": "a.mp3"}).json() == {
        "duration_s": None, "denoise_metrics": None
    }


def test_audio_serving(client, monkeypatch, tmp_path):
    os.makedirs(tmp_path / "data" / "sample_calls", exist_ok=True)
    monkeypatch.setattr("api.main.SAMPLE_CALLS_DIR", str(tmp_path / "data" / "sample_calls"))
    (tmp_path / "data" / "sample_calls" / "a.mp3").write_bytes(b"audio")
    res = client.get("/audio/a.mp3")
    assert res.status_code == 200
    assert res.content == b"audio"
    assert res.headers["content-type"] == "audio/mpeg"
    assert client.get("/audio/missing.mp3").status_code == 404
