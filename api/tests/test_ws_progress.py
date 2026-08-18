import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from api.main import create_app
from pipeline.job_store import InMemoryJobStore
from pipeline.results_repo import InMemoryResultsRepository


class FakeSubscription:
    def __init__(self, messages=None):
        self._queue = list(messages or [])
        self.received = []
        self.closed = False

    def get_message(self, timeout=None):
        if self._queue:
            msg = self._queue.pop(0)
            self.received.append(msg)
            return msg
        return None

    def close(self):
        self.closed = True


def make_app(sub, job_store=None):
    return create_app(
        job_store=job_store or InMemoryJobStore(),
        results_repo=InMemoryResultsRepository(),
        enqueue=lambda f: None,
        job_alive=lambda f: True,
        subscriber=lambda filename: sub,
    )


def test_ws_sends_current_snapshot_on_connect():
    job_store = InMemoryJobStore()
    job_store.create("a.mp3")
    app = make_app(FakeSubscription(), job_store=job_store)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/progress?file=a.mp3") as ws:
            data = json.loads(ws.receive_text())
            assert data["status"] == "queued"
            assert "stages" in data


def test_ws_completed_job_snapshot_then_close():
    job_store = InMemoryJobStore()
    job_store.create("a.mp3")
    job_store.finish("a.mp3")
    app = make_app(FakeSubscription(), job_store=job_store)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/progress?file=a.mp3") as ws:
            data = json.loads(ws.receive_text())
            assert data["status"] == "completed"
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()


def test_ws_forwards_messages_verbatim_and_closes_on_terminal():
    sub = FakeSubscription(messages=[
        {"type": "message", "data": json.dumps({"status": "running", "percent": 50, "stages": {}})},
        {"type": "message", "data": json.dumps({"status": "completed", "percent": 100, "stages": {}})},
    ])
    app = make_app(sub)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/progress?file=a.mp3") as ws:
            assert json.loads(ws.receive_text())["status"] == "running"
            assert json.loads(ws.receive_text())["status"] == "completed"
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()

    assert sub.closed
    assert len(sub.received) == 2


def test_ws_client_disconnect_closes_subscriber():
    sub = FakeSubscription()
    app = make_app(sub)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/progress?file=a.mp3"):
            pass  # nothing is sent (no snapshot, empty queue); just connect and disconnect

    assert sub.closed

class FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, filename, progress):
        self.published.append((filename, dict(progress)))


def make_app_with_publisher(pub, job_store=None, results_repo=None):
    return create_app(
        job_store=job_store or InMemoryJobStore(),
        results_repo=results_repo or InMemoryResultsRepository(),
        enqueue=lambda f: None,
        job_alive=lambda f: True,
        subscriber=lambda filename: FakeSubscription(),
        publisher=pub,
    )


def test_already_analyzed_publishes_completed_snapshot(monkeypatch, tmp_path):
    import os
    from api import main as api_main
    os.makedirs(tmp_path / "data" / "sample_calls", exist_ok=True)
    monkeypatch.setattr(api_main, "SAMPLE_CALLS_DIR", str(tmp_path / "data" / "sample_calls"))
    (tmp_path / "data" / "sample_calls" / "a.mp3").write_bytes(b"x")
    job_store = InMemoryJobStore()
    results_repo = InMemoryResultsRepository()
    results_repo.save("a.mp3", {"qa": {"score": 70}})
    pub = FakePublisher()
    app = make_app_with_publisher(pub, job_store=job_store, results_repo=results_repo)

    with TestClient(app) as client:
        res = client.post("/api/analyze", json={"filename": "a.mp3"})
        assert res.status_code == 200
        assert res.json()["status"] == "completed"

    assert len(pub.published) == 1
    filename, snapshot = pub.published[0]
    assert filename == "a.mp3"
    assert snapshot["status"] == "completed"


def test_fresh_analyze_does_not_publish():
    pub = FakePublisher()
    app = make_app_with_publisher(pub)

    with TestClient(app) as client:
        client.post("/api/analyze", json={"filename": "a.mp3"})

    assert pub.published == []
