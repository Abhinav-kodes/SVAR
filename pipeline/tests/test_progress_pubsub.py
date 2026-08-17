import json

import pytest

from pipeline.progress_pubsub import NoopProgressPublisher, RedisProgressPublisher


class FakeRedis:
    def __init__(self):
        self.published = []

    def publish(self, channel, message):
        self.published.append((channel, message))
        return 1


@pytest.fixture
def fake_redis(monkeypatch):
    import redis

    fake = FakeRedis()
    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, url: fake))
    return fake


def test_publish_sends_json_to_per_file_channel(fake_redis):
    pub = RedisProgressPublisher("redis://test/0")
    pub.publish("a.mp3", {"status": "running", "percent": 11})
    assert fake_redis.published == [
        ("svar:progress:a.mp3", json.dumps({"status": "running", "percent": 11}))
    ]


def test_publish_swallows_redis_errors(fake_redis, caplog):
    def boom(channel, message):
        raise RuntimeError("redis down")

    fake_redis.publish = boom
    pub = RedisProgressPublisher("redis://test/0")
    pub.publish("a.mp3", {"status": "running"})  # must not raise
    assert "redis down" in caplog.text


def test_noop_publisher_does_nothing():
    pub = NoopProgressPublisher()
    pub.publish("a.mp3", {"status": "running"})  # must not raise

STAGE_FN = {"text_emo": "stage_text_emotion", "compliance": "stage_audit"}


class RecordingPublisher:
    def __init__(self):
        self.published = []

    def publish(self, filename, progress):
        self.published.append((filename, dict(progress)))


class RaisingPublisher:
    def publish(self, filename, progress):
        raise RuntimeError("publish boom")


def _mock_stages(monkeypatch, tmp_path, fail_at=None):
    from pipeline import runner, stages
    from pipeline.job_store import STAGES

    monkeypatch.setattr(runner, "SAMPLE_CALLS_DIR", str(tmp_path))
    for sid in [s["id"] for s in STAGES]:
        def make_stage(sid=sid):
            def stage(ctx):
                if fail_at and sid == fail_at:
                    raise ValueError(f"stage {sid} failed")
                ctx.cache[sid] = "done"
            return stage
        monkeypatch.setattr(stages, STAGE_FN.get(sid, f"stage_{sid}"), make_stage())
    monkeypatch.setattr(stages, "free_gpu", lambda label="": None)


def test_run_pipeline_publishes_snapshot_per_transition(monkeypatch, tmp_path):
    from pipeline import runner
    from pipeline.job_store import InMemoryJobStore, STAGES
    from pipeline.results_repo import InMemoryResultsRepository

    job_store = InMemoryJobStore()
    pub = RecordingPublisher()
    _mock_stages(monkeypatch, tmp_path)
    (tmp_path / "x.wav").write_bytes(b"not really audio")

    runner.run_pipeline("x.wav", job_store=job_store,
                        results_repo=InMemoryResultsRepository(), publisher=pub)

    filenames = [f for f, _ in pub.published]
    assert filenames == ["x.wav"] * len(pub.published)
    statuses = [p["status"] for _, p in pub.published]
    assert statuses[0] == "queued"
    assert "running" in statuses
    assert statuses[-1] == "completed"
    for _, p in pub.published:
        assert set(p["stages"]) == {s["id"] for s in STAGES}


def test_run_pipeline_publishes_terminal_error_snapshot(monkeypatch, tmp_path):
    from pipeline import runner
    from pipeline.job_store import InMemoryJobStore
    from pipeline.results_repo import InMemoryResultsRepository

    job_store = InMemoryJobStore()
    pub = RecordingPublisher()
    _mock_stages(monkeypatch, tmp_path, fail_at="crm")
    (tmp_path / "x.wav").write_bytes(b"not really audio")

    with pytest.raises(ValueError, match="stage crm failed"):
        runner.run_pipeline("x.wav", job_store=job_store,
                            results_repo=InMemoryResultsRepository(), publisher=pub)

    assert pub.published[-1][1]["status"] == "error"


def test_raising_publisher_does_not_fail_pipeline(monkeypatch, tmp_path):
    from pipeline import runner
    from pipeline.job_store import InMemoryJobStore
    from pipeline.results_repo import InMemoryResultsRepository

    job_store = InMemoryJobStore()
    _mock_stages(monkeypatch, tmp_path)
    (tmp_path / "x.wav").write_bytes(b"not really audio")

    runner.run_pipeline("x.wav", job_store=job_store,
                        results_repo=InMemoryResultsRepository(),
                        publisher=RaisingPublisher())

    assert job_store.get("x.wav")["status"] == "completed"


def test_noop_publisher_completes_silently(monkeypatch, tmp_path):
    from pipeline import runner
    from pipeline.job_store import InMemoryJobStore
    from pipeline.results_repo import InMemoryResultsRepository

    job_store = InMemoryJobStore()
    _mock_stages(monkeypatch, tmp_path)
    (tmp_path / "x.wav").write_bytes(b"not really audio")

    runner.run_pipeline("x.wav", job_store=job_store,
                        results_repo=InMemoryResultsRepository(),
                        publisher=NoopProgressPublisher())

    assert job_store.get("x.wav")["status"] == "completed"
