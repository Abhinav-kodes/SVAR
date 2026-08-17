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