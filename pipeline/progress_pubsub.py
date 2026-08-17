import json
import logging

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ProgressPublisher(ABC):
    @abstractmethod
    def publish(self, filename: str, progress: dict) -> None:
        raise NotImplementedError


class RedisProgressPublisher(ProgressPublisher):
    def __init__(self, url: str = "redis://localhost:6379/0"):
        import redis

        self._redis = redis.Redis.from_url(url)

    def publish(self, filename: str, progress: dict) -> None:
        try:
            self._redis.publish(f"svar:progress:{filename}", json.dumps(progress))
        except Exception as e:
            logger.warning("progress publish failed for %s: %s", filename, e)


class NoopProgressPublisher(ProgressPublisher):
    def publish(self, filename: str, progress: dict) -> None:
        pass