import json
import threading
import time
from typing import Dict, Optional

STAGES = [
    {"id": "denoise", "label": "Denoising"},
    {"id": "diarize", "label": "Diarization"},
    {"id": "stt", "label": "Speech-to-Text"},
    {"id": "acoustic", "label": "Acoustic Emotion"},
    {"id": "text_emo", "label": "Text Emotion"},
    {"id": "compliance", "label": "Compliance"},
    {"id": "fusion", "label": "Emotion Fusion"},
    {"id": "qa", "label": "QA Scoring"},
    {"id": "crm", "label": "CRM Note"},
]


def _new_progress() -> dict:
    return {
        "status": "queued",
        "current_stage": "",
        "percent": 0,
        "stages": {s["id"]: {"status": "pending", "time_s": 0} for s in STAGES},
        "error": None,
        "start_time": time.time(),
    }


class JobStore:
    def create(self, filename: str) -> None:
        raise NotImplementedError

    def update_stage(self, filename: str, stage_id: str, status: str, time_s: float = 0.0) -> None:
        raise NotImplementedError

    def finish(self, filename: str, error: Optional[str] = None) -> None:
        raise NotImplementedError

    def get(self, filename: str) -> Optional[dict]:
        raise NotImplementedError


class InMemoryJobStore(JobStore):
    def __init__(self):
        self._jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, filename: str) -> None:
        with self._lock:
            self._jobs[filename] = _new_progress()

    def update_stage(self, filename: str, stage_id: str, status: str, time_s: float = 0.0) -> None:
        with self._lock:
            p = self._jobs.get(filename)
            if not p:
                return
            p["current_stage"] = stage_id
            if status == "done":
                p["stages"][stage_id]["status"] = "done"
                p["stages"][stage_id]["time_s"] = time_s
            elif status == "error":
                p["stages"][stage_id]["status"] = "error"
            else:
                p["stages"][stage_id]["status"] = "running"
                p["status"] = "running"
            done = sum(1 for s in p["stages"].values() if s["status"] == "done")
            p["percent"] = round(done / len(STAGES) * 100)

    def finish(self, filename: str, error: Optional[str] = None) -> None:
        with self._lock:
            p = self._jobs.get(filename)
            if not p:
                return
            p["status"] = "error" if error else "completed"
            p["percent"] = 100 if not error else p["percent"]
            p["error"] = error
            p["time_s"] = round(time.time() - p["start_time"], 1)

    def get(self, filename: str) -> Optional[dict]:
        with self._lock:
            p = self._jobs.get(filename)
            return dict(p) if p else None


class RedisJobStore(JobStore):
    TTL_SECONDS = 86400

    def __init__(self, url: str = "redis://localhost:6379/0"):
        import redis
        self._redis = redis.Redis.from_url(url)
        self._lock = threading.Lock()

    def _key(self, filename: str) -> str:
        return f"svar:job:{filename}"

    def create(self, filename: str) -> None:
        self._redis.set(self._key(filename), json.dumps(_new_progress()), ex=self.TTL_SECONDS)

    def update_stage(self, filename: str, stage_id: str, status: str, time_s: float = 0.0) -> None:
        with self._lock:
            raw = self._redis.get(self._key(filename))
            if raw is None:
                return
            p = json.loads(raw)
            p["current_stage"] = stage_id
            if status == "done":
                p["stages"][stage_id]["status"] = "done"
                p["stages"][stage_id]["time_s"] = time_s
            elif status == "error":
                p["stages"][stage_id]["status"] = "error"
            else:
                p["stages"][stage_id]["status"] = "running"
                p["status"] = "running"
            done = sum(1 for s in p["stages"].values() if s["status"] == "done")
            p["percent"] = round(done / len(STAGES) * 100)
            self._redis.set(self._key(filename), json.dumps(p), ex=self.TTL_SECONDS)

    def finish(self, filename: str, error: Optional[str] = None) -> None:
        raw = self._redis.get(self._key(filename))
        if raw is None:
            return
        p = json.loads(raw)
        p["status"] = "error" if error else "completed"
        p["percent"] = 100 if not error else p["percent"]
        p["error"] = error
        p["time_s"] = round(time.time() - p["start_time"], 1)
        self._redis.set(self._key(filename), json.dumps(p), ex=self.TTL_SECONDS)

    def get(self, filename: str) -> Optional[dict]:
        raw = self._redis.get(self._key(filename))
        return json.loads(raw) if raw else None