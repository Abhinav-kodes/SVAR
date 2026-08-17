import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sentiment.stt.stt_transcriber import (
    CHUNK_SECONDS,
    VAD_FRAME_MS,
    VAD_GAP_TOLERANCE_S,
    VAD_PADDING_S,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_DIR = os.environ.get(
    "SVAR_TRANSCRIPT_CACHE_DIR",
    os.path.join(_REPO_ROOT, "data", "transcripts"),
)
CACHE_TTL_DAYS = 30
_MODEL = "chirp_3"


def sha1_of_file(path: str) -> str:
    """Streamed sha1 hexdigest of a file's bytes."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_fingerprint(lang: str) -> Dict[str, Any]:
    vad_disabled = os.environ.get("STT_VAD_DISABLED", "").lower() in ("1", "true", "yes")
    return {
        "model": _MODEL,
        "language": lang,
        "vad_enabled": not vad_disabled,
        "VAD_FRAME_MS": VAD_FRAME_MS,
        "VAD_PADDING_S": VAD_PADDING_S,
        "VAD_GAP_TOLERANCE_S": VAD_GAP_TOLERANCE_S,
        "CHUNK_SECONDS": CHUNK_SECONDS,
    }


def _entry_path(sha1: str, lang: str) -> str:
    return os.path.join(CACHE_DIR, f"{sha1}-{lang}.json")


def get_words(sha1: str, lang: str) -> Optional[List[Dict[str, Any]]]:
    """Return the cached word stream, or None on miss/corrupt/expired/mismatch."""
    path = _entry_path(sha1, lang)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        created = datetime.fromisoformat(data["created_at"])
        if datetime.utcnow() - created > timedelta(days=CACHE_TTL_DAYS):
            os.remove(path)
            return None
        if data.get("config") != _config_fingerprint(lang):
            return None
        words = data.get("words")
        if not isinstance(words, list):
            return None
        return words
    except Exception as e:
        print(f"  [transcript-cache] read failed for {path}: {e}")
        return None


def put_words(sha1: str, lang: str, words: List[Dict[str, Any]]) -> None:
    """Atomically write the word stream (tmp + os.replace); failures never raise."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = _entry_path(sha1, lang)
        tmp_path = path + ".tmp"
        payload = {
            "config": _config_fingerprint(lang),
            "created_at": datetime.utcnow().isoformat(),
            "words": words,
        }
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"  [transcript-cache] write failed for {sha1}-{lang}: {e}")