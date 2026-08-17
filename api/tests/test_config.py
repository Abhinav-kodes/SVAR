import importlib
import sys


def test_config_defaults(monkeypatch):
    for var in ("SVAR_PORT", "SVAR_REDIS_URL", "SVAR_DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)
    sys.modules.pop("api.config", None)
    cfg = importlib.import_module("api.config")
    assert cfg.PORT == 8050
    assert cfg.REDIS_URL == "redis://localhost:6379/0"
    assert cfg.DATABASE_URL == "postgresql://svar:svar@localhost:5432/svar"
    assert cfg.SAMPLE_CALLS_DIR.endswith(("sample_calls", "sample_calls/"))


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("SVAR_PORT", "9000")
    monkeypatch.setenv("SVAR_REDIS_URL", "redis://example:6390/2")
    sys.modules.pop("api.config", None)
    cfg = importlib.import_module("api.config")
    assert cfg.PORT == 9000
    assert cfg.REDIS_URL == "redis://example:6390/2"
