import os

import pytest

from pipeline.results_repo import InMemoryResultsRepository, PostgresResultsRepository


def make_repo():
    return InMemoryResultsRepository()


def test_save_and_get_roundtrip():
    repo = make_repo()
    repo.save("a.mp3", {"qa": {"score": 70}, "crm_note": "hi"})
    res = repo.get("a.mp3")
    assert res["qa"]["score"] == 70
    assert res["crm_note"] == "hi"


def test_get_missing_returns_none():
    assert make_repo().get("nope.mp3") is None


def test_save_upserts():
    repo = make_repo()
    repo.save("a.mp3", {"v": 1})
    repo.save("a.mp3", {"v": 2})
    assert repo.get("a.mp3")["v"] == 2


@pytest.mark.skipif(os.getenv("SVAR_TEST_POSTGRES") != "1", reason="needs postgres")
class TestPostgresResultsRepository:
    def make_repo(self):
        return PostgresResultsRepository()

    def setup_method(self):
        self.repo = self.make_repo()
        self.repo.init_db()

    def test_save_and_get_roundtrip(self):
        self.repo.save("a.mp3", {"qa": {"score": 70}, "crm_note": "hi"})
        res = self.repo.get("a.mp3")
        assert res["qa"]["score"] == 70
        assert res["crm_note"] == "hi"

    def test_get_missing_returns_none(self):
        assert self.repo.get("nope.mp3") is None

    def test_save_upserts(self):
        self.repo.save("a.mp3", {"v": 1})
        self.repo.save("a.mp3", {"v": 2})
        assert self.repo.get("a.mp3")["v"] == 2

def test_default_url_reads_svar_database_url(monkeypatch):
    monkeypatch.setenv("SVAR_DATABASE_URL", "postgresql://u:p@db:5432/x")
    assert PostgresResultsRepository._default_url() == "postgresql://u:p@db:5432/x"


def test_default_url_falls_back_to_localhost(monkeypatch):
    monkeypatch.delenv("SVAR_DATABASE_URL", raising=False)
    assert PostgresResultsRepository._default_url() == "postgresql://svar:svar@localhost:5432/svar"
