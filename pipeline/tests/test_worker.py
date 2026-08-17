from pipeline import worker


def test_run_pipeline_job_wrapper_delegates(monkeypatch):
    captured = {}

    def fake_run(filename, **kwargs):
        captured["filename"] = filename
        return {"done": True}

    monkeypatch.setattr(worker, "run_pipeline", fake_run)
    assert worker.run_pipeline_job("a.mp3") == {"done": True}
    assert captured["filename"] == "a.mp3"