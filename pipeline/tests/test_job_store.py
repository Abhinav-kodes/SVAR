from pipeline.job_store import InMemoryJobStore, STAGES


def make_store():
    return InMemoryJobStore()


def test_create_sets_queued_progress():
    store = make_store()
    store.create("a.mp3")
    p = store.get("a.mp3")
    assert p["status"] == "queued"
    assert p["percent"] == 0
    assert set(p["stages"].keys()) == {s["id"] for s in STAGES}


def test_update_stage_running_and_done():
    store = make_store()
    store.create("a.mp3")
    store.update_stage("a.mp3", "denoise", "running")
    p = store.get("a.mp3")
    assert p["current_stage"] == "denoise"
    assert p["stages"]["denoise"]["status"] == "running"
    store.update_stage("a.mp3", "denoise", "done", time_s=1.5)
    p = store.get("a.mp3")
    assert p["stages"]["denoise"]["status"] == "done"
    assert p["stages"]["denoise"]["time_s"] == 1.5
    assert p["percent"] == round(1 / len(STAGES) * 100)


def test_finish_completed_sets_time_s():
    store = make_store()
    store.create("a.mp3")
    store.finish("a.mp3")
    p = store.get("a.mp3")
    assert p["status"] == "completed"
    assert p["percent"] == 100
    assert p["time_s"] >= 0


def test_finish_error():
    store = make_store()
    store.create("a.mp3")
    store.finish("a.mp3", error="boom")
    p = store.get("a.mp3")
    assert p["status"] == "error"
    assert p["error"] == "boom"


def test_get_missing_returns_none():
    assert make_store().get("nope.mp3") is None
