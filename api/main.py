import os
import urllib.parse

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from api.config import DASHBOARD_DIR, REDIS_URL, SAMPLE_CALLS_DIR
from api.schemas import AnalyzeRequest
from pipeline.job_store import JobStore, RedisJobStore
from pipeline.results_repo import PostgresResultsRepository, ResultsRepository
from pipeline.runner import _stage_results

STAGE_SLICES = {
    "denoise": ["duration_s", "denoise_metrics"],
    "diarize": ["duration_s", "segments", "talk_ratio", "separability", "confidence_method", "role_resolution"],
    "transcribe": ["duration_s", "segments", "talk_ratio", "role_resolution"],
    "emotion": ["duration_s", "segments", "fusion"],
    "compliance": ["compliance", "crm_note"],
    "qa-score": ["qa", "crm_note", "compliance"],
    "crm-note": ["crm_note", "compliance", "qa"],
}

AUDIO_CONTENT_TYPES = {
    ".mp3": "audio/mpeg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
}

DIST_MIME_TYPES = {
    ".html": "text/html",
    ".js": "application/javascript",
    ".css": "text/css",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
    ".json": "application/json",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
}

JOB_TIMEOUT_SECONDS = 3600
JOB_ALIVE_STATUSES = {"queued", "started", "deferred", "scheduled"}


def _rq_job_id(filename: str) -> str:
    return f"svar:{filename}"




def _default_enqueue(filename: str):
    import redis
    from rq import Queue
    conn = redis.Redis.from_url(REDIS_URL)
    Queue("svar", connection=conn).enqueue(
        "pipeline.worker.run_pipeline_job",
        filename,
        job_id=_rq_job_id(filename),
        job_timeout=JOB_TIMEOUT_SECONDS,
    )


def _default_job_alive(filename: str) -> bool:
    import redis
    from rq import Job
    from rq.exceptions import InvalidJobOperation, NoSuchJobError

    conn = redis.Redis.from_url(REDIS_URL)
    try:
        job = Job.fetch(_rq_job_id(filename), connection=conn)
        return job.get_status() in JOB_ALIVE_STATUSES
    except (NoSuchJobError, InvalidJobOperation):
        return False


class _LazyResultsRepository:
    def __init__(self):
        self._repo = None

    def _resolve(self):
        if self._repo is None:
            self._repo = PostgresResultsRepository()
        return self._repo

    def get(self, filename):
        return self._resolve().get(filename)


def create_app(
    job_store: JobStore = None,
    results_repo: ResultsRepository = None,
    enqueue=None,
    job_alive=None,
) -> FastAPI:
    job_store = job_store or RedisJobStore(REDIS_URL)
    results_repo = results_repo or _LazyResultsRepository()
    enqueue = enqueue or _default_enqueue
    job_alive = job_alive or _default_job_alive

    app = FastAPI(title="SVAR API")
    app.state.job_store = job_store
    app.state.results_repo = results_repo

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/sample_calls")
    def sample_calls():
        return sorted(
            f for f in os.listdir(SAMPLE_CALLS_DIR)
            if f.endswith((".wav", ".mp3", ".opus")) and not f.endswith("_denoised.wav")
        )

    @app.post("/api/analyze")
    def analyze(req: AnalyzeRequest):
        filepath = os.path.join(SAMPLE_CALLS_DIR, req.filename)
        if not os.path.exists(filepath):
            return JSONResponse(status_code=404, content={"error": f"{req.filename} not found"})
        if results_repo.get(req.filename) is not None:
            job_store.create(req.filename)
            job_store.finish(req.filename)
            return {"status": "completed", "message": "Already analyzed"}
        p = job_store.get(req.filename)
        if p and p["status"] == "running" and job_alive(req.filename):
            return {"status": "running", "message": "Pipeline already running"}
        try:
            job_store.create(req.filename)
            enqueue(req.filename)
        except Exception as e:
            try:
                job_store.delete(req.filename)
            except Exception:
                pass
            return JSONResponse(status_code=503, content={"error": f"queue unavailable: {e}"})
        return {"status": "queued", "message": "Pipeline queued"}

    @app.get("/api/progress")
    def progress(file: str = ""):
        p = job_store.get(file)
        if p:
            return p
        if file and results_repo.get(file) is not None:
            return {"status": "completed", "current_stage": "", "percent": 100, "stages": {}}
        return {"status": "idle", "percent": 0, "stages": {}}

    @app.post("/api/results")
    def results(req: AnalyzeRequest):
        res = results_repo.get(req.filename)
        if res is None:
            return JSONResponse(status_code=404, content={"error": "no results yet"})
        return res

    for route, keys in STAGE_SLICES.items():

        @app.post(f"/api/{route}")
        def stage_endpoint(req: AnalyzeRequest, _keys: list = Query(keys), _route: str = route):
            res = results_repo.get(req.filename)
            if res is None:
                return JSONResponse(status_code=404, content={"error": "no results yet"})
            return _stage_results(res, _keys)

    @app.get("/audio/{filename:path}")
    def audio(filename: str):
        filepath = os.path.join(SAMPLE_CALLS_DIR, urllib.parse.unquote(filename))
        if not os.path.exists(filepath):
            return JSONResponse(status_code=404, content={"error": "Audio file not found"})
        ext = os.path.splitext(filepath)[1].lower()
        return FileResponse(filepath, media_type=AUDIO_CONTENT_TYPES.get(ext, "application/octet-stream"))

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    dist = os.path.join(DASHBOARD_DIR, "dist")
    if os.path.isdir(dist):
        app.mount("/assets", StaticFiles(directory=os.path.join(dist, "assets")), name="assets")

    @app.get("/")
    def index():
        dist_index = os.path.join(dist, "index.html") if os.path.isdir(dist) else None
        fallback = os.path.join(DASHBOARD_DIR, "index.html")
        target = dist_index if dist_index and os.path.exists(dist_index) else fallback
        if not os.path.exists(target):
            return JSONResponse(status_code=404, content={"error": "dashboard not built"})
        return FileResponse(target, media_type="text/html")

    if os.path.isdir(dist):

        @app.get("/{path:path}")
        def dist_files(path: str):
            if path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found")
            if path:
                filepath = os.path.join(dist, path)
                if os.path.isfile(filepath):
                    ext = os.path.splitext(filepath)[1].lower()
                    return FileResponse(
                        filepath,
                        media_type=DIST_MIME_TYPES.get(ext, "application/octet-stream"),
                    )
            return JSONResponse(status_code=404, content={"error": "Not found"})

    return app


app = create_app()