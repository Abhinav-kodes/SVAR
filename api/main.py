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


def _default_enqueue(filename: str):
    import redis
    from rq import Queue
    conn = redis.Redis.from_url(REDIS_URL)
    Queue("svar", connection=conn).enqueue("pipeline.worker.run_pipeline_job", filename)


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
) -> FastAPI:
    job_store = job_store or RedisJobStore(REDIS_URL)
    results_repo = results_repo or _LazyResultsRepository()
    enqueue = enqueue or _default_enqueue

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
        p = job_store.get(req.filename)
        if p and p["status"] == "running":
            return {"status": "running", "message": "Pipeline already running"}
        if results_repo.get(req.filename) is not None:
            return {"status": "completed", "message": "Already analyzed"}
        try:
            job_store.create(req.filename)
            enqueue(req.filename)
        except Exception as e:
            return JSONResponse(status_code=503, content={"error": f"queue unavailable: {e}"})
        return {"status": "queued", "message": "Pipeline queued"}

    @app.get("/api/progress")
    def progress(file: str = ""):
        p = job_store.get(file)
        return p or {"status": "idle", "percent": 0, "stages": {}}

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