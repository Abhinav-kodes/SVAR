import os
import time
from typing import List, Optional

from api.config import SAMPLE_CALLS_DIR
from pipeline.job_store import JobStore, RedisJobStore
from pipeline.results_repo import PostgresResultsRepository, ResultsRepository
from pipeline import stages
from pipeline.stages import JobContext, free_gpu


def _timed_stage(filename: str, stage_id: str, job_store: JobStore, fn):
    job_store.update_stage(filename, stage_id, "running")
    t0 = time.time()
    try:
        result = fn()
        elapsed = round(time.time() - t0, 2)
        job_store.update_stage(filename, stage_id, "done", time_s=elapsed)
        return result
    except Exception:
        job_store.update_stage(filename, stage_id, "error")
        raise


def run_pipeline(
    filename: str,
    job_store: Optional[JobStore] = None,
    results_repo: Optional[ResultsRepository] = None,
) -> dict:
    job_store = job_store or RedisJobStore()
    results_repo = results_repo or PostgresResultsRepository()

    filepath = os.path.join(SAMPLE_CALLS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filename} not found")

    ctx = JobContext(filepath=filepath, filename=filename, cache={}, job_store=job_store)
    job_store.create(filename)
    t0 = time.time()
    try:
        _timed_stage(filename, "denoise", job_store, lambda: stages.stage_denoise(ctx))
        _timed_stage(filename, "diarize", job_store, lambda: stages.stage_diarize(ctx))
        _timed_stage(filename, "stt", job_store, lambda: stages.stage_stt(ctx))
        _timed_stage(filename, "acoustic", job_store, lambda: stages.stage_acoustic(ctx))
        free_gpu("stt+acoustic done")

        _timed_stage(filename, "text_emo", job_store, lambda: stages.stage_text_emotion(ctx))
        free_gpu("text_emo done")

        _timed_stage(filename, "compliance", job_store, lambda: stages.stage_audit(ctx))
        _timed_stage(filename, "fusion", job_store, lambda: stages.stage_fusion(ctx))
        _timed_stage(filename, "qa", job_store, lambda: stages.stage_qa(ctx))
        _timed_stage(filename, "crm", job_store, lambda: stages.stage_crm(ctx))

        results = {
            "duration_s": ctx.cache.get("duration"),
            "processing_time_s": round(time.time() - t0, 1),
            "segments": ctx.cache.get("segments"),
            "talk_ratio": ctx.cache.get("talk_ratio", {}),
            "denoise_metrics": ctx.cache.get("denoise_metrics"),
            "role_resolution": ctx.cache.get("role_resolution", {}),
            "fusion": ctx.cache.get("fusion", []),
            "compliance": ctx.cache.get("compliance"),
            "qa": ctx.cache.get("qa"),
            "crm_note": ctx.cache.get("crm_note"),
        }
        results_repo.save(filename, results)
        job_store.finish(filename)
        return results
    except Exception as e:
        job_store.finish(filename, error=str(e))
        raise


def _stage_results(results: dict, keys: List[str]) -> dict:
    return {k: results.get(k) for k in keys}
