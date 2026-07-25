import os
import sys
import gc
import json
import time
import threading
import http.server
import socketserver
import urllib.parse
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from denoising.audio_loader import load_audio
from denoising.pipeline import DenoiserPipeline

PORT = 8050
SAMPLE_CALLS_DIR = os.path.join(REPO_ROOT, "data", "sample_calls")
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))

STAGES = [
    {"id": "denoise",    "label": "Denoising"},
    {"id": "diarize",    "label": "Diarization"},
    {"id": "stt",        "label": "Speech-to-Text"},
    {"id": "acoustic",   "label": "Acoustic Emotion"},
    {"id": "text_emo",   "label": "Text Emotion"},
    {"id": "compliance", "label": "Compliance"},
    {"id": "fusion",     "label": "Emotion Fusion"},
    {"id": "qa",         "label": "QA Scoring"},
    {"id": "crm",        "label": "CRM Note"},
]

_cache = {}
_progress = {}
_progress_lock = threading.Lock()
_denoiser = None
_diarizer = None
_stt = None
_acoustic_pipeline = None
_emotion_classifier = None
_role_engine = None


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)


def _get_denoiser():
    global _denoiser
    if _denoiser is None:
        log("Loading DenoiserPipeline...")
        _denoiser = DenoiserPipeline()
    return _denoiser


def _get_diarizer():
    global _diarizer
    if _diarizer is None:
        log("Loading DiarizationPipeline...")
        from diarization.pipeline import DiarizationPipeline
        _diarizer = DiarizationPipeline()
    return _diarizer


def _get_stt():
    global _stt
    if _stt is None:
        log("Loading Google Cloud STT...")
        from sentiment.stt.stt_transcriber import SpeechToTextTranscriber
        _stt = SpeechToTextTranscriber()
    return _stt


def _get_acoustic():
    global _acoustic_pipeline
    if _acoustic_pipeline is None:
        log("Loading acoustic emotion pipeline...")
        from sentiment.acoustic_emotion.acoustic_pipeline import analyze_acoustic_emotions
        _acoustic_pipeline = analyze_acoustic_emotions
    return _acoustic_pipeline


def _get_emotion_classifier():
    global _emotion_classifier
    if _emotion_classifier is not None:
        return _emotion_classifier
    from sentiment.emotion_classifier import classify_emotions_batch
    _emotion_classifier = classify_emotions_batch
    return _emotion_classifier


def _get_role_engine():
    global _role_engine
    if _role_engine is None:
        from svar.role_inference import RoleInferenceEngine
        _role_engine = RoleInferenceEngine()
    return _role_engine


def _ensure_cache(filename):
    filepath = os.path.join(SAMPLE_CALLS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filename} not found")
    if filename not in _cache:
        _cache[filename] = {}
    return _cache[filename]


def _send_json(handler, data, status=200):
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(json.dumps(data).encode("utf-8"))


# ── Progress tracking ──

def _init_progress(filename):
    with _progress_lock:
        _progress[filename] = {
            "status": "running",
            "current_stage": "",
            "percent": 0,
            "stages": {s["id"]: {"status": "pending", "time_s": 0} for s in STAGES},
            "error": None,
            "start_time": time.time(),
        }


def _set_stage(filename, stage_id, status):
    with _progress_lock:
        p = _progress.get(filename)
        if p:
            p["current_stage"] = stage_id
            if status == "running":
                p["stages"][stage_id]["status"] = "running"
            elif status == "done":
                p["stages"][stage_id]["status"] = "done"
            elif status == "error":
                p["stages"][stage_id]["status"] = "error"
            done = sum(1 for s in p["stages"].values() if s["status"] == "done")
            p["percent"] = round(done / len(STAGES) * 100)


def _finish_progress(filename, error=None):
    with _progress_lock:
        p = _progress.get(filename)
        if p:
            p["status"] = "error" if error else "completed"
            p["percent"] = 100 if not error else p["percent"]
            p["error"] = error
            p["time_s"] = round(time.time() - p["start_time"], 1)


def _timed_stage(filename, stage_id, fn):
    _set_stage(filename, stage_id, "running")
    t0 = time.time()
    try:
        result = fn()
        elapsed = round(time.time() - t0, 2)
        with _progress_lock:
            p = _progress.get(filename)
            if p:
                p["stages"][stage_id]["time_s"] = elapsed
        _set_stage(filename, stage_id, "done")
        log(f"  [{stage_id}] done in {elapsed}s")
        return result
    except Exception as e:
        _set_stage(filename, stage_id, "error")
        raise


# ── Pipeline stages ──

def stage_denoise(filename):
    c = _ensure_cache(filename)
    if "denoise_metrics" in c:
        return
    filepath = os.path.join(SAMPLE_CALLS_DIR, filename)
    audio, sr = load_audio(filepath, target_sr=16000)
    c["audio"] = audio
    c["sr"] = sr
    c["duration"] = round(float(len(audio) / sr), 2)
    clean_audio, metrics = _get_denoiser().process(audio, sr)
    c["clean_audio"] = clean_audio
    c["denoise_metrics"] = metrics


def stage_diarize(filename):
    c = _ensure_cache(filename)
    if "segments" in c:
        return
    if "clean_audio" not in c:
        stage_denoise(filename)
    res = _get_diarizer().process(c["clean_audio"], c["sr"])
    c["segments"] = res["segments"]
    c["talk_ratio"] = res.get("talk_ratio", {})
    c["separability"] = res.get("separability", [])
    c["confidence_method"] = res.get("confidence_method", "")
    import torch
    if torch.cuda.is_available():
        from diarization.pipeline import DiarizationPipeline
        DiarizationPipeline.offload_to_cpu()


def stage_stt(filename):
    c = _ensure_cache(filename)
    if c.get("transcribed"):
        return
    if "segments" not in c:
        stage_diarize(filename)
    stt = _get_stt()
    transcript = stt.transcribe_diarized_segments(c["segments"], c["clean_audio"], c["sr"], language="hi")
    c["segments"] = transcript
    c["transcribed"] = True

    from sentiment.role_resolver_llm import GeminiRoleResolver
    gemini = GeminiRoleResolver()
    gemini_mapping = gemini.resolve(c["segments"])

    if gemini_mapping:
        for seg in c["segments"]:
            spk = seg.get("speaker", "")
            if spk in gemini_mapping:
                seg["speaker"] = gemini_mapping[spk]
        c["role_resolution"] = {
            "mapping": gemini_mapping,
            "method": "gemini",
            "applied": True,
            "confidence": 0.9,
            "status": "resolved",
        }
        log("  [role] Gemini resolved roles: " + str(gemini_mapping))
    else:
        engine = _get_role_engine()
        resolution = engine.resolve(c["segments"])
        engine.apply_mapping(c["segments"], resolution)
        c["role_resolution"] = {
            "mapping": resolution.role_mapping,
            "method": resolution.method,
            "applied": resolution.applied,
        }
        if resolution.result:
            c["role_resolution"]["confidence"] = resolution.result.confidence
            c["role_resolution"]["status"] = resolution.result.status
            c["role_resolution"]["turns_used"] = resolution.result.turns_used
        elif resolution.method in ("heuristic", "fallback") and resolution.applied:
            c["role_resolution"]["confidence"] = 1.0
            c["role_resolution"]["status"] = "resolved"


def stage_acoustic(filename):
    c = _ensure_cache(filename)
    if c.get("acoustic_done"):
        return
    if "segments" not in c:
        stage_diarize(filename)
    analyze = _get_acoustic()
    clean_audio = c.get("clean_audio")
    sr = c.get("sr", 16000)
    if clean_audio is not None:
        analyze(clean_audio, sr, c["segments"])
    else:
        for seg in c["segments"]:
            seg["acoustic_emotion"] = {
                "emotion": "neutral", "confidence": 0.0,
                "indeterminate": True, "all_scores": {},
                "prosodic_features": {}, "deltas": {},
            }
    c["acoustic_done"] = True


def stage_text_emotion(filename):
    c = _ensure_cache(filename)
    if c.get("text_emo_done"):
        return
    if not c.get("transcribed"):
        stage_stt(filename)
    classify_batch = _get_emotion_classifier()
    texts = [seg.get("text", "") for seg in c["segments"]]
    batch_results = classify_batch(texts, batch_size=16)
    c["text_emotions"] = []
    for i, seg in enumerate(c["segments"]):
        r = batch_results[i] if i < len(batch_results) else {"emotion": "neutral", "sentiment": "neutral", "confidence": 0.0}
        c["text_emotions"].append({
            "emotion": r["emotion"],
            "confidence": r["confidence"],
            "sentiment": r["sentiment"],
        })
    c["text_emo_done"] = True


def stage_compliance(filename):
    c = _ensure_cache(filename)
    if "compliance" in c:
        return
    if not c.get("transcribed"):
        stage_stt(filename)
    from sentiment.compliance_engine import analyze_call
    c["compliance"] = analyze_call(c["segments"])


def stage_fusion(filename):
    c = _ensure_cache(filename)
    if c.get("fusion"):
        return
    if not c.get("text_emo_done"):
        stage_text_emotion(filename)
    if not c.get("acoustic_done"):
        stage_acoustic(filename)
    from sentiment.fusion_layer import fuse_segments
    fused = fuse_segments(c["text_emotions"], [
        seg.get("acoustic_emotion", {
            "emotion": "neutral", "confidence": 0.0, "indeterminate": True
        }) for seg in c["segments"]
    ])
    for i, seg in enumerate(c["segments"]):
        if i < len(fused):
            seg["emotion"] = fused[i]["emotion"]
            seg["sentiment"] = fused[i]["sentiment"]
            seg["confidence"] = fused[i]["confidence"]
            seg["fusion_source"] = fused[i].get("source", "text")
    c["fusion"] = fused


def stage_qa(filename):
    c = _ensure_cache(filename)
    if "qa" in c:
        return
    if "compliance" not in c:
        stage_compliance(filename)
    if not c.get("fusion"):
        stage_fusion(filename)
    from sentiment.qa_scorer import score_call
    emotions_with_speakers = []
    for seg in c["segments"]:
        emotions_with_speakers.append({
            "speaker": seg.get("speaker", ""),
            "emotion": seg.get("emotion", "neutral"),
            "sentiment": seg.get("sentiment", "neutral"),
            "confidence": seg.get("confidence", 0.0),
        })
    c["qa"] = score_call(c["segments"], emotions_with_speakers, c["compliance"])


def stage_crm(filename):
    c = _ensure_cache(filename)
    if "crm_note" in c:
        return
    if "qa" not in c:
        stage_qa(filename)
    from sentiment.crm_note_generator import generate_crm_note
    transcript = " ".join(s.get("text", "") for s in c["segments"] if s.get("text"))
    c["crm_note"] = generate_crm_note(transcript, c["fusion"], c["compliance"], c["qa"])


# ── Background pipeline runner ──

_pipeline_lock = threading.Lock()


def _free_gpu(label=""):
    global _role_engine, _emotion_classifier, _diarizer
    import torch
    if not torch.cuda.is_available():
        return
    _role_engine = None
    _emotion_classifier = None
    _diarizer = None
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  [gpu] freed memory ({label})")


def _run_pipeline(filename):
    with _pipeline_lock:
        _init_progress(filename)
        try:
            c = _ensure_cache(filename)
            t0 = time.time()

            _timed_stage(filename, "denoise", lambda: stage_denoise(filename))
            _timed_stage(filename, "diarize", lambda: stage_diarize(filename))

            with ThreadPoolExecutor(max_workers=2) as pool:
                stt_f = pool.submit(_timed_stage, filename, "stt", lambda: stage_stt(filename))
                ac_f = pool.submit(_timed_stage, filename, "acoustic", lambda: stage_acoustic(filename))
                stt_f.result()
                ac_f.result()

            _free_gpu("stt+acoustic done")

            with ThreadPoolExecutor(max_workers=2) as pool:
                te_f = pool.submit(_timed_stage, filename, "text_emo", lambda: stage_text_emotion(filename))
                co_f = pool.submit(_timed_stage, filename, "compliance", lambda: stage_compliance(filename))
                te_f.result()
                co_f.result()

            _timed_stage(filename, "fusion", lambda: stage_fusion(filename))
            _timed_stage(filename, "qa", lambda: stage_qa(filename))
            _timed_stage(filename, "crm", lambda: stage_crm(filename))

            total = round(time.time() - t0, 1)
            c["processing_time_s"] = total
            _finish_progress(filename)
            log(f"  Pipeline complete: {filename} ({total}s)")

        except Exception as e:
            import traceback
            log(f"  Pipeline ERROR: {e}")
            traceback.print_exc()
            _finish_progress(filename, error=str(e))


# ── API handler functions ──

def api_analyze(handler, filename):
    c = _ensure_cache(filename)
    with _progress_lock:
        existing = _progress.get(filename)
    if existing and existing["status"] == "running":
        return {"status": "running", "message": "Pipeline already running"}
    if c.get("crm_note") and existing and existing["status"] == "completed":
        return {"status": "completed", "message": "Already analyzed"}
    thread = threading.Thread(target=_run_pipeline, args=(filename,), daemon=True)
    thread.start()
    return {"status": "started", "message": "Pipeline started"}


def api_denoise(handler, filename):
    c = _ensure_cache(filename)
    return {
        "duration_s": c.get("duration"),
        "denoise_metrics": c.get("denoise_metrics"),
    }


def api_diarize(handler, filename):
    c = _ensure_cache(filename)
    return {
        "duration_s": c.get("duration"),
        "segments": c.get("segments"),
        "talk_ratio": c.get("talk_ratio", {}),
        "separability": c.get("separability", []),
        "confidence_method": c.get("confidence_method", ""),
        "role_resolution": c.get("role_resolution", {}),
    }


def api_transcribe(handler, filename):
    c = _ensure_cache(filename)
    return {
        "duration_s": c.get("duration"),
        "segments": c.get("segments"),
        "talk_ratio": c.get("talk_ratio", {}),
        "role_resolution": c.get("role_resolution", {}),
    }


def api_emotion(handler, filename):
    c = _ensure_cache(filename)
    return {
        "duration_s": c.get("duration"),
        "segments": c.get("segments"),
        "fusion": c.get("fusion", []),
    }


def api_compliance(handler, filename):
    c = _ensure_cache(filename)
    return {"compliance": c.get("compliance")}


def api_qa_score(handler, filename):
    c = _ensure_cache(filename)
    return {"qa": c.get("qa")}


def api_crm_note(handler, filename):
    c = _ensure_cache(filename)
    return {"crm_note": c.get("crm_note")}


API_ROUTES = {
    "/api/analyze": api_analyze,
    "/api/denoise": api_denoise,
    "/api/diarize": api_diarize,
    "/api/transcribe": api_transcribe,
    "/api/emotion": api_emotion,
    "/api/compliance": api_compliance,
    "/api/qa-score": api_qa_score,
    "/api/crm-note": api_crm_note,
}


class DashboardHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(os.path.join(DASHBOARD_DIR, "index.html"), "rb") as f:
                self.wfile.write(f.read())
            return

        if path == "/api/progress":
            filename = qs.get("file", [""])[0]
            with _progress_lock:
                p = _progress.get(filename, {"status": "idle", "percent": 0, "stages": {}})
            _send_json(self, p)
            return

        if path == "/api/sample_calls":
            files = sorted([
                f for f in os.listdir(SAMPLE_CALLS_DIR)
                if f.endswith(('.wav', '.mp3', '.opus')) and not f.endswith('_denoised.wav')
            ])
            _send_json(self, files)
            return

        if path.startswith("/audio/"):
            filename = urllib.parse.unquote(path[len("/audio/"):])
            filepath = os.path.join(SAMPLE_CALLS_DIR, filename)
            if os.path.exists(filepath):
                self.send_response(200)
                ct = "audio/mpeg" if filename.endswith(".mp3") else "audio/ogg" if filename.endswith(".opus") else "audio/wav"
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(os.path.getsize(filepath)))
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "Audio file not found")
                return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in API_ROUTES:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode("utf-8"))
                filename = payload.get("filename", "")

                handler_fn = API_ROUTES[path]
                result = handler_fn(self, filename)
                _send_json(self, result)
            except FileNotFoundError as e:
                _send_json(self, {"error": str(e)}, 404)
            except Exception as e:
                import traceback
                log(f"  ERROR: {e}")
                traceback.print_exc()
                _send_json(self, {"error": str(e)}, 500)
            return

        self.send_error(404, "Unknown endpoint")


class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    log(f"Starting SVAR Dashboard at http://localhost:{PORT}")
    log(f"Dashboard: {DASHBOARD_DIR}")
    log(f"Sample calls: {SAMPLE_CALLS_DIR}")
    log(f"APIs: {', '.join(sorted(API_ROUTES.keys()))}")
    server = ReusableTCPServer(("", PORT), DashboardHTTPRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
