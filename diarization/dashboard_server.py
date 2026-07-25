import os
import sys
import gc
import json
import time
import http.server
import socketserver
import urllib.parse
import numpy as np
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from denoising.audio_loader import load_audio
from denoising.pipeline import DenoiserPipeline

PORT = 8050
SAMPLE_CALLS_DIR = os.path.join(REPO_ROOT, "data", "sample_calls")
DASHBOARD_DIR = os.path.join(REPO_ROOT, "dashboard")

_cache = {}
_denoiser = None
_diarizer = None
_stt = None
_acoustic_pipeline = None
_muril_model = None
_muril_tokenizer = None
_device = "cpu"


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


def _get_muril():
    global _muril_model, _muril_tokenizer, _device
    if _muril_model is not None:
        return _muril_model, _muril_tokenizer, _device
    import torch
    from sentiment.models.muril_model import MultiTaskMuRIL
    from transformers import AutoTokenizer

    checkpoint_path = os.path.join(REPO_ROOT, "sentiment", "models", "checkpoints", "best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if _stt is not None and hasattr(_stt, 'model') and _stt.model is not None:
        try:
            del _stt.model
            _stt.model = None
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    free_vram_gb = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0
    log(f"Loading MuRIL model... (free VRAM: {free_vram_gb:.2f} GB)")
    _device = "cuda" if free_vram_gb >= 0.8 else "cpu"
    try:
        model = MultiTaskMuRIL()
        state = torch.load(checkpoint_path, map_location=_device, weights_only=False)
        sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(sd, strict=False)
        model = model.to(_device).eval()
        _muril_model = model
        _muril_tokenizer = AutoTokenizer.from_pretrained(model.model_name)
        log(f"MuRIL loaded on {_device}")
    except Exception as e:
        log(f"MuRIL GPU load failed: {e}. Loading on CPU...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _device = "cpu"
        model = MultiTaskMuRIL()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(sd, strict=False)
        model = model.to("cpu").eval()
        _muril_model = model
        _muril_tokenizer = AutoTokenizer.from_pretrained(model.model_name)
        log(f"MuRIL loaded on CPU")
    return _muril_model, _muril_tokenizer, _device


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


def api_denoise(handler, filename):
    c = _ensure_cache(filename)
    if "denoise_metrics" in c:
        return c["denoise_metrics"]

    filepath = os.path.join(SAMPLE_CALLS_DIR, filename)
    t0 = time.time()
    audio, sr = load_audio(filepath, target_sr=16000)
    c["audio"] = audio
    c["sr"] = sr
    c["duration"] = round(float(len(audio) / sr), 2)
    log(f"  Audio loaded: {c['duration']}s ({time.time()-t0:.2f}s)")

    t0 = time.time()
    clean_audio, metrics = _get_denoiser().process(audio, sr)
    c["clean_audio"] = clean_audio
    c["denoise_metrics"] = metrics
    log(f"  Denoised: SNR {metrics['snr_before_db']:.1f}→{metrics['snr_after_db']:.1f}dB ({time.time()-t0:.2f}s)")

    return {
        "duration_s": c["duration"],
        "denoise_metrics": metrics,
    }


def api_diarize(handler, filename):
    c = _ensure_cache(filename)
    if "segments" in c:
        return {
            "duration_s": c["duration"],
            "segments": c["segments"],
            "talk_ratio": c.get("talk_ratio", {}),
            "separability": c.get("separability", []),
            "confidence_method": c.get("confidence_method", ""),
        }

    if "clean_audio" not in c:
        api_denoise(handler, filename)

    t0 = time.time()
    res = _get_diarizer().process(c["clean_audio"], c["sr"])
    c["segments"] = res["segments"]
    c["talk_ratio"] = res.get("talk_ratio", {})
    c["separability"] = res.get("separability", [])
    c["confidence_method"] = res.get("confidence_method", "")
    log(f"  Diarized: {len(res['segments'])} segments ({time.time()-t0:.1f}s)")

    import torch
    if torch.cuda.is_available():
        from diarization.pipeline import DiarizationPipeline
        DiarizationPipeline.offload_to_cpu()
        free = torch.cuda.mem_get_info()[0] / (1024**3)
        log(f"  GPU freed, {free:.1f}GB available")

    return {
        "duration_s": c["duration"],
        "segments": c["segments"],
        "talk_ratio": c["talk_ratio"],
        "separability": c["separability"],
        "confidence_method": c["confidence_method"],
    }


def api_transcribe(handler, filename):
    c = _ensure_cache(filename)
    if c.get("transcribed"):
        return {
            "duration_s": c["duration"],
            "segments": c["segments"],
            "talk_ratio": c.get("talk_ratio", {}),
        }

    if "segments" not in c:
        api_diarize(handler, filename)

    t0 = time.time()
    stt = _get_stt()
    sr = c["sr"]
    audio = c["clean_audio"]

    transcript = stt.transcribe_diarized_segments(c["segments"], audio, sr, language="hi")
    log(f"  Transcribed: {len(transcript)} segments ({time.time()-t0:.1f}s)")

    c["segments"] = transcript
    c["transcribed"] = True
    n_text = sum(1 for s in transcript if s.get("text"))
    log(f"  Non-empty: {n_text}/{len(transcript)} segments ({time.time()-t0:.1f}s)")

    return {
        "duration_s": c["duration"],
        "segments": c["segments"],
        "talk_ratio": c.get("talk_ratio", {}),
    }


def api_text_emotion(handler, filename):
    c = _ensure_cache(filename)
    if c.get("emotion_analyzed"):
        return {
            "duration_s": c["duration"],
            "segments": c["segments"],
            "fusion": c.get("fusion", []),
        }

    if not c.get("transcribed"):
        api_transcribe(handler, filename)

    t0 = time.time()
    model, tokenizer, device = _get_muril()
    from sentiment.models.dataset import EMOTION_ID2LABEL, SENTIMENT_ID2LABEL
    import torch

    text_results = []
    for seg in c["segments"]:
        text = seg.get("text", "")
        if text.strip():
            encoding = tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            emo_logits = outputs["emotion_logits"][0].cpu().numpy()
            sent_logits = outputs["sentiment_logits"][0].cpu().numpy()
            emo_probs = np.exp(emo_logits) / np.sum(np.exp(emo_logits))
            sent_probs = np.exp(sent_logits) / np.sum(np.exp(sent_logits))
            emo_idx = int(np.argmax(emo_probs))
            sent_idx = int(np.argmax(sent_probs))
            text_results.append({
                "emotion": EMOTION_ID2LABEL.get(emo_idx, "neutral"),
                "confidence": float(emo_probs[emo_idx]),
                "sentiment": SENTIMENT_ID2LABEL.get(sent_idx, "neutral"),
            })
        else:
            text_results.append({"emotion": "neutral", "confidence": 0.0, "sentiment": "neutral"})

    log(f"  Text emotion done ({time.time()-t0:.1f}s)")

    t1 = time.time()
    analyze = _get_acoustic()
    clean_audio = c.get("clean_audio")
    sr = c.get("sr", 16000)
    if clean_audio is not None:
        analyze(clean_audio, sr, c["segments"])
        log(f"  Acoustic emotion done ({time.time()-t1:.1f}s)")
    else:
        for seg in c["segments"]:
            seg["acoustic_emotion"] = {
                "emotion": "neutral", "confidence": 0.0,
                "indeterminate": True, "all_scores": {},
                "prosodic_features": {}, "deltas": {},
            }

    t2 = time.time()
    from sentiment.fusion_layer import fuse_segments
    fused = fuse_segments(text_results, [
        seg.get("acoustic_emotion", {
            "emotion": "neutral", "confidence": 0.0, "indeterminate": True
        }) for seg in c["segments"]
    ])
    log(f"  Fusion done ({time.time()-t2:.1f}s)")

    for i, seg in enumerate(c["segments"]):
        if i < len(fused):
            seg["emotion"] = fused[i]["emotion"]
            seg["sentiment"] = fused[i]["sentiment"]
            seg["confidence"] = fused[i]["confidence"]
            seg["fusion_source"] = fused[i].get("source", "text")

    c["fusion"] = fused
    c["emotion_analyzed"] = True
    log(f"  Emotion pipeline done ({time.time()-t0:.1f}s)")

    return {
        "duration_s": c["duration"],
        "segments": c["segments"],
        "fusion": c["fusion"],
    }


def api_compliance(handler, filename):
    c = _ensure_cache(filename)
    if "compliance" in c:
        return c["compliance"]

    if not c.get("emotion_analyzed"):
        api_text_emotion(handler, filename)

    from sentiment.compliance_engine import analyze_call
    c["compliance"] = analyze_call(c["segments"])
    return c["compliance"]


def api_qa_score(handler, filename):
    c = _ensure_cache(filename)
    if "qa" in c:
        return c["qa"]

    if "compliance" not in c:
        api_compliance(handler, filename)
    if "fusion" not in c:
        api_text_emotion(handler, filename)

    from sentiment.qa_scorer import score_call
    c["qa"] = score_call(c["segments"], c["fusion"], c["compliance"])
    return c["qa"]


def api_crm_note(handler, filename):
    c = _ensure_cache(filename)
    if "crm_note" in c:
        return c["crm_note"]

    if "qa" not in c:
        api_qa_score(handler, filename)

    from sentiment.crm_note_generator import generate_crm_note
    transcript = " ".join(s.get("text", "") for s in c["segments"] if s.get("text"))
    c["crm_note"] = generate_crm_note(transcript, c["fusion"], c["compliance"], c["qa"])
    return c["crm_note"]


def api_full(handler, filename):
    c = _ensure_cache(filename)
    api_denoise(handler, filename)
    api_diarize(handler, filename)
    api_transcribe(handler, filename)
    api_text_emotion(handler, filename)
    api_compliance(handler, filename)
    api_qa_score(handler, filename)
    api_crm_note(handler, filename)
    return {
        "duration_s": c["duration"],
        "processing_time_s": 0,
        "talk_ratio": c.get("talk_ratio", {}),
        "segments": c["segments"],
        "separability": c.get("separability", []),
        "confidence_method": c.get("confidence_method", ""),
        "denoise_metrics": c.get("denoise_metrics", {}),
        "fusion": c.get("fusion", []),
        "acoustic_results": [],
        "compliance": c.get("compliance", {}),
        "qa": c.get("qa", {}),
        "crm_note": c.get("crm_note", {}),
    }


API_ROUTES = {
    "/api/denoise": api_denoise,
    "/api/diarize": api_diarize,
    "/api/transcribe": api_transcribe,
    "/api/text-emotion": api_text_emotion,
    "/api/compliance": api_compliance,
    "/api/qa-score": api_qa_score,
    "/api/crm-note": api_crm_note,
    "/api/analyze": api_full,
}


class DashboardHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

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

        if path == "/api/model_status":
            _send_json(self, {
                "denoiser": _denoiser is not None,
                "diarizer": _diarizer is not None,
                "stt": _stt is not None,
                "muril": _muril_model is not None,
            })
            return

        if path == "/api/sample_calls":
            files = sorted([
                f for f in os.listdir(SAMPLE_CALLS_DIR)
                if f.endswith(('.wav', '.mp3', '.opus')) and not f.endswith('_denoised.wav')
            ])
            log(f"GET /api/sample_calls → {len(files)} files")
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
                log(f"POST {path} → '{filename}'")
                t0 = time.time()
                result = handler_fn(self, filename)
                log(f"  {path} done in {time.time()-t0:.1f}s")
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
