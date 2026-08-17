import os
import gc
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict

from denoising.audio_loader import load_audio
from denoising.pipeline import DenoiserPipeline
from pipeline.job_store import JobStore


@dataclass
class JobContext:
    filepath: str
    filename: str
    cache: Dict[str, Any] = field(default_factory=dict)
    job_store: JobStore = None


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


_denoiser = None
_diarizer = None
_stt = None
_acoustic_pipeline = None
_emotion_classifier = None
_role_engine = None


# ── Pipeline stages ──

def stage_denoise(ctx: JobContext):
    c = ctx.cache
    if "denoise_metrics" in c:
        return
    filepath = ctx.filepath
    audio, sr = load_audio(filepath, target_sr=16000)
    c["audio"] = audio
    c["sr"] = sr
    c["duration"] = round(float(len(audio) / sr), 2)
    clean_audio, metrics = _get_denoiser().process(audio, sr)
    c["clean_audio"] = clean_audio
    c["denoise_metrics"] = metrics


def stage_diarize(ctx: JobContext):
    c = ctx.cache
    if "segments" in c:
        return
    if "clean_audio" not in c:
        stage_denoise(ctx)
    res = _get_diarizer().process(c["clean_audio"], c["sr"])
    c["segments"] = res["segments"]
    c["talk_ratio"] = res.get("talk_ratio", {})
    c["separability"] = res.get("separability", [])
    c["confidence_method"] = res.get("confidence_method", "")
    import torch
    if torch.cuda.is_available():
        from diarization.pipeline import DiarizationPipeline
        DiarizationPipeline.offload_to_cpu()


def stage_stt(ctx: JobContext):
    c = ctx.cache
    if c.get("transcribed"):
        return
    if "segments" not in c:
        stage_diarize(ctx)
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


def stage_acoustic(ctx: JobContext):
    c = ctx.cache
    if c.get("acoustic_done"):
        return
    if "segments" not in c:
        stage_diarize(ctx)
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


def stage_text_emotion(ctx: JobContext):
    c = ctx.cache
    if c.get("text_emo_done"):
        return
    if not c.get("transcribed"):
        stage_stt(ctx)
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


def stage_compliance(ctx: JobContext):
    c = ctx.cache
    if "compliance" in c:
        return
    if not c.get("transcribed"):
        stage_stt(ctx)
    from sentiment.compliance_engine import analyze_call
    c["compliance"] = analyze_call(c["segments"])


def stage_fusion(ctx: JobContext):
    c = ctx.cache
    if c.get("fusion"):
        return
    if not c.get("text_emo_done"):
        stage_text_emotion(ctx)
    if not c.get("acoustic_done"):
        stage_acoustic(ctx)
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


def stage_qa(ctx: JobContext):
    c = ctx.cache
    if "qa" in c:
        return
    if "compliance" not in c:
        stage_compliance(ctx)
    if not c.get("fusion"):
        stage_fusion(ctx)
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


def stage_crm(ctx: JobContext):
    c = ctx.cache
    if "crm_note" in c:
        return
    if "qa" not in c:
        stage_qa(ctx)
    from sentiment.crm_note_generator import generate_crm_note
    transcript = " ".join(s.get("text", "") for s in c["segments"] if s.get("text"))
    c["crm_note"] = generate_crm_note(transcript, c["fusion"], c["compliance"], c["qa"])


def stage_audit(ctx: JobContext):
    """Execute unified Gemini audit for compliance, QA scorecard & CRM note in 1 call."""
    c = ctx.cache
    if "compliance" in c and "qa" in c and "crm_note" in c:
        return
    if not c.get("transcribed"):
        stage_stt(ctx)
    if not c.get("fusion"):
        stage_fusion(ctx)

    if not any(s.get("text", "").strip() for s in c.get("segments", [])):
        c["audit_skipped"] = "no transcript"
        log("  [audit] skipped: no transcript")
        return

    try:
        from sentiment.audit_llm import run_unified_audit
        audit_res = run_unified_audit(c["segments"], c.get("talk_ratio"))
        if audit_res:
            c["compliance"] = audit_res["compliance"]
            c["qa"] = audit_res["qa"]
            c["crm_note"] = audit_res["crm_note"]
            log("  [audit] Unified Gemini Audit completed for compliance, QA, and CRM note")
            return
    except Exception as e:
        log(f"  [audit] Unified Gemini Audit fallback: {e}")

    if "compliance" not in c:
        stage_compliance(ctx)
    if "qa" not in c:
        stage_qa(ctx)
    if "crm_note" not in c:
        stage_crm(ctx)


def free_gpu(label: str = ""):
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