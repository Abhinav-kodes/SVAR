import os
import torch
import numpy as np
from typing import Dict, Any, List, Optional

_MODEL = None


def _get_pyannote_pipeline():
    """Lazy-load pyannote diarization pipeline (singleton)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    from pyannote.audio import Pipeline
    _MODEL = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=True,
    )
    _MODEL.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return _MODEL


class DiarizationPipeline:
    """Speaker diarization pipeline using pyannote.audio.

    Runs pyannote/speaker-diarization-3.1 and computes embedding-based
    confidence scores via silhouette analysis on internal speaker embeddings.
    Also computes a rolling separability curve that auto-detects ambiguous
    regions anywhere in the call.
    """

    def __init__(self, num_speakers: int = 2, min_speakers: int = 2, max_speakers: int = 2):
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def process(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        if len(audio) == 0 or sr <= 0:
            return self._empty_result()

        pipeline = _get_pyannote_pipeline()

        captured = {}

        def _hook(step_name, step_artefact, **kwargs):
            captured[step_name] = step_artefact

        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        inputs = {"waveform": waveform, "sample_rate": sr}

        diarization = pipeline(
            inputs,
            num_speakers=self.num_speakers,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
            hook=_hook,
        )

        annotation = diarization.speaker_diarization

        raw_segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            raw_segments.append({
                "start_time_s": float(turn.start),
                "end_time_s": float(turn.end),
                "duration_s": float(turn.end - turn.start),
                "pyannote_speaker": speaker,
            })

        if not raw_segments:
            return self._empty_result()

        chunk_embeddings = captured.get("embeddings")
        segmentation = captured.get("segmentation")
        global_centroids = diarization.speaker_embeddings

        from diarization.speaker_embedder import (
            split_merged_turns,
            reassign_speakers,
            decode_region,
        )

        raw_segments = split_merged_turns(audio, sr, raw_segments)

        pyannote_segments = []
        for seg in raw_segments:
            pyannote_segments.append({
                "start_sample": int(round(seg["start_time_s"] * sr)),
                "end_sample": int(round(seg["end_time_s"] * sr)),
                "start_time_s": seg["start_time_s"],
                "end_time_s": seg["end_time_s"],
                "duration_s": seg["duration_s"],
                "speaker": "spk_0" if seg.get("pyannote_speaker") == raw_segments[0].get("pyannote_speaker") else "spk_1",
                "_needs_reclass": seg.get("_needs_reclass", False),
            })

        from diarization.confidence import (
            compute_rolling_separability,
            detect_low_separability_regions,
        )

        total_duration_s = float(len(audio) / sr)

        if chunk_embeddings is not None and segmentation is not None and global_centroids is not None:
            separability_curve = compute_rolling_separability(
                chunk_embeddings,
                segmentation.sliding_window,
                global_centroids,
                total_duration_s=total_duration_s,
            )
            low_regions = detect_low_separability_regions(separability_curve)
        else:
            separability_curve = []
            low_regions = []

        def _in_low_region(seg):
            center = (seg["start_time_s"] + seg["end_time_s"]) / 2.0
            return any(r["start_s"] <= center <= r["end_s"] for r in low_regions)

        if len(pyannote_segments) >= 3:
            sb_results = reassign_speakers(audio, sr, raw_segments, self.num_speakers)
            for ps, sb in zip(pyannote_segments, sb_results):
                ps["sb_d_agent"] = sb["sb_d_agent"]
                ps["sb_d_customer"] = sb["sb_d_customer"]
                ps["sb_margin"] = sb["sb_margin"]
                is_split = ps.get("_needs_reclass", False)
                in_low = _in_low_region(ps)
                if is_split or (in_low and ps["speaker"] != sb["speaker"]):
                    ps["speaker"] = sb["speaker"]

        segments = pyannote_segments

        if (
            chunk_embeddings is not None
            and segmentation is not None
            and len(segments) >= 3
        ):
            from diarization.change_detector import merge_false_splits, detect_single_speaker

            seg_np = segmentation.data if hasattr(segmentation, "data") else segmentation
            emb_np = chunk_embeddings if not hasattr(chunk_embeddings, "cpu") else chunk_embeddings.cpu().numpy()

            is_single_speaker, single_speaker_score = detect_single_speaker(
                segments, seg_np, emb_np,
            )
            if is_single_speaker:
                for seg in segments:
                    seg["speaker"] = "spk_0"
                segments = [segments[0].copy()]
                segments[0]["end_time_s"] = pyannote_segments[-1]["end_time_s"]
                segments[0]["end_sample"] = pyannote_segments[-1].get("end_sample", 0)
                segments[0]["duration_s"] = segments[0]["end_time_s"] - segments[0]["start_time_s"]
            else:
                segments = merge_false_splits(segments, seg_np, emb_np)

        if (
            chunk_embeddings is not None
            and segmentation is not None
            and global_centroids is not None
            and len(segments) >= 3
        ):
            from diarization.confidence import (
                compute_segment_confidence,
            )

            role_labels = sorted(set(
                s["speaker"] for s in segments if s["speaker"] != "overlap"
            ))
            segments = compute_segment_confidence(
                chunk_embeddings,
                segmentation.data,
                segmentation.sliding_window,
                segments,
                role_labels,
            )

            confidence_method = "silhouette"
        else:
            for seg in segments:
                seg["confidence"] = self._duration_confidence(seg["duration_s"])
                seg["uncertain"] = seg["confidence"] < 0.5
            separability_curve = []
            low_regions = []
            confidence_method = "duration_fallback"

        SHORT_TURN_THRESHOLD_S = 0.5
        SHORT_MARGIN_THRESHOLD = 0.1
        for seg in segments:
            if seg["duration_s"] < SHORT_TURN_THRESHOLD_S and abs(seg.get("sb_margin", 0.0)) < SHORT_MARGIN_THRESHOLD:
                seg["uncertain"] = True
                seg["needs_review"] = True

        agent_dur = 0.0
        customer_dur = 0.0
        overlap_dur = 0.0
        agent_count = 0
        customer_count = 0
        overlap_count = 0
        for seg in segments:
            dur = seg["duration_s"]
            if seg["speaker"] == "spk_0":
                agent_dur += dur
                agent_count += 1
            elif seg["speaker"] == "spk_1":
                customer_dur += dur
                customer_count += 1
            else:
                overlap_dur += dur
                overlap_count += 1

        total_speech_s = agent_dur + customer_dur
        agent_ratio = float(agent_dur / total_speech_s) if total_speech_s > 0 else 0.0
        customer_ratio = float(customer_dur / total_speech_s) if total_speech_s > 0 else 0.0

        result = {
            "talk_ratio": {
                "agent_duration_s": round(agent_dur, 2),
                "customer_duration_s": round(customer_dur, 2),
                "overlap_duration_s": round(overlap_dur, 2),
                "total_speech_s": round(total_speech_s, 2),
                "agent_ratio": round(agent_ratio, 4),
                "customer_ratio": round(customer_ratio, 4),
            },
            "speakers": {
                "spk_0": {"segment_count": agent_count, "has_baseline": False},
                "spk_1": {"segment_count": customer_count, "has_baseline": False},
                "overlap": {"segment_count": overlap_count},
            },
            "segments": segments,
            "separability": separability_curve,
            "low_separability_regions": low_regions,
            "confidence_method": confidence_method,
        }

        return result

    @staticmethod
    def offload_to_cpu():
        global _MODEL
        if _MODEL is None:
            return False
        del _MODEL
        _MODEL = None
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return True

    @staticmethod
    def _duration_confidence(dur_s: float) -> float:
        """Fallback confidence based on segment duration only."""
        if dur_s >= 2.0:
            return 1.0
        if dur_s >= 1.0:
            return 0.85
        if dur_s >= 0.5:
            return 0.65
        if dur_s >= 0.3:
            return 0.4
        return 0.2

    @staticmethod
    def _empty_result():
        return {
            "talk_ratio": {
                "agent_duration_s": 0.0,
                "customer_duration_s": 0.0,
                "overlap_duration_s": 0.0,
                "total_speech_s": 0.0,
                "agent_ratio": 0.0,
                "customer_ratio": 0.0,
            },
            "speakers": {
                "spk_0": {"segment_count": 0, "has_baseline": False},
                "spk_1": {"segment_count": 0, "has_baseline": False},
                "overlap": {"segment_count": 0},
            },
            "segments": [],
            "separability": [],
            "low_separability_regions": [],
            "confidence_method": "none",
        }
