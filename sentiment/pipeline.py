import os
import json
import time
from typing import Dict, Any, List, Optional

from sentiment.fusion_layer import fuse_segments
from sentiment.compliance_engine import analyze_call, analyze_transcript
from sentiment.qa_scorer import score_call
from sentiment.crm_note_generator import generate_crm_note


class SentimentPipeline:
    """
    End-to-end sentiment analysis pipeline.

    Consumes Part 2 diarized segments and runs:
      acoustic emotion → STT transcript → text emotion → fusion → compliance → QA scoring
    """

    def __init__(
        self,
        acoustic_pipeline=None,
        muril_model=None,
        muril_tokenizer=None,
        stt_transcriber=None,
        use_acoustic: bool = True,
        use_text: bool = True,
        weights_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.acoustic_pipeline = acoustic_pipeline
        self.muril_model = muril_model
        self.muril_tokenizer = muril_tokenizer
        self.stt_transcriber = stt_transcriber
        self.use_acoustic = use_acoustic
        self.use_text = use_text
        self.weights_path = weights_path
        self.device = device

    def _transcribe_segment(self, segment: Dict[str, Any], audio_path: str) -> str:
        """Transcribe a single segment using STT."""
        if self.stt_transcriber is None:
            return segment.get("text", "")
        try:
            start = float(segment.get("start", 0))
            end = float(segment.get("end", 0))
            if end <= start:
                return segment.get("text", "")
            result = self.stt_transcriber.transcribe_segment(audio_path, start, end)
            return result.get("text", "") if isinstance(result, dict) else str(result)
        except Exception as e:
            print(f"STT error: {e}")
            return segment.get("text", "")

    def _text_emotion(self, text: str) -> Dict[str, Any]:
        """Classify text emotion using MuRIL model."""
        if self.muril_model is None or self.muril_tokenizer is None:
            return {"emotion": "neutral", "confidence": 0.5, "sentiment": "neutral"}

        import torch
        from sentiment.models.dataset import EMOTION_ID2LABEL, SENTIMENT_ID2LABEL

        encoding = self.muril_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        self.muril_model.eval()
        with torch.no_grad():
            outputs = self.muril_model(input_ids=input_ids, attention_mask=attention_mask)

        emo_logits = outputs["emotion_logits"][0].cpu().numpy()
        sent_logits = outputs["sentiment_logits"][0].cpu().numpy()

        import numpy as np
        emo_probs = np.exp(emo_logits) / np.sum(np.exp(emo_logits))
        sent_probs = np.exp(sent_logits) / np.sum(np.exp(sent_logits))

        emo_idx = int(np.argmax(emo_probs))
        sent_idx = int(np.argmax(sent_probs))

        return {
            "emotion": EMOTION_ID2LABEL.get(emo_idx, "neutral"),
            "confidence": float(emo_probs[emo_idx]),
            "sentiment": SENTIMENT_ID2LABEL.get(sent_idx, "neutral"),
        }

    def process_call(
        self,
        segments: List[Dict[str, Any]],
        audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a single call's diarized segments.

        Args:
            segments: List of dicts from diarization pipeline with keys:
                      'speaker', 'start', 'end', 'audio_path' (optional), 'text' (optional).
            audio_path: Path to audio file for STT and acoustic analysis.

        Returns:
            Dict with 'emotion_timeline', 'compliance', 'qa', 'crm_note', 'summary'.
        """
        t0 = time.time()

        if not audio_path:
            audio_path = segments[0].get("audio_path", "") if segments else ""

        for seg in segments:
            if not seg.get("text") and audio_path:
                seg["text"] = self._transcribe_segment(seg, audio_path)

        acoustic_results = []
        if self.use_acoustic and self.acoustic_pipeline and audio_path:
            try:
                acoustic_results = self.acoustic_pipeline.process_call(segments, audio_path)
            except Exception as e:
                print(f"Acoustic pipeline error: {e}")

        text_results = []
        if self.use_text:
            for seg in segments:
                text = seg.get("text", "")
                if text.strip():
                    text_results.append(self._text_emotion(text))
                else:
                    text_results.append({"emotion": "neutral", "confidence": 0.0, "sentiment": "neutral"})
        else:
            text_results = [{"emotion": "neutral", "confidence": 0.5, "sentiment": "neutral"}] * len(segments)

        if acoustic_results and text_results:
            fused = fuse_segments(text_results, acoustic_results)
        elif text_results:
            fused = text_results
        elif acoustic_results:
            fused = acoustic_results
        else:
            fused = [{"emotion": "neutral", "confidence": 0.0, "sentiment": "neutral"}] * len(segments)

        for i, seg in enumerate(segments):
            if i < len(fused):
                seg["emotion"] = fused[i].get("emotion", "neutral")
                seg["sentiment"] = fused[i].get("sentiment", "neutral")
                seg["confidence"] = fused[i].get("confidence", 0.0)
                seg["fusion_source"] = fused[i].get("source", "none")

        compliance_result = analyze_call(segments)
        qa_result = score_call(
            segments, fused, compliance_result, weights_path=self.weights_path,
        )

        transcript = " ".join(seg.get("text", "") for seg in segments if seg.get("text"))
        crm_note = generate_crm_note(transcript, fused, compliance_result, qa_result)

        elapsed = time.time() - t0

        return {
            "emotion_timeline": fused,
            "compliance": compliance_result,
            "qa": qa_result,
            "crm_note": crm_note,
            "segments": segments,
            "processing_time_s": round(elapsed, 2),
            "summary": {
                "total_segments": len(segments),
                "total_violations": compliance_result.get("total_violations", 0),
                "qa_score": qa_result.get("qa_score", 0),
                "grade": qa_result.get("grade", "D"),
                "compliant": compliance_result.get("compliant", True),
            },
        }


def process_call(
    segments: List[Dict[str, Any]],
    audio_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Convenience function to process a call with default pipeline."""
    pipeline = SentimentPipeline(weights_path=weights_path, device=device)
    return pipeline.process_call(segments, audio_path)
