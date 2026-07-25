"""
Inference pipeline for the SVAR emotion recognition system.

Takes diarized segments → repaired turns → contextual inputs →
multi-task predictions → calibrated outputs.

Audio path: builds speaker baselines from early-call turns,
computes baseline-relative features, runs trajectory model,
and fuses with text model via learned fusion.
"""
from __future__ import annotations
import time
from typing import Dict, List, Optional, Any
import torch
import numpy as np

from .schemas import (
    DiarizedSegment, AnalysisTurn, ContextInput, TurnPrediction,
    EmotionOutput, SentimentOutput, InteractionStateOutput,
    ConductRiskOutput, UncertaintyOutput, ModalityInfo, AcousticOutput,
    EMOINHINDI_EMOTIONS, INTERACTION_STATES, CONDUCT_RISKS, SENTIMENTS,
)
from .turn_repair import repair_turns
from .context_builder import build_contexts
from .calibration import TemperatureScaler, EvidenceQualityGate
from .acoustic.speaker_baseline import SpeakerBaselineBuilder
from .acoustic.baseline_features import extract_baseline_features
from .acoustic.relative_features import build_relative_acoustic_vector, build_sequence_input


class EmotionPipeline:
    """
    End-to-end emotion analysis pipeline.

    Usage:
        pipeline = EmotionPipeline(
            text_model_path="checkpoints/muril_emotion/best_model.pt",
            audio_model_path="checkpoints/wavlm_emotion/wavlm_emotion.pt",
            trajectory_model_path="checkpoints/trajectory/trajectory.pt",
            fusion_model_path="checkpoints/fusion/fusion.pt",
        )
        predictions = pipeline.analyze(segments)
    """

    def __init__(
        self,
        text_model_path: Optional[str] = None,
        audio_model_path: Optional[str] = None,
        trajectory_model_path: Optional[str] = None,
        fusion_model_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        device: str = "cuda",
        text_model_name: str = "google/muril-base-cased",
        left_context: int = 2,
        right_context: int = 1,
        merge_gap_s: float = 0.7,
        min_semantic_duration_s: float = 0.8,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.left_context = left_context
        self.right_context = right_context
        self.merge_gap_s = merge_gap_s
        self.min_semantic_duration_s = min_semantic_duration_s

        # Models (loaded lazily)
        self._text_model = None
        self._audio_model = None
        self._trajectory_model = None
        self._fusion_model = None
        self._tokenizer = None

        # Calibration and quality gating
        self._scaler = TemperatureScaler()
        self._quality_gate = EvidenceQualityGate()
        self._baseline_builder = SpeakerBaselineBuilder()

        # Load models
        if text_model_path:
            self._load_text_model(text_model_path, text_model_name)
        if audio_model_path:
            self._load_audio_model(audio_model_path)
        if trajectory_model_path:
            self._load_trajectory_model(trajectory_model_path)
        if fusion_model_path:
            self._load_fusion_model(fusion_model_path)
        if calibration_path:
            self._load_calibration(calibration_path)

    def _load_text_model(self, path: str, model_name: str) -> None:
        from .models.contextual_muril import ContextualCallEmotionModel
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})
        self._text_model = ContextualCallEmotionModel(
            model_name=config.get("model_name", model_name),
            num_emotions=config.get("num_emotions", 16),
        ).to(self.device)
        self._text_model.load_state_dict(checkpoint["model_state_dict"])
        self._text_model.eval()
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(config.get("model_name", model_name))
        print(f"Loaded text model from {path}")

    def _load_audio_model(self, path: str) -> None:
        from .models.wavlm_emotion import WavLMEmotionModel
        self._audio_model = WavLMEmotionModel().to(self.device)
        state = torch.load(path, map_location=self.device, weights_only=True)
        self._audio_model.load_state_dict(state)
        self._audio_model.eval()
        print(f"Loaded audio model from {path}")

    def _load_trajectory_model(self, path: str) -> None:
        from .acoustic.trajectory_model import SpeakerShiftTemporalModel
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})
        self._trajectory_model = SpeakerShiftTemporalModel(
            input_dim=config.get("input_dim", 515),
            hidden_dim=config.get("hidden_dim", 128),
        ).to(self.device)
        self._trajectory_model.load_state_dict(checkpoint["model_state_dict"])
        self._trajectory_model.eval()
        print(f"Loaded trajectory model from {path}")

    def _load_fusion_model(self, path: str) -> None:
        from .models.fusion_net import MultimodalFusionNet
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})
        self._fusion_model = MultimodalFusionNet(
            audio_relative_dim=config.get("audio_relative_dim", 803),
        ).to(self.device)
        self._fusion_model.load_state_dict(checkpoint["model_state_dict"])
        self._fusion_model.eval()
        print(f"Loaded fusion model from {path}")

    def _load_calibration(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "temperatures" in state:
            self._scaler.temperatures.data = state["temperatures"].to(self.device)
        print(f"Loaded calibration from {path}")

    @torch.no_grad()
    def analyze(
        self,
        segments: List[DiarizedSegment],
        audio_arrays: Optional[Dict[int, np.ndarray]] = None,
        audio_sample_rate: int = 16000,
    ) -> List[TurnPrediction]:
        """
        Analyze diarized segments and return per-turn predictions.

        Args:
            segments: List of diarized ASR segments.
            audio_arrays: Optional dict mapping segment ID to mono audio array.
            audio_sample_rate: Sample rate of audio arrays.

        Returns:
            List of TurnPrediction objects.
        """
        t0 = time.time()

        # Step 1: Repair turns (inference only)
        turns = repair_turns(
            segments,
            merge_gap_s=self.merge_gap_s,
            min_semantic_duration_s=self.min_semantic_duration_s,
        )

        # Step 2: Build contexts
        contexts = build_contexts(
            turns,
            left_context=self.left_context,
            right_context=self.right_context,
        )

        # Step 3: Text inference
        text_outputs = self._infer_text(contexts)

        # Step 4: Build speaker baselines and extract acoustic features
        acoustic_outputs = None
        if audio_arrays:
            acoustic_outputs = self._infer_acoustic(turns, audio_arrays, audio_sample_rate)

        # Step 5: Fuse (if fusion model available and both streams ready)
        fused_outputs = None
        if text_outputs and acoustic_outputs and self._fusion_model:
            fused_outputs = self._fuse(text_outputs, acoustic_outputs, turns)

        # Step 6: Calibrate and build predictions
        predictions = self._calibrate_and_build(
            turns, contexts, text_outputs, acoustic_outputs, fused_outputs,
        )

        elapsed = time.time() - t0
        print(f"Emotion analysis: {len(predictions)} turns in {elapsed:.2f}s")
        return predictions

    @torch.no_grad()
    def _infer_text(self, contexts: List[ContextInput]) -> Optional[Dict[str, Any]]:
        if not self._text_model or not self._tokenizer or not contexts:
            return None

        texts = [c.full_sequence for c in contexts]
        encoded = self._tokenizer(
            texts, max_length=256, padding=True, truncation=True, return_tensors="pt",
        ).to(self.device)

        outputs = self._text_model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        return {k: v for k, v in outputs.items() if k != "cls_embedding" or True}

    def _infer_acoustic(
        self,
        turns: List[AnalysisTurn],
        audio_arrays: Dict[int, np.ndarray],
        sample_rate: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Real acoustic inference:
        1. Extract low-level features + WavLM embeddings per segment.
        2. Build per-speaker baselines from early-call turns.
        3. Compute baseline-relative features.
        4. Run trajectory model for arousal/shift/escalation.
        """
        if not audio_arrays:
            return None

        # ── Per-segment feature extraction ──
        prosody_vectors = {}
        wavlm_embeddings = {}

        for turn in turns:
            for sid in turn.segment_ids:
                if sid in audio_arrays:
                    audio = audio_arrays[sid]
                    if len(audio) < sample_rate * 0.3:
                        continue

                    prosody_vectors[sid] = extract_baseline_features(audio, sample_rate)

                    if self._audio_model:
                        import torch
                        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
                        if audio_tensor.shape[1] > sample_rate * 30:
                            audio_tensor = audio_tensor[:, :sample_rate * 30]
                        mask = torch.ones(1, audio_tensor.shape[1], dtype=torch.long, device=self.device)
                        audio_out = self._audio_model(input_values=audio_tensor, attention_mask=mask)
                        wavlm_embeddings[sid] = audio_out["embedding"].squeeze(0).cpu().numpy()

        if not prosody_vectors:
            return None

        # ── Build per-speaker baselines from early-call turns ──
        baselines: Dict[str, Any] = {}
        for speaker in ["agent", "customer", "Agent", "Customer"]:
            turn_records = []
            for turn in turns:
                if turn.speaker.lower() != speaker.lower():
                    continue
                has_audio = any(sid in prosody_vectors for sid in turn.segment_ids)
                has_wavlm = any(sid in wavlm_embeddings for sid in turn.segment_ids)

                # Get representative features for this turn
                turn_prosody = None
                turn_wavlm = None
                for sid in turn.segment_ids:
                    if sid in prosody_vectors:
                        turn_prosody = prosody_vectors[sid]
                    if sid in wavlm_embeddings:
                        turn_wavlm = wavlm_embeddings[sid]

                if turn_prosody is not None and turn_wavlm is not None:
                    turn_records.append({
                        "turn_id": turn.turn_id,
                        "duration": turn.duration,
                        "prosody_vector": turn_prosody,
                        "wavlm_embedding": turn_wavlm,
                        "audio_available": True,
                        "overlap": False,
                        "quality_score": turn.asr_confidence if turn.asr_confidence is not None else 0.8,
                    })

            if turn_records:
                baselines[speaker.lower()] = self._baseline_builder.build(speaker, turn_records)

        # ── Compute baseline-relative features per turn ──
        relative_vectors = []
        turn_durations = []
        turn_quality = []
        baseline_ready_flags = []
        arousal_values = []
        shift_values = []
        escalation_values = []

        for turn in turns:
            profile = baselines.get(turn.speaker.lower())

            # Get this turn's features
            turn_prosody = None
            turn_wavlm = None
            for sid in turn.segment_ids:
                if sid in prosody_vectors:
                    turn_prosody = prosody_vectors[sid]
                if sid in wavlm_embeddings:
                    turn_wavlm = wavlm_embeddings[sid]

            if turn_prosody is None or turn_wavlm is None:
                relative_vectors.append(np.zeros(803, dtype=np.float32))
                turn_durations.append(turn.duration)
                turn_quality.append(0.0)
                baseline_ready_flags.append(False)
                arousal_values.append(None)
                shift_values.append(None)
                escalation_values.append(None)
                continue

            rel = build_relative_acoustic_vector(turn_prosody, turn_wavlm, profile)
            relative_vectors.append(rel)
            turn_durations.append(turn.duration)
            turn_quality.append(turn.asr_confidence if turn.asr_confidence is not None else 0.8)
            baseline_ready_flags.append(profile.ready if profile else False)
            arousal_values.append(None)
            shift_values.append(None)
            escalation_values.append(None)

        # ── Run trajectory model per speaker ──
        if self._trajectory_model and relative_vectors:
            for speaker in set(t.speaker.lower() for t in turns):
                speaker_turns = [
                    (i, t) for i, t in enumerate(turns) if t.speaker.lower() == speaker
                ]
                if len(speaker_turns) < 2:
                    continue

                indices = [i for i, _ in speaker_turns]
                speaker_vecs = [relative_vectors[i] for i in indices]
                speaker_durations = [turn_durations[i] for i in indices]
                speaker_quality = [turn_quality[i] for i in indices]

                seq_input = build_sequence_input(speaker_vecs, speaker_durations, speaker_quality)
                if seq_input.shape[0] == 0 or seq_input.shape[1] == 0:
                    continue

                x = torch.from_numpy(seq_input).float().unsqueeze(0).to(self.device)
                mask = torch.ones(1, x.shape[1], dtype=torch.long, device=self.device)

                traj_out = self._trajectory_model(x, mask)

                for j, orig_idx in enumerate(indices):
                    if j < traj_out["arousal"].shape[1]:
                        arousal_values[orig_idx] = traj_out["arousal"][0, j].item()
                        shift_values[orig_idx] = traj_out["voice_shift"][0, j].item()
                        escalation_values[orig_idx] = traj_out["escalation"][0, j].item()

        # ── Build acoustic outputs ──
        acoustic_outputs_list = []
        for i, turn in enumerate(turns):
            acoustic_outputs_list.append(AcousticOutput(
                arousal=arousal_values[i],
                valence=None,  # WavLM valence head not yet trained
                voice_shift=shift_values[i],
                escalation=escalation_values[i],
                relative_features=relative_vectors[i],
                baseline_ready=baseline_ready_flags[i],
                quality_score=turn_quality[i],
                audio_available=any(sid in audio_arrays for sid in turn.segment_ids),
                acoustic_status=(
                    "ready" if baseline_ready_flags[i]
                    else "baseline_warming_up" if any(sid in prosody_vectors for sid in turn.segment_ids)
                    else "no_audio"
                ),
            ))

        return {
            "acoustic_outputs": acoustic_outputs_list,
            "relative_features": relative_vectors,
        }

    def _fuse(
        self,
        text_outputs: Dict[str, Any],
        acoustic_outputs: Dict[str, Any],
        turns: List[AnalysisTurn],
    ) -> Optional[Dict[str, Any]]:
        if not self._fusion_model:
            return None

        text_emb = text_outputs.get("cls_embedding")
        if text_emb is None:
            return None

        relative_features = acoustic_outputs["relative_features"]
        min_b = min(text_emb.shape[0], len(relative_features))
        text_emb = text_emb[:min_b]

        # Stack relative features into tensor
        audio_rel = torch.from_numpy(
            np.stack(relative_features[:min_b])
        ).float().to(self.device)

        text_available = torch.ones(min_b, dtype=torch.bool, device=self.device)
        audio_available = torch.tensor(
            [acoustic_outputs["acoustic_outputs"][i].baseline_ready for i in range(min_b)],
            dtype=torch.bool, device=self.device,
        )

        return self._fusion_model(
            text_embedding=text_emb,
            audio_relative=audio_rel,
            text_available=text_available,
            audio_available=audio_available,
        )

    def _calibrate_and_build(
        self,
        turns: List[AnalysisTurn],
        contexts: List[ContextInput],
        text_outputs: Optional[Dict[str, Any]],
        acoustic_outputs: Optional[Dict[str, Any]],
        fused_outputs: Optional[Dict[str, Any]],
    ) -> List[TurnPrediction]:
        predictions = []

        for i, (turn, ctx) in enumerate(zip(turns, contexts)):
            # Select logits from best available source
            if fused_outputs is not None and i < fused_outputs["emotion_logits"].shape[0]:
                emo_logits = fused_outputs["emotion_logits"][i]
                sent_logits = fused_outputs["sentiment_logits"][i]
                is_logits = fused_outputs["interaction_state_logits"][i]
                cr_logits = fused_outputs["conduct_risk_logits"][i]
                fusion_mode = "learned_fusion"
            elif text_outputs is not None and i < text_outputs["emotion_logits"].shape[0]:
                emo_logits = text_outputs["emotion_logits"][i]
                sent_logits = text_outputs["sentiment_logits"][i]
                is_logits = text_outputs["interaction_state_logits"][i]
                cr_logits = text_outputs["conduct_risk_logits"][i]
                fusion_mode = "text_only"
            else:
                continue

            # Calibrate
            cal = self._scaler.calibrate(emo_logits, sent_logits, is_logits, cr_logits)

            # Evidence quality assessment (deterministic, NOT a prediction)
            evidence_quality = self._quality_gate.assess(
                turn.text,
                turn.duration,
                turn.asr_confidence,
                cal.emotion_entropy,
                cal.emotion_margin,
            )

            # Determine primary emotion
            primary_emotion = max(cal.emotion, key=cal.emotion.get)

            # ── CRITICAL: output "uncertain", NEVER "neutral" ──
            is_uncertain = evidence_quality != "good"
            if is_uncertain:
                primary_emotion = "uncertain"

            # Get acoustic output
            ac_out = acoustic_outputs["acoustic_outputs"][i] if acoustic_outputs and i < len(acoustic_outputs.get("acoustic_outputs", [])) else AcousticOutput()

            pred = TurnPrediction(
                turn_id=turn.turn_id,
                speaker=turn.speaker,
                start=turn.start,
                end=turn.end,
                text=turn.text,
                segment_ids=turn.segment_ids,
                emotion=EmotionOutput(
                    probabilities=cal.emotion,
                    primary=primary_emotion,
                ),
                sentiment=SentimentOutput(
                    label=max(cal.sentiment, key=cal.sentiment.get),
                    confidence=max(cal.sentiment.values()),
                ),
                interaction_state=InteractionStateOutput(
                    label=max(cal.interaction_state, key=cal.interaction_state.get),
                    confidence=max(cal.interaction_state.values()),
                ),
                conduct_risk=ConductRiskOutput(
                    probabilities=cal.conduct_risk,
                ),
                uncertainty=UncertaintyOutput(
                    label="uncertain" if is_uncertain else "confident",
                    score=cal.emotion_entropy,
                    evidence_quality=evidence_quality,
                ),
                acoustic=ac_out,
                modality=ModalityInfo(
                    used_text=(text_outputs is not None),
                    used_audio=(acoustic_outputs is not None),
                    fusion_mode=fusion_mode,
                ),
                inference_mode=fusion_mode,
            )

            predictions.append(pred)

        return predictions
