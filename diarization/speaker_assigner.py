import numpy as np
from typing import List, Dict, Any
from diarization.speaker_fingerprinter import extract_speaker_fingerprint
from diarization.baseline_builder import SpeakerBaselineTracker


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes cosine similarity between two 1D vectors.

    Args:
        vec1: 1D numpy array.
        vec2: 1D numpy array.

    Returns:
        Cosine similarity scalar value in range [-1.0, 1.0].
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 <= 1e-10 or norm2 <= 1e-10:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


class SpeakerAssigner:
    """
    Assigns speaker roles ('agent' or 'customer') to audio segments using voice identity fingerprints.
    """
    def __init__(
        self,
        similarity_threshold_delta: float = 0.05,
        new_speaker_threshold: float = 0.95,
        alpha: float = 0.1
    ):
        """
        Args:
            similarity_threshold_delta: Margin below which an assignment is marked uncertain (default: 0.05).
            new_speaker_threshold: Similarity limit below which a new voice profile is initialized (default: 0.95).
            alpha: Learning rate parameter for EWA baseline updates (default: 0.1).
        """
        self.similarity_threshold_delta = similarity_threshold_delta
        self.new_speaker_threshold = new_speaker_threshold
        self.alpha = alpha
        self.tracker = SpeakerBaselineTracker(vector_dim=32)

    def assign_speakers(
        self,
        segments: List[Dict[str, Any]],
        sr: int
    ) -> List[Dict[str, Any]]:
        """
        Assigns speaker roles ('agent' or 'customer') to audio segments.

        Args:
            segments: List of segment dictionaries (containing 'audio' array).
            sr: Audio sample rate in Hz.

        Returns:
            List of updated segment dictionaries with added keys:
                - 'speaker': 'agent' or 'customer'
                - 'fingerprint': 32-dim np.ndarray
                - 'confidence': float
                - 'uncertain': bool
        """
        updated_segments = []

        for segment in segments:
            audio = segment["audio"]
            fp = extract_speaker_fingerprint(audio, sr)

            has_agent = self.tracker.has_baseline("agent")
            has_customer = self.tracker.has_baseline("customer")

            if not has_agent:
                # 1st speech segment = Agent (opening greeting convention)
                speaker = "agent"
                self.tracker.set_baseline("agent", fp)
                confidence = 1.0
                uncertain = False
            elif not has_customer:
                # Evaluate if this segment belongs to a 2nd distinct speaker (Customer)
                agent_fp = self.tracker.get_baseline("agent")
                sim_agent = cosine_similarity(fp, agent_fp)

                if sim_agent < self.new_speaker_threshold:
                    # New distinct voice detected -> Customer
                    speaker = "customer"
                    self.tracker.set_baseline("customer", fp)
                    confidence = float(1.0 - sim_agent)
                    uncertain = False
                else:
                    speaker = "agent"
                    self.tracker.update_baseline("agent", fp, alpha=self.alpha)
                    confidence = float(sim_agent)
                    uncertain = False
            else:
                # Both Agent and Customer baselines exist
                agent_fp = self.tracker.get_baseline("agent")
                cust_fp = self.tracker.get_baseline("customer")

                sim_agent = cosine_similarity(fp, agent_fp)
                sim_customer = cosine_similarity(fp, cust_fp)

                if sim_agent >= sim_customer:
                    speaker = "agent"
                    confidence = sim_agent
                    diff = sim_agent - sim_customer
                else:
                    speaker = "customer"
                    confidence = sim_customer
                    diff = sim_customer - sim_agent

                uncertain = bool(diff < self.similarity_threshold_delta)

                # Update baseline if assignment is relatively confident
                if not uncertain:
                    self.tracker.update_baseline(speaker, fp, alpha=self.alpha)

            seg_copy = dict(segment)
            seg_copy["speaker"] = speaker
            seg_copy["fingerprint"] = fp
            seg_copy["confidence"] = float(confidence)
            seg_copy["uncertain"] = uncertain
            updated_segments.append(seg_copy)

        return updated_segments
