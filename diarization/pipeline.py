import numpy as np
from typing import List, Dict, Any, Optional
from diarization.pause_segmenter import segment_audio_by_pauses
from diarization.speaker_assigner import SpeakerAssigner


class DiarizationPipeline:
    """
    Master pipeline orchestrating audio pause segmentation and speaker role assignment.
    """
    def __init__(
        self,
        min_pause_duration_s: float = 0.4,
        min_segment_duration_s: float = 0.5,
        similarity_threshold_delta: float = 0.05,
        new_speaker_threshold: float = 0.95,
        alpha: float = 0.1
    ):
        """
        Args:
            min_pause_duration_s: Minimum pause duration in seconds to trigger segment boundary.
            min_segment_duration_s: Minimum duration for a valid speech segment.
            similarity_threshold_delta: Confidence margin below which assignment is marked uncertain.
            new_speaker_threshold: Similarity limit below which a new voice profile is initialized.
            alpha: Learning rate parameter for EWA baseline updates.
        """
        self.min_pause_duration_s = min_pause_duration_s
        self.min_segment_duration_s = min_segment_duration_s
        self.speaker_assigner = SpeakerAssigner(
            similarity_threshold_delta=similarity_threshold_delta,
            new_speaker_threshold=new_speaker_threshold,
            alpha=alpha
        )

    def process(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Processes audio to segment speech turns and assign speaker roles ('agent' vs 'customer').

        Args:
            audio: 1D numpy array of audio samples.
            sr: Sample rate in Hz.

        Returns:
            Dictionary containing:
                - 'talk_ratio': dict with agent_duration_s, customer_duration_s, total_speech_s, agent_ratio, customer_ratio
                - 'speakers': dict with agent and customer baseline info & segment counts
                - 'segments': list of segment dictionaries with timestamps and role labels
        """
        if len(audio) == 0 or sr <= 0:
            return {
                "talk_ratio": {
                    "agent_duration_s": 0.0,
                    "customer_duration_s": 0.0,
                    "total_speech_s": 0.0,
                    "agent_ratio": 0.0,
                    "customer_ratio": 0.0
                },
                "speakers": {
                    "agent": {"segment_count": 0, "has_baseline": False},
                    "customer": {"segment_count": 0, "has_baseline": False}
                },
                "segments": []
            }

        # 1. Segment audio by pauses
        raw_segments = segment_audio_by_pauses(
            audio,
            sr,
            min_pause_duration_s=self.min_pause_duration_s,
            min_segment_duration_s=self.min_segment_duration_s
        )

        # 2. Assign speaker roles
        assigned_segments = self.speaker_assigner.assign_speakers(raw_segments, sr)

        # 3. Calculate talk statistics
        agent_dur = 0.0
        customer_dur = 0.0
        agent_count = 0
        customer_count = 0

        clean_segments = []
        for seg in assigned_segments:
            speaker = seg.get("speaker", "agent")
            dur = seg.get("duration_s", 0.0)

            if speaker == "agent":
                agent_dur += dur
                agent_count += 1
            else:
                customer_dur += dur
                customer_count += 1

            # Prepare lightweight segment dict for reporting
            clean_segments.append({
                "start_sample": seg["start_sample"],
                "end_sample": seg["end_sample"],
                "start_time_s": seg["start_time_s"],
                "end_time_s": seg["end_time_s"],
                "duration_s": seg["duration_s"],
                "speaker": speaker,
                "confidence": seg["confidence"],
                "uncertain": seg["uncertain"]
            })

        total_speech_s = agent_dur + customer_dur
        agent_ratio = float(agent_dur / total_speech_s) if total_speech_s > 0 else 0.0
        customer_ratio = float(customer_dur / total_speech_s) if total_speech_s > 0 else 0.0

        agent_fp = self.speaker_assigner.tracker.get_baseline("agent")
        cust_fp = self.speaker_assigner.tracker.get_baseline("customer")

        return {
            "talk_ratio": {
                "agent_duration_s": round(agent_dur, 2),
                "customer_duration_s": round(customer_dur, 2),
                "total_speech_s": round(total_speech_s, 2),
                "agent_ratio": round(agent_ratio, 4),
                "customer_ratio": round(customer_ratio, 4)
            },
            "speakers": {
                "agent": {
                    "segment_count": agent_count,
                    "has_baseline": agent_fp is not None,
                    "baseline_vector": agent_fp.tolist() if agent_fp is not None else []
                },
                "customer": {
                    "segment_count": customer_count,
                    "has_baseline": cust_fp is not None,
                    "baseline_vector": cust_fp.tolist() if cust_fp is not None else []
                }
            },
            "segments": clean_segments
        }
