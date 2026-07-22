import numpy as np
from typing import List, Dict, Any


def segment_audio_by_pauses(
    audio: np.ndarray,
    sr: int,
    min_pause_duration_s: float = 0.4,
    min_segment_duration_s: float = 0.5,
    frame_size_s: float = 0.025,
    frame_stride_s: float = 0.010,
    silence_threshold_ratio: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Segments continuous audio into active speech turns separated by silence pauses.

    Args:
        audio: 1D numpy array of audio samples.
        sr: Sample rate in Hz.
        min_pause_duration_s: Minimum silence duration in seconds to trigger a split (default: 0.4s).
        min_segment_duration_s: Minimum duration in seconds for a valid speech segment (default: 0.5s).
        frame_size_s: Frame length in seconds for energy calculation (default: 25ms).
        frame_stride_s: Hop duration in seconds (default: 10ms).
        silence_threshold_ratio: Ratio of peak frame RMS energy below which a frame is silent.

    Returns:
        List of dictionaries with segment details:
        [
            {
                "start_sample": int,
                "end_sample": int,
                "start_time_s": float,
                "end_time_s": float,
                "duration_s": float,
                "audio": np.ndarray
            },
            ...
        ]
    """
    if len(audio) == 0 or sr <= 0:
        return []

    frame_len = int(round(frame_size_s * sr))
    hop_len = int(round(frame_stride_s * sr))

    if len(audio) < frame_len:
        # If audio is shorter than a frame, pad it or evaluate directly
        if len(audio) / sr >= min_segment_duration_s:
            return [{
                "start_sample": 0,
                "end_sample": len(audio),
                "start_time_s": 0.0,
                "end_time_s": float(len(audio) / sr),
                "duration_s": float(len(audio) / sr),
                "audio": audio
            }]
        return []

    num_frames = 1 + (len(audio) - frame_len) // hop_len
    rms_energies = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_len
        end = start + frame_len
        rms_energies[i] = np.sqrt(np.mean(audio[start:end] ** 2))

    max_rms = np.max(rms_energies)
    silence_thresh = max(1e-5, silence_threshold_ratio * max_rms)
    is_speech = rms_energies >= silence_thresh

    min_pause_frames = int(np.ceil(min_pause_duration_s / frame_stride_s))
    min_segment_samples = int(round(min_segment_duration_s * sr))

    # Group continuous speech runs and bridge short pause gaps
    speech_regions = []
    in_speech = False
    start_frame = 0

    for i in range(num_frames):
        if is_speech[i] and not in_speech:
            in_speech = True
            start_frame = i
        elif not is_speech[i] and in_speech:
            # Check if silence run is shorter than min_pause_frames
            silence_run = 0
            j = i
            while j < num_frames and not is_speech[j]:
                silence_run += 1
                j += 1

            if silence_run < min_pause_frames and j < num_frames:
                # Bridge gap, continue speech region
                continue
            else:
                in_speech = False
                end_frame = i
                speech_regions.append((start_frame, end_frame))

    if in_speech:
        speech_regions.append((start_frame, num_frames))

    segments = []
    for s_frame, e_frame in speech_regions:
        start_sample = s_frame * hop_len
        end_sample = min(len(audio), e_frame * hop_len + frame_len)
        duration_samples = end_sample - start_sample

        if duration_samples >= min_segment_samples:
            seg_audio = audio[start_sample:end_sample]
            start_t = float(start_sample / sr)
            end_t = float(end_sample / sr)
            segments.append({
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_time_s": start_t,
                "end_time_s": end_t,
                "duration_s": float(end_t - start_t),
                "audio": seg_audio
            })

    # Fallback if energy thresholding filtered everything out but audio duration is sufficient
    if len(segments) == 0 and len(audio) >= min_segment_samples:
        segments.append({
            "start_sample": 0,
            "end_sample": len(audio),
            "start_time_s": 0.0,
            "end_time_s": float(len(audio) / sr),
            "duration_s": float(len(audio) / sr),
            "audio": audio
        })

    return segments
