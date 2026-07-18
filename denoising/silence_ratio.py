import numpy as np
from denoising.vad_basic import compute_vad

def calculate_silence_ratio(audio: np.ndarray, sr: int, frame_duration_ms: int = 25, threshold_multiplier: float = 1.5) -> float:
    """
    Calculate the ratio of silence frames to total frames in the audio signal.
    
    Args:
        audio: 1D numpy array of audio samples
        sr: Sample rate
        frame_duration_ms: Duration of each frame in milliseconds
        threshold_multiplier: Multiplier over the noise floor to set the speech threshold
        
    Returns:
        Float ratio of silence frames to total frames (0.0 to 1.0).
    """
    if len(audio) == 0:
        return 1.0
        
    vad_mask = compute_vad(audio, sr, frame_duration_ms, threshold_multiplier)
    
    if len(vad_mask) == 0:
        return 1.0
        
    silence_frames = np.sum(~vad_mask)
    return float(silence_frames / len(vad_mask))
