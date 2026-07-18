import numpy as np
from scipy.interpolate import CubicSpline

def declip_audio(audio: np.ndarray, threshold: float = 0.99) -> np.ndarray:
    """
    Simple interpolation-based declipper.
    Replaces clipped samples (amplitude >= threshold) using cubic spline interpolation.
    
    Args:
        audio: 1D numpy array of audio samples
        threshold: Absolute value threshold for detecting clipping
        
    Returns:
        Declipped 1D numpy array
    """
    if len(audio) == 0:
        return audio
        
    declipped = audio.copy()
    abs_audio = np.abs(audio)
    clipped_indices = np.where(abs_audio >= threshold)[0]
    
    if len(clipped_indices) == 0:
        return declipped
        
    # Group consecutive clipped indices
    diffs = np.diff(clipped_indices)
    split_indices = np.where(diffs > 1)[0] + 1
    groups = np.split(clipped_indices, split_indices)
    
    for group in groups:
        if len(group) == 0:
            continue
            
        start_idx = group[0]
        end_idx = group[-1]
        
        # Take a window of unclipped samples before and after the clipped region
        window_size = 5
        left_samples = np.arange(max(0, start_idx - window_size), start_idx)
        right_samples = np.arange(end_idx + 1, min(len(audio), end_idx + 1 + window_size))
        
        # Filter out any samples in the window that are also clipped
        left_samples = left_samples[abs_audio[left_samples] < threshold]
        right_samples = right_samples[abs_audio[right_samples] < threshold]
        
        x_train = np.concatenate([left_samples, right_samples])
        y_train = audio[x_train]
        
        if len(x_train) < 4:
            # Not enough points for cubic spline, fall back
            continue
            
        try:
            cs = CubicSpline(x_train, y_train, extrapolate=False)
            x_pred = group
            y_pred = cs(x_pred)
            y_pred = np.clip(y_pred, -1.2, 1.2)
            declipped[group] = y_pred
        except ValueError:
            continue
            
    return np.clip(declipped, -1.0, 1.0)
