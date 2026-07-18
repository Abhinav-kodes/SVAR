import numpy as np
from scipy.signal import butter, filtfilt

def highpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 80.0, order: int = 5) -> np.ndarray:
    """
    Butterworth high-pass filter to remove low-frequency rumble.
    
    Args:
        audio: 1D numpy array of audio samples
        sr: Sample rate in Hz
        cutoff_hz: High-pass cutoff frequency in Hz
        order: Filter order
        
    Returns:
        Filtered 1D numpy array
    """
    if len(audio) == 0:
        return audio
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    # Zero-phase filtering (filtfilt) to prevent phase distortion
    return filtfilt(b, a, audio)
