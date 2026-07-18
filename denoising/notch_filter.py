import numpy as np
from scipy.signal import iirnotch, filtfilt

def notch_filter(audio: np.ndarray, sr: int, notch_hz: float = 50.0, quality_factor: float = 30.0) -> np.ndarray:
    """
    IIR Notch filter to remove constant electrical mains hum (50 Hz / 60 Hz).
    
    Args:
        audio: 1D numpy array of audio samples
        sr: Sample rate in Hz
        notch_hz: Center frequency of the notch filter
        quality_factor: Q-factor (higher means narrower notch)
        
    Returns:
        Filtered 1D numpy array
    """
    if len(audio) == 0:
        return audio
    nyquist = 0.5 * sr
    w0 = notch_hz / nyquist
    b, a = iirnotch(w0, quality_factor)
    return filtfilt(b, a, audio)
