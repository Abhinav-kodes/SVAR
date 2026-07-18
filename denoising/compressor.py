import numpy as np
from scipy.signal import lfilter

def compress_dynamic_range(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -20.0,
    ratio: float = 3.0,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    makeup_gain_db: float = 0.0,
) -> np.ndarray:
    """
    Fast RMS-based dynamic range compressor.
    
    Args:
        audio: 1D numpy array of audio samples
        sr: Sample rate in Hz
        threshold_db: Threshold in dB relative to full scale (0 dBFS)
        ratio: Compression ratio (e.g. 3.0 for 3:1)
        attack_ms: Attack time in milliseconds
        release_ms: Release time in milliseconds
        makeup_gain_db: Makeup gain in dB
        
    Returns:
        Compressed 1D numpy array
    """
    if len(audio) == 0:
        return audio
        
    threshold = 10 ** (threshold_db / 20.0)
    makeup_gain = 10 ** (makeup_gain_db / 20.0)
    
    # Calculate RMS energy with a 20ms sliding window
    window_len = int(sr * 0.02)
    if window_len < 1:
        window_len = 1
        
    squared = audio ** 2
    # Fast sliding window average using cumsum
    cumsum = np.cumsum(np.insert(squared, 0, 0))
    rms_energy = np.sqrt((cumsum[window_len:] - cumsum[:-window_len]) / window_len + 1e-10)
    
    # Pad to match original length
    pad_left = window_len // 2
    pad_right = len(audio) - len(rms_energy) - pad_left
    rms_energy = np.pad(rms_energy, (pad_left, pad_right), mode='edge')
    
    # Target gain (un-smoothed)
    target_gain = np.ones_like(audio)
    over_threshold = rms_energy > threshold
    target_gain[over_threshold] = (threshold / rms_energy[over_threshold]) ** (1.0 - 1.0 / ratio)
    
    # Smooth the target gain using a simple IIR filter (exponential moving average)
    # We use release time constant as it is the dominant smoothing factor
    tau = release_ms / 1000.0
    alpha = np.exp(-1.0 / (sr * tau))
    
    b = [1.0 - alpha]
    a = [1.0, -alpha]
    
    smoothed_gain = lfilter(b, a, target_gain)
    smoothed_gain = np.clip(smoothed_gain, 0.0, 1.0)
    
    compressed_audio = audio * smoothed_gain * makeup_gain
    return np.clip(compressed_audio, -1.0, 1.0)
