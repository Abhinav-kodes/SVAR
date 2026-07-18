import numpy as np
from denoising.highpass_filter import highpass_filter
from denoising.notch_filter import notch_filter
from denoising.compressor import compress_dynamic_range
from denoising.declipper import declip_audio

def enhance_audio(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float = 80.0,
    notch_hz: float = 50.0,
    threshold_db: float = -20.0,
    ratio: float = 3.0,
    makeup_gain_db: float = 0.0,
    declip_threshold: float = 0.99,
) -> np.ndarray:
    """
    Chains high-pass filtering, notch filtering, dynamic range compression, and declipping.
    
    Args:
        audio: 1D numpy array of audio samples
        sr: Sample rate in Hz
        cutoff_hz: Highpass filter cutoff frequency in Hz
        notch_hz: Notch filter center frequency in Hz
        threshold_db: Compressor threshold in dB
        ratio: Compressor ratio
        makeup_gain_db: Compressor makeup gain in dB
        declip_threshold: Declipper detection threshold
        
    Returns:
        Enhanced 1D numpy array
    """
    # 1. De-clip first (reconstruct peaks before applying filters/compression)
    x = declip_audio(audio, threshold=declip_threshold)
    
    # 2. High-pass filter to remove low rumble
    x = highpass_filter(x, sr, cutoff_hz=cutoff_hz)
    
    # 3. Notch filter to remove mains hum
    x = notch_filter(x, sr, notch_hz=notch_hz)
    
    # 4. Dynamic range compression
    x = compress_dynamic_range(
        x,
        sr,
        threshold_db=threshold_db,
        ratio=ratio,
        makeup_gain_db=makeup_gain_db
    )
    
    return x
