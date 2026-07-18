import numpy as np

def detect_clipping(audio: np.ndarray, threshold: float = 0.99, min_run: int = 3) -> float:
    """
    Detect clipping in audio by checking for consecutive samples stuck at or above the threshold.
    This separates digital flatline distortion from clean, transient peaks.
    
    Args:
        audio: 1D numpy array of audio samples
        threshold: Amplitude threshold above which a sample is considered clipping (default: 0.99)
        min_run: Minimum number of consecutive samples at/above threshold to classify as clipping (default: 3)
        
    Returns:
        clipping_ratio: Float representing the fraction of clipped samples in the audio
    """
    if len(audio) < min_run:
        return 0.0
        
    is_clipped = np.abs(audio) >= threshold
    
    # Fast vectorized consecutive runs check using shifts
    runs_len = len(audio) - min_run + 1
    runs = np.ones(runs_len, dtype=bool)
    for i in range(min_run):
        runs &= is_clipped[i : runs_len + i]
            
    # Mark all samples that are part of any valid run
    clipped_mask = np.zeros_like(is_clipped, dtype=bool)
    for i in range(min_run):
        clipped_mask[i : runs_len + i] |= runs
            
    return float(np.sum(clipped_mask) / len(audio))

if __name__ == "__main__":
    from denoising.audio_loader import load_audio
    audio, sr = load_audio("data/sample_calls/sample_audio.mp3")
    clipping_ratio = detect_clipping(audio, threshold=0.99, min_run=3)
    print(f"Clipping Ratio (Consecutive runs >= 3): {clipping_ratio:.6f} ({clipping_ratio * 100:.4f}%)")