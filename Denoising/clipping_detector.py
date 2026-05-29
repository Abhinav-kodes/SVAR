import numpy as np
from audio_loader import load_audio

def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> float:
    """Detect clipping in audio"""
    
    clipped_count = np.sum(np.abs(audio) >= threshold)
    total_samples = len(audio)
    
    return clipped_count / total_samples

if __name__ == "__main__":
    audio, sr = load_audio("data/sample_audio.mp3")
    clipping_ratio = detect_clipping(audio, threshold=0.99)
    print(f"Clipping Ratio: {clipping_ratio:.6f} ({clipping_ratio * 100:.4f}%)")