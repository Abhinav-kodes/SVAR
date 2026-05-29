import soundfile as sf
import numpy as np
import librosa

def load_audio(file_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio, convert to mono, and resample"""
    
    data, sr = sf.read(file_path)

    # Stereo -> mono
    if data.ndim == 2:
        data = np.mean(data, axis=1)

    # Resample to 16000Hz
    if sr != target_sr:
        data = librosa.resample(
            data,
            orig_sr=sr,
            target_sr=target_sr
        )
        sr = target_sr

    return data.astype(np.float32), sr

if __name__ == "__main__":
    data, sr = load_audio("data/sample_audio.mp3")

    print(data.shape, sr)