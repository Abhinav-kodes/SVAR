import soundfile as sf
import numpy as np
import librosa
import matplotlib.pyplot as plt

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

def estimate_noise_power(audio: np.ndarray, sr: int, frame_duration: float = 0.02, lowest_frac: float = 0.1) -> float:
    """Estimate noise power from the lowest-energy frames."""
    frame_size = int(frame_duration * sr)

    if len(audio) < frame_size:
        raise ValueError("Audio is too short for noise estimation.")

    energies = []
    for i in range(0, len(audio) - frame_size + 1, frame_size):
        frame = audio[i:i + frame_size]
        energies.append(np.mean(frame ** 2))

    energies = np.array(energies)

    k = max(1, int(lowest_frac * len(energies)))
    noise_power = np.mean(np.sort(energies)[:k])

    return float(noise_power)

def estimate_snr(audio: np.ndarray, noise_power: float) -> float:
    """Estimate SNR in dB."""
    signal_power = np.mean(audio ** 2)

    if noise_power <= 0:
        return float("inf")

    return float(10 * np.log10(signal_power / noise_power))

def plot_first_half_second(audio: np.ndarray, sr: int) -> None:
    """Plot first 0.5 seconds of audio."""

    samples = int(2 * sr)

    segment = audio[:samples]

    time = np.arange(len(segment)) / sr

    plt.figure(figsize=(12, 4))
    plt.plot(time, segment)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("First 2 Seconds of Audio")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data, sr = load_audio("data/sample_calls/sample_audio_2.opus")

    noise_power = estimate_noise_power(data, sr)
    snr_db = estimate_snr(data, noise_power)

    print("Shape:", data.shape)
    print("Sample rate:", sr)
    print("Estimated noise power:", noise_power)
    print(f"SNR: {snr_db:.2f} dB")
    plot_first_half_second(data, sr)