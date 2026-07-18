import numpy as np
from scipy.signal import stft, istft

def wiener_denoise(
    audio: np.ndarray,
    sr: int,
    noise_duration_s: float = 0.5,
    gain_floor: float = 0.1,
    over_subtraction: float = 1.0,
) -> np.ndarray:
    """
    Spectral Wiener Denoiser for background noise reduction.
    
    Args:
        audio: 1D numpy array of audio samples
        sr: Sample rate in Hz
        noise_duration_s: Duration in seconds at the start of the audio to estimate noise
        gain_floor: Minimum gain threshold (spectral floor) to prevent musical noise
        over_subtraction: Factor to scale estimated noise power for stronger suppression
        
    Returns:
        Denoised 1D numpy array (float32)
    """
    if len(audio) == 0:
        return audio
        
    # 1. STFT Setup: 25ms window (400 samples at 16kHz), 50% overlap (200 samples hop)
    nperseg = int(0.025 * sr)
    if nperseg < 4:
        nperseg = 4  # Minimum segment size
    noverlap = nperseg // 2
    
    f, t, Zxx = stft(
        audio,
        fs=sr,
        window='hamming',
        nperseg=nperseg,
        noverlap=noverlap,
        boundary='zeros',
        padded=True
    )
    
    # 2. Estimate Noise PSD from the initial silence region
    # Find all frame indices corresponding to the first noise_duration_s seconds
    noise_frame_indices = np.where(t <= noise_duration_s)[0]
    if len(noise_frame_indices) == 0:
        # Fall back to using the first frame if audio is too short
        noise_frames = Zxx[:, :1]
    else:
        noise_frames = Zxx[:, noise_frame_indices]
        
    # PSD of noise per frequency bin: mean(|X_noise|^2) over time frames
    noise_psd = np.mean(np.abs(noise_frames) ** 2, axis=1, keepdims=True)
    
    # 3. Wiener Gain Calculation
    # Power spectral density of the noisy signal
    noisy_psd = np.abs(Zxx) ** 2
    
    # Estimate clean signal power: max(noisy_power - alpha * noise_power, 0)
    clean_psd = np.maximum(noisy_psd - over_subtraction * noise_psd, 0.0)
    
    # Wiener Gain: P_clean / (P_clean + P_noise)
    gain = clean_psd / (clean_psd + noise_psd + 1e-10)
    
    # 4. Spectral Flooring to reduce "musical noise" bubbling artifacts
    gain = np.maximum(gain, gain_floor)
    
    # 5. Apply gain to the complex spectrogram
    Zxx_clean = gain * Zxx
    
    # 6. Reconstruct time-domain signal using iSTFT
    _, audio_clean = istft(
        Zxx_clean,
        fs=sr,
        window='hamming',
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=True
    )
    
    # Crop or pad to match the original audio length
    if len(audio_clean) > len(audio):
        audio_clean = audio_clean[:len(audio)]
    elif len(audio_clean) < len(audio):
        audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)), mode='constant')
        
    return audio_clean.astype(np.float32)

if __name__ == "__main__":
    import os
    from denoising.audio_loader import load_audio
    
    data_dir = "data/sample_calls"
    files = [f for f in os.listdir(data_dir) if f.lower().endswith((".mp3", ".opus", ".wav"))]
    if files:
        for filename in files:
            filepath = os.path.join(data_dir, filename)
            print("-" * 50)
            print(f"Testing Wiener denoiser on: {filename}")
            audio, sr = load_audio(filepath)
            denoised = wiener_denoise(audio, sr)
            print(f"  Shape: {audio.shape}")
            
            # Calculate difference to prove signal values were modified
            diff = np.abs(audio - denoised)
            print(f"  Max absolute difference: {np.max(diff):.6f}")
            print(f"  Mean absolute difference: {np.mean(diff):.6f}")
        print("-" * 50)
        print("Denoising test complete!")
    else:
        print("No sample calls found to test.")
