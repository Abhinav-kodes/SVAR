import numpy as np

def pre_emphasis(signal: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """
    Applies a pre-emphasis filter to the audio signal.
    y[n] = x[n] - alpha * x[n-1]
    """
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def hz_to_mel(hz: float) -> float:
    """Converts a frequency in Hertz to the Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel: float) -> float:
    """Converts a Mel scale value back to Hertz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def get_mel_filterbank(sr: int, nfft: int, num_filters: int = 26) -> np.ndarray:
    """
    Generates a Mel filterbank matrix of shape (num_filters, nfft // 2 + 1).
    Uses continuous frequency interpolation to match librosa.
    """
    # FFT bin frequencies
    fft_freqs = np.linspace(0, sr / 2, nfft // 2 + 1)
    
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)
    
    # Mel frequency points (including edges)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    
    # Construct triangular filterbank
    fbank = np.zeros((num_filters, len(fft_freqs)))
    for m in range(1, num_filters + 1):
        f_m_minus = hz_points[m - 1]
        f_m = hz_points[m]
        f_m_plus = hz_points[m + 1]
        
        for k in range(len(fft_freqs)):
            freq = fft_freqs[k]
            if f_m_minus <= freq < f_m:
                fbank[m - 1, k] = (freq - f_m_minus) / (f_m - f_m_minus)
            elif f_m <= freq <= f_m_plus:
                fbank[m - 1, k] = (f_m_plus - freq) / (f_m_plus - f_m)
                
        # Slaney area normalization
        enorm = 2.0 / (f_m_plus - f_m_minus)
        fbank[m - 1] *= enorm
                
    return fbank

def get_dct2_matrix(num_filters: int, num_ceps: int = 13) -> np.ndarray:
    """
    Generates an orthonormal DCT-II matrix of shape (num_ceps, num_filters).
    """
    n = np.arange(num_filters)
    k = np.arange(num_ceps)[:, None]
    
    # Base DCT-II formula
    dct_coef = np.cos(np.pi * (n + 0.5) * k / num_filters)
    
    # Orthonormal scaling factors
    dct_coef[0] *= np.sqrt(1.0 / num_filters)
    dct_coef[1:] *= np.sqrt(2.0 / num_filters)
    
    return dct_coef

def extract_mfcc(
    audio: np.ndarray,
    sr: int,
    num_ceps: int = 13,
    num_filters: int = 26,
    nfft: int = 512,
    frame_size_s: float = 0.025,
    frame_stride_s: float = 0.010,
    alpha: float = 0.97
) -> np.ndarray:
    """
    Extracts 13 MFCCs from scratch for a given audio signal.
    
    Args:
        audio: 1D numpy array of audio samples
        sr: Sample rate in Hz
        num_ceps: Number of cepstral coefficients to keep (default: 13)
        num_filters: Number of Mel filters (default: 26)
        nfft: FFT size (default: 512)
        frame_size_s: Frame length in seconds (default: 25ms)
        frame_stride_s: Frame stride in seconds (default: 10ms)
        alpha: Pre-emphasis coefficient (default: 0.97)
        
    Returns:
        mfcc: numpy array of shape (num_frames, num_ceps)
    """
    if len(audio) == 0:
        return np.zeros((0, num_ceps), dtype=np.float32)
        
    # 1. Pre-emphasis filtering
    emphasized_audio = pre_emphasis(audio, alpha=alpha)
    
    # 2. Frame the signal (match librosa center=False behavior)
    frame_length = int(round(frame_size_s * sr))
    frame_step = int(round(frame_stride_s * sr))
    audio_len = len(emphasized_audio)
    
    if audio_len < nfft:
        return np.zeros((0, num_ceps), dtype=np.float32)
        
    num_frames = 1 + (audio_len - nfft) // frame_step
    
    # Build indexing matrix
    indices = np.tile(np.arange(0, nfft), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (nfft, 1)).T
    frames = emphasized_audio[indices.astype(np.int32, copy=False)]
    
    # 3. Apply Hamming Window (centered inside nfft frame)
    win = np.hamming(frame_length)
    pad_left = (nfft - frame_length) // 2
    pad_right = nfft - frame_length - pad_left
    win_padded = np.pad(win, (pad_left, pad_right), mode='constant')
    
    frames = frames * win_padded
    
    # 4. Compute Power Spectrum (|FFT|^2)
    mag_frames = np.abs(np.fft.rfft(frames, n=nfft, axis=-1))
    power_frames = mag_frames ** 2
    
    # 5. Get Mel Filterbank & Map to Mel scale
    fbank = get_mel_filterbank(sr, nfft, num_filters)
    mel_energies = np.dot(power_frames, fbank.T)
    
    # Avoid log of zero
    mel_energies = np.maximum(mel_energies, 1e-10)
    log_mel_energies = np.log(mel_energies)
    
    # 6. Apply DCT-II to obtain MFCCs
    dct_matrix = get_dct2_matrix(num_filters, num_ceps)
    mfcc = np.dot(log_mel_energies, dct_matrix.T)
    
    return mfcc.astype(np.float32)
