import numpy as np
import librosa

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
    Extracts 13 MFCCs using optimized librosa library functions.
    
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
    if alpha > 0:
        audio = librosa.effects.preemphasis(audio, coef=alpha)
    
    # Convert seconds to sample counts
    frame_length = int(round(frame_size_s * sr))
    hop_length = int(round(frame_stride_s * sr))
    
    # 2. Extract Mel Spectrogram
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=nfft,
        hop_length=hop_length,
        win_length=frame_length,
        window='hamming',
        center=False,
        n_mels=num_filters,
        power=2.0,
        htk=True
    )
    
    # 3. Orthonormal DCT-II of the natural log
    log_S = np.log(np.maximum(S, 1e-10))
    mfcc = librosa.feature.mfcc(
        S=log_S,
        sr=sr,
        n_mfcc=num_ceps,
        dct_type=2,
        norm='ortho'
    )
    
    return mfcc.T.astype(np.float32)
