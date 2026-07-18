import numpy as np

def compute_vad(audio: np.ndarray, sr: int, frame_duration_ms: int = 25, threshold_multiplier: float = 1.5) -> np.ndarray:
    """
    Compute Voice Activity Detection (VAD) using an RMS energy threshold.
    
    Args:
        audio: 1D numpy array of audio samples
        sr: Sample rate
        frame_duration_ms: Duration of each frame in milliseconds
        threshold_multiplier: Multiplier over the noise floor to set the speech threshold
        
    Returns:
        Boolean numpy array where True indicates speech, False indicates silence. Length is number of frames.
    """
    frame_size = int(sr * (frame_duration_ms / 1000.0))
    
    if len(audio) < frame_size:
        return np.array([], dtype=bool)
        
    # Calculate total number of full frames
    num_frames = len(audio) // frame_size
    
    # Truncate audio to exact number of frames and reshape
    audio_frames = audio[:num_frames * frame_size].reshape(num_frames, frame_size)
    
    # Compute RMS energy for each frame
    frame_rms = np.sqrt(np.mean(audio_frames**2, axis=1) + 1e-10)
    
    # Estimate the noise floor as the 10th percentile of frame RMS energy
    noise_floor = np.percentile(frame_rms, 10)
    
    # Threshold for speech
    threshold = noise_floor * threshold_multiplier
    
    # Generate boolean mask
    vad_mask = frame_rms > threshold
    
    return vad_mask
