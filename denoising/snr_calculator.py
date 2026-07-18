import numpy as np

def calculate_snr(audio_data: np.ndarray, sr: int, debug: bool = False) -> float:
    frame_size = int(0.02 * sr) # 20ms frame
    hop_size   = frame_size // 2 # 10ms hop

    if len(audio_data) < frame_size:
        raise ValueError("Audio too short.")
    
    num_frames = (len(audio_data) - frame_size) // hop_size + 1
    
    # Slide a window across the audio to extract frames. 
    # For each frame, square the amplitudes to calculate positive power, 
    # then find the mean energy of that specific chunk.
    frame_energies = np.array([
        np.mean(audio_data[i * hop_size: i * hop_size + frame_size] ** 2)
        for i in range(num_frames)
    ])

    has_silence, silence_thresh = _has_sustained_silence(frame_energies, min_run_frames=5)

    if debug:
        path = "VAD" if has_silence else "spectral"
        print(f"  has_sustained_silence={has_silence} → {path} path")

    if has_silence:
        P_speech, P_noise = _vad_snr(frame_energies, silence_thresh)
    else:
        P_speech, P_noise = _spectral_snr(audio_data, frame_size, hop_size, num_frames)

    return 10 * np.log10(max(P_speech, 1e-10) / max(P_noise, 1e-10))


def _has_sustained_silence(
    frame_energies: np.ndarray,
    min_run_frames: int = 5,         # 5 × 10ms hop = 50ms minimum silence
    silence_floor_db: float = 25.0,  # quiet frames must be ≥ this many dB below median
) -> tuple[bool, float]:
    
    """
    Detects the presence of genuine, prolonged silence, ignoring brief volume drops caused by hard cuts or soft speech.

    A valid silent segment must satisfy two criteria:
      1. Duration: Must last for at least `min_run_frames` consecutively.
      2. Depth: The frame energy must drop at least `silence_floor_db` below the track's median energy. This prevents heavily compressed or edited audio from being falsely flagged.

    Returns:
        tuple[bool, float]: A flag indicating if sustained silence was found, and the energy threshold used to define it (passed down for VAD noise estimation).
    """
    median_energy = np.median(frame_energies)
    if median_energy <= 0:
        return False, 0.0

    # Absolute floor: quiet frames must be at least 25 dB below median
    absolute_floor = median_energy * 10 ** (-silence_floor_db / 10)
    quiet_threshold = min(np.percentile(frame_energies, 15), absolute_floor)

    max_run = 0
    current_run = 0
    
    for energy in frame_energies:
        if energy <= quiet_threshold:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    return max_run >= min_run_frames, quiet_threshold


def _vad_snr(frame_energies: np.ndarray, silence_thresh: float) -> tuple[float, float]:
    """
    Estimate speech/noise power using the silence threshold from
    _has_sustained_silence.  Noise is measured only from frames that
    are genuinely silent — not an arbitrary bottom-percentile that
    would include quieter speech in clean recordings.
    """
    noise_frames  = frame_energies[frame_energies <= silence_thresh]
    speech_frames = frame_energies[frame_energies > silence_thresh]

    # Use top 60th percentile of speech frames for robust speech power
    if len(speech_frames) > 0:
        P_speech = np.mean(speech_frames[speech_frames >= np.percentile(speech_frames, 60)])
    else:
        P_speech = np.mean(frame_energies)

    P_noise = np.mean(noise_frames) if len(noise_frames) > 0 else 1e-10
    return P_speech, P_noise


def _spectral_snr(
    audio_data: np.ndarray,
    frame_size: int,
    hop_size: int,
    num_frames: int,
    noise_pct: float = 5.0,
) -> tuple[float, float]:
    """
    Per-bin spectral percentile noise floor estimation.
    Works on cut audio by exploiting spectral sparsity:
    even dense speech leaves each frequency bin quiet some of the time.
    """
    window   = np.hanning(frame_size)
    spectra, frame_powers = [], []

    for i in range(num_frames):
        chunk = audio_data[i * hop_size: i * hop_size + frame_size]
        if len(chunk) < frame_size:
            break
        spectra.append(np.abs(np.fft.rfft(chunk * window)))
        frame_powers.append(np.mean(chunk ** 2))

    spectra      = np.array(spectra)       # (frames, freq_bins)
    frame_powers = np.array(frame_powers)

    noise_floor_per_bin = np.percentile(spectra, noise_pct, axis=0)
    P_noise  = np.mean(noise_floor_per_bin ** 2)
    P_speech = np.mean(frame_powers[frame_powers >= np.percentile(frame_powers, 60)])

    return P_speech, P_noise