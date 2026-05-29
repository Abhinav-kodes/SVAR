import numpy as np

def calculate_snr(audio_data: np.ndarray, sr: int) -> float:
    """
    Calculates the Signal-to-Noise Ratio (SNR) in dB using FFT.
    Searches the entire file to find the absolute quietest 0.5-second window.
    Args:
        audio_data: 1D numpy array of audio samples.
        sr: Sampling rate of the audio.
        
    Returns:
        float: SNR in decibels (dB).
    """
    # 1. Find the quietest 0.5s window in the audio to estimate the noise floor
    window_size = int(0.1 * sr)
    num_chunks = len(audio_data) // window_size
    min_energy = float('inf')
    noise_segment = None
    
    for i in range(num_chunks):
        chunk = audio_data[i * window_size : (i + 1) * window_size]
        energy = np.mean(chunk ** 2)
        if energy < min_energy:
            min_energy = energy
            noise_segment = chunk


    # 2. Compute FFT for noise
    N_noise = len(noise_segment)
    Y_noise = np.fft.rfft(noise_segment) / N_noise
    P_noise = np.sum(np.abs(Y_noise)**2)
    
    # 3. Compute FFT for full audio
    N_sig = len(audio_data)
    Y_sig = np.fft.rfft(audio_data) / N_sig
    P_sig = np.sum(np.abs(Y_sig)**2)
    
    # 4. Calculate SNR in dB
    # Avoid log of negative or zero values
    numerator = max(P_sig - P_noise, 1e-10)
    denominator = P_noise if P_noise > 0 else 1e-10
    
    #Formula:(power log)
    #SNR = 10 * log10(Signal Power / Noise Power)
    snr_db = 10 * np.log10(numerator / denominator)
    return snr_db

if __name__ == "__main__":
    from audio_loader import load_audio
    
    audio_path = "data/sample_audio_2.opus"
    data, sr = load_audio(audio_path)
    
    snr = calculate_snr(data, sr)
    print(f"File: {audio_path}")
    print(f"Calculated SNR: {snr:.2f} dB")
