import os
import numpy as np
import matplotlib.pyplot as plt
from denoising.audio_loader import load_audio
from denoising.enhancement_pipeline import enhance_audio

def main():
    data_dir = "data/sample_calls"
    audio_extensions = (".mp3", ".opus", ".wav")
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(audio_extensions)]
    if not files:
        print("No audio files found in data/sample_calls/")
        return
        
    filename = files[0]
    filepath = os.path.join(data_dir, filename)
    print(f"Loading {filename} for enhancement visualization...")
    
    audio, sr = load_audio(filepath)
    
    # Run enhancement
    print("Running enhancement pipeline...")
    enhanced = enhance_audio(audio, sr)
    
    # Plot FFT comparison
    plt.figure(figsize=(14, 8))
    
    # 1. FFT plot
    plt.subplot(2, 1, 1)
    
    # Compute FFT
    fft_orig = np.abs(np.fft.rfft(audio))
    fft_enh = np.abs(np.fft.rfft(enhanced))
    freqs = np.fft.rfftfreq(len(audio), 1/sr)
    
    # Convert to dB
    fft_orig_db = 20 * np.log10(fft_orig + 1e-10)
    fft_enh_db = 20 * np.log10(fft_enh + 1e-10)
    
    plt.plot(freqs, fft_orig_db, label="Original Spectrum", color="blue", alpha=0.5)
    plt.plot(freqs, fft_enh_db, label="Enhanced Spectrum", color="red", alpha=0.5)
    
    plt.xlim(0, 1000)  # Focus on low frequencies to see 50Hz and 80Hz cutoff
    plt.title("Frequency Spectrum Comparison (0-1000 Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.legend()
    
    # 2. Waveform comparison
    plt.subplot(2, 1, 2)
    time_axis = np.arange(len(audio)) / sr
    plt.plot(time_axis, audio, label="Original Waveform", color="blue", alpha=0.5)
    plt.plot(time_axis, enhanced, label="Enhanced Waveform", color="green", alpha=0.5)
    plt.title("Waveform Comparison")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    output_plot = os.path.join(data_dir, f"enhancement_comparison_{filename}.png")
    plt.savefig(output_plot)
    plt.close()
    print(f"Enhancement visualization saved to {output_plot}")

if __name__ == "__main__":
    main()
