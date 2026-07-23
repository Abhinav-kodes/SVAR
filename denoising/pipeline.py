import numpy as np
from denoising.audio_loader import load_audio
from denoising.snr_calculator import calculate_snr
from denoising.clipping_detector import detect_clipping
from denoising.silence_ratio import calculate_silence_ratio
from denoising.enhancement_pipeline import enhance_audio
from denoising.spectral_denoiser import wiener_denoise

class DenoiserPipeline:
    def __init__(self, sr: int = 16000):
        """
        Initializes the integrated audio quality assessment and enhancement pipeline.
        
        Args:
            sr: Sample rate to standardise audio files to (default: 16000 Hz)
        """
        self.sr = sr
        
    def _measure_hum_drop(self, raw_audio: np.ndarray, clean_audio: np.ndarray) -> float:
        """
        Measures the decibel attenuation of the 50 Hz mains hum frequency.
        """
        if len(raw_audio) == 0:
            return 0.0
            
        fft_raw = np.abs(np.fft.rfft(raw_audio))
        fft_clean = np.abs(np.fft.rfft(clean_audio))
        freqs = np.fft.rfftfreq(len(raw_audio), 1 / self.sr)
        
        # Select band around 50 Hz (48 Hz to 52 Hz)
        hum_indices = np.where((freqs >= 48.0) & (freqs <= 52.0))[0]
        if len(hum_indices) == 0:
            # Fallback to closest single bin
            hum_indices = [np.argmin(np.abs(freqs - 50.0))]
            
        power_raw = np.sum(fft_raw[hum_indices] ** 2)
        power_clean = np.sum(fft_clean[hum_indices] ** 2)
        
        if power_clean == 0:
            return 100.0  # Infinite attenuation if fully silenced
            
        ratio = (power_raw + 1e-10) / (power_clean + 1e-10)
        return float(10 * np.log10(ratio))

    def process(self, raw_audio: np.ndarray, sr: int = None) -> tuple[np.ndarray, dict]:
        """
        Processes raw audio samples through the full enhancement pipeline and calculates metrics.
        
        Args:
            raw_audio: 1D numpy array of audio samples
            sr: Sample rate in Hz (defaults to self.sr if None)
            
        Returns:
            processed_audio: 1D numpy array (float32) of the enhanced/denoised audio
            metrics: dict containing quality metrics
        """
        if sr is None:
            sr = self.sr
            
        # 1. Pre-enhancement metrics
        snr_before = calculate_snr(raw_audio, sr)
        clipping_ratio = detect_clipping(raw_audio)
        silence_ratio = calculate_silence_ratio(raw_audio, sr)
        
        # 2. Apply enhancement chain (declip -> HPF -> notch -> compressor)
        enhanced_audio = enhance_audio(raw_audio, sr)
        
        # 3. Apply Wiener Denoiser
        final_audio = wiener_denoise(enhanced_audio, sr, noise_duration_s=0.5, gain_floor=0.15, over_subtraction=1.2)
        
        # 4. Post-enhancement metrics
        snr_after = calculate_snr(final_audio, sr)
        snr_improvement = snr_after - snr_before
        
        # 5. Assess hum removal (50 Hz band drop)
        hum_drop_db = self._measure_hum_drop(raw_audio, final_audio)
        hum_removed = hum_drop_db >= 15.0
        
        # 6. Dynamic range compression check (flagged True as compressor is always run in pipeline)
        compression_applied = True
        
        # 7. Determine quality grade
        # PASS if final SNR is >= 15dB or we improved it by >= 2dB (while maintaining positive final SNR)
        if snr_after >= 15.0 or (snr_improvement >= 2.0 and snr_after > 5.0):
            grade = "PASS"
        else:
            grade = "FAIL"
            
        metrics = {
            "snr_before_db": round(float(snr_before), 2),
            "snr_after_db": round(float(snr_after), 2),
            "snr_improvement_db": round(float(snr_improvement), 2),
            "clipping_ratio": round(float(clipping_ratio), 6),
            "silence_ratio": round(float(silence_ratio), 4),
            "hum_removed": bool(hum_removed),
            "compression_applied": bool(compression_applied),
            "audio_quality_grade": grade
        }
        
        return final_audio, metrics

    def process_file(self, filepath: str) -> tuple[np.ndarray, dict]:
        """
        Processes a raw audio file through the full enhancement pipeline and calculates metrics.
        
        Args:
            filepath: Path to the input audio file
            
        Returns:
            processed_audio: 1D numpy array (float32) of the enhanced/denoised audio
            metrics: dict containing quality metrics
        """
        raw_audio, sr = load_audio(filepath, target_sr=self.sr)
        return self.process(raw_audio, sr)

if __name__ == "__main__":
    import os
    import json
    
    pipeline = DenoiserPipeline()
    data_dir = "data/sample_calls"
    files = [f for f in os.listdir(data_dir) if f.lower().endswith((".mp3", ".opus", ".wav"))]
    raw_files = [f for f in files if "denoised" not in f.lower()]
    
    if raw_files:
        filepath = os.path.join(data_dir, raw_files[0])
        print(f"Testing DenoiserPipeline on: {raw_files[0]}")
        clean_audio, metrics = pipeline.process_file(filepath)
        print("Pipeline Execution Metrics:")
        print(json.dumps(metrics, indent=4))
    else:
        print("No raw audio files found in data/sample_calls/ to test.")
