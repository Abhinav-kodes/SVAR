import os
import json
from Denoising.audio_loader import load_audio
from Denoising.clipping_detector import detect_clipping
from Denoising.snr_calculator import calculate_snr
from Denoising.silence_ratio import calculate_silence_ratio
from Denoising.visualize_timeline import plot_speech_timeline
def run_quality_report():
    data_dir = "data"
    report = {}
    
    # 1. Gather all audio files in the data directory
    audio_extensions = (".mp3", ".opus", ".wav")
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(audio_extensions)]
    
    print(f"Starting processing for {len(files)} file(s)...")
    print("-" * 50)
    
    for i, filename in enumerate(files):
        filepath = os.path.join(data_dir, filename)
        print(f"Processing: {filename}...")
        
        try:
            # 2. Load and resample
            audio_data, sr = load_audio(filepath)
            
            # 3. Detect clipping
            clipping = detect_clipping(audio_data)
            
            # 4. Calculate SNR
            snr = calculate_snr(audio_data, sr)
            
            # 5. Calculate Silence Ratio
            silence = calculate_silence_ratio(audio_data, sr)
            
            # 6. Generate Timeline Plot for the first file
            if i == 0:
                plot_filepath = os.path.join(data_dir, f"timeline_{filename}.png")
                plot_speech_timeline(audio_data, sr, filepath=plot_filepath)
            
            # 7. Save results to report dict
            report[filename] = {
                "snr_db": round(float(snr), 2),
                "clipping_ratio": round(float(clipping), 6),
                "silence_ratio": round(float(silence), 4)
            }
            
            print(f"  -> SNR: {snr:.2f} dB, Clipping: {clipping:.6f}, Silence: {silence:.4f}")
        except Exception as e:
            print(f"  -> Error processing {filename}: {e}")
            
    # 6. Save report to quality_report.json
    with open("quality_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print("-" * 50)
    print("Quality report saved to 'quality_report.json'. Day 1 & 2 Complete!")

if __name__ == "__main__":
    run_quality_report()

