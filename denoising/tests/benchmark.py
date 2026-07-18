import os
import csv
import soundfile as sf
from denoising.pipeline import DenoiserPipeline

def run_benchmark():
    data_dir = "data/sample_calls"
    pipeline = DenoiserPipeline()
    
    # 1. Discover files
    audio_extensions = (".mp3", ".opus", ".wav")
    all_files = [f for f in os.listdir(data_dir) if f.lower().endswith(audio_extensions)]
    # Skip any files that are already marked as denoised
    raw_files = [f for f in all_files if "denoised" not in f.lower()]
    
    if not raw_files:
        print("No raw audio files found in data/sample_calls/ to benchmark.")
        return
        
    print(f"Discovered {len(raw_files)} raw call(s) for benchmarking.")
    print("Running DenoiserPipeline...")
    print("-" * 100)
    print(f"{'Filename':<30} | {'SNR Before':<10} | {'SNR After':<9} | {'SNR Delta':<9} | {'Silence':<7} | {'Grade':<5}")
    print("-" * 100)
    
    results = []
    
    for filename in raw_files:
        filepath = os.path.join(data_dir, filename)
        
        try:
            # Process file
            clean_audio, metrics = pipeline.process_file(filepath)
            
            # Save denoised file
            name_part, _ = os.path.splitext(filename)
            output_filename = f"{name_part}_denoised.wav"
            output_filepath = os.path.join(data_dir, output_filename)
            sf.write(output_filepath, clean_audio, pipeline.sr)
            
            # Print row
            print(f"{filename:<30} | {metrics['snr_before_db']:<10.2f} | {metrics['snr_after_db']:<9.2f} | {metrics['snr_improvement_db']:<9.2f} | {metrics['silence_ratio']:<7.4f} | {metrics['audio_quality_grade']:<5}")
            
            # Keep record
            record = {
                "filename": filename,
                "output_filename": output_filename,
                **metrics
            }
            results.append(record)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    print("-" * 100)
    
    # 2. Write CSV report
    csv_path = os.path.join(data_dir, "benchmark_results.csv")
    if results:
        headers = list(results[0].keys())
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        print(f"Benchmark results successfully saved to: {csv_path}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    run_benchmark()
