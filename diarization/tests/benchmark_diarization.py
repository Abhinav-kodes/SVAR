import os
import csv
import numpy as np
from diarization.pipeline import DiarizationPipeline
from denoising.audio_loader import load_audio


def run_diarization_benchmark():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_calls"))
    output_csv = os.path.join(data_dir, "diarization_results.csv")

    audio_files = [f for f in os.listdir(data_dir) if f.endswith(('.wav', '.mp3', '.opus')) and not f.endswith('_denoised.wav')]

    pipeline = DiarizationPipeline()
    results = []

    print(f"Running Diarization Benchmark on {len(audio_files)} sample call recordings...")

    for file_name in sorted(audio_files):
        file_path = os.path.join(data_dir, file_name)
        print(f"Processing: {file_name}...")

        try:
            audio, sr = load_audio(file_path, target_sr=16000)
            res = pipeline.process(audio, sr)
            talk_info = res["talk_ratio"]
            speakers_info = res["speakers"]

            row = {
                "filename": file_name,
                "duration_s": round(len(audio) / sr, 2),
                "total_speech_s": talk_info["total_speech_s"],
                "agent_speech_s": talk_info["agent_duration_s"],
                "customer_speech_s": talk_info["customer_duration_s"],
                "agent_ratio": talk_info["agent_ratio"],
                "customer_ratio": talk_info["customer_ratio"],
                "agent_segments": speakers_info["agent"]["segment_count"],
                "customer_segments": speakers_info["customer"]["segment_count"]
            }
            results.append(row)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Write to CSV
    fieldnames = [
        "filename", "duration_s", "total_speech_s", "agent_speech_s",
        "customer_speech_s", "agent_ratio", "customer_ratio",
        "agent_segments", "customer_segments"
    ]

    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nBenchmarking Complete. Results saved to: {output_csv}")
    print("\nSummary Table:")
    print(f"{'Filename':<25} | {'Total Speech':<12} | {'Agent %':<8} | {'Customer %':<10} | {'Segments (A/C)':<14}")
    print("-" * 75)
    for r in results:
        agent_pct = f"{r['agent_ratio']*100:.1f}%"
        cust_pct = f"{r['customer_ratio']*100:.1f}%"
        segs = f"{r['agent_segments']}/{r['customer_segments']}"
        print(f"{r['filename']:<25} | {r['total_speech_s']:<12} | {agent_pct:<8} | {cust_pct:<10} | {segs:<14}")


if __name__ == "__main__":
    run_diarization_benchmark()
