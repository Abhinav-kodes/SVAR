import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Denoising.vad_basic import compute_vad

def plot_speech_timeline(audio: np.ndarray, sr: int, filepath: str = "speech_timeline_plot.png", frame_duration_ms: int = 25):
    """
    Plot the audio waveform with speech regions highlighted.
    
    Args:
        audio: 1D numpy array of audio samples
        sr: Sample rate
        filepath: Path to save the generated plot
        frame_duration_ms: Duration of each frame in milliseconds
    """
    vad_mask = compute_vad(audio, sr, frame_duration_ms)
    
    frame_size = int(sr * (frame_duration_ms / 1000.0))
    time_axis = np.arange(len(audio)) / sr
    
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, audio, label='Audio Waveform', color='blue', alpha=0.5)
    
    # Highlight speech regions
    for i, is_speech in enumerate(vad_mask):
        if is_speech:
            start_time = i * frame_size / sr
            end_time = (i + 1) * frame_size / sr
            plt.axvspan(start_time, end_time, color='red', alpha=0.3, lw=0)
            
    # Add a proxy artist for the speech region in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    speech_patch = mpatches.Patch(color='red', alpha=0.3, label='Speech Region')
    by_label['Speech Region'] = speech_patch
    
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.title('Audio Waveform with Speech/Silence Timeline')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Timeline plot saved to {filepath}")
