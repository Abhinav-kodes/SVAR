import unittest
import numpy as np
import librosa
from diarization.mfcc_extractor import extract_mfcc
from denoising.pipeline import DenoiserPipeline

class TestMFCCExtractor(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        self.duration = 2.0  # seconds
        self.t = np.arange(int(self.sr * self.duration)) / self.sr
        # Create a synthetic speech-like signal: sine wave with some harmonics and noise
        self.clean_signal = (
            0.5 * np.sin(2 * np.pi * 220 * self.t) +
            0.25 * np.sin(2 * np.pi * 440 * self.t) +
            0.1 * np.sin(2 * np.pi * 880 * self.t)
        )
        self.noise = 0.1 * np.random.randn(len(self.clean_signal))
        self.noisy_signal = self.clean_signal + self.noise

    def test_output_shape(self):
        mfcc = extract_mfcc(self.clean_signal, self.sr)
        
        # Expected frames: 1 + (len - 512) // 160
        # len = 32000. 1 + (32000 - 512) // 160 = 1 + 31488 // 160 = 1 + 196 = 197 frames
        self.assertEqual(mfcc.shape[1], 13)
        self.assertEqual(mfcc.shape[0], 197)

    def test_empty_audio(self):
        mfcc = extract_mfcc(np.array([]), self.sr)
        self.assertEqual(mfcc.shape, (0, 13))

    def test_match_librosa(self):
        # Apply pre-emphasis manually to match librosa inputs
        audio_pre = librosa.effects.preemphasis(self.clean_signal, coef=0.97)
        
        # Extract using our implementation
        mfcc_scratch = extract_mfcc(self.clean_signal, self.sr, num_ceps=13, num_filters=26, alpha=0.97)
        
        # Extract using librosa
        # 1. Mel Spectrogram with matching parameters (Hamming window, 512 FFT, 400 win, 160 hop, center=False)
        S_librosa = librosa.feature.melspectrogram(
            y=audio_pre,
            sr=self.sr,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window='hamming',
            center=False,
            n_mels=26,
            power=2.0,
            htk=True
        )
        
        # 2. Orthonormal DCT-II of the natural log
        log_S_librosa = np.log(np.maximum(S_librosa, 1e-10))
        mfcc_librosa = librosa.feature.mfcc(
            S=log_S_librosa,
            sr=self.sr,
            n_mfcc=13,
            dct_type=2,
            norm='ortho'
        ).T
        
        # Compare shape and values
        self.assertEqual(mfcc_scratch.shape, mfcc_librosa.shape)
        
        # Allow a small mean absolute error due to floating point and boundary padding differences
        mae = np.mean(np.abs(mfcc_scratch - mfcc_librosa))
        self.assertLess(mae, 0.05, f"Mean absolute error ({mae:.6f}) exceeds tolerance.")

    def test_enhanced_vs_noisy_noise_floor(self):
        # Generate a synthetic silence + sine wave signal
        np.random.seed(42)
        sr = 16000
        t = np.arange(int(sr * 1.5)) / sr
        signal = np.zeros_like(t)
        signal[int(sr * 0.5):] = 0.5 * np.sin(2 * np.pi * 440 * t[int(sr * 0.5):])
        noise = 0.05 * np.random.randn(len(t))
        noisy_signal = signal + noise
        
        pipeline = DenoiserPipeline(sr=sr)
        
        # Process the noisy signal using the pipeline
        temp_wav = "data/sample_calls/temp_noise_floor_test.wav"
        import soundfile as sf
        import os
        sf.write(temp_wav, noisy_signal, sr)
        
        try:
            denoised, _ = pipeline.process_file(temp_wav)
            
            # Extract MFCCs
            mfcc_noisy = extract_mfcc(noisy_signal, sr)
            mfcc_denoised = extract_mfcc(denoised, sr)
            
            # Silence frames (first 40 frames correspond to first 0.4s)
            energy_noisy = np.mean(mfcc_noisy[:40, 0])
            energy_denoised = np.mean(mfcc_denoised[:40, 0])
            
            # The denoised silence energy (MFCC 0) should be significantly lower (more negative)
            self.assertLess(energy_denoised, energy_noisy)
            
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

if __name__ == '__main__':
    unittest.main()
