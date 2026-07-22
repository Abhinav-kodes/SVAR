import numpy as np
from typing import Dict, Optional


class SpeakerBaselineTracker:
    """
    Tracks and updates speaker voice baseline fingerprint vectors dynamically.
    """
    def __init__(self, vector_dim: int = 32):
        self.vector_dim = vector_dim
        self._baselines: Dict[str, np.ndarray] = {}

    def set_baseline(self, speaker_label: str, fingerprint: np.ndarray) -> None:
        """
        Sets the baseline vector for a speaker label.

        Args:
            speaker_label: Speaker role identifier (e.g. 'agent', 'customer').
            fingerprint: 1D numpy array representing speaker voice embedding.
        """
        if len(fingerprint) != self.vector_dim:
            raise ValueError(f"Fingerprint dimension must be {self.vector_dim}, got {len(fingerprint)}")

        norm = np.linalg.norm(fingerprint)
        if norm > 1e-10:
            self._baselines[speaker_label] = (fingerprint / norm).astype(np.float32)
        else:
            self._baselines[speaker_label] = np.zeros(self.vector_dim, dtype=np.float32)

    def update_baseline(self, speaker_label: str, fingerprint: np.ndarray, alpha: float = 0.1) -> None:
        """
        Updates an existing baseline vector using Exponential Weighted Average (EWA):
            baseline = (1 - alpha) * baseline + alpha * new_fingerprint

        Args:
            speaker_label: Speaker role identifier.
            fingerprint: New segment speaker fingerprint vector.
            alpha: Learning rate parameter for EWA (default: 0.1).
        """
        if speaker_label not in self._baselines:
            self.set_baseline(speaker_label, fingerprint)
            return

        old_baseline = self._baselines[speaker_label]
        updated = (1.0 - alpha) * old_baseline + alpha * fingerprint
        norm = np.linalg.norm(updated)
        if norm > 1e-10:
            self._baselines[speaker_label] = (updated / norm).astype(np.float32)
        else:
            self._baselines[speaker_label] = updated.astype(np.float32)

    def get_baseline(self, speaker_label: str) -> Optional[np.ndarray]:
        """Returns the baseline vector for a given speaker label."""
        return self._baselines.get(speaker_label)

    def has_baseline(self, speaker_label: str) -> bool:
        """Returns True if a baseline exists for the speaker label."""
        return speaker_label in self._baselines
