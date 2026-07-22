import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from diarization.speaker_fingerprinter import extract_speaker_fingerprint
from diarization.baseline_builder import SpeakerBaselineTracker


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


class SpeakerAssigner:
    """
    Assigns speaker roles ('agent' or 'customer') to audio segments.

    Supports both 192-dimensional Deep Neural Embeddings (ECAPA-TDNN) and
    classical 32-dimensional acoustic fingerprints.
    """

    def __init__(
        self,
        similarity_threshold_delta: float = 0.05,
        new_speaker_threshold: float = 0.95,
        alpha: float = 0.1,
        kmeans_n_init: int = 15,
        kmeans_random_state: int = 42,
        single_speaker_threshold: float = 2.0,
        overlap_ratio_threshold: float = 0.97,
    ):
        self.similarity_threshold_delta = similarity_threshold_delta
        self.new_speaker_threshold = new_speaker_threshold
        self.alpha = alpha
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_random_state = kmeans_random_state
        self.single_speaker_threshold = single_speaker_threshold
        self.overlap_ratio_threshold = overlap_ratio_threshold
        self.tracker = SpeakerBaselineTracker(vector_dim=32)

    def _smooth_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Re-assigns a segment only when BOTH immediate neighbours agree on the
        opposite label (isolated-flip smoothing).
        """
        if len(labels) < 3:
            return labels.copy()

        smoothed = labels.copy()
        for i in range(1, len(labels) - 1):
            left = labels[i - 1]
            right = labels[i + 1]
            if left == right and left != labels[i]:
                smoothed[i] = int(left)
        return smoothed

    def assign_speakers(
        self,
        segments: List[Dict[str, Any]],
        sr: int,
        fingerprinter: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Assigns speaker roles using Spectral / Cosine Clustering on segment fingerprints.

        Args:
            segments: List of segment dicts (must contain 'audio').
            sr: Audio sample rate in Hz.
            fingerprinter: Optional object/function with `extract_embedding(audio, sr)`
                or `extract_speaker_fingerprint`.

        Returns:
            List of updated segment dicts.
        """
        if not segments:
            return []

        # ── Step 1: Extract feature embeddings per segment ─────────────
        fingerprints = []
        for seg in segments:
            audio = seg["audio"]
            if fingerprinter is not None and hasattr(fingerprinter, "extract_embedding"):
                fp = fingerprinter.extract_embedding(audio, sr)
            elif fingerprinter is not None and callable(fingerprinter):
                fp = fingerprinter(audio, sr)
            else:
                fp = extract_speaker_fingerprint(audio, sr)
            fingerprints.append(fp)

        fingerprints = np.array(fingerprints, dtype=np.float32)
        fingerprints = np.nan_to_num(fingerprints, nan=0.0, posinf=0.0, neginf=0.0)
        vec_dim = fingerprints.shape[1]

        # ── Edge case: single segment ───────────────────────────────────
        if len(segments) == 1:
            seg_copy = dict(segments[0])
            seg_copy["speaker"] = "agent"
            seg_copy["fingerprint"] = fingerprints[0]
            seg_copy["confidence"] = 1.0
            seg_copy["uncertain"] = False
            self.tracker.set_baseline("agent", fingerprints[0][:32])
            return [seg_copy]

        # ── Step 2: Spectral / Cosine Affinity Clustering ──────────────
        is_deep = vec_dim > 32
        if is_deep:
            norms = np.linalg.norm(fingerprints, axis=1, keepdims=True) + 1e-10
            fps_norm = fingerprints / norms

            # Compute Cosine Affinity Matrix (scaled to [0, 1])
            sim_matrix = 0.5 * (1.0 + np.dot(fps_norm, fps_norm.T))

            try:
                clustering = SpectralClustering(
                    n_clusters=2,
                    affinity='precomputed',
                    random_state=self.kmeans_random_state
                )
                cluster_labels = clustering.fit_predict(sim_matrix)
            except Exception:
                kmeans = KMeans(n_clusters=2, n_init=self.kmeans_n_init, random_state=self.kmeans_random_state)
                cluster_labels = kmeans.fit_predict(fps_norm)

            X_s = fps_norm
        else:
            scaler = StandardScaler()
            try:
                X_s = scaler.fit_transform(fingerprints)
            except Exception:
                X_s = fingerprints.copy()

            kmeans = KMeans(
                n_clusters=2,
                n_init=self.kmeans_n_init,
                random_state=self.kmeans_random_state,
            )
            try:
                cluster_labels = kmeans.fit_predict(X_s)
            except Exception:
                cluster_labels = np.zeros(len(segments), dtype=int)

        # Single-speaker detection
        centroid_0 = np.mean(X_s[cluster_labels == 0], axis=0) if np.any(cluster_labels == 0) else np.zeros(X_s.shape[1])
        centroid_1 = np.mean(X_s[cluster_labels == 1], axis=0) if np.any(cluster_labels == 1) else np.zeros(X_s.shape[1])

        if is_deep:
            c0_norm = np.linalg.norm(centroid_0) + 1e-10
            c1_norm = np.linalg.norm(centroid_1) + 1e-10
            centroid_sim = float(np.dot(centroid_0, centroid_1) / (c0_norm * c1_norm))
            is_single_speaker = centroid_sim > 0.85
        else:
            centroid_dist = float(np.linalg.norm(centroid_0 - centroid_1))
            is_single_speaker = centroid_dist < self.single_speaker_threshold

        if is_single_speaker:
            cluster_labels = np.zeros(len(segments), dtype=int)

        # ── Step 3: Agent vs Customer mapping (segment 0 -> agent) ──────
        agent_cluster = int(cluster_labels[0])
        customer_cluster = 1 - agent_cluster
        has_customer = (not is_single_speaker) and np.any(cluster_labels == customer_cluster)

        # ── Step 4: Isolated-segment smoothing ──────────────────────────
        cluster_labels = self._smooth_labels(cluster_labels)

        # ── Step 5: Per-segment confidence & overlap calculation ─────────
        if has_customer:
            centroid_agent = np.mean(X_s[cluster_labels == agent_cluster], axis=0)
            centroid_customer = np.mean(X_s[cluster_labels == customer_cluster], axis=0)
            inter_dist = float(np.linalg.norm(centroid_agent - centroid_customer))
        else:
            centroid_agent = np.mean(X_s, axis=0)
            centroid_customer = centroid_agent
            inter_dist = 1.0

        updated_segments = []
        for i, (seg, fp, cl) in enumerate(zip(segments, fingerprints, cluster_labels)):
            if not has_customer:
                speaker = "agent"
                confidence = 1.0
                uncertain = False
            else:
                dist_agent = float(np.linalg.norm(X_s[i] - centroid_agent))
                dist_customer = float(np.linalg.norm(X_s[i] - centroid_customer))

                min_d = min(dist_agent, dist_customer) + 1e-10
                max_d = max(dist_agent, dist_customer) + 1e-10

                overlap_thresh = 0.95 if is_deep else self.overlap_ratio_threshold

                if (min_d / max_d) > overlap_thresh:
                    speaker = "overlap"
                    confidence = float(min_d / max_d)
                    uncertain = True
                elif int(cl) == agent_cluster:
                    speaker = "agent"
                    confidence = float(np.clip(dist_customer / (inter_dist + 1e-10), 0.0, 1.0))
                    uncertain = confidence < 0.3
                else:
                    speaker = "customer"
                    confidence = float(np.clip(dist_agent / (inter_dist + 1e-10), 0.0, 1.0))
                    uncertain = confidence < 0.3

            seg_copy = dict(seg)
            seg_copy["speaker"] = speaker
            seg_copy["fingerprint"] = fp
            seg_copy["confidence"] = float(confidence)
            seg_copy["uncertain"] = bool(uncertain)
            updated_segments.append(seg_copy)

        # Update legacy tracker baselines
        agent_mask = np.array([s["speaker"] == "agent" for s in updated_segments])
        cust_mask = np.array([s["speaker"] == "customer" for s in updated_segments])
        if agent_mask.any():
            self.tracker.set_baseline("agent", fingerprints[agent_mask].mean(axis=0)[:32])
        if cust_mask.any():
            self.tracker.set_baseline("customer", fingerprints[cust_mask].mean(axis=0)[:32])

        return updated_segments
