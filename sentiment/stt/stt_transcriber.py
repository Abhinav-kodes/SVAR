import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "credentials", "gcloud-stt.json",
)

CHUNK_SECONDS = 50
PROJECT_ID = "sunohq"
REGION = "us"
MAX_STT_WORKERS = 5
VAD_FRAME_MS = 25
VAD_PADDING_S = 0.3
VAD_GAP_TOLERANCE_S = 0.5


def _build_vad_chunks(audio_int16: np.ndarray, sr: int) -> List[Tuple[bytes, float]]:
    """Slice audio into <=CHUNK_SECONDS chunks covering only speech regions.

    Speech regions come from denoising.vad_basic.compute_vad (VAD_FRAME_MS
    frames). Gaps <= VAD_GAP_TOLERANCE_S between speech frames are merged;
    each region is padded by VAD_PADDING_S on both sides (clamped to audio
    bounds). Returns (chunk_bytes, absolute_offset_s) pairs.
    """
    from denoising.vad_basic import compute_vad

    vad = compute_vad(audio_int16.astype(np.float32), sr, frame_duration_ms=VAD_FRAME_MS)
    if not vad.any():
        return []

    frame_s = VAD_FRAME_MS / 1000.0
    gap_frames = max(1, int(round(VAD_GAP_TOLERANCE_S / frame_s)))

    regions = []
    start = None
    prev = -1
    for i, active in enumerate(vad):
        if active:
            if start is None or i - prev > gap_frames + 1:
                if start is not None:
                    regions.append((start, prev))
                start = i
            prev = i
    if start is not None:
        regions.append((start, prev))

    duration_s = len(audio_int16) / sr
    chunks = []
    chunk_samples = CHUNK_SECONDS * sr
    for start_f, end_f in regions:
        region_start = max(0.0, start_f * frame_s - VAD_PADDING_S)
        region_end = min(duration_s, (end_f + 1) * frame_s + VAD_PADDING_S)
        seg = audio_int16[int(region_start * sr) : int(region_end * sr)]
        if len(seg) == 0:
            continue
        for i in range(0, len(seg), chunk_samples):
            piece = seg[i : i + chunk_samples]
            if len(piece) < sr:
                break
            chunks.append((piece.tobytes(), region_start + i / sr))
    return chunks


class SpeechToTextTranscriber:
    """
    Google Cloud Speech-to-Text V2 with Chirp 3 model.
    Chunks long audio into ≤50s segments and sends them in PARALLEL.
    Diarization handled by local pyannote.
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google.cloud.speech_v2 import SpeechClient
            from google.api_core.client_options import ClientOptions
            self._client = SpeechClient(
                client_options=ClientOptions(
                    api_endpoint=f"{REGION}-speech.googleapis.com",
                )
            )
        return self._client

    def _transcribe_chunk(
        self,
        chunk_bytes: bytes,
        chunk_offset_s: float,
        config,
        recognizer_path: str,
    ) -> List[Dict[str, Any]]:
        """Transcribe a single audio chunk. Thread-safe — called in parallel."""
        from google.cloud.speech_v2.types import cloud_speech

        client = self._get_client()
        request = cloud_speech.RecognizeRequest(
            recognizer=recognizer_path,
            config=config,
            content=chunk_bytes,
        )
        result = client.recognize(request=request)

        words = []
        for r in result.results:
            alt = r.alternatives[0]
            for w in alt.words:
                words.append({
                    "start": round(w.start_offset.total_seconds() + chunk_offset_s, 3),
                    "end": round(w.end_offset.total_seconds() + chunk_offset_s, 3),
                    "text": w.word,
                    "probability": round(w.confidence or 0.8, 3),
                })
        return words

    def _transcribe_api(
        self,
        audio: np.ndarray,
        sr: int,
        language: str = "hi",
    ) -> Optional[List[Dict[str, Any]]]:
        from google.cloud.speech_v2.types import cloud_speech

        if sr != 16000:
            import scipy.signal
            target = int(round(len(audio) * 16000 / sr))
            audio = scipy.signal.resample(audio, target)
            sr = 16000

        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

        lang_map = {"hi": "hi-IN", "en": "en-US"}
        api_lang = lang_map.get(language, language)
        recognizer_path = f"projects/{PROJECT_ID}/locations/{REGION}/recognizers/_"

        config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sr,
                audio_channel_count=1,
            ),
            language_codes=[api_lang],
            model="chirp_3",
            features=cloud_speech.RecognitionFeatures(
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
            ),
        )

        chunk_samples = CHUNK_SECONDS * sr
        chunks = []
        for i in range(0, len(audio_int16), chunk_samples):
            chunk = audio_int16[i : i + chunk_samples]
            if len(chunk) < sr:
                break
            chunks.append((chunk.tobytes(), i / sr))

        if not chunks:
            return None

        all_words = []
        try:
            num_workers = min(len(chunks), MAX_STT_WORKERS)
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = {
                    pool.submit(
                        self._transcribe_chunk, chunk_bytes, offset, config, recognizer_path
                    ): idx
                    for idx, (chunk_bytes, offset) in enumerate(chunks)
                }
                results = [None] * len(chunks)
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
                for chunk_words in results:
                    if chunk_words:
                        all_words.extend(chunk_words)
        except Exception as e:
            print(f"Google Cloud STT Chirp 3 error: {e}")
            return None

        return all_words

    def transcribe_full(
        self,
        audio: np.ndarray,
        sr: int,
        language: str = "hi",
    ) -> List[Dict[str, Any]]:
        words = self._transcribe_api(audio, sr, language)
        if not words:
            return []

        return [
            {
                "start": w["start"],
                "end": w["end"],
                "text": w["text"],
                "avg_logprob": -0.5,
                "no_speech_prob": 0,
                "words": [
                    {
                        "start": w["start"],
                        "end": w["end"],
                        "word": w["text"],
                        "probability": w["probability"],
                    }
                ],
            }
            for w in words
        ]

    def transcribe_diarized_segments(
        self,
        diarization_segments: List[Dict[str, Any]],
        full_audio: np.ndarray,
        sr: int,
        language: str = "hi",
    ) -> List[Dict[str, Any]]:
        merged = self._merge_segments(diarization_segments, max_gap=0.5)
        words = self._transcribe_api(full_audio, sr, language)

        if not words:
            return [
                {
                    "start_time_s": d.get("start_time_s", 0),
                    "end_time_s": d.get("end_time_s", 0),
                    "text": "",
                    "speaker": d.get("speaker", "spk_0"),
                    "words": [],
                    "avg_logprob": 0,
                    "no_speech_prob": 0,
                }
                for d in merged
            ]

        word_to_seg = [None] * len(words)
        for wi, w in enumerate(words):
            best_overlap = 0
            best_idx = -1
            for di, dseg in enumerate(merged):
                d_start = dseg.get("start_time_s", 0)
                d_end = dseg.get("end_time_s", 0)
                overlap = max(0, min(w["end"], d_end) - max(w["start"], d_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = di
            if best_idx >= 0:
                word_to_seg[wi] = best_idx

        seg_word_lists = [[] for _ in merged]
        for wi, w in enumerate(words):
            idx = word_to_seg[wi]
            if idx is not None:
                seg_word_lists[idx].append(w)

        output = []
        for di, dseg in enumerate(merged):
            d_start = dseg.get("start_time_s", 0)
            d_end = dseg.get("end_time_s", 0)
            seg_words = seg_word_lists[di]
            text = " ".join(w.get("text", "") for w in seg_words).strip()

            entry = {
                "start_time_s": d_start,
                "end_time_s": d_end,
                "duration_s": round(d_end - d_start, 6),
                "text": text,
                "speaker": dseg.get("speaker", "spk_0"),
                "words": [
                    {
                        "start": round(w["start"], 3),
                        "end": round(w["end"], 3),
                        "word": w["text"],
                        "probability": w["probability"],
                    }
                    for w in seg_words
                ],
                "avg_logprob": round(
                    sum(w["probability"] for w in seg_words) / max(len(seg_words), 1), 3
                ),
                "no_speech_prob": 0,
            }
            for field in ("confidence", "uncertain", "sb_margin"):
                if field in dseg:
                    entry[field] = dseg[field]
            output.append(entry)

        return output

    @staticmethod
    def _merge_segments(
        segments: List[Dict[str, Any]], max_gap: float = 0.5
    ) -> List[Dict[str, Any]]:
        if not segments:
            return []
        merged = [dict(segments[0])]
        for seg in segments[1:]:
            prev = merged[-1]
            same_speaker = seg.get("speaker") == prev.get("speaker")
            gap = seg.get("start_time_s", 0) - prev.get("end_time_s", 0)
            if same_speaker and gap <= max_gap:
                prev["end_time_s"] = max(prev.get("end_time_s", 0), seg.get("end_time_s", 0))
            else:
                merged.append(dict(seg))
        return merged
