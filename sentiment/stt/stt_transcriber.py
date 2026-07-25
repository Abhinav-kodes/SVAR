import os
import numpy as np
from typing import List, Dict, Any, Optional

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "credentials", "gcloud-stt.json",
)

CHUNK_SECONDS = 50
PROJECT_ID = "sunohq"
REGION = "us"


class SpeechToTextTranscriber:
    """
    Google Cloud Speech-to-Text V2 with Chirp 3 model.
    Chunks long audio into ≤50s segments for Recognize method.
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

        client = self._get_client()
        chunk_samples = CHUNK_SECONDS * sr
        all_words = []

        try:
            for i in range(0, len(audio_int16), chunk_samples):
                chunk = audio_int16[i : i + chunk_samples]
                if len(chunk) < sr:
                    break
                offset = i / sr

                request = cloud_speech.RecognizeRequest(
                    recognizer=recognizer_path,
                    config=config,
                    content=chunk.tobytes(),
                )
                result = client.recognize(request=request)

                for r in result.results:
                    alt = r.alternatives[0]
                    for w in alt.words:
                        start_s = w.start_offset.total_seconds() + offset
                        end_s = w.end_offset.total_seconds() + offset
                        all_words.append({
                            "start": round(start_s, 3),
                            "end": round(end_s, 3),
                            "text": w.word,
                            "probability": round(w.confidence or 0.8, 3),
                        })
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
                    "speaker": d.get("speaker", "agent"),
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

            output.append({
                "start_time_s": d_start,
                "end_time_s": d_end,
                "text": text,
                "speaker": dseg.get("speaker", "agent"),
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
            })

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
