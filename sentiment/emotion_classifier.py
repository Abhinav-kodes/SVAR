"""
Emotion classifier: Translate Hindi → English → DistilRoBERTa emotion model.

Pipeline:
  1. Detect if text contains Devanagari (Hindi)
  2. If yes, translate to English via deep-translator (free Google Translate)
  3. Classify emotion via j-hartmann/emotion-english-distilroberta-base
  4. Map to standard emotion/sentiment labels
"""
import os
import re
import gc
from typing import Dict, Any, List

_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

_translator = None
_emotion_pipe = None
_emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"

LABEL_MAP = {
    "anger": ("anger", "negative"),
    "disgust": ("disgust", "negative"),
    "fear": ("fear", "negative"),
    "joy": ("happiness", "positive"),
    "neutral": ("neutral", "neutral"),
    "sadness": ("sadness", "negative"),
    "surprise": ("surprise", "neutral"),
}


def _get_translator():
    global _translator
    if _translator is not None:
        return _translator
    from deep_translator import GoogleTranslator
    _translator = GoogleTranslator(source="hi", target="en")
    print("  Google Translate (deep-translator) loaded")
    return _translator


def _get_emotion_pipe():
    global _emotion_pipe
    if _emotion_pipe is not None:
        return _emotion_pipe
    from transformers import pipeline as hf_pipeline
    import torch
    device = 0 if torch.cuda.is_available() else -1
    print(f"  Loading emotion model: {_emotion_model_name}...")
    _emotion_pipe = hf_pipeline(
        "text-classification",
        model=_emotion_model_name,
        top_k=None,
        device=device,
    )
    print(f"  Emotion model loaded (device={device})")
    return _emotion_pipe


def _translate_batch(texts: List[str]) -> List[str]:
    """Translate a batch of Hindi texts to English."""
    if not texts:
        return texts
    translator = _get_translator()
    translated = []
    for t in texts:
        try:
            translated.append(translator.translate(t))
        except Exception:
            translated.append(t)
    return translated


def _has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI_RE.search(text))


def classify_emotions_batch(
    texts: List[str],
    batch_size: int = 16,
    max_length: int = 128,
) -> List[Dict[str, Any]]:
    """
    Classify emotions for a batch of texts.

    Hindi/Devanagari text is translated to English first,
    then classified with a strong English emotion model.
    """
    pipe = _get_emotion_pipe()

    needs_translation = [_has_devanagari(t) for t in texts]
    translated_texts = list(texts)

    hindi_indices = [i for i, nt in enumerate(needs_translation) if nt]
    if hindi_indices:
        hindi_texts = [texts[i] for i in hindi_indices]
        for start in range(0, len(hindi_texts), batch_size):
            batch = hindi_texts[start:start + batch_size]
            batch_translated = _translate_batch(batch)
            for j, t_idx in enumerate(hindi_indices[start:start + batch_size]):
                translated_texts[t_idx] = batch_translated[j]

    results = []
    for start in range(0, len(translated_texts), batch_size):
        batch = translated_texts[start:start + batch_size]
        batch = [t[:512] if len(t) > 512 else t for t in batch]
        batch = [t if t.strip() else "neutral" for t in batch]

        preds = pipe(batch)

        for pred in preds:
            if isinstance(pred, list):
                scores = pred
            else:
                scores = [pred]

            score_dict = {p["label"]: p["score"] for p in scores}
            top = max(scores, key=lambda x: x["score"])
            emotion, sentiment = LABEL_MAP.get(top["label"], ("neutral", "neutral"))

            results.append({
                "emotion": emotion,
                "sentiment": sentiment,
                "confidence": round(top["score"], 4),
                "all_scores": {k: round(v, 4) for k, v in score_dict.items()},
            })

    return results


def classify_emotion(text: str) -> Dict[str, Any]:
    """Classify emotion for a single text string."""
    results = classify_emotions_batch([text], batch_size=1)
    return results[0]


def offload_to_cpu():
    """Free GPU memory."""
    global _emotion_pipe
    if _emotion_pipe is not None:
        _emotion_pipe.model.to("cpu")
        del _emotion_pipe
        _emotion_pipe = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
