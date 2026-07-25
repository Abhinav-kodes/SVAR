"""
Emotion classifier: Local Hindi→English translation → DistilRoBERTa emotion model.

Pipeline:
  1. Detect if text contains Devanagari (Hindi)
  2. If yes, translate to English via Helsinki-NLP/opus-mt-hi-en (local GPU)
  3. Classify emotion via j-hartmann/emotion-english-distilroberta-base (local GPU)
  4. Map to standard emotion/sentiment labels

No external API calls. All inference is local.
"""
import os
import re
import gc
from typing import Dict, Any, List

_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

_translator_model = None
_translator_tokenizer = None
_emotion_pipe = None
_device = None

EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-hi-en"

LABEL_MAP = {
    "anger": ("anger", "negative"),
    "disgust": ("disgust", "negative"),
    "fear": ("fear", "negative"),
    "joy": ("happiness", "positive"),
    "neutral": ("neutral", "neutral"),
    "sadness": ("sadness", "negative"),
    "surprise": ("surprise", "neutral"),
}


def _get_device():
    global _device
    if _device is None:
        import torch
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _load_translator():
    global _translator_model, _translator_tokenizer
    if _translator_model is not None:
        return
    from transformers import MarianMTModel, MarianTokenizer
    device = _get_device()
    print(f"  Loading translation model: {TRANSLATION_MODEL}...")
    _translator_tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL)
    _translator_model = MarianMTModel.from_pretrained(TRANSLATION_MODEL).to(device)
    _translator_model.eval()
    print(f"  Translation model loaded on {device}")


def _load_emotion_pipe():
    global _emotion_pipe
    if _emotion_pipe is not None:
        return
    from transformers import pipeline as hf_pipeline
    device = 0 if _get_device().type == "cuda" else -1
    print(f"  Loading emotion model: {EMOTION_MODEL}...")
    _emotion_pipe = hf_pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        top_k=None,
        device=device,
    )
    print(f"  Emotion model loaded (device={device})")


def _translate_texts(texts: List[str], batch_size: int = 32) -> List[str]:
    """Translate Hindi texts to English using local model."""
    import torch
    _load_translator()
    device = _get_device()
    all_translated = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = _translator_tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            translated = _translator_model.generate(**encoded)
        decoded = _translator_tokenizer.batch_decode(translated, skip_special_tokens=True)
        all_translated.extend(decoded)

    return all_translated


def classify_emotions_batch(
    texts: List[str],
    batch_size: int = 16,
    max_length: int = 128,
) -> List[Dict[str, Any]]:
    """
    Classify emotions for a batch of texts.

    Hindi/Devanagari text is translated to English locally via opus-mt-hi-en,
    then classified with DistilRoBERTa emotion model.
    """
    _load_emotion_pipe()

    needs_translation = [_DEVANAGARI_RE.search(t) is not None for t in texts]
    translated_texts = list(texts)

    hindi_indices = [i for i, nt in enumerate(needs_translation) if nt]
    if hindi_indices:
        hindi_texts = [texts[i] for i in hindi_indices]
        translated_hindi = _translate_texts(hindi_texts, batch_size=32)
        for j, t_idx in enumerate(hindi_indices):
            translated_texts[t_idx] = translated_hindi[j]

    results = []
    for start in range(0, len(translated_texts), batch_size):
        batch = translated_texts[start:start + batch_size]
        batch = [t[:512] if len(t) > 512 else t for t in batch]
        batch = [t.strip() if t.strip() else "neutral" for t in batch]

        preds = _emotion_pipe(batch)

        for pred in preds:
            scores = pred if isinstance(pred, list) else [pred]
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
    global _translator_model, _emotion_pipe
    if _translator_model is not None:
        _translator_model.to("cpu")
        del _translator_model
        _translator_model = None
    if _emotion_pipe is not None:
        _emotion_pipe.model.to("cpu")
        del _emotion_pipe
        _emotion_pipe = None
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
