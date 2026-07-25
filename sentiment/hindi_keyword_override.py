"""
Hindi/Hinglish keyword-based emotion override layer.

When the MuRIL model fails to detect obvious emotions (especially anger from
profanity, disgust from insults, sadness from distress signals), this module
provides a rule-based override with high confidence.

This is a fallback layer — it only activates when keywords strongly indicate
a specific non-neutral emotion.
"""
import re
from typing import Dict, Any, Optional


# Hindi profanity and strong anger markers (romanized + Devanagari)
# Note: \b does not work with Devanagari in Python regex, so we use plain matching
ANGER_PATTERNS = [
    'मादरचोद', 'मादर चोद', 'भानचोद', 'बहनचोद', 'बहन के लौड़े', 'बहन के लौड़े',
    'तेरी मां', 'तेरी माँ', 'तेरी मा', 'मां चुदा', 'माँ चुदा', 'मां चोद', 'माँ चोद',
    'भोसड़ी', 'भोसड़ी के', 'भोसड़ीके', 'फटू',
    'कुत्ते की तरह', 'कुत्ते की', 'कुत्ता',
    'हरामी', 'हरामिया', 'घटिया आदमी', 'घटिया',
    'गांडू', 'लौंडे', 'लौड़े', 'लौड़ा',
    'मादरचोद', 'बहनचोद',
    'madarchod', 'bhenchod', 'bhosadi', 'benchod', 'bhosdike',
    'kutta', 'kutte', 'harami', 'haramkhor', 'ghatiya',
    'gandu', 'launde', 'laude',
    'tera maa', 'teri maa',
]

# Customer anger directed at agent
ANGER_DIRECTED_PATTERNS = [
    'क्यों लिखते हो', 'क्यों लिखा', 'क्यों लिखता',
    'कुत्ते की तरह', 'कुत्ते की',
    'मिल गया ना', 'मिल गया', 'जिस दिन तू', 'जिस दिन',
    'तू मुझे मिल', 'तू मिल गया',
    'तेरी मां चोद', 'तेरी माँ चोद', 'तेरी मा चोद',
    'पैसे लेते हो', 'क्या पैसे',
    'क्यों लिखते हो', 'तू इतना घटिया',
    'मुझे मिल गया',
]

# Sadness / distress markers
SADNESS_PATTERNS = [
    'डिप्रेशन', 'दौर से गुजर', 'परेशान',
    'तकलीफ', 'दुख', 'रो रहा',
    'depression', 'going through',
]

# Fear patterns
FEAR_PATTERNS = [
    'धमकी', 'धमका',
    'dhamki',
]

# Disgust patterns
DISGUST_PATTERNS = [
    'घटिया आदमी', 'घटिया', 'निकम्मा', 'बेकार',
    'ghatiya', 'nikamma', 'bekar',
]


def _match_any(text: str, patterns: list) -> bool:
    """Check if text contains any of the pattern strings."""
    text_lower = text.lower()
    for pat in patterns:
        if pat.lower() in text_lower:
            return True
    return False


def override_emotion(
    text: str,
    ml_emotion: str,
    ml_confidence: float,
    speaker: str = "customer"
) -> Dict[str, Any]:
    """
    Apply keyword-based override to MuRIL emotion prediction.

    Only overrides when:
    1. Keywords strongly indicate a specific emotion
    2. Either MuRIL said neutral, or MuRIL confidence is moderate (<0.85)

    Returns dict with 'emotion', 'sentiment', 'confidence', 'source' keys.
    """
    if not text or not text.strip():
        return {"emotion": ml_emotion, "sentiment": "neutral", "confidence": ml_confidence, "source": "ml"}

    text_clean = text.strip()

    # Strong anger detection (profanity is unambiguous)
    if _match_any(text_clean, ANGER_PATTERNS):
        if ml_emotion != "anger" or ml_confidence < 0.85:
            return {"emotion": "anger", "sentiment": "negative", "confidence": 0.92, "source": "keyword"}

    # Directed anger (threats, confrontational language)
    if _match_any(text_clean, ANGER_DIRECTED_PATTERNS):
        if speaker == "customer" and (ml_emotion == "neutral" or ml_confidence < 0.75):
            return {"emotion": "anger", "sentiment": "negative", "confidence": 0.80, "source": "keyword"}

    # Disgust detection
    if _match_any(text_clean, DISGUST_PATTERNS):
        if ml_emotion not in ("disgust", "anger") or ml_confidence < 0.75:
            return {"emotion": "disgust", "sentiment": "negative", "confidence": 0.78, "source": "keyword"}

    # Sadness detection
    if _match_any(text_clean, SADNESS_PATTERNS):
        if ml_emotion not in ("sadness", "fear") or ml_confidence < 0.75:
            return {"emotion": "sadness", "sentiment": "negative", "confidence": 0.75, "source": "keyword"}

    # Fear detection
    if _match_any(text_clean, FEAR_PATTERNS):
        if ml_emotion not in ("fear", "sadness") or ml_confidence < 0.75:
            return {"emotion": "fear", "sentiment": "negative", "confidence": 0.70, "source": "keyword"}

    # No override — trust ML model
    return {"emotion": ml_emotion, "sentiment": "neutral" if ml_emotion == "neutral" else ("positive" if ml_emotion == "happiness" else "negative"), "confidence": ml_confidence, "source": "ml"}
