# Target Domain Annotation Schema

## Overview

This document defines the annotation schema for collecting target domain
(Hindi/Hinglish call center) emotion data for the SVAR system.

## Why Annotation is Needed

The base model is trained on EmoInHindi (broadcast/media data). Call center
conversations have different characteristics:
- Formal/semi-formal register
- Scripted openings/closings
- Different emotional expression patterns
- Hindi/Hinglish code-mixing

Target domain annotation allows fine-tuning the model on call-specific data.

## Annotation Units

### Turn
A single speaker's continuous speech segment, bounded by speaker change
or a pause > 2 seconds.

### Dialogue
A complete call recording containing multiple turns.

## Labels per Turn

### 1. Emotion (Multi-label, Intensity)

Each turn can have multiple emotions with intensity scores.

| Emotion | Description | Example Utterance |
|---------|-------------|-------------------|
| neutral | No strong emotion | "Haan, bataiye." |
| anger | Irritation, frustration | "Yeh kya bakwas hai!" |
| annoyed | Mild irritation | "Theek hai, theek hai." |
| disgusted | Disgust, repulsion | "Yeh toh bilkul galat hai." |
| fear | Anxiety, worry | "Mujhe darr lag raha hai." |
| happy | Happiness, satisfaction | "Bahut accha, thank you!" |
| sad | Sadness, disappointment | "Mujhe bahut dukh hua." |
| surprise | Surprise, shock | "Arre waah! Yeh kaise hua?" |
| shame | Embarrassment | "Maaf kijiye, galat ho gaya." |
| guilt | Guilt, remorse | "Meri galti thi, sorry." |
| excited | Enthusiasm | "Main bahut excited hoon!" |
| confused | Confusion | "Mujhe samajh nahi aaya." |
| disappointed | Let down | "Mujhe umeed nahi thi aisa." |
| proud | Pride | "Maine bahut mehnat ki." |
| grateful | Gratitude | "Bahut bahut shukriya." |
| love | Warmth, affection | "Aap bahut acche hain." |

Intensity: 0.0 (absent) to 1.0 (max)

### 2. Sentiment (Single-label)

| Sentiment | Description |
|-----------|-------------|
| positive | Favorable, satisfied |
| neutral | Neither positive nor negative |
| negative | Unfavorable, dissatisfied |

### 3. Interaction State (Single-label)

| State | Description |
|-------|-------------|
| calm | Normal conversation flow |
| tension | Rising tension, disagreement |
| escalation | Active arguing, raised voices |
| peak_conflict | Maximum conflict, shouting |
| recovery | De-escalating, resolving |

### 4. Conduct Risk (Multi-label)

| Risk | Description |
|------|-------------|
| none | No risk detected |
| insult_or_degradation | Name-calling, belittling |
| profanity | Profane language |
| intimidation_or_threat | Threats, intimidation |
| harassment | Persistent unwanted behavior |

### 5. Uncertainty

| Label | When to use |
|-------|-------------|
| confident | Model is confident in prediction |
| uncertain | Model is unsure, prediction may be wrong |
| insufficient_evidence | Turn too short/ambiguous to judge |

## Annotation Guidelines

### General Rules
1. Annotate the **dominant** emotion first, then secondary emotions.
2. Use intensity to distinguish between mild and strong expressions.
3. Consider **context**: the same words may mean different things in different contexts.
4. Mark conduct risks **consistently** — what constitutes harassment varies by culture.

### Code-Mixing
- Hinglish (Hindi + English) is common.
- Annotate based on **meaning**, not language.
- Profanity in Hindi is still profanity.

### Silence/Pauses
- Long pauses (> 2s) should be marked as a turn boundary.
- Silence itself is not an emotion label.

## Quality Control

- Minimum 2 annotators per dialogue.
- Inter-annotator agreement target: Cohen's kappa > 0.7 for emotion, > 0.8 for sentiment.
- Disagreements resolved by majority vote or third annotator.

## Data Format

### JSONL (preferred)

```json
{
  "dialogue_id": "D001",
  "utterance_id": "U001",
  "speaker": "agent",
  "start": 0.0,
  "end": 2.5,
  "text": "Hello, how can I help you today?",
  "emotion_label": ["neutral"],
  "intensity": {"neutral": 0.9},
  "sentiment": "neutral",
  "interaction_state": "calm",
  "conduct_risk": ["none"],
  "uncertainty": "confident",
  "annotator_id": "A001"
}
```

### CSV (alternative)

```
dialogue_id,utterance_id,speaker,start,end,text,emotion_label,intensity,sentiment,interaction_state,conduct_risk,uncertainty,annotator_id
```

## Split Ratios

| Split | Ratio | Notes |
|-------|-------|-------|
| Train | 70% | Used for model training |
| Validation | 15% | Used for early stopping, calibration |
| Test | 15% | Used for final evaluation |

**Critical**: Split at dialogue level to prevent data leakage.

## Recommended Annotation Tools

- **Label Studio**: Open source, supports audio + text annotation.
- **Prodigy**: Commercial, good for iterative annotation.
- **Doccano**: Simple, open source, good for text-only annotation.

## Minimum Dataset Size

For target domain fine-tuning:
- **500 dialogues** minimum for meaningful improvement.
- **2000+ dialogues** for robust performance.
- Each dialogue should have 10-30 turns on average.
