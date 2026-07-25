"""
Turn repair: merge adjacent same-speaker diarized segments into analysis turns.

Merges short same-speaker fragments separated by small gaps. Marks
segments that are too short for independent semantic analysis.
"""
from __future__ import annotations
from typing import List, Optional
from .schemas import DiarizedSegment, AnalysisTurn


def repair_turns(
    segments: List[DiarizedSegment],
    merge_gap_s: float = 0.7,
    min_semantic_duration_s: float = 0.8,
) -> List[AnalysisTurn]:
    """
    Merge adjacent same-speaker segments into analysis turns.

    Rules:
    - Merge if same speaker and gap < merge_gap_s.
    - Mark turns shorter than min_semantic_duration_s as too_short.
    - Preserve original segment IDs.
    - No lexical rules for merging.

    Args:
        segments: Sorted diarized segments.
        merge_gap_s: Maximum silence gap (seconds) to merge across.
        min_semantic_duration_s: Turns shorter than this are marked too_short.

    Returns:
        List of AnalysisTurn objects.
    """
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: (s.start, s.id))

    turns: List[_TurnAccumulator] = []
    current = _TurnAccumulator(sorted_segs[0])

    for seg in sorted_segs[1:]:
        gap = seg.start - current.end
        same_speaker = seg.speaker == current.speaker
        adjacent = gap <= merge_gap_s and gap >= 0

        if same_speaker and adjacent:
            current.add(seg)
        else:
            turns.append(current)
            current = _TurnAccumulator(seg)

    turns.append(current)

    result: List[AnalysisTurn] = []
    for i, acc in enumerate(turns):
        result.append(AnalysisTurn(
            turn_id=i,
            speaker=acc.speaker,
            start=acc.start,
            end=acc.end,
            duration=acc.end - acc.start,
            text=acc.text,
            segment_ids=acc.segment_ids,
            too_short=(acc.end - acc.start) < min_semantic_duration_s,
            asr_confidence=acc.avg_confidence,
        ))

    return result


class _TurnAccumulator:
    """Internal accumulator for merging consecutive same-speaker segments."""

    __slots__ = ("speaker", "start", "end", "_texts", "_seg_ids", "_confidences")

    def __init__(self, seg: DiarizedSegment):
        self.speaker = seg.speaker
        self.start = seg.start
        self.end = seg.end
        self._texts: List[str] = [seg.text]
        self._seg_ids: List[int] = [seg.id]
        self._confidences: List[float] = (
            [seg.asr_confidence] if seg.asr_confidence is not None else []
        )

    def add(self, seg: DiarizedSegment) -> None:
        self.end = max(self.end, seg.end)
        self._texts.append(seg.text)
        self._seg_ids.append(seg.id)
        if seg.asr_confidence is not None:
            self._confidences.append(seg.asr_confidence)

    @property
    def text(self) -> str:
        return " ".join(t for t in self._texts if t and t.strip())

    @property
    def segment_ids(self) -> List[int]:
        return list(self._seg_ids)

    @property
    def avg_confidence(self) -> Optional[float]:
        if not self._confidences:
            return None
        return sum(self._confidences) / len(self._confidences)
