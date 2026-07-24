import os
import re
import csv
from typing import Dict, Any, List, Optional, Tuple


RBI_PATTERNS = [
    r"(?i)(?:unauthorized\s+access|data\s+breach|confidential\s+information)",
    r"(?i)(?:share\s+(?:your|my)\s+(?:pin|password|otp|cvv|aadhaar))",
    r"(?i)(?:send\s+(?:money|payment)\s+to\s+(?:this|that|my)\s+(?:account|number))",
    r"(?i)(?:bank\s+account\s+(?:number|details))",
    r"(?i)(?:atm\s+pin|net\s+banking\s+(?:password|pin))",
    r"(?i)(?:upi\s+(?:pin|password)|paytm\s+pin)",
    r"हिंदी|बैंक(?:\s+खाता|\s+विवरण)|पिन(?:\s+शेयर|\s+बताओ)|ओटीपी(?:\s+भेजो|\s+बताओ)",
    r"(?i)आधार\s*(?:नंबर|संख्या|शेयर|बताओ)",
    r"(?i)(?:पैसे?\s+भेजो|पेमेंट\s+करो|ट्रांसफर\s+करो)",
]

IRDAI_PATTERNS = [
    r"(?i)(?:pre-existing\s+disease|not\s+covered|claim\s+denied|policy\s+lapsed)",
    r"(?i)(?:misrepresent|false\s+(?:information|declaration)|fraud(?:ulent)?)",
    r"(?i)(?:without\s+consent|solicited\s+(?:call|contact))",
    r"(?i)(?:cooling.off\s+period|free\s+look\s+period)",
    r"(?i)(?:agent\s+commission|mis.sell(?:ing)?)",
    r"पहले\s+से\s+बीमार|बीमा\s+(?:दावा|क्लेम)\s+(?:अस्वीकृत|खारिज)|पॉलिसी\s+बंद",
    r"(?i)बिना\s+सहमति|ठगी|धोखाधड़ी",
]

ABUSIVE_KEYWORDS = [
    # Hindi abusive
    "गाली", "गंदी", "बेवकूफ", "मूर्ख", "कमीना", "साला", "भोसड़ी", "मादरचोद",
    "बहनचोद", "लंड", "चूत", "रंडी", "हरामी", "सुअर", "कुत्ता",
    # English abusive
    "idiot", "stupid", "moron", "damn", "hell", "bastard", "asshole",
    "bitch", "crap", "fuck", "shit", "dick", "screw you", "shut up",
    "worthless", "useless", "incompetent",
]

ABUSIVE_PATTERNS = [re.compile(re.escape(kw), re.IGNORECASE) for kw in ABUSIVE_KEYWORDS]
RBI_COMPILED = [re.compile(p) for p in RBI_PATTERNS]
IRDAI_COMPILED = [re.compile(p) for p in IRDAI_PATTERNS]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance via dynamic programming."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                prev_row[j + 1] + 1,
                curr_row[j] + 1,
                prev_row[j] + cost,
            ))
        prev_row = curr_row
    return prev_row[-1]


def fuzzy_match_abusive(text: str, max_distance: int = 2) -> List[Dict[str, Any]]:
    """Scan text for abusive keywords using exact match + Levenshtein fuzzy match."""
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    matches = []
    seen = set()

    for kw in ABUSIVE_KEYWORDS:
        if kw in text_lower and kw not in seen:
            matches.append({"keyword": kw, "type": "exact", "distance": 0})
            seen.add(kw)

    for word in words:
        if word in seen or len(word) < 3:
            continue
        for kw in ABUSIVE_KEYWORDS:
            if abs(len(word) - len(kw)) > max_distance:
                continue
            dist = levenshtein_distance(word, kw)
            if 0 < dist <= max_distance:
                matches.append({"keyword": kw, "matched": word, "type": "fuzzy", "distance": dist})
                seen.add(word)
                break

    return matches


def check_regulatory(text: str) -> Dict[str, Any]:
    """Check text against RBI and IRDAI violation regex patterns."""
    rbi_hits = []
    irdai_hits = []

    for pattern in RBI_COMPILED:
        m = pattern.search(text)
        if m:
            rbi_hits.append({"pattern": pattern.pattern, "match": m.group()})

    for pattern in IRDAI_COMPILED:
        m = pattern.search(text)
        if m:
            irdai_hits.append({"pattern": pattern.pattern, "match": m.group()})

    return {
        "rbi_violations": rbi_hits,
        "irdai_violations": irdai_hits,
        "total_violations": len(rbi_hits) + len(irdai_hits),
    }


def analyze_transcript(transcript: str) -> Dict[str, Any]:
    """
    Full compliance analysis of a transcript segment.

    Returns:
        Dict with 'abusive', 'regulatory', 'compliant' (bool), 'flags' (list).
    """
    abusive = fuzzy_match_abusive(transcript)
    regulatory = check_regulatory(transcript)
    flags = []

    if abusive:
        flags.extend([f"ABUSIVE:{m['keyword']}" for m in abusive])
    if regulatory["rbi_violations"]:
        flags.extend([f"RBI:{v['match']}" for v in regulatory["rbi_violations"]])
    if regulatory["irdai_violations"]:
        flags.extend([f"IRDAI:{v['match']}" for v in regulatory["irdai_violations"]])

    return {
        "abusive_matches": abusive,
        "regulatory": regulatory,
        "compliant": len(flags) == 0,
        "flags": flags,
        "violation_count": len(flags),
    }


def analyze_call(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze a full call's segments for compliance.

    Args:
        segments: List of dicts with 'text', 'speaker', 'start', 'end' keys.

    Returns:
        Dict with per-segment results, total violations, overall compliant status.
    """
    results = []
    total_violations = 0
    agent_violations = 0
    customer_violations = 0

    for seg in segments:
        text = seg.get("text", "")
        speaker = seg.get("speaker", "unknown")
        analysis = analyze_transcript(text)
        analysis["speaker"] = speaker
        analysis["start"] = seg.get("start", 0)
        analysis["end"] = seg.get("end", 0)
        results.append(analysis)

        vcount = analysis["violation_count"]
        total_violations += vcount
        if "agent" in speaker.lower():
            agent_violations += vcount
        else:
            customer_violations += vcount

    return {
        "segment_results": results,
        "total_violations": total_violations,
        "agent_violations": agent_violations,
        "customer_violations": customer_violations,
        "compliant": total_violations == 0,
    }
