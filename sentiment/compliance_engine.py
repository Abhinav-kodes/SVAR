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
    # Hindi abusive — single words
    "गाली", "गंदी", "बेवकूफ", "मूर्ख", "कमीना", "साला", "भोसड़ी", "मादरचोद",
    "बहनचोद", "लंड", "चूत", "रंडी", "हरामी", "सुअर", "कुत्ता", "कुत्ते",
    "घटिया", "लौड़े", "लौंडा", "रंडी", "खोटा", "नालायक", "बेहया",
    "चोर", "झूठा", "धोखेबाज", "कमीने", "साले", "हरामजादे",
    # English abusive
    "idiot", "stupid", "moron", "damn", "hell", "bastard", "asshole",
    "bitch", "crap", "fuck", "shit", "dick", "screw you", "shut up",
    "worthless", "useless", "incompetent",
]

# Compound Hindi abuse phrases (multi-word patterns not covered by single keywords)
ABUSIVE_COMPOUND_PATTERNS = [
    re.compile(r"तेरी\s*मां\s*चोद"),
    re.compile(r"तेरी\s*मां\s*की\s*चूत"),
    re.compile(r"मां\s*चुदा"),
    re.compile(r"बहन\s*के\s*लौड़े"),
    re.compile(r"भोसड़ी\s*के"),
    re.compile(r"बहन\s*की\s*लौड़ी"),
    re.compile(r"तेरी\s*बेटी\s*को\s*बेच"),
    re.compile(r"मां\s*चोद"),
    re.compile(r"बाप\s*चोद"),
    re.compile(r"सुअर\s*के\s*बच्चे"),
    re.compile(r"कुत्ते\s*की\s*तरह"),
    re.compile(r"गांडू"),
    re.compile(r"गांड\w*"),
]

# Threat/harassment patterns (Hindi + English)
THREAT_PATTERNS = [
    re.compile(r"(?:तेरी|तू)\s*(?:मां|बेटी|बहन)\s*(?:.*?(?:सोना|बेच|उठा|मार|काट|जला))"),
    re.compile(r"(?:मिल\s*गया|मिल\s*गई)\s*(?:ना|तो)"),
    re.compile(r"(?:मार\s*(?:ऊंगा|दूंगा|डालूंगा|देंगे))"),
    re.compile(r"(?:काट\s*(?:ऊंगा|दूंगा|डालूंगा))"),
    re.compile(r"(?:जला\s*(?:दूंगा|देंगे))"),
    re.compile(r"(?:खत्म\s*(?:कर\s*दूंगा|कर\s*देंगे))"),
    re.compile(r"(?:बर्बाद\s*(?:कर\s*दूंगा|कर\s*देंगे))"),
    re.compile(r"(?:तबाह\s*(?:कर\s*दूंगा|कर\s*देंगे))"),
    re.compile(r"(?:सुसाइड|आत्महत्या)"),
    re.compile(r"(?:kill\s*you|murder\s*you|destroy\s*you|burn\s*you)"),
    re.compile(r"(?:rape|raped|raping)"),
    re.compile(r"(?:threaten|threatening|threat)"),
    re.compile(r"(?:you\s*will\s*regret|you'll\s*be\s*sorry)"),
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
    """Scan text for abusive keywords using exact match + compound patterns + threat patterns + Levenshtein fuzzy match."""
    text_lower = text.lower()
    words = re.findall(r'[\w\u0900-\u097F]+', text_lower)
    matches = []
    seen_keywords = set()

    for kw in ABUSIVE_KEYWORDS:
        if kw in text_lower and kw not in seen_keywords:
            matches.append({"keyword": kw, "type": "exact", "distance": 0})
            seen_keywords.add(kw)

    for pattern in ABUSIVE_COMPOUND_PATTERNS:
        m = pattern.search(text)
        if m:
            matched = m.group()
            is_dup = any(matched in sk or sk in matched for sk in seen_keywords)
            if not is_dup:
                matches.append({"keyword": matched, "type": "compound", "distance": 0})
                seen_keywords.add(matched)

    for pattern in THREAT_PATTERNS:
        m = pattern.search(text)
        if m:
            matched = m.group()
            if matched not in seen_keywords:
                matches.append({"keyword": matched, "type": "threat", "distance": 0})
                seen_keywords.add(matched)

    for word in words:
        if word in seen_keywords or len(word) < 5:
            continue
        for kw in ABUSIVE_KEYWORDS:
            if abs(len(word) - len(kw)) > max_distance:
                continue
            dist = levenshtein_distance(word, kw)
            if 0 < dist <= max_distance:
                matches.append({"keyword": kw, "matched": word, "type": "fuzzy", "distance": dist})
                seen_keywords.add(word)
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
        for m in abusive:
            mtype = m.get("type", "exact")
            if mtype == "threat":
                flags.append(f"THREAT:{m['keyword']}")
            else:
                flags.append(f"ABUSIVE:{m['keyword']}")
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
        analysis["start"] = seg.get("start", seg.get("start_time_s", 0))
        analysis["end"] = seg.get("end", seg.get("end_time_s", 0))
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
