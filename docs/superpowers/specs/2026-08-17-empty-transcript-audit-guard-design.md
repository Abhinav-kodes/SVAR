# Empty-Transcript Audit Guard — Design

**Date:** 2026-08-17
**Status:** Approved

## Problem

When STT fails (e.g., GCP `BILLING_DISABLED`), diarized segments contain empty
`text` fields. `sentiment/audit_llm.py` still sends the segment list to Gemini
(only speaker/start/end), and the model **hallucinates** a full compliance
report — fabricated violations, QA score/grade, and CRM narrative — from
timestamps alone. The unified audit must never run on an empty transcript.

## Behavior

When no segment has non-empty text, the audit is **skipped entirely**:

- No Gemini call (saves quota; no fabrication)
- No local compliance/QA/CRM fallbacks (they produce meaningless output on empty input)
- `audit_skipped = "no transcript"` is recorded in the pipeline cache and
  surfaced in the API results
- `compliance` / `qa` / `crm_note` keys are left absent — the dashboard tabs
  already render "No … data available" for missing keys

## Changes

### 1. `sentiment/audit_llm.py` (defense in depth)

- `GeminiAuditEngine.audit_call()`: if no segment has non-empty stripped text,
  log and return `None` before building the prompt.
- When some segments have text: filter empty-text segments out of the Gemini
  batch input, **preserving original segment indices** so the existing
  `segment_results` index→timestamp mapping continues to work.

### 2. `pipeline/stages.py:stage_audit` (primary guard)

- After fusion, before the Gemini call: if no segment has non-empty text, log
  `[audit] skipped: no transcript`, set `c["audit_skipped"] = "no transcript"`,
  and return without running `run_unified_audit` or the local fallbacks.
- Behavior with transcript present is unchanged.

### 3. `pipeline/runner.py`

- Include `audit_skipped` in the results dict (from cache, when present).

## Testing

- `audit_llm`: all-empty segments → `None`; mixed segments → empty ones filtered
  from the batch with original indices preserved (no network — monkeypatch).
- `stages`: empty transcript → flag set, fallbacks not invoked; transcript
  present → normal path unchanged.
- `runner`: `audit_skipped` present in results when cache carries it.