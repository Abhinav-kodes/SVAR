# Disk Transcript Cache — Design

**Date:** 2026-08-17
**Status:** Approved

## Problem

Re-analyzing a file whose results were lost (job-store reset, restart,
deleted Postgres row) re-runs the whole pipeline, including the paid
Google Chirp 3 STT call. Same file → same words, but the API is billed
again each time. `pipeline/stages.py::stage_stt` currently has no way to
reuse a previous transcription.

## Behavior

Before calling Chirp 3, `stage_stt` computes the sha1 of the **source
audio file** and checks `data/transcripts/` for a cached word stream:

- Cache hit → skip the API entirely; rerun only the local word→segment
  mapping (free).
- Cache miss → call Chirp 3 as today, then store the raw word stream.
- Cache entry is invalidated when the STT config fingerprint changes or
  the entry is older than 30 days (TTL), so server-side model changes
  eventually flow in.
- Cache failures (corrupt JSON, write errors, missing fields) never fail
  the pipeline — they behave as a miss and are logged.

The cache stores the **raw Chirp word stream** (`start`, `end`, `text`,
`probability` per word), not the final per-segment transcript, so
diarization or segment-mapping changes do not invalidate paid output.

## Changes

### `sentiment/stt/transcript_cache.py` (new module, pure — no google imports)

- `sha1_of_file(path) -> str` — streamed sha1 of source file bytes
- `get_words(sha1, lang) -> Optional[List[Dict]]` — raw word stream or
  `None` on miss / corrupt file / expired entry / fingerprint mismatch
- `put_words(sha1, lang, words)` — atomic write via `.tmp` + `os.replace`
- Constants:
  - `CACHE_DIR = data/transcripts/` (override: `SVAR_TRANSCRIPT_CACHE_DIR`)
  - `CACHE_TTL_DAYS = 30`
- Cache file layout: `data/transcripts/<sha1>-<lang>.json` containing
  `{"config": {...}, "created_at": "<ISO>", "words": [...]}`
  - Language in the filename so `hi` and `en` transcriptions of the same
    file coexist
  - `config` fingerprint = `{model, language, vad_enabled,
    VAD_FRAME_MS, VAD_PADDING_S, VAD_GAP_TOLERANCE_S, CHUNK_SECONDS}`;
    any difference → miss
  - Expired entries are deleted on read; corrupt entries treated as miss

### `sentiment/stt/stt_transcriber.py`

- New `transcribe_words(audio, sr, language) -> Optional[List[Dict]]` —
  public wrapper around the existing `_transcribe_api` (raw word stream)
- New `build_diarized_transcript(segments, words)` — the word→segment
  mapping extracted verbatim from `transcribe_diarized_segments`,
  including the initial `_merge_segments(segments, max_gap=0.5)` and the
  empty-transcript fallback (empty segment dicts when `words` is falsy)
- `transcribe_diarized_segments` becomes a thin convenience calling both
  (existing callers/tests unchanged)

### `pipeline/stages.py` — `stage_stt`

```python
sha1 = transcript_cache.sha1_of_file(ctx.filepath)
words = transcript_cache.get_words(sha1, "hi")
if words is None:
    words = stt.transcribe_words(c["clean_audio"], c["sr"], language="hi")
    if words:
        transcript_cache.put_words(sha1, "hi", words)
    log("  [stt] Chirp 3 API (%d words)")        # miss path
else:
    log("  [stt] transcript cache hit (%d words)")  # hit path
c["segments"] = stt.build_diarized_transcript(c["segments"], words)
c["transcribed"] = True
```

`transcribe_words` returning `None` (API failure) is NOT cached, so
transient failures retry next run.

## Error handling

- Read failures → miss, no exception, log
- Write failures → catch, log, continue (result unaffected)
- Concurrent workers transcribing the same file: both valid writes via
  `os.replace`; last writer wins
- `sha1_of_file` on missing file → raise (pipeline guarantees existence)

## Testing

### `sentiment/tests/test_transcript_cache.py` (no network)

1. `sha1_of_file` known-content → known sha1
2. `put_words`/`get_words` roundtrip
3. Language separation — `hi` and `en` entries coexist
4. Fingerprint mismatch (tampered `config`) → miss
5. TTL expiry (monkeypatched TTL) → miss + file removed
6. Corrupt file → miss, no exception
7. No `.tmp` left after `put_words` (atomic replace)

### `pipeline/tests` — stage-level

8. `stage_stt` first run: `transcribe_words` called once, cache file
   written; second run: cache hit, API not called, same transcript
   (monkeypatched `transcribe_words`; tmp `SVAR_TRANSCRIPT_CACHE_DIR`)

## Docs

- Roadmap.md: mark "STT cost: disk transcript cache" ✅ (bullet:
  `data/transcripts/<sha1>-<lang>.json`, 30-day TTL, config fingerprint)
- README.md env table: `SVAR_TRANSCRIPT_CACHE_DIR` row
- `.gitignore`: add `data/transcripts/`