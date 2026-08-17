# VAD-Gated STT Chunking — Design

**Date:** 2026-08-17
**Status:** Approved

## Problem

`sentiment/stt/stt_transcriber.py::_transcribe_api` slices the **entire** audio
into 50s chunks and sends every chunk to Google Chirp 3 — silence included.
Each chunk is billed, so call silence (typically 20-40%) is pure wasted cost.

## Behavior

Audio is chunked only around speech regions detected by the existing
`denoising/vad_basic.compute_vad` (25ms RMS frames vs 10th-percentile noise
floor × 1.5):

- Speech frames are grouped into regions; gaps ≤ 0.5s between speech frames
  are merged into the same region
- Each region is padded by **0.3s** on both sides (clamped to audio bounds)
  so boundary words are never clipped
- Regions longer than 50s (`CHUNK_SECONDS`) are split into ≤50s chunks;
  a trailing sub-chunk shorter than 1s is dropped (same rule as today)
- Only the resulting chunks are sent; word offsets remain absolute
  (region start + intra-chunk offset), so downstream word→segment mapping
  in `transcribe_diarized_segments` is unchanged
- No speech at all → return `None` (same as today's no-chunks path)
- `STT_VAD_DISABLED=1` env var falls back to the current full-audio
  chunking (escape hatch for the cost-critical cloud path)

## Changes

### `sentiment/stt/stt_transcriber.py`

- New constants: `VAD_FRAME_MS = 25`, `VAD_PADDING_S = 0.3`,
  `VAD_GAP_TOLERANCE_S = 0.5`
- New module function `_build_vad_chunks(audio_int16, sr) -> List[Tuple[bytes, float]]`
  implementing the gating above (pure, testable, no I/O)
- `_transcribe_api` replaces its fixed 50s chunk loop with
  `_build_vad_chunks`, unless `STT_VAD_DISABLED=1` (then the old loop)

## Testing

- `sentiment/tests/test_vad_chunking.py` (no network):
  - tone bursts separated by silence → chunks cover only speech + padding;
    offsets correct
  - long speech region → split into ≤50s chunks; trailing <1s piece dropped
  - gaps ≤ 0.5s merged into one region
  - all-silence audio → no chunks (empty list)

## Roadmap

- Mark "hallucination guard" ✅ (already implemented) and "VAD-gated STT
  chunks" ✅; renumber remaining items.