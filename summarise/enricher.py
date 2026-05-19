"""
enricher.py — Frame intelligence for the YouTube summariser.

Enriches weak transcript segments (thin speech signal) with OCR text and/or
visual captions extracted from video frames via ffmpeg.  Strong segments are
wrapped unchanged so the rest of the pipeline always works with a uniform
``EnrichedSegment`` type.

Heavy libraries (easyocr, transformers) are imported lazily and their absence
is handled gracefully — the segment is simply left un-enriched.
"""

import logging
import re
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .transcript import Segment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level lazy cache for the caption pipeline
# ---------------------------------------------------------------------------

_caption_pipeline = None
_ocr_warned: bool = False
_caption_warned: bool = False

# Whisper noise/event tags produced by the model
_NOISE_TAG_RE = re.compile(r"^\[.*?\]$")

# Exact placeholder strings considered "empty" speech
_EMPTY_TEXTS = {"[music]", "[applause]", "[laughter]", "[silence]", "[noise]", ""}


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class EnrichedSegment:
    """Wraps a :class:`~transcript.Segment` with optional frame intelligence data."""

    segment: Segment  # original, unchanged
    ocr_text: str = ""  # text read from frames via easyocr
    caption: str = ""  # visual scene description via moondream2
    enriched: bool = False  # True if frame analysis ran and found something
    source_badge: str = ""  # "📄 auto" | "🎙️ whisper" | "🖼️ ocr" | "🖼️ caption"

    @property
    def full_text(self) -> str:
        """Combined text for downstream scoring and LLM consumption.

        Concatenates the raw transcript text with any OCR or caption data
        that was discovered, clearly bracketed so the consumer can distinguish
        sources.
        """
        parts = [self.segment.text]
        if self.ocr_text:
            parts.append(f"[ON SCREEN: {self.ocr_text}]")
        if self.caption:
            parts.append(f"[VISUAL: {self.caption}]")
        return " ".join(p for p in parts if p.strip())

    @property
    def start(self) -> float:
        return self.segment.start

    @property
    def end(self) -> float:
        return self.segment.end

    @property
    def text(self) -> str:
        return self.segment.text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enrich_segments(
    segments: List[Segment],
    stream_url: str,
    tmp_dir: str,
    enable_captions: bool = True,
) -> List[EnrichedSegment]:
    """Enrich weak segments with frame-derived OCR text and/or visual captions.

    All segments are wrapped as :class:`EnrichedSegment` objects.  Only those
    whose speech signal is considered *weak* (see :func:`_is_weak`) trigger
    frame extraction and analysis.

    Parameters
    ----------
    segments:
        Time-ordered list of transcript :class:`~transcript.Segment` objects.
    stream_url:
        Direct video stream URL (or local path) passed to ffmpeg via ``-i``.
    tmp_dir:
        Existing directory where extracted frame PNGs will be written.
    enable_captions:
        When *True*, run moondream2 captioning on frames that yield no useful
        OCR text.  Set to *False* to skip (faster but less informative).

    Returns
    -------
    list[EnrichedSegment]
        One entry per input segment, in the same order.
    """
    tmp_path = Path(tmp_dir)

    # 1. Wrap every segment — even strong ones get the dataclass shell
    enriched: List[EnrichedSegment] = []
    for seg in segments:
        badge = "📄 auto" if seg.source == "subtitles" else "🎙️ whisper"
        enriched.append(EnrichedSegment(segment=seg, source_badge=badge))

    # 2. Identify weak segments
    weak_indices = [i for i, seg in enumerate(segments) if _is_weak(seg)]

    if not weak_indices:
        return enriched

    # 3. Extract frames for all weak segments in one batch
    weak_segs = [segments[i] for i in weak_indices]
    frames_by_local_idx = _extract_frames(stream_url, weak_segs, tmp_path)

    # 4. Enrich each weak segment
    for local_idx, global_idx in enumerate(weak_indices):
        frame_paths = frames_by_local_idx.get(local_idx, [])
        if not frame_paths:
            continue

        enriched_seg = enriched[global_idx]

        # a. Try OCR first
        ocr_text = _run_ocr(frame_paths)
        if len(ocr_text.strip()) > 3:
            enriched_seg.ocr_text = ocr_text
            enriched_seg.enriched = True
            enriched_seg.source_badge = "🖼️ ocr"

        elif enable_captions:
            # b. Fall back to visual captioning
            caption = _run_caption(frame_paths)
            if len(caption.strip()) > 3:
                enriched_seg.caption = caption
                enriched_seg.enriched = True
                enriched_seg.source_badge = "🖼️ caption"

    return enriched


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_weak(seg: Segment) -> bool:
    """Return *True* when the speech signal in *seg* is too thin to be useful.

    A segment is weak when ANY of the following hold:

    * Fewer than 5 words.
    * Fewer than 0.5 words per second.
    * Text (normalised) is one of the known placeholder strings.
    * Every token in the text matches the Whisper noise-tag pattern ``[...]``.
    """
    text_lower = seg.text.strip().lower()

    if text_lower in _EMPTY_TEXTS:
        return True

    tokens = text_lower.split()
    word_count = len(tokens)

    if word_count < 5:
        return True

    duration = max(seg.end - seg.start, 1.0)
    if (word_count / duration) < 0.5:
        return True

    # All tokens are noise tags like [music], [applause], etc.
    if tokens and all(_NOISE_TAG_RE.match(tok) for tok in tokens):
        return True

    return False


def _extract_frames(
    stream_url: str,
    segments: List[Segment],
    tmp_dir: Path,
) -> Dict[int, List[Path]]:
    """Extract representative frames for each segment via ffmpeg.

    For each segment two frames are extracted — at 25 % and 75 % of the
    segment's duration.  If the segment is shorter than 1 second a single
    mid-point frame is used instead.

    Parameters
    ----------
    stream_url:
        Passed directly to ffmpeg as ``-i``.
    segments:
        The weak segments whose frames we want.
    tmp_dir:
        Directory where PNG files will be written.

    Returns
    -------
    dict
        ``{segment_index: [Path, ...]}`` — index matches position in *segments*.
        An empty list is stored for any segment where ffmpeg fails.
    """
    result: Dict[int, List[Path]] = {}

    for idx, seg in enumerate(segments):
        duration = seg.end - seg.start
        paths: List[Path] = []

        if duration < 1.0:
            # Single mid-point frame
            timestamps = [seg.start + duration * 0.5]
            suffixes = [0]
        else:
            timestamps = [seg.start + duration * 0.25, seg.start + duration * 0.75]
            suffixes = [0, 1]

        all_ok = True
        for ts, suffix in zip(timestamps, suffixes):
            out_path = tmp_dir / f"frame_{idx}_{suffix}.png"
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(ts),
                "-i",
                stream_url,
                "-frames:v",
                "1",
                str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode != 0:
                logger.warning(
                    "ffmpeg failed for segment %d frame %d (exit %d): %s",
                    idx,
                    suffix,
                    proc.returncode,
                    proc.stderr.decode(errors="replace").strip(),
                )
                all_ok = False
            elif out_path.exists():
                paths.append(out_path)

        result[idx] = paths if (all_ok or paths) else []

    return result


def _run_ocr(frame_paths: List[Path]) -> str:
    """Run EasyOCR on *frame_paths* and return deduplicated text.

    Confidence threshold: 0.4.  Words are deduplicated while preserving order.
    Returns ``""`` on any failure or if easyocr is not installed.
    """
    global _ocr_warned

    if not frame_paths:
        return ""

    try:
        import easyocr  # noqa: PLC0415
    except ImportError:
        if not _ocr_warned:
            warnings.warn(
                "easyocr is not installed; OCR enrichment will be skipped. "
                "Install with: pip install easyocr",
                ImportWarning,
                stacklevel=2,
            )
            _ocr_warned = True
        return ""

    try:
        reader = easyocr.Reader(["en"], verbose=False)
        all_fragments: List[str] = []
        for frame_path in frame_paths:
            results = reader.readtext(str(frame_path))
            for _bbox, text, confidence in results:
                if confidence > 0.4:
                    all_fragments.append(text)

        # Deduplicate words while preserving order
        seen: set = set()
        deduped: List[str] = []
        for fragment in all_fragments:
            for word in fragment.split():
                normalised = word.lower().strip(".,!?;:")
                if normalised not in seen:
                    seen.add(normalised)
                    deduped.append(word)

        return " ".join(deduped).strip()

    except Exception as exc:
        logger.warning("OCR failed: %s", exc)
        return ""


def _run_caption(frame_paths: List[Path]) -> str:
    """Generate a visual caption for the first frame in *frame_paths*.

    Uses the ``vikhyatk/moondream2`` model via the ``transformers`` pipeline
    API.  The pipeline is cached in the module-level ``_caption_pipeline``
    variable so it is only loaded once per process.

    Returns ``""`` on any failure or if transformers is not installed.
    """
    global _caption_pipeline, _caption_warned

    if not frame_paths:
        return ""

    try:
        from transformers import pipeline as hf_pipeline  # noqa: PLC0415
    except ImportError:
        if not _caption_warned:
            warnings.warn(
                "transformers is not installed; caption enrichment will be skipped. "
                "Install with: pip install transformers",
                ImportWarning,
                stacklevel=2,
            )
            _caption_warned = True
        return ""

    try:
        if _caption_pipeline is None:
            _caption_pipeline = hf_pipeline(
                "image-to-text",
                model="vikhyatk/moondream2",
            )

        result = _caption_pipeline(str(frame_paths[0]))
        if result and isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"].strip()
        return ""

    except Exception as exc:
        logger.warning("Caption generation failed: %s", exc)
        return ""
