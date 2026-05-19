"""
summarise.py — CLI entry point for the YouTube educational video summariser.

Usage:
    python -m summarise <youtube_url> [<url> ...] [options]

Pipeline stages (per video):
    1. Metadata      — title, duration via yt-dlp
    2. Transcript    — YouTube captions → Whisper fallback
    3. Enrichment    — OCR + captioning on weak segments
    4. Scoring       — TF-IDF, semantic, density, energy
    5. Segmentation  — chapter boundary detection
    6. Selection     — greedy highlight selection
    7. Synthesis     — local LLM (Mistral-7B) grounded summary
    8. References    — Wikipedia on-demand for [[LOOKUP: X]] terms
    9. Clipping      — ffmpeg highlight video (optional)
   10. Report        — Markdown output
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

from .clipper import ClipperError, build_highlight_video, get_stream_url
from .enricher import EnrichedSegment, enrich_segments
from .references import resolve_lookups
from .reporter import ReportConfig, write_report
from .scorer import ScoringWeights, score_segments
from .segmenter import detect_chapters
from .selector import SelectionConfig, select_segments, total_duration
from .synthesiser import synthesise
from .transcript import Segment as _Segment
from .transcript import get_transcript, get_video_duration, get_video_title
from .utils import youtube_playlist_urls, youtube_video_id

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _progress(video_id: str, stage: int, total: int, description: str) -> None:
    """Print a timestamped progress line."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {video_id} | Stage {stage}/{total}: {description}")


def _log(message: str) -> None:
    """Print a plain timestamped info line."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}")


def _fmt_dur(seconds: float) -> str:
    """Short duration string: 1h23m or 23m45s or 45s."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m}m"
    if m:
        return f"{m}m{sec}s"
    return f"{sec}s"


def _setup_logging(level_str: str) -> None:
    import logging

    level = getattr(logging, level_str.upper(), logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        prog="summarise",
        description="Summarise YouTube educational videos into Markdown reports.",
    )
    p.add_argument("urls", nargs="+", help="YouTube video or playlist URLs")
    p.add_argument(
        "--target-pct",
        type=float,
        default=0.10,
        help="Highlight duration as fraction of video (default: 0.10)",
    )
    p.add_argument(
        "--target-mins",
        type=float,
        default=None,
        help="Override --target-pct with exact minutes",
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Skip ffmpeg clipping",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip local LLM synthesis",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: ./output)",
    )
    p.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        help="Whisper model size (default: base)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (default: 1)",
    )
    p.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip frame OCR/caption enrichment",
    )
    p.add_argument(
        "--no-captions",
        action="store_true",
        help="Skip moondream2 captioning (OCR only)",
    )
    p.add_argument(
        "--loglevel",
        type=str,
        default="warning",
        choices=["debug", "info", "warning"],
        help="Logging level (default: warning)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _run_pipeline(video_id: str, url: str, args, output_dir: Path) -> str:
    """Execute all pipeline stages for a single video.

    Returns
    -------
    str
        ``"ok"`` on success.  Raises on unrecoverable failure.
    """
    output_dir = Path(output_dir)

    # ------------------------------------------------------------------ Stage 1 — Metadata
    _progress(video_id, 1, 10, "Fetching metadata")
    title = get_video_title(url)
    duration = get_video_duration(url)
    if not title:
        title = video_id
    if duration == 0.0:
        _log(f"{video_id} | Warning: could not determine duration")

    # ------------------------------------------------------------------ Stage 2 — Transcript
    _progress(video_id, 2, 10, "Acquiring transcript")
    segments = get_transcript(url, whisper_model=args.whisper_model)
    _log(
        f"{video_id} | {len(segments)} segments from "
        f"{segments[0].source if segments else 'unknown'}"
    )

    if not segments:
        raise RuntimeError("No transcript segments obtained")

    # ------------------------------------------------------------------ Stage 3 — Stream URL
    stream_url = None
    if not args.no_enrich or not args.no_video:
        _progress(video_id, 3, 10, "Resolving stream URL")
        try:
            stream_url = get_stream_url(url)
        except ClipperError as exc:
            _log(f"{video_id} | Warning: could not resolve stream URL: {exc}")

    # ------------------------------------------------------------------ Stage 4 — Enrichment
    _progress(video_id, 4, 10, "Enriching weak segments")
    with tempfile.TemporaryDirectory(prefix=f"yts_{video_id}_") as td:
        if not args.no_enrich and stream_url:
            enriched = enrich_segments(
                segments,
                stream_url,
                td,
                enable_captions=not args.no_captions,
            )
        else:
            # Wrap every segment without frame analysis
            enriched = [
                EnrichedSegment(
                    segment=seg,
                    source_badge=(
                        "📄 auto" if seg.source == "subtitles" else "🎙️ whisper"
                    ),
                )
                for seg in segments
            ]

    enriched_count = sum(1 for e in enriched if e.enriched)
    _log(f"{video_id} | {enriched_count} segments enriched via frame analysis")

    # ------------------------------------------------------------------ Stage 5 — Scoring
    _progress(video_id, 5, 10, "Scoring segments")
    weights = (
        ScoringWeights()
    )  # defaults: tfidf=0.3, semantic=0.3, density=0.2, energy=0.2

    # Build proxy Segment objects whose .text carries the enriched full_text so
    # the scorer sees OCR/caption text, not just the bare transcript text.
    proxy_segments = [
        _Segment(
            start=e.start,
            end=e.end,
            text=e.full_text,
            source=e.segment.source,
        )
        for e in enriched
    ]
    scored = score_segments(proxy_segments, url, title, weights)

    # Attach the score from ScoredSegment back to a thin wrapper that also
    # exposes the enriched-segment fields needed by select_segments and reporter.
    class _ScoredEnriched:
        """Lightweight container that merges a ScoredSegment score with an
        EnrichedSegment's display properties."""

        __slots__ = (
            "_e",
            "_s",
            "score",
            "full_text",
            "source_badge",
            "start",
            "end",
        )

        def __init__(self, enriched_seg: EnrichedSegment, scored_seg) -> None:
            self._e = enriched_seg
            self._s = scored_seg
            self.score = scored_seg.score
            self.full_text = enriched_seg.full_text
            self.source_badge = enriched_seg.source_badge
            self.start = enriched_seg.start
            self.end = enriched_seg.end

    scored_enriched = [_ScoredEnriched(e, s) for e, s in zip(enriched, scored)]

    # ------------------------------------------------------------------ Stage 6 — Chapter segmentation
    _progress(video_id, 6, 10, "Detecting chapters")
    chapters = detect_chapters(enriched)
    _log(f"{video_id} | {len(chapters)} chapter(s) detected")

    # ------------------------------------------------------------------ Stage 7 — Selection
    _progress(video_id, 7, 10, "Selecting highlights")
    if args.target_mins is not None:
        target_secs = args.target_mins * 60.0
    elif duration > 0:
        target_secs = duration * args.target_pct
    else:
        target_secs = 600.0  # fallback: 10 minutes

    # Clamp: at least 30 s, at most the full video length (when known)
    target_secs = max(30.0, min(target_secs, duration if duration > 0 else target_secs))

    config = SelectionConfig(target_duration=target_secs)
    selected = select_segments(scored_enriched, config)
    _log(
        f"{video_id} | {len(selected)} highlight segment(s) "
        f"({_fmt_dur(total_duration(selected))})"
    )

    # ------------------------------------------------------------------ Stage 8 — Synthesis
    synthesis = None
    if not args.no_llm:
        _progress(video_id, 8, 10, "Synthesising insights (local LLM)")
        try:
            synthesis = synthesise(enriched, chapters, title)
            _log(f"{video_id} | {len(synthesis.lookup_terms)} [[LOOKUP]] term(s) found")
        except Exception as exc:
            _log(
                f"{video_id} | Warning: synthesis failed ({exc}), "
                "continuing without LLM"
            )
    else:
        _log(f"{video_id} | --no-llm: skipping synthesis")

    # ------------------------------------------------------------------ Stage 9 — References
    references = {}
    if synthesis and synthesis.lookup_terms:
        _progress(
            video_id,
            9,
            10,
            f"Resolving {len(synthesis.lookup_terms)} Wikipedia lookup(s)",
        )
        references = resolve_lookups(synthesis.lookup_terms)
        found = sum(1 for r in references.values() if r.found)
        _log(f"{video_id} | {found}/{len(references)} Wikipedia reference(s) found")

    # ------------------------------------------------------------------ Stage 10 — Clipping
    highlight_path = None
    if not args.no_video and stream_url and selected:
        _progress(video_id, 10, 10, "Building highlight video")
        highlight_path = output_dir / f"{video_id}_highlights.mp4"
        try:
            build_highlight_video(selected, stream_url, highlight_path)
            _log(f"{video_id} | Highlight video → {highlight_path.name}")
        except (ClipperError, ValueError) as exc:
            _log(
                f"{video_id} | Warning: clipping failed ({exc}), "
                "continuing without video"
            )
            highlight_path = None

    # ------------------------------------------------------------------ Stage 11 — Report
    _progress(video_id, 10, 10, "Writing Markdown report")
    cfg = ReportConfig(
        video_id=video_id,
        video_url=url,
        video_title=title,
        duration=duration,
        processed_at=datetime.now(),
        enriched_segments=enriched,
        selected=selected,
        chapters=chapters,
        synthesis=synthesis,
        references=references,
        highlight_path=highlight_path,
        output_dir=output_dir,
    )
    report_path = write_report(cfg)
    _log(f"{video_id} | ✅ Report → {report_path}")

    return "ok"


# ---------------------------------------------------------------------------
# Per-video dispatcher (wraps _run_pipeline with skip / error handling)
# ---------------------------------------------------------------------------


def _process_one(url: str, args, output_dir: Path) -> str:
    """Run the full pipeline for *url*.

    Returns
    -------
    str
        One of ``"ok"``, ``"skipped"``, or ``"failed"``.
    """
    try:
        video_id = youtube_video_id(url)
    except ValueError:
        _log(f"Cannot extract video ID from {url!r} — skipping")
        return "failed"

    md_path = Path(output_dir) / f"{video_id}.md"
    if md_path.exists():
        _log(f"{video_id} | already done — skipping")
        return "skipped"

    try:
        return _run_pipeline(video_id, url, args, output_dir)
    except Exception as exc:
        _log(f"{video_id} | FAILED: {exc}")
        import traceback

        traceback.print_exc()
        return "failed"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    _setup_logging(args.loglevel)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Expand all URLs (playlist support)
    all_urls = []
    for url in args.urls:
        expanded = youtube_playlist_urls(url)
        all_urls.extend(expanded)

    if not all_urls:
        print("No valid URLs provided.", file=sys.stderr)
        sys.exit(1)

    if len(all_urls) > 15:
        print(f"Warning: {len(all_urls)} URLs provided. Processing first 15 only.")
        all_urls = all_urls[:15]

    print(f"Processing {len(all_urls)} video(s) → {output_dir}/")

    # Dispatch — serial or parallel
    results: dict = {"ok": [], "failed": [], "skipped": []}

    if args.workers <= 1:
        for url in all_urls:
            outcome = _process_one(url, args, output_dir)
            results[outcome].append(url)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_process_one, url, args, output_dir): url
                for url in all_urls
            }
            for fut in as_completed(futures):
                url = futures[fut]
                try:
                    outcome = fut.result()
                    results[outcome].append(url)
                except Exception as exc:
                    print(f"[ERROR] {url}: {exc}")
                    results["failed"].append(url)

    # Final summary
    print(f"\n{'=' * 60}")
    print(
        f"Done: {len(results['ok'])} succeeded, "
        f"{len(results['skipped'])} skipped, "
        f"{len(results['failed'])} failed"
    )
    if results["failed"]:
        print("Failed URLs:")
        for u in results["failed"]:
            print(f"  {u}")


if __name__ == "__main__":
    main()
