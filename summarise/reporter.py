"""
reporter.py — Markdown report renderer for the YouTube summariser.

This is the final output stage.  It consumes all upstream artefacts
(transcript, chapters, scoring, selection, synthesis, references, highlight
video) and produces a single, self-contained ``.md`` file.

Design principles
-----------------
* Honest about data sources (source badges, method disclosure).
* Every timestamp is a YouTube deep-link.
* [[LOOKUP: X]] markers are resolved via ``references.inject_references()``
  before any text hits the report.
* Sections that have no data are silently omitted.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from . import utils
from .enricher import EnrichedSegment
from .references import WikiReference, inject_references
from .segmenter import Chapter
from .selector import SelectedSegment, total_duration
from .synthesiser import SynthesisResult

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReportConfig:
    """All data needed to render the final Markdown report."""

    video_id: str
    video_url: str
    video_title: str
    duration: float  # total video duration in seconds
    processed_at: datetime
    enriched_segments: List[EnrichedSegment]
    selected: List[SelectedSegment]
    chapters: List[Chapter]
    synthesis: Optional[SynthesisResult]
    references: Dict[str, WikiReference]
    highlight_path: Optional[Path]
    output_dir: Path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_report(config: ReportConfig) -> Path:
    """Render and write the Markdown report to disk.

    Parameters
    ----------
    config:
        Fully-populated :class:`ReportConfig`.

    Returns
    -------
    Path
        The path of the written ``.md`` file.
    """
    md = _render_markdown(config)
    out_path = Path(config.output_dir) / f"{config.video_id}.md"
    out_path.write_text(md, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Private — top-level renderer
# ---------------------------------------------------------------------------


def _render_markdown(config: ReportConfig) -> str:
    """Build and return the complete Markdown string."""
    parts: List[str] = []

    # ------------------------------------------------------------------ Title
    parts.append(f"# {config.video_title}\n")

    # ----------------------------------------------------------- Meta table
    parts.append(_meta_table(config))

    parts.append("---\n")

    # ---------------------------------------------------------------- TL;DR
    if config.synthesis is not None:
        tldr_text = inject_references(config.synthesis.tldr, config.references)
        parts.append("## 🎯 TL;DR\n")
        parts.append(f"{tldr_text}\n")
        parts.append("---\n")

    # ------------------------------------------------------------- Key Concepts
    if config.synthesis is not None and config.synthesis.key_concepts:
        parts.append("## 🏷️ Key Concepts\n")
        badges = " ".join(f"`{c}`" for c in config.synthesis.key_concepts)
        parts.append(f"{badges}\n")
        parts.append("---\n")

    # --------------------------------------------------------------- Chapters
    if config.chapters:
        parts.append("## 📖 Chapters\n")
        for ch in config.chapters:
            parts.append(_render_chapter(ch, config))
        parts.append("---\n")

    # --------------------------------------------------------------- Insights
    if config.synthesis is not None and config.synthesis.insights:
        parts.append("## 🔍 Insights\n")
        for insight in config.synthesis.insights:
            injected = inject_references(insight, config.references)
            linked = _linkify_timestamps(injected, config.video_id)
            parts.append(f"> {linked}\n")
        parts.append("---\n")

    # --------------------------------------------------------- Highlight table
    if config.selected:
        parts.append("## ✂️ Highlight Segments\n")
        parts.append(_render_highlights_table(config))
        parts.append("---\n")

    # ------------------------------------------------------------ Takeaways
    if config.synthesis is not None and config.synthesis.takeaways:
        parts.append("## 💡 Actionable Takeaways\n")
        for i, tw in enumerate(config.synthesis.takeaways, start=1):
            injected = inject_references(tw, config.references)
            parts.append(f"{i}. {injected}\n")
        parts.append("---\n")

    # --------------------------------------------------------------- References
    found_refs = {t: r for t, r in config.references.items() if r.found}
    if found_refs:
        parts.append("## 📚 References\n")
        for term, ref in found_refs.items():
            parts.append(f"**{term}** — {ref.summary} · [Wikipedia]({ref.url})\n")
        parts.append("---\n")

    # ----------------------------------------------------------- Highlight video
    if config.highlight_path is not None:
        parts.append("## 🎬 Highlight Video\n")
        filename = Path(config.highlight_path).name
        highlight_secs = total_duration(config.selected)
        n_clips = len(config.selected)
        parts.append(f"[▶ Watch highlights]({filename})\n")
        parts.append(
            f"*{n_clips} segment{'s' if n_clips != 1 else ''} · "
            f"{_format_duration(highlight_secs)} of "
            f"{_format_duration(config.duration)}*\n"
        )
        parts.append("---\n")

    # ----------------------------------------------------------------- Footer
    parts.append(
        f"*Generated by yt-summariser · {config.processed_at.isoformat(timespec='seconds')}*\n"
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Private — section renderers
# ---------------------------------------------------------------------------


def _meta_table(config: ReportConfig) -> str:
    """Render the video metadata table."""
    source_counts: Dict[str, int] = {}
    for seg in config.enriched_segments:
        source_counts[seg.source_badge] = source_counts.get(seg.source_badge, 0) + 1

    # Build "source — N segments" string
    if source_counts:
        source_str = ", ".join(
            f"{badge} {count}" for badge, count in sorted(source_counts.items())
        )
        source_str += f" — {len(config.enriched_segments)} segments total"
    else:
        source_str = f"{len(config.enriched_segments)} segments"

    highlight_secs = total_duration(config.selected)

    lines = [
        "| | |",
        "|---|---|",
        f"| **URL** | [{config.video_url}]({config.video_url}) |",
        f"| **Duration** | {_format_duration(config.duration)} |",
        f"| **Processed** | {config.processed_at.isoformat(timespec='seconds')} |",
        f"| **Transcript** | {source_str} |",
        f"| **Highlights** | {len(config.selected)} clips · {_format_duration(highlight_secs)} |",
    ]
    return "\n".join(lines) + "\n"


def _render_chapter(ch: Chapter, config: ReportConfig) -> str:
    """Render one chapter as a Markdown section."""
    start_ts = utils.format_timestamp(ch.start)
    # Format as HH:MM:SS only (drop milliseconds) for the heading
    hms_start = start_ts[:8]

    deep_link = _make_deep_link(config.video_id, ch.start)
    heading = (
        f"### Chapter {ch.index + 1} · {ch.title} · [[{hms_start}]]({deep_link})\n"
    )

    # Chapter summary from synthesis, or a placeholder
    summary = ""
    if config.synthesis is not None:
        raw_summary = config.synthesis.chapter_summaries.get(str(ch.index), "")
        if raw_summary:
            injected = inject_references(raw_summary, config.references)
            linked = _linkify_timestamps(injected, config.video_id)
            summary = f"{linked}\n"

    return heading + (summary or "*(no summary available)*\n") + "\n"


def _render_highlights_table(config: ReportConfig) -> str:
    """Render the segment highlights table."""
    header = "| # | Time | Score | Source | Excerpt |\n|---|---|---|---|---|\n"
    rows: List[str] = []

    for i, seg in enumerate(config.selected, start=1):
        ts_display = utils.format_timestamp(seg.start)[:8]  # HH:MM:SS
        deep_link = _make_deep_link(config.video_id, seg.start)
        score_str = f"{seg.score:.2f}"
        badge = seg.source_badge

        # Truncate excerpt to 140 chars
        excerpt = seg.full_text.replace("|", "\\|").replace("\n", " ")
        if len(excerpt) > 140:
            excerpt = excerpt[:137] + "..."

        rows.append(
            f"| {i} | [{ts_display}]({deep_link}) | {score_str} | {badge} | {excerpt} |"
        )

    return header + "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# Private — formatting helpers
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    """Convert *seconds* to a human-readable string.

    Examples
    --------
    * 5045.0  → "1h 24m 5s"
    * 183.0   → "3m 3s"
    * 47.0    → "47s"
    * 0.0     → "0s"
    """
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


def _make_deep_link(video_id: str, seconds: float) -> str:
    """Return a YouTube deep-link URL for *seconds* into *video_id*.

    Example: ``https://youtu.be/dQw4w9WgXcQ?t=73``
    """
    return f"https://youtu.be/{video_id}?t={int(seconds)}"


def _linkify_timestamps(text: str, video_id: str) -> str:
    """Replace bare ``[HH:MM:SS]`` timestamps in *text* with Markdown links.

    Each ``[HH:MM:SS]`` becomes ``[[HH:MM:SS]](deep_link)`` where the deep
    link is computed from the parsed seconds value.

    Parameters
    ----------
    text:
        Any string — typically a synthesis insight or chapter summary.
    video_id:
        YouTube video ID for constructing the deep-link URL.

    Returns
    -------
    str
        *text* with all bare timestamp patterns replaced.
    """
    _TS_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]")

    def _replace(match: re.Match) -> str:
        ts_str = match.group(1)
        try:
            secs = utils.parse_timestamp(ts_str)
        except ValueError:
            return match.group(0)  # leave unchanged on parse failure
        deep_link = _make_deep_link(video_id, secs)
        return f"[[{ts_str}]]({deep_link})"

    return _TS_RE.sub(_replace, text)
