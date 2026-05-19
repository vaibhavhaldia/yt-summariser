"""
worker.py — Background pipeline worker for yt-summariser desktop app.

Runs the full summarise pipeline in a QThread so the UI never freezes.
Emits signals at each stage for live progress updates.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal

# ---------------------------------------------------------------------------
# Summarise package — make sure it's importable regardless of how the app is
# launched (e.g. from zed/app/ directly vs. from the project root).
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from summarise.clipper import ClipperError, build_highlight_video, get_stream_url
from summarise.enricher import EnrichedSegment, enrich_segments
from summarise.references import resolve_lookups
from summarise.reporter import ReportConfig, write_report
from summarise.scorer import ScoringWeights, score_segments
from summarise.segmenter import detect_chapters
from summarise.selector import SelectionConfig, select_segments, total_duration
from summarise.synthesiser import synthesise
from summarise.transcript import Segment as _Segment
from summarise.transcript import get_transcript, get_video_duration, get_video_title
from summarise.utils import youtube_video_id

from .database import Database, VideoRecord
from .settings import AppSettings

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_TOTAL_STAGES = 10


class PipelineWorker(QThread):
    """QThread that runs the full summarise pipeline for a single YouTube URL.

    Signals
    -------
    stage_changed(current_stage, total_stages, description)
        Emitted at every stage transition so the UI can update a step label.
    log_message(text)
        Emitted for every informational line (replaces print / _log).
    finished_ok(video_id, report_path)
        Emitted when the pipeline completes successfully.
    finished_error(video_id, error_message)
        Emitted when the pipeline raises an unhandled exception.
    progress(percent)
        Emitted with an integer 0–100 representing overall job progress.
    """

    # ------------------------------------------------------------------ signals
    stage_changed = Signal(int, int, str)  # (current_stage, total_stages, description)
    log_message = Signal(str)  # plain log line
    finished_ok = Signal(str, str)  # (video_id, report_path)
    finished_error = Signal(str, str)  # (video_id, error_message)
    progress = Signal(int)  # 0–100

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        url: str,
        settings: AppSettings,
        database: Database,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.url = url
        self.settings = settings
        self.database = database
        self._cancelled = False

    # ------------------------------------------------------------------ public API
    def cancel(self) -> None:
        """Request cancellation.  The pipeline checks this flag between stages."""
        self._cancelled = True
        self.requestInterruption()

    # ------------------------------------------------------------------ QThread entry
    def run(self) -> None:
        """Entry point called by QThread.start().  Delegates to _execute()."""
        try:
            self._execute()
        except Exception as exc:
            tb = traceback.format_exc()
            video_id = "unknown"
            try:
                video_id = youtube_video_id(self.url)
            except Exception:
                pass
            self.finished_error.emit(video_id, f"{exc}\n\n{tb}")

    # ------------------------------------------------------------------ pipeline
    def _execute(self) -> None:
        """Full pipeline, mirroring summarise._run_pipeline() with Qt integration."""

        s = self.settings  # shorthand
        output_dir = Path(s.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------- Stage 1 — Metadata
        self._emit_stage(1, _TOTAL_STAGES, "Fetching metadata")
        video_id = youtube_video_id(self.url)
        title = get_video_title(self.url)
        duration = get_video_duration(self.url)
        if not title:
            title = video_id
        if duration == 0.0:
            self._emit_log(f"Warning: could not determine duration for {video_id}")
        self.progress.emit(10)

        if self._cancelled:
            return

        # -------------------------------------------------------- Stage 2 — Transcript
        self._emit_stage(2, _TOTAL_STAGES, "Acquiring transcript")
        segments = get_transcript(self.url, whisper_model=s.whisper_model)
        if not segments:
            raise RuntimeError("No transcript segments obtained")
        source = segments[0].source if segments else "unknown"
        self._emit_log(f"{len(segments)} segments from {source}")
        self.progress.emit(20)

        if self._cancelled:
            return

        # -------------------------------------------------------- Stage 3 — Stream URL
        stream_url = None
        if not s.no_enrich or not s.no_video:
            self._emit_stage(3, _TOTAL_STAGES, "Resolving stream URL")
            try:
                stream_url = get_stream_url(self.url)
            except ClipperError as exc:
                self._emit_log(f"Warning: could not resolve stream URL: {exc}")
        self.progress.emit(28)

        if self._cancelled:
            return

        # -------------------------------------------------------- Stage 4 — Enrichment
        self._emit_stage(4, _TOTAL_STAGES, "Enriching weak segments")
        with tempfile.TemporaryDirectory(prefix=f"yts_{video_id}_") as td:
            if not s.no_enrich and stream_url:
                enriched = enrich_segments(
                    segments,
                    stream_url,
                    td,
                    enable_captions=not s.no_captions,
                )
            else:
                # Wrap every segment without running frame analysis
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
        self._emit_log(f"{enriched_count} segment(s) enriched via frame analysis")
        self.progress.emit(40)

        if self._cancelled:
            return

        # -------------------------------------------------------- Stage 5 — Scoring
        self._emit_stage(5, _TOTAL_STAGES, "Scoring segments")
        weights = (
            ScoringWeights()
        )  # defaults: tfidf=0.3, semantic=0.3, density=0.2, energy=0.2

        # Build proxy Segment objects whose .text carries enriched full_text so the
        # scorer sees OCR/caption text, not just the bare transcript text.
        proxy_segments = [
            _Segment(
                start=e.start,
                end=e.end,
                text=e.full_text,
                source=e.segment.source,
            )
            for e in enriched
        ]
        scored = score_segments(proxy_segments, self.url, title, weights)

        # Thin wrapper that merges a ScoredSegment's score with an EnrichedSegment's
        # display properties — mirrors _ScoredEnriched in summarise._run_pipeline().
        class _SE:
            __slots__ = ("score", "full_text", "source_badge", "start", "end")

            def __init__(self, e: EnrichedSegment, s_) -> None:
                self.score = s_.score
                self.full_text = e.full_text
                self.source_badge = e.source_badge
                self.start = e.start
                self.end = e.end

        scored_enriched = [_SE(e, s_) for e, s_ in zip(enriched, scored)]
        self.progress.emit(50)

        if self._cancelled:
            return

        # -------------------------------------------------------- Stage 6 — Chapters
        self._emit_stage(6, _TOTAL_STAGES, "Detecting chapters")
        chapters = detect_chapters(enriched)
        self._emit_log(f"{len(chapters)} chapter(s) detected")
        self.progress.emit(58)

        if self._cancelled:
            return

        # -------------------------------------------------------- Stage 7 — Selection
        self._emit_stage(7, _TOTAL_STAGES, "Selecting highlights")
        if s.target_mins is not None:
            target_secs = s.target_mins * 60.0
        elif duration > 0:
            target_secs = duration * s.target_pct
        else:
            target_secs = 600.0  # fallback: 10 minutes

        # Clamp: at least 30 s, at most the full video length (when known)
        target_secs = max(
            30.0,
            min(target_secs, duration if duration > 0 else target_secs),
        )

        sel_config = SelectionConfig(target_duration=target_secs)
        selected = select_segments(scored_enriched, sel_config)
        highlight_dur = total_duration(selected)
        self._emit_log(f"{len(selected)} highlight segment(s) selected")
        self.progress.emit(65)

        if self._cancelled:
            return

        # -------------------------------------------------------- Stage 8 — Synthesis
        synthesis = None
        if not s.no_llm:
            self._emit_stage(8, _TOTAL_STAGES, "Synthesising insights (local LLM)")
            try:
                model_path = s.llm_model_path if s.llm_model_path else None
                synthesis = synthesise(enriched, chapters, title, model_path=model_path)
                self._emit_log(
                    f"{len(synthesis.lookup_terms)} [[LOOKUP]] term(s) found"
                )
            except Exception as exc:
                self._emit_log(
                    f"Warning: synthesis failed ({exc}), continuing without LLM"
                )
        else:
            self._emit_log("no_llm=True: skipping synthesis")
        self.progress.emit(75)

        if self._cancelled:
            return

        # -------------------------------------------------------- Stage 9 — References
        references: dict = {}
        if synthesis and synthesis.lookup_terms:
            self._emit_stage(
                9,
                _TOTAL_STAGES,
                f"Resolving {len(synthesis.lookup_terms)} Wikipedia lookup(s)",
            )
            references = resolve_lookups(synthesis.lookup_terms)
            found = sum(1 for r in references.values() if r.found)
            self._emit_log(f"{found}/{len(references)} Wikipedia reference(s) found")
        self.progress.emit(83)

        if self._cancelled:
            return

        # -------------------------------------------------------- Stage 10 — Clipping
        highlight_path = None
        if not s.no_video and stream_url and selected:
            self._emit_stage(10, _TOTAL_STAGES, "Building highlight video")
            highlight_path = output_dir / f"{video_id}_highlights.mp4"
            try:
                build_highlight_video(selected, stream_url, highlight_path)
                self._emit_log(f"Highlight video → {highlight_path.name}")
            except (ClipperError, ValueError) as exc:
                self._emit_log(
                    f"Warning: clipping failed ({exc}), continuing without video"
                )
                highlight_path = None
        self.progress.emit(90)

        if self._cancelled:
            return

        # -------------------------------------------------------- Stage 11 — Report
        # We re-use stage 10 label here (report writing is the tail of stage 10).
        self._emit_stage(10, _TOTAL_STAGES, "Writing Markdown report")
        cfg = ReportConfig(
            video_id=video_id,
            video_url=self.url,
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
        self.progress.emit(95)

        # -------------------------------------------------------- Database insert
        topics = synthesis.key_concepts if synthesis else []
        tldr = synthesis.tldr[:300] if synthesis else ""
        record = VideoRecord(
            video_id=video_id,
            url=self.url,
            title=title,
            duration=duration,
            processed_at=datetime.now().isoformat(),
            report_path=str(report_path),
            highlight_path=str(highlight_path) if highlight_path else "",
            source=source,
            n_segments=len(segments),
            n_chapters=len(chapters),
            n_highlights=len(selected),
            highlight_secs=highlight_dur,
            tldr=tldr,
            topics=topics,
            tags=topics[:5],  # auto-tag with top concepts
        )
        self.database.add_video(record)
        self.progress.emit(100)

        self._emit_log(f"✅ Done → {report_path}")
        self.finished_ok.emit(video_id, str(report_path))

    # ------------------------------------------------------------------ private helpers
    def _emit_stage(self, current: int, total: int, description: str) -> None:
        """Emit stage_changed and a corresponding log_message."""
        ts = datetime.now().strftime("%H:%M:%S")
        self.stage_changed.emit(current, total, description)
        self.log_message.emit(f"[{ts}] Stage {current}/{total}: {description}")

    def _emit_log(self, message: str) -> None:
        """Emit a plain timestamped log_message."""
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_message.emit(f"[{ts}] {message}")
