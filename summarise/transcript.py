"""
transcript.py — Transcript retrieval for the YouTube summariser.

Tries auto-generated subtitles first; falls back to Whisper ASR.
"""

import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import utils

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    start: float
    end: float
    text: str
    source: str  # "subtitles" | "whisper"


class TranscriptError(Exception):
    """Raised when no transcript can be obtained by any available method."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_transcript(
    url: str,
    whisper_model: str = "base",
    tmp_dir: Optional[str] = None,
) -> List[Segment]:
    """Return a time-ordered list of :class:`Segment` objects for *url*.

    Strategy
    --------
    1. Download auto-generated / manual subtitles via yt-dlp (VTT format).
    2. If no subtitles are available, transcribe the audio with Whisper.

    Parameters
    ----------
    url:
        YouTube video URL.
    whisper_model:
        Whisper model size string (e.g. ``"base"``, ``"small"``, ``"medium"``).
        Only used when subtitles are unavailable.
    tmp_dir:
        Optional directory for intermediate files.  A fresh temporary directory
        is created (and removed on exit) when *None*.

    Raises
    ------
    TranscriptError
        If both subtitle download and Whisper transcription fail.
    """
    if tmp_dir is not None:
        return _fetch_transcript(url, whisper_model, Path(tmp_dir))

    # Manage our own temp directory
    with tempfile.TemporaryDirectory(prefix="yt_summariser_") as td:
        return _fetch_transcript(url, whisper_model, Path(td))


def get_video_title(url: str) -> str:
    """Return the video title for *url*, or an empty string on failure."""
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-check-certificates",
        "--get-title",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return ""


def get_video_duration(url: str) -> float:
    """Return the video duration in seconds for *url*, or 0.0 on failure."""
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-check-certificates",
        "--get-duration",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        raw = result.stdout.strip()
        return utils.parse_timestamp(raw)
    except Exception:
        return 0.0


def segments_to_text(segments: List[Segment]) -> str:
    """Concatenate all segment texts, separated by newlines."""
    return "\n".join(seg.text for seg in segments)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_transcript(url: str, whisper_model: str, tmp_dir: Path) -> List[Segment]:
    """Core logic; expects *tmp_dir* to already exist."""
    raw = utils.mappings_from_subs(url, tmp_dir)

    if raw:
        segments = [
            Segment(
                start=cue["start"], end=cue["end"], text=cue["text"], source="subtitles"
            )
            for cue in raw
        ]
        return sorted(segments, key=lambda s: s.start)

    # No subtitles — try Whisper
    try:
        return _whisper_fallback(url, whisper_model, tmp_dir)
    except Exception as exc:
        raise TranscriptError(
            f"Could not obtain a transcript for {url!r}. "
            f"Subtitle download yielded nothing and Whisper failed: {exc}"
        ) from exc


def _whisper_fallback(url: str, model_name: str, tmp_dir: Path) -> List[Segment]:
    """Download audio and transcribe with OpenAI Whisper.

    Parameters
    ----------
    url:
        YouTube video URL.
    model_name:
        Whisper model size (e.g. ``"base"``).
    tmp_dir:
        Existing directory where the audio file will be written.

    Returns
    -------
    list[Segment]
        Segments sorted by start time, tagged ``source="whisper"``.

    Raises
    ------
    RuntimeError
        If audio download fails or no audio file is found after download.
    ImportError
        If the ``whisper`` package is not installed.
    """
    # 1. Download audio
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-check-certificates",
        "-x",
        "--audio-format",
        "mp3",
        "--output",
        f"{tmp_dir}/audio.%(ext)s",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp audio download failed (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )

    # 2. Locate the downloaded mp3
    mp3_files = list(tmp_dir.glob("*.mp3"))
    if not mp3_files:
        raise RuntimeError(
            f"yt-dlp reported success but no .mp3 file was found in {tmp_dir}"
        )
    audio_path = mp3_files[0]

    # 3. Lazy Whisper import
    import whisper  # noqa: PLC0415  (intentional late import)

    # 4 & 5. Load model and transcribe
    model = whisper.load_model(model_name)
    result_data = model.transcribe(str(audio_path))

    # 6. Convert to Segment list
    segments = [
        Segment(
            start=seg["start"],
            end=seg["end"],
            text=seg["text"].strip(),
            source="whisper",
        )
        for seg in result_data.get("segments", [])
    ]

    # 7. Return sorted by start time
    return sorted(segments, key=lambda s: s.start)
