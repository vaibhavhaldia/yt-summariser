"""
clipper.py — Highlight clip extraction for the YouTube summariser.

Downloads individual clip segments from the YouTube stream via yt-dlp and
stitches them together into a single mp4 using ffmpeg.  No ffmpeg-python
wrapper is used — all subprocess calls are direct.

Requirements
------------
* ``yt-dlp``  — must be on PATH
* ``ffmpeg``  — must be on PATH
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from .selector import SelectedSegment

# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------


class ClipperError(Exception):
    """Raised when yt-dlp or ffmpeg operations fail unrecoverably."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_stream_url(video_url: str) -> str:
    """Resolve the direct video stream URL for *video_url* using yt-dlp.

    Parameters
    ----------
    video_url:
        Standard YouTube watch URL.

    Returns
    -------
    str
        Direct stream URL suitable for passing to ffmpeg as ``-i``.

    Raises
    ------
    ClipperError
        If yt-dlp exits with a non-zero code or produces no output.
    """
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-check-certificates",
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--get-url",
        video_url,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ClipperError(
            f"yt-dlp failed to resolve stream URL for {video_url!r} "
            f"(exit {exc.returncode}):\n{exc.stderr.strip()}"
        ) from exc

    # yt-dlp may emit two lines (video + audio) for DASH streams; take the first
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        raise ClipperError(
            f"yt-dlp produced no output for {video_url!r}. "
            "The video may be unavailable or geo-restricted."
        )

    return lines[0]


def build_highlight_video(
    segments: List[SelectedSegment],
    stream_url: str,
    output_path: Path,
    tmp_dir: Optional[str] = None,
) -> Path:
    """Extract and concatenate highlight segments into a single mp4.

    Parameters
    ----------
    segments:
        Non-empty list of :class:`~selector.SelectedSegment` objects whose
        ``.start`` / ``.end`` attributes define the clip boundaries (seconds).
    stream_url:
        Direct video stream URL as returned by :func:`get_stream_url`.
    output_path:
        Desired path for the final mp4 file.  Parent directories must exist.
    tmp_dir:
        Optional path to an existing directory for intermediate clip files.
        A temporary directory is created (and cleaned up) when ``None``.

    Returns
    -------
    Path
        The resolved ``output_path``.

    Raises
    ------
    ValueError
        If *segments* is empty.
    ClipperError
        If any ffmpeg operation fails.
    """
    if not segments:
        raise ValueError("segments must not be empty.")

    output_path = Path(output_path)

    # Choose between managed and caller-supplied temp directory
    if tmp_dir is not None:
        _run_build(segments, stream_url, output_path, Path(tmp_dir))
        return output_path

    with tempfile.TemporaryDirectory(prefix="yt_clipper_") as td:
        _run_build(segments, stream_url, output_path, Path(td))

    return output_path


# ---------------------------------------------------------------------------
# Private — orchestration
# ---------------------------------------------------------------------------


def _run_build(
    segments: List[SelectedSegment],
    stream_url: str,
    output_path: Path,
    work_dir: Path,
) -> None:
    """Internal: extract individual clips and concatenate them."""
    clip_files: List[Path] = []

    for i, seg in enumerate(segments):
        clip_path = work_dir / f"clip_{i:04d}.mp4"
        try:
            _clip_segment(stream_url, seg.start, seg.end, clip_path)
        except subprocess.CalledProcessError as exc:
            raise ClipperError(
                f"ffmpeg failed to clip segment {i} "
                f"({seg.start:.3f}s – {seg.end:.3f}s) "
                f"(exit {exc.returncode}):\n{exc.stderr.decode(errors='replace').strip()}"
            ) from exc
        clip_files.append(clip_path)

    try:
        _concat_clips(clip_files, output_path, work_dir)
    except subprocess.CalledProcessError as exc:
        raise ClipperError(
            f"ffmpeg concat failed (exit {exc.returncode}):\n"
            f"{exc.stderr.decode(errors='replace').strip()}"
        ) from exc


# ---------------------------------------------------------------------------
# Private — ffmpeg wrappers
# ---------------------------------------------------------------------------


def _clip_segment(
    stream_url: str,
    start: float,
    end: float,
    output_path: Path,
) -> None:
    """Extract a single time range from *stream_url* to *output_path*.

    Uses stream copy (``-c copy``) to avoid re-encoding; ``-avoid_negative_ts
    make_zero`` prevents DTS issues on the first frame after seeking.

    Raises :exc:`subprocess.CalledProcessError` on failure.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        stream_url,
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _concat_clips(
    clip_paths: List[Path],
    output_path: Path,
    tmp_dir: Path,
) -> None:
    """Concatenate *clip_paths* into *output_path*.

    For a single clip, the file is simply copied rather than invoking ffmpeg's
    concat demuxer (which requires at least two inputs to be safe).

    For multiple clips a ``concat.txt`` file list is written and ffmpeg is
    called with ``-f concat -safe 0``.

    Raises :exc:`subprocess.CalledProcessError` on failure.
    """
    if len(clip_paths) == 1:
        shutil.copy(str(clip_paths[0]), str(output_path))
        return

    # Write the concat list (absolute paths, escaped for ffmpeg)
    concat_file = tmp_dir / "concat.txt"
    with open(concat_file, "w", encoding="utf-8") as fh:
        for clip_path in clip_paths:
            # ffmpeg concat list format requires forward slashes on all platforms
            safe_path = str(clip_path.resolve()).replace("\\", "/")
            fh.write(f"file '{safe_path}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
