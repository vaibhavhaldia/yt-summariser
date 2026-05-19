"""
utils.py — Shared primitives for the YouTube summariser.

No external imports except yt-dlp, which is invoked as a subprocess.
"""

import re
import subprocess
import warnings
from collections import namedtuple
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Named types
# ---------------------------------------------------------------------------

Resource = namedtuple("Resource", ["start", "final"])
Mapping = namedtuple("Mapping", ["target"])


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def parse_timestamp(s: str) -> float:
    """Parse a timestamp string into float seconds.

    Accepted formats:
        "HH:MM:SS.mmm"
        "HH:MM:SS"
        "MM:SS.mmm"
        "MM:SS"
        bare numeric string (e.g. "123.45")

    Raises ValueError on unrecognised input.
    """
    s = s.strip()

    # Bare numeric (e.g. "123", "45.6")
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)

    # Normalise the leading field if it encodes more than 59 (hours >= 60)
    t = s.split(":")
    if t[1:] and int(t[0]) >= 60:
        s = ":".join((str(int(t[0]) // 60), str(int(t[0]) % 60).zfill(2), *t[1:]))

    formats = ["%H:%M:%S.%f", "%H:%M:%S", "%M:%S.%f", "%M:%S"]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            return (
                dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000
            )
        except ValueError:
            continue

    raise ValueError(f"Unrecognised timestamp format: {s!r}")


def format_timestamp(t: float) -> str:
    """Return *t* (seconds) formatted as 'HH:MM:SS.mmm'."""
    return datetime.utcfromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]


# ---------------------------------------------------------------------------
# YouTube URL helpers
# ---------------------------------------------------------------------------


def youtube_video_id(url: str) -> str:
    """Extract the 11-character YouTube video ID from a URL.

    Handles:
        youtu.be/<id>
        ?v=<id>
        /embed/<id>
        /shorts/<id>
        /v/<id>

    Raises ValueError if no ID is found.
    """
    patterns = [
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"[?&]v=([A-Za-z0-9_-]{11})",
        r"/embed/([A-Za-z0-9_-]{11})",
        r"/shorts/([A-Za-z0-9_-]{11})",
        r"/v/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    raise ValueError(f"Could not extract a YouTube video ID from URL: {url!r}")


def youtube_playlist_urls(url: str) -> list:
    """Return a list of video URLs from a YouTube playlist URL.

    If the URL is a plain video (not a playlist), returns ``[url]`` directly
    without invoking yt-dlp.  Only calls yt-dlp for genuine playlist URLs.
    Falls back to ``[url]`` with a warning on subprocess failure.
    """
    # Plain video URLs — return immediately, no subprocess needed.
    # yt-dlp's --print url returns 'NA' for non-playlist URLs which breaks
    # the pipeline, so we skip the yt-dlp call entirely for plain videos.
    _PLAYLIST_INDICATORS = ("playlist?list=", "/playlist", "&list=")
    if not any(indicator in url for indicator in _PLAYLIST_INDICATORS):
        return [url]

    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-check-certificates",
        "--flat-playlist",
        "--print",
        "url",
        url,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        urls = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() and line.strip() != "NA"
        ]
        return urls if urls else [url]
    except subprocess.CalledProcessError as exc:
        warnings.warn(
            f"yt-dlp failed while listing playlist URLs "
            f"(exit {exc.returncode}); falling back to single URL.\n"
            f"stderr: {exc.stderr.strip()}",
            RuntimeWarning,
            stacklevel=2,
        )
        return [url]


# ---------------------------------------------------------------------------
# VTT subtitle parsing
# ---------------------------------------------------------------------------

# Matches a VTT cue timing line, e.g. "00:00:01.000 --> 00:00:04.000 align:start"
_VTT_TIMING_RE = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"  # start
    r"\s+-->\s+"
    r"(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"  # end
)

# Strips VTT inline tags such as <c>, </c>, <00:00:01.000>, <lang en>, etc.
_VTT_TAG_RE = re.compile(r"<[^>]+>")

# Header lines to skip wholesale
_VTT_HEADER_RE = re.compile(r"^(WEBVTT|Kind:|Language:)")


def mappings_from_subs(url: str, tmp_dir: Path) -> list:
    """Download auto-generated subtitles for *url* into *tmp_dir* and parse them.

    Returns a list of dicts::

        {"start": float, "end": float, "text": str}

    Returns an empty list if no ``.vtt`` file is found or the subtitles are
    empty after deduplication.

    The ``Mapping`` namedtuple (and ``Resource``) defined in this module are
    available for callers that need timestamp-only plumbing; the richer dict
    form is returned here so that ``transcript.py`` has access to the text.
    """
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-check-certificates",
        "--write-auto-sub",
        "--write-sub",
        "--sub-lang",
        "en",
        "--sub-format",
        "vtt",
        "--skip-download",
        "--output",
        f"{tmp_dir}/%(id)s",
        url,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        warnings.warn(
            f"yt-dlp subtitle download failed (exit {exc.returncode}): "
            f"{exc.stderr.strip()}",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    # Locate the first .vtt file written into tmp_dir
    vtt_files = list(Path(tmp_dir).glob("*.vtt"))
    if not vtt_files:
        return []

    vtt_path = vtt_files[0]
    return _parse_vtt(vtt_path)


def _parse_vtt(vtt_path: Path) -> list:
    """Parse a VTT file and return ``[{"start", "end", "text"}, …]``."""
    cues = []
    prev_text = None

    with open(vtt_path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")

        # Skip header / blank / metadata lines that aren't cue content
        if not line.strip() or _VTT_HEADER_RE.match(line):
            i += 1
            continue

        # Cue timing line
        timing_match = _VTT_TIMING_RE.search(line)
        if timing_match:
            start_str, end_str = timing_match.group(1), timing_match.group(2)
            try:
                start = parse_timestamp(start_str)
                end = parse_timestamp(end_str)
            except ValueError:
                i += 1
                continue

            # Collect all following non-blank lines as the cue's payload
            i += 1
            text_lines = []
            while i < len(lines):
                payload = lines[i].rstrip("\n")
                if not payload.strip():
                    break
                # Skip cue identifier lines (pure digits or "NOTE …" etc.)
                if not _VTT_TIMING_RE.search(payload):
                    text_lines.append(payload)
                i += 1

            raw_text = " ".join(text_lines)
            clean_text = _VTT_TAG_RE.sub("", raw_text).strip()

            # Skip VTT repetition artefacts
            if clean_text and clean_text != prev_text:
                cues.append({"start": start, "end": end, "text": clean_text})
                prev_text = clean_text

            continue  # outer while already advanced by inner while

        # Anything else (cue identifiers, NOTE blocks, etc.) — skip
        i += 1

    return cues
