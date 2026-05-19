"""
database.py — SQLite history store for yt-summariser desktop app.

Schema
------
videos(
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id        TEXT UNIQUE NOT NULL,
    url             TEXT NOT NULL,
    title           TEXT NOT NULL,
    duration        REAL,
    processed_at    TEXT,          -- ISO-8601
    report_path     TEXT,          -- absolute path to .md file
    highlight_path  TEXT,          -- absolute path to .mp4 or NULL
    source          TEXT,          -- "subtitles" | "whisper"
    n_segments      INTEGER,
    n_chapters      INTEGER,
    n_highlights    INTEGER,
    highlight_secs  REAL,
    tldr            TEXT,          -- first 300 chars of tldr
    topics          TEXT           -- JSON array of topic strings
)

tags(
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    video_fk    INTEGER REFERENCES videos(id) ON DELETE CASCADE,
    tag         TEXT NOT NULL
)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class VideoRecord:
    video_id: str
    url: str
    title: str
    duration: float = 0.0
    processed_at: str = ""  # ISO-8601 string
    report_path: str = ""
    highlight_path: str = ""
    source: str = ""
    n_segments: int = 0
    n_chapters: int = 0
    n_highlights: int = 0
    highlight_secs: float = 0.0
    tldr: str = ""
    topics: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    row_id: int = 0


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

_CREATE_VIDEOS = """
CREATE TABLE IF NOT EXISTS videos (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id        TEXT UNIQUE NOT NULL,
    url             TEXT NOT NULL,
    title           TEXT NOT NULL,
    duration        REAL,
    processed_at    TEXT,
    report_path     TEXT,
    highlight_path  TEXT,
    source          TEXT,
    n_segments      INTEGER,
    n_chapters      INTEGER,
    n_highlights    INTEGER,
    highlight_secs  REAL,
    tldr            TEXT,
    topics          TEXT
);
"""

_CREATE_TAGS = """
CREATE TABLE IF NOT EXISTS tags (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    video_fk    INTEGER REFERENCES videos(id) ON DELETE CASCADE,
    tag         TEXT NOT NULL
);
"""

_CREATE_TAGS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_tags_video_fk ON tags(video_fk);
"""

_CREATE_TAGS_UNIQUE = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_tags_unique ON tags(video_fk, tag);
"""


def _row_to_record(row: sqlite3.Row, tags: list[str]) -> VideoRecord:
    """Convert a sqlite3.Row from the videos table into a VideoRecord."""
    topics: list = []
    raw_topics = row["topics"]
    if raw_topics:
        try:
            topics = json.loads(raw_topics)
        except (json.JSONDecodeError, TypeError):
            topics = []

    return VideoRecord(
        video_id=row["video_id"],
        url=row["url"],
        title=row["title"],
        duration=row["duration"] or 0.0,
        processed_at=row["processed_at"] or "",
        report_path=row["report_path"] or "",
        highlight_path=row["highlight_path"] or "",
        source=row["source"] or "",
        n_segments=row["n_segments"] or 0,
        n_chapters=row["n_chapters"] or 0,
        n_highlights=row["n_highlights"] or 0,
        highlight_secs=row["highlight_secs"] or 0.0,
        tldr=row["tldr"] or "",
        topics=topics,
        tags=tags,
        row_id=row["id"],
    )


class Database:
    """SQLite-backed history store for processed YouTube videos."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.execute("PRAGMA journal_mode = WAL;")
            self._init_schema()
        except Exception:
            log.exception("Failed to open database at %s", self._db_path)
            raise

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create tables if they do not already exist. Safe to call on every startup."""
        try:
            with self._conn:
                self._conn.execute(_CREATE_VIDEOS)
                self._conn.execute(_CREATE_TAGS)
                self._conn.execute(_CREATE_TAGS_INDEX)
                self._conn.execute(_CREATE_TAGS_UNIQUE)
        except Exception:
            log.exception("Failed to initialise database schema")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_video(self, record: VideoRecord) -> int:
        """Insert or replace a video record. Returns the row id."""
        topics_json = json.dumps(record.topics) if record.topics else "[]"
        sql = """
            INSERT INTO videos (
                video_id, url, title, duration, processed_at,
                report_path, highlight_path, source,
                n_segments, n_chapters, n_highlights, highlight_secs,
                tldr, topics
            ) VALUES (
                :video_id, :url, :title, :duration, :processed_at,
                :report_path, :highlight_path, :source,
                :n_segments, :n_chapters, :n_highlights, :highlight_secs,
                :tldr, :topics
            )
            ON CONFLICT(video_id) DO UPDATE SET
                url            = excluded.url,
                title          = excluded.title,
                duration       = excluded.duration,
                processed_at   = excluded.processed_at,
                report_path    = excluded.report_path,
                highlight_path = excluded.highlight_path,
                source         = excluded.source,
                n_segments     = excluded.n_segments,
                n_chapters     = excluded.n_chapters,
                n_highlights   = excluded.n_highlights,
                highlight_secs = excluded.highlight_secs,
                tldr           = excluded.tldr,
                topics         = excluded.topics
        """
        try:
            with self._conn:
                cur = self._conn.execute(
                    sql,
                    {
                        "video_id": record.video_id,
                        "url": record.url,
                        "title": record.title,
                        "duration": record.duration,
                        "processed_at": record.processed_at,
                        "report_path": record.report_path,
                        "highlight_path": record.highlight_path,
                        "source": record.source,
                        "n_segments": record.n_segments,
                        "n_chapters": record.n_chapters,
                        "n_highlights": record.n_highlights,
                        "highlight_secs": record.highlight_secs,
                        "tldr": record.tldr,
                        "topics": topics_json,
                    },
                )
                row_id: int = cur.lastrowid or self._row_id_for(record.video_id)

            # Re-sync tags: clear existing tags and re-insert from record.tags
            if record.tags:
                self._sync_tags(record.video_id, record.tags)

            return row_id
        except Exception:
            log.exception("add_video failed for video_id=%s", record.video_id)
            return 0

    def delete(self, video_id: str) -> None:
        """Delete a video record and its tags (cascade)."""
        try:
            with self._conn:
                self._conn.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
        except Exception:
            log.exception("delete failed for video_id=%s", video_id)

    def add_tag(self, video_id: str, tag: str) -> None:
        """Add a tag to a video. Silently ignores duplicates."""
        try:
            row_id = self._row_id_for(video_id)
            if row_id == 0:
                log.warning("add_tag: unknown video_id=%s", video_id)
                return
            with self._conn:
                self._conn.execute(
                    "INSERT OR IGNORE INTO tags (video_fk, tag) VALUES (?, ?)",
                    (row_id, tag.strip()),
                )
        except Exception:
            log.exception("add_tag failed for video_id=%s tag=%s", video_id, tag)

    def remove_tag(self, video_id: str, tag: str) -> None:
        """Remove a tag from a video."""
        try:
            row_id = self._row_id_for(video_id)
            if row_id == 0:
                return
            with self._conn:
                self._conn.execute(
                    "DELETE FROM tags WHERE video_fk = ? AND tag = ?",
                    (row_id, tag.strip()),
                )
        except Exception:
            log.exception("remove_tag failed for video_id=%s tag=%s", video_id, tag)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_all(self, search: str = "", tag: str = "") -> list[VideoRecord]:
        """Return all videos, optionally filtered by search text or tag.

        search  — case-insensitive LIKE match against title and tldr.
        tag     — exact match against any tag associated with the video.
        Results are ordered newest-first by processed_at.
        """
        try:
            params: list = []
            conditions: list[str] = []

            if search:
                conditions.append(
                    "(LOWER(v.title) LIKE LOWER(?) OR LOWER(v.tldr) LIKE LOWER(?))"
                )
                like = f"%{search}%"
                params.extend([like, like])

            if tag:
                conditions.append("v.id IN (SELECT video_fk FROM tags WHERE tag = ?)")
                params.append(tag.strip())

            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            sql = f"""
                SELECT v.*
                FROM videos v
                {where}
                ORDER BY v.processed_at DESC
            """
            rows = self._conn.execute(sql, params).fetchall()
            return [_row_to_record(r, self._tags_for(r["id"])) for r in rows]
        except Exception:
            log.exception("get_all failed (search=%r, tag=%r)", search, tag)
            return []

    def get_by_video_id(self, video_id: str) -> VideoRecord | None:
        """Return the record for a specific YouTube video ID, or None."""
        try:
            row = self._conn.execute(
                "SELECT * FROM videos WHERE video_id = ?", (video_id,)
            ).fetchone()
            if row is None:
                return None
            return _row_to_record(row, self._tags_for(row["id"]))
        except Exception:
            log.exception("get_by_video_id failed for video_id=%s", video_id)
            return None

    def get_all_tags(self) -> list[str]:
        """Return sorted list of all unique tags across all videos."""
        try:
            rows = self._conn.execute(
                "SELECT DISTINCT tag FROM tags ORDER BY tag"
            ).fetchall()
            return [r["tag"] for r in rows]
        except Exception:
            log.exception("get_all_tags failed")
            return []

    def get_stats(self) -> dict:
        """Return dict: total_videos, total_hours, most_common_tags (top 5)."""
        try:
            total_videos: int = (
                self._conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] or 0
            )
            total_secs = (
                self._conn.execute("SELECT SUM(duration) FROM videos").fetchone()[0]
                or 0.0
            )
            total_hours = round(total_secs / 3600, 2)

            tag_rows = self._conn.execute(
                """
                SELECT tag, COUNT(*) AS cnt
                FROM tags
                GROUP BY tag
                ORDER BY cnt DESC
                LIMIT 5
                """
            ).fetchall()
            most_common_tags = [{"tag": r["tag"], "count": r["cnt"]} for r in tag_rows]

            return {
                "total_videos": total_videos,
                "total_hours": total_hours,
                "most_common_tags": most_common_tags,
            }
        except Exception:
            log.exception("get_stats failed")
            return {"total_videos": 0, "total_hours": 0.0, "most_common_tags": []}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_id_for(self, video_id: str) -> int:
        """Return the integer primary-key for a video_id, or 0 if not found."""
        row = self._conn.execute(
            "SELECT id FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        return int(row["id"]) if row else 0

    def _tags_for(self, row_id: int) -> list[str]:
        """Return all tags for the given videos.id primary key."""
        rows = self._conn.execute(
            "SELECT tag FROM tags WHERE video_fk = ? ORDER BY tag", (row_id,)
        ).fetchall()
        return [r["tag"] for r in rows]

    def _sync_tags(self, video_id: str, tags: list[str]) -> None:
        """Replace all tags for a video with the given list."""
        row_id = self._row_id_for(video_id)
        if row_id == 0:
            return
        try:
            with self._conn:
                self._conn.execute("DELETE FROM tags WHERE video_fk = ?", (row_id,))
                self._conn.executemany(
                    "INSERT OR IGNORE INTO tags (video_fk, tag) VALUES (?, ?)",
                    [(row_id, t.strip()) for t in tags if t.strip()],
                )
        except Exception:
            log.exception("_sync_tags failed for video_id=%s", video_id)

    def close(self) -> None:
        """Close the underlying database connection."""
        try:
            self._conn.close()
        except Exception:
            log.exception("Failed to close database connection")
