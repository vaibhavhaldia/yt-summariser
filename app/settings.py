"""
settings.py — Persistent settings for yt-summariser desktop app.

Stored at: ~/.ytsummariser/settings.json
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class AppSettings:
    # Output
    output_dir: str = str(Path.home() / "yt-summaries")

    # Pipeline options
    target_pct: float = 0.10
    target_mins: float | None = None  # None = use target_pct
    no_video: bool = False
    no_llm: bool = False
    no_enrich: bool = False
    no_captions: bool = False
    whisper_model: str = "base"  # "tiny" | "base" | "small" | "medium"

    # Models
    llm_model_path: str = ""  # empty = auto-download Mistral-7B

    # App behaviour
    workers: int = 1
    theme: str = "dark"  # "dark" | "light"
    window_width: int = 1200
    window_height: int = 800
    sidebar_width: int = 280

    # yt-dlp
    ytdlp_path: str = "yt-dlp"  # path to binary or just "yt-dlp"


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class SettingsManager:
    """Load, persist, and update application settings."""

    CONFIG_DIR = Path.home() / ".ytsummariser"
    CONFIG_FILE = CONFIG_DIR / "settings.json"

    def __init__(self):
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self._settings = self._load()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def settings(self) -> AppSettings:
        return self._settings

    def save(self) -> None:
        """Write current settings to JSON file atomically (write to .tmp, rename)."""
        tmp_path = self.CONFIG_FILE.with_suffix(".json.tmp")
        try:
            data = dataclasses.asdict(self._settings)
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp_path, self.CONFIG_FILE)
        except Exception:
            log.exception("Failed to save settings to %s", self.CONFIG_FILE)
            # Clean up orphaned .tmp file if it was written
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

    def update(self, **kwargs) -> None:
        """Update one or more settings fields and save immediately."""
        valid_fields = {f.name for f in fields(AppSettings)}
        for key, value in kwargs.items():
            if key not in valid_fields:
                log.warning("update: unknown settings field %r — ignoring", key)
                continue
            setattr(self._settings, key, value)
        self.save()

    def reset(self) -> None:
        """Reset to defaults and save."""
        self._settings = AppSettings()
        self.save()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> AppSettings:
        """Load from JSON file, falling back to defaults on any error."""
        try:
            if not self.CONFIG_FILE.exists():
                return AppSettings()

            raw = self.CONFIG_FILE.read_text(encoding="utf-8")
            data: dict = json.loads(raw)

            # Only set fields that exist in the current dataclass definition;
            # unknown keys from older/newer versions are silently ignored.
            known = {f.name: f for f in fields(AppSettings)}
            kwargs: dict = {}
            for name, fld in known.items():
                if name not in data:
                    continue
                raw_value = data[name]
                kwargs[name] = _coerce(raw_value, fld)

            return AppSettings(**kwargs)
        except Exception:
            log.exception(
                "Failed to load settings from %s — using defaults", self.CONFIG_FILE
            )
            return AppSettings()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce(value: object, fld: dataclasses.Field) -> object:
    """Best-effort type coercion for a single field value loaded from JSON.

    Handles the common cases: bool, int, float, str, None for Optional fields.
    Falls back to the raw value when the type cannot be determined.
    """
    # Resolve the field's type annotation
    type_hint = fld.type

    # Handle "float | None" / "Optional[float]" — allow None pass-through
    if value is None:
        return None

    # Map simple type strings to Python builtins
    _SIMPLE: dict[str, type] = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
    }

    type_str = str(type_hint)

    for name, cast in _SIMPLE.items():
        if name in type_str:
            try:
                # Special-case bool: JSON "true"/"false" already decoded correctly
                if cast is bool:
                    if isinstance(value, bool):
                        return value
                    return str(value).lower() in ("true", "1", "yes")
                return cast(value)
            except (ValueError, TypeError):
                return value

    return value
