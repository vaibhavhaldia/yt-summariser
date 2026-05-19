"""
app.py — PySide6 main window for the yt-summariser desktop application.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .database import Database, VideoRecord
from .settings import SettingsManager
from .worker import PipelineWorker

# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _fmt_dur(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Examples: "1h 23m", "23m 45s", "45s"
    """
    seconds = int(seconds)
    if seconds <= 0:
        return "0s"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Primary application window for yt-summariser."""

    def __init__(self) -> None:
        super().__init__()
        self._settings_mgr = SettingsManager()
        self._db = Database(Path.home() / ".ytsummariser" / "history.db")
        self._worker: PipelineWorker | None = None
        self._current_video_id: str | None = None

        self.setWindowTitle("🎓 yt-summariser")
        self.resize(
            self._settings_mgr.settings.window_width,
            self._settings_mgr.settings.window_height,
        )

        self._build_ui()
        self._apply_theme(self._settings_mgr.settings.theme)
        self._refresh_history()
        self._update_stats()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build and assemble the complete widget tree."""

        # ── Root container ─────────────────────────────────────────────
        root_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(root_splitter)

        # ── LEFT PANEL ─────────────────────────────────────────────────
        left_widget = QWidget()
        left_widget.setFixedWidth(280)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        # App title
        app_title_label = QLabel("🎓 yt-summariser")
        app_title_label.setObjectName("app_title")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        app_title_label.setFont(title_font)
        app_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(app_title_label)

        # Search box
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search videos...")
        left_layout.addWidget(self.search_input)

        # Tag filter combo
        self.tag_filter = QComboBox()
        self.tag_filter.addItem("All Topics")
        left_layout.addWidget(self.tag_filter)

        # History list
        self.history_list = QListWidget()
        self.history_list.setAlternatingRowColors(True)
        self.history_list.setSpacing(2)
        left_layout.addWidget(self.history_list)

        # Stats label
        self.stats_label = QLabel("0 videos · 0.0h total")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stats_font = QFont()
        stats_font.setPointSize(9)
        self.stats_label.setFont(stats_font)
        left_layout.addWidget(self.stats_label)

        root_splitter.addWidget(left_widget)

        # ── RIGHT PANEL ────────────────────────────────────────────────
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # ── TOP: Input + Progress area ─────────────────────────────────
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(8, 8, 8, 8)
        top_layout.setSpacing(6)

        # URL input
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Paste YouTube URL or playlist...")
        top_layout.addWidget(self.url_input)

        # Action buttons row
        btn_row = QHBoxLayout()
        self.analyse_btn = QPushButton("▶  Analyse")
        self.cancel_btn = QPushButton("✕  Cancel")
        self.cancel_btn.setEnabled(False)
        self.settings_btn = QPushButton("⚙  Settings")
        btn_row.addWidget(self.analyse_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.settings_btn)
        top_layout.addLayout(btn_row)

        # Progress bar (hidden when idle)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        top_layout.addWidget(self.progress_bar)

        # Stage label (hidden when idle)
        self.stage_label = QLabel("Stage 0/10: —")
        self.stage_label.setVisible(False)
        top_layout.addWidget(self.stage_label)

        # Log area
        self.log_area = QPlainTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumBlockCount(200)
        mono_font = QFont("Courier New")
        mono_font.setStyleHint(QFont.StyleHint.Monospace)
        mono_font.setPointSize(10)
        self.log_area.setFont(mono_font)
        self.log_area.setPlaceholderText("Pipeline log output will appear here…")
        top_layout.addWidget(self.log_area)

        right_splitter.addWidget(top_widget)

        # ── BOTTOM: Preview area ───────────────────────────────────────
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(8, 8, 8, 8)
        bottom_layout.setSpacing(6)

        # Video title
        self.preview_title = QLabel("")
        preview_title_font = QFont()
        preview_title_font.setBold(True)
        preview_title_font.setPointSize(13)
        self.preview_title.setFont(preview_title_font)
        self.preview_title.setWordWrap(True)
        bottom_layout.addWidget(self.preview_title)

        # Meta label: duration · date · source
        self.meta_label = QLabel("")
        meta_font = QFont()
        meta_font.setPointSize(9)
        self.meta_label.setFont(meta_font)
        bottom_layout.addWidget(self.meta_label)

        # TL;DR
        self.tldr_label = QLabel("")
        self.tldr_label.setWordWrap(True)
        tldr_font = QFont()
        tldr_font.setItalic(True)
        self.tldr_label.setFont(tldr_font)
        bottom_layout.addWidget(self.tldr_label)

        # Tab widget: Report | Chapters | Highlights
        self.tab_widget = QTabWidget()

        # Tab 1 — Report (Markdown)
        self.report_browser = QTextBrowser()
        self.report_browser.setOpenExternalLinks(True)
        self.tab_widget.addTab(self.report_browser, "📄 Report")

        # Tab 2 — Chapters
        self.chapters_list = QListWidget()
        self.tab_widget.addTab(self.chapters_list, "📖 Chapters")

        # Tab 3 — Highlights
        self.highlights_table = QTableWidget()
        self.highlights_table.setColumnCount(4)
        self.highlights_table.setHorizontalHeaderLabels(
            ["Time", "Score", "Source", "Excerpt"]
        )
        self.highlights_table.horizontalHeader().setStretchLastSection(True)
        self.highlights_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.highlights_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.tab_widget.addTab(self.highlights_table, "✂️ Highlights")

        bottom_layout.addWidget(self.tab_widget)

        # Bottom action buttons row
        action_row = QHBoxLayout()
        self.open_report_btn = QPushButton("📄 Open Report")
        self.open_video_btn = QPushButton("🎬 Open Video")
        self.open_folder_btn = QPushButton("📁 Open Folder")
        self.delete_btn = QPushButton("🗑 Delete")
        action_row.addWidget(self.open_report_btn)
        action_row.addWidget(self.open_video_btn)
        action_row.addWidget(self.open_folder_btn)
        action_row.addStretch()
        action_row.addWidget(self.delete_btn)
        bottom_layout.addLayout(action_row)

        right_splitter.addWidget(bottom_widget)

        # Give the top pane slightly more space than the bottom
        right_splitter.setSizes([320, 480])

        root_splitter.addWidget(right_splitter)
        root_splitter.setSizes([280, 920])

        # ── Signal connections ─────────────────────────────────────────
        self.analyse_btn.clicked.connect(self._on_analyse)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.settings_btn.clicked.connect(self._on_settings)
        self.search_input.textChanged.connect(self._refresh_history)
        self.tag_filter.currentTextChanged.connect(self._refresh_history)
        self.history_list.currentItemChanged.connect(self._on_history_select)
        self.url_input.returnPressed.connect(self._on_analyse)
        self.open_report_btn.clicked.connect(self._on_open_report)
        self.open_video_btn.clicked.connect(self._on_open_video)
        self.open_folder_btn.clicked.connect(self._on_open_folder)
        self.delete_btn.clicked.connect(self._on_delete)

    # ------------------------------------------------------------------
    # Pipeline slots
    # ------------------------------------------------------------------

    def _on_analyse(self) -> None:
        """Start the pipeline for the URL currently in url_input."""
        url = self.url_input.text().strip()
        if not url:
            return

        if self._worker and self._worker.isRunning():
            return  # already running

        # Show progress UI
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.stage_label.setVisible(True)
        self.analyse_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.log_area.clear()

        self._worker = PipelineWorker(url, self._settings_mgr.settings, self._db)
        self._worker.stage_changed.connect(self._on_stage_changed)
        self._worker.log_message.connect(self._on_log_message)
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.finished_ok.connect(self._on_pipeline_ok)
        self._worker.finished_error.connect(self._on_pipeline_error)
        self._worker.start()

    def _on_cancel(self) -> None:
        """Cancel the running pipeline."""
        if self._worker:
            self._worker.cancel()
            self._on_log_message("[cancelled by user]")
        self._reset_progress_ui()

    def _on_stage_changed(self, current: int, total: int, description: str) -> None:
        self.stage_label.setText(f"Stage {current}/{total}: {description}")

    def _on_log_message(self, message: str) -> None:
        self.log_area.appendPlainText(message)
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_pipeline_ok(self, video_id: str, report_path: str) -> None:
        self._reset_progress_ui()
        self.url_input.clear()
        self._refresh_history()
        self._update_stats()
        self._update_tag_filter()
        self._select_history_item(video_id)

    def _on_pipeline_error(self, video_id: str, error_message: str) -> None:
        self._reset_progress_ui()
        QMessageBox.critical(
            self,
            "Pipeline Error",
            f"Failed to process video:\n\n{error_message[:500]}",
        )

    def _reset_progress_ui(self) -> None:
        self.progress_bar.setVisible(False)
        self.stage_label.setVisible(False)
        self.analyse_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # History list
    # ------------------------------------------------------------------

    def _refresh_history(self) -> None:
        """Reload the history list from the database, respecting current filters."""
        search = self.search_input.text().strip()
        tag_text = self.tag_filter.currentText()
        tag = "" if tag_text == "All Topics" else tag_text

        records = self._db.get_all(search=search, tag=tag)

        # Remember the current selection so we can restore it
        current_item = self.history_list.currentItem()
        selected_id = (
            current_item.data(Qt.ItemDataRole.UserRole) if current_item else None
        )

        self.history_list.blockSignals(True)
        self.history_list.clear()

        for record in records:
            date_str = record.processed_at[:10] if record.processed_at else "—"
            item = QListWidgetItem(f"{record.title}\n{date_str}")
            item.setData(Qt.ItemDataRole.UserRole, record.video_id)
            tooltip = record.tldr[:100] if record.tldr else record.title
            item.setToolTip(tooltip)
            self.history_list.addItem(item)

        self.history_list.blockSignals(False)

        # Restore previous selection if still present
        if selected_id:
            self._select_history_item(selected_id)

    def _on_history_select(
        self, current: QListWidgetItem | None, previous: QListWidgetItem | None
    ) -> None:
        if current is None:
            return
        video_id = current.data(Qt.ItemDataRole.UserRole)
        self._current_video_id = video_id
        record = self._db.get_by_video_id(video_id)
        if record is None:
            return
        self._populate_preview(record)

    # ------------------------------------------------------------------
    # Preview panel
    # ------------------------------------------------------------------

    def _populate_preview(self, record: VideoRecord) -> None:
        """Fill the right-hand preview panel with data from *record*."""

        self.preview_title.setText(record.title)

        date_str = record.processed_at[:10] if record.processed_at else "—"
        meta = f"{_fmt_dur(record.duration)} · {date_str} · {record.source}"
        self.meta_label.setText(meta)

        self.tldr_label.setText(record.tldr)

        # ── Report tab ─────────────────────────────────────────────────
        if record.report_path:
            report_file = Path(record.report_path)
            if report_file.exists():
                try:
                    markdown_text = report_file.read_text(encoding="utf-8")
                    self.report_browser.setMarkdown(markdown_text)
                except Exception as exc:
                    self.report_browser.setPlainText(
                        f"[Error reading report file: {exc}]"
                    )
            else:
                self.report_browser.setPlainText(
                    f"[Report file not found: {record.report_path}]"
                )
        else:
            self.report_browser.clear()

        # ── Chapters tab ───────────────────────────────────────────────
        self.chapters_list.clear()
        chapters = _parse_chapters_from_report(record)
        for ch in chapters:
            self.chapters_list.addItem(ch)

        # ── Highlights tab ─────────────────────────────────────────────
        highlights = _parse_highlights_from_report(record)
        self.highlights_table.setRowCount(len(highlights))
        for row_idx, (time_str, score_str, source_str, excerpt) in enumerate(
            highlights
        ):
            self.highlights_table.setItem(row_idx, 0, QTableWidgetItem(time_str))
            self.highlights_table.setItem(row_idx, 1, QTableWidgetItem(score_str))
            self.highlights_table.setItem(row_idx, 2, QTableWidgetItem(source_str))
            self.highlights_table.setItem(row_idx, 3, QTableWidgetItem(excerpt))
        self.highlights_table.resizeColumnsToContents()
        self.highlights_table.horizontalHeader().setStretchLastSection(True)

        # ── Enable / disable Open Video button ─────────────────────────
        has_video = bool(record.highlight_path and Path(record.highlight_path).exists())
        self.open_video_btn.setEnabled(has_video)

    # ------------------------------------------------------------------
    # Open / Delete actions
    # ------------------------------------------------------------------

    def _on_open_report(self) -> None:
        if not self._current_video_id:
            return
        record = self._db.get_by_video_id(self._current_video_id)
        if record and record.report_path:
            _open_path(record.report_path)

    def _on_open_video(self) -> None:
        if not self._current_video_id:
            return
        record = self._db.get_by_video_id(self._current_video_id)
        if record and record.highlight_path:
            _open_path(record.highlight_path)

    def _on_open_folder(self) -> None:
        if not self._current_video_id:
            return
        record = self._db.get_by_video_id(self._current_video_id)
        if not record:
            return
        # Use report path's directory; fall back to highlight path's directory
        target_path = record.report_path or record.highlight_path
        if target_path:
            folder = str(Path(target_path).parent)
            _open_path(folder)

    def _on_delete(self) -> None:
        if not self._current_video_id:
            return
        reply = QMessageBox.question(
            self,
            "Delete",
            "Remove this video from history?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._db.delete(self._current_video_id)
            self._current_video_id = None
            self._refresh_history()
            self._update_stats()
            # Clear preview panel
            self.preview_title.setText("")
            self.meta_label.setText("")
            self.tldr_label.setText("")
            self.report_browser.clear()
            self.chapters_list.clear()
            self.highlights_table.setRowCount(0)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _on_settings(self) -> None:
        dialog = SettingsDialog(self._settings_mgr, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Re-apply theme in case it changed
            self._apply_theme(self._settings_mgr.settings.theme)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self, theme: str) -> None:
        if theme == "dark":
            self.setStyleSheet(
                """
                QMainWindow, QWidget {
                    background-color: #1e1e2e;
                    color: #cdd6f4;
                }
                QLineEdit, QPlainTextEdit, QTextBrowser, QListWidget,
                QTableWidget, QComboBox {
                    background-color: #313244;
                    color: #cdd6f4;
                    border: 1px solid #45475a;
                    border-radius: 4px;
                    padding: 4px;
                }
                QPushButton {
                    background-color: #89b4fa;
                    color: #1e1e2e;
                    border: none;
                    border-radius: 4px;
                    padding: 6px 14px;
                    font-weight: bold;
                }
                QPushButton:disabled {
                    background-color: #45475a;
                    color: #6c7086;
                }
                QPushButton:hover {
                    background-color: #b4befe;
                }
                QProgressBar {
                    border: 1px solid #45475a;
                    border-radius: 4px;
                    background-color: #313244;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #89b4fa;
                    border-radius: 3px;
                }
                QSplitter::handle {
                    background-color: #45475a;
                }
                QTabBar::tab {
                    background-color: #313244;
                    color: #cdd6f4;
                    padding: 6px 12px;
                }
                QTabBar::tab:selected {
                    background-color: #89b4fa;
                    color: #1e1e2e;
                }
                QListWidget::item:selected {
                    background-color: #89b4fa;
                    color: #1e1e2e;
                }
                QLabel#app_title {
                    font-size: 18px;
                    font-weight: bold;
                    color: #89b4fa;
                }
                QHeaderView::section {
                    background-color: #313244;
                    color: #cdd6f4;
                    border: 1px solid #45475a;
                    padding: 4px;
                }
                QTableWidget {
                    gridline-color: #45475a;
                }
                QScrollBar:vertical {
                    background: #313244;
                    width: 10px;
                    border-radius: 5px;
                }
                QScrollBar::handle:vertical {
                    background: #585b70;
                    border-radius: 5px;
                }
                QScrollBar::add-line:vertical,
                QScrollBar::sub-line:vertical {
                    height: 0px;
                }
                """
            )
        else:
            # Light theme — revert to Qt defaults
            self.setStyleSheet("")

    # ------------------------------------------------------------------
    # Stats & tag filter
    # ------------------------------------------------------------------

    def _update_stats(self) -> None:
        stats = self._db.get_stats()
        n = stats.get("total_videos", 0)
        h = stats.get("total_hours", 0.0)
        self.stats_label.setText(f"{n} video{'s' if n != 1 else ''} · {h:.1f}h total")

    def _update_tag_filter(self) -> None:
        """Repopulate the tag ComboBox, keeping 'All Topics' first."""
        current_tag = self.tag_filter.currentText()
        all_tags = self._db.get_all_tags()

        self.tag_filter.blockSignals(True)
        self.tag_filter.clear()
        self.tag_filter.addItem("All Topics")
        for tag in all_tags:
            self.tag_filter.addItem(tag)

        # Restore previous selection if still available
        idx = self.tag_filter.findText(current_tag)
        if idx >= 0:
            self.tag_filter.setCurrentIndex(idx)
        else:
            self.tag_filter.setCurrentIndex(0)

        self.tag_filter.blockSignals(False)

    # ------------------------------------------------------------------
    # History selection helper
    # ------------------------------------------------------------------

    def _select_history_item(self, video_id: str) -> None:
        """Programmatically select the history item whose video_id matches."""
        for i in range(self.history_list.count()):
            item = self.history_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == video_id:
                self.history_list.setCurrentItem(item)
                self.history_list.scrollToItem(item)
                return


# ---------------------------------------------------------------------------
# File-open helper
# ---------------------------------------------------------------------------


def _open_path(path: str) -> None:
    """Open a file or directory with the OS default application."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        elif sys.platform == "win32":
            subprocess.run(["start", path], shell=True, check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Report-parsing helpers (best-effort extraction from the Markdown report)
# ---------------------------------------------------------------------------


def _parse_chapters_from_report(record: VideoRecord) -> list[str]:
    """Extract chapter headings from the Markdown report file.

    Falls back to an empty list on any error.
    """
    chapters: list[str] = []
    if not record.report_path:
        return chapters
    report_file = Path(record.report_path)
    if not report_file.exists():
        return chapters
    try:
        text = report_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            # Look for lines that start with "##" and contain timestamps
            # (chapter headings commonly look like: "## 0:01:23 — Chapter title")
            if stripped.startswith("## ") or (
                stripped.startswith("### ") and any(c.isdigit() for c in stripped[:20])
            ):
                # Strip markdown heading markers for display
                heading = stripped.lstrip("#").strip()
                if heading:
                    chapters.append(heading)
    except Exception:
        pass
    return chapters


def _parse_highlights_from_report(
    record: VideoRecord,
) -> list[tuple[str, str, str, str]]:
    """Extract highlight segment rows from the Markdown report file.

    Returns a list of (time_str, score_str, source_str, excerpt) tuples.
    Falls back to an empty list on any error.
    """
    rows: list[tuple[str, str, str, str]] = []
    if not record.report_path:
        return rows
    report_file = Path(record.report_path)
    if not report_file.exists():
        return rows
    try:
        text = report_file.read_text(encoding="utf-8")
        in_highlights = False
        for line in text.splitlines():
            stripped = line.strip()
            # Detect the highlights section
            if "highlight" in stripped.lower() and stripped.startswith("#"):
                in_highlights = True
                continue
            # Stop at the next top-level section
            if in_highlights and stripped.startswith("# "):
                break
            if not in_highlights:
                continue
            # Parse Markdown table rows: | time | score | source | excerpt |
            if stripped.startswith("|") and stripped.endswith("|"):
                cells = [c.strip() for c in stripped.split("|")]
                # cells[0] is empty (before first |), cells[-1] is empty (after last |)
                cells = [c for c in cells if c]
                # Skip header and separator rows
                if len(cells) < 4:
                    continue
                if set(cells[0]) <= set("-: "):
                    continue
                if cells[0].lower() in ("time", "start", "timestamp"):
                    continue
                rows.append((cells[0], cells[1], cells[2], " ".join(cells[3:])))
    except Exception:
        pass
    return rows


# ---------------------------------------------------------------------------
# Settings Dialog
# ---------------------------------------------------------------------------


class SettingsDialog(QDialog):
    """Modal settings dialog."""

    def __init__(
        self, settings_mgr: SettingsManager, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._settings_mgr = settings_mgr
        self.setWindowTitle("⚙ Settings")
        self.setMinimumWidth(520)
        self._build_ui()
        self._load_values()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setSpacing(12)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setSpacing(8)

        # Output directory
        out_row = QHBoxLayout()
        self._output_dir_edit = QLineEdit()
        out_browse_btn = QPushButton("Browse…")
        out_browse_btn.clicked.connect(self._browse_output_dir)
        out_row.addWidget(self._output_dir_edit)
        out_row.addWidget(out_browse_btn)
        form.addRow("Output directory:", out_row)

        # Target %
        self._target_pct_spin = QDoubleSpinBox()
        self._target_pct_spin.setRange(1.0, 50.0)
        self._target_pct_spin.setSingleStep(1.0)
        self._target_pct_spin.setSuffix(" %")
        self._target_pct_spin.setDecimals(1)
        form.addRow("Target %:", self._target_pct_spin)

        # Target minutes (0 = disabled → use %)
        self._target_mins_spin = QDoubleSpinBox()
        self._target_mins_spin.setRange(0.0, 120.0)
        self._target_mins_spin.setSingleStep(1.0)
        self._target_mins_spin.setSuffix(" min")
        self._target_mins_spin.setDecimals(1)
        self._target_mins_spin.setSpecialValueText("disabled (use %)")
        form.addRow("Target minutes:", self._target_mins_spin)

        # LLM model path
        llm_row = QHBoxLayout()
        self._llm_path_edit = QLineEdit()
        self._llm_path_edit.setPlaceholderText("Leave blank for auto-download")
        llm_browse_btn = QPushButton("Browse…")
        llm_browse_btn.clicked.connect(self._browse_llm_model)
        llm_row.addWidget(self._llm_path_edit)
        llm_row.addWidget(llm_browse_btn)
        form.addRow("LLM model path:", llm_row)

        # Whisper model
        self._whisper_combo = QComboBox()
        for model in ("tiny", "base", "small", "medium"):
            self._whisper_combo.addItem(model)
        form.addRow("Whisper model:", self._whisper_combo)

        # Skip checkboxes
        self._skip_video_cb = QCheckBox("Skip video clipping")
        form.addRow("", self._skip_video_cb)

        self._skip_llm_cb = QCheckBox("Skip LLM synthesis")
        form.addRow("", self._skip_llm_cb)

        self._skip_enrich_cb = QCheckBox("Skip enrichment")
        form.addRow("", self._skip_enrich_cb)

        # Theme
        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["dark", "light"])
        form.addRow("Theme:", self._theme_combo)

        # yt-dlp path
        self._ytdlp_edit = QLineEdit()
        self._ytdlp_edit.setPlaceholderText("yt-dlp")
        form.addRow("yt-dlp path:", self._ytdlp_edit)

        outer_layout.addLayout(form)

        # yt-dlp update button
        update_btn = QPushButton("🔄 Check for yt-dlp update")
        update_btn.clicked.connect(self._update_ytdlp)
        outer_layout.addWidget(update_btn)

        # Standard dialog buttons
        btn_box = QDialogButtonBox()
        save_btn = btn_box.addButton("Save", QDialogButtonBox.ButtonRole.AcceptRole)
        reset_btn = btn_box.addButton(
            "Reset to Defaults", QDialogButtonBox.ButtonRole.ResetRole
        )
        btn_box.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)

        save_btn.clicked.connect(self._on_save)
        reset_btn.clicked.connect(self._on_reset)
        btn_box.rejected.connect(self.reject)

        outer_layout.addWidget(btn_box)

    # ------------------------------------------------------------------
    # Value loading / saving
    # ------------------------------------------------------------------

    def _load_values(self) -> None:
        s = self._settings_mgr.settings
        self._output_dir_edit.setText(s.output_dir)
        self._target_pct_spin.setValue(s.target_pct * 100.0)
        self._target_mins_spin.setValue(
            s.target_mins if s.target_mins is not None else 0.0
        )
        self._llm_path_edit.setText(s.llm_model_path)

        idx = self._whisper_combo.findText(s.whisper_model)
        self._whisper_combo.setCurrentIndex(idx if idx >= 0 else 1)

        self._skip_video_cb.setChecked(s.no_video)
        self._skip_llm_cb.setChecked(s.no_llm)
        self._skip_enrich_cb.setChecked(s.no_enrich)

        theme_idx = self._theme_combo.findText(s.theme)
        self._theme_combo.setCurrentIndex(theme_idx if theme_idx >= 0 else 0)

        self._ytdlp_edit.setText(s.ytdlp_path)

    def _on_save(self) -> None:
        s = self._settings_mgr.settings

        s.output_dir = self._output_dir_edit.text().strip() or s.output_dir
        s.target_pct = self._target_pct_spin.value() / 100.0

        mins_val = self._target_mins_spin.value()
        s.target_mins = mins_val if mins_val > 0.0 else None

        s.llm_model_path = self._llm_path_edit.text().strip()
        s.whisper_model = self._whisper_combo.currentText()
        s.no_video = self._skip_video_cb.isChecked()
        s.no_llm = self._skip_llm_cb.isChecked()
        s.no_enrich = self._skip_enrich_cb.isChecked()
        s.theme = self._theme_combo.currentText()
        s.ytdlp_path = self._ytdlp_edit.text().strip() or "yt-dlp"

        self._settings_mgr.save()
        self.accept()

    def _on_reset(self) -> None:
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset all settings to their defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._settings_mgr.reset()
            self._load_values()

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def _browse_output_dir(self) -> None:
        current = self._output_dir_edit.text() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", current
        )
        if folder:
            self._output_dir_edit.setText(folder)

    def _browse_llm_model(self) -> None:
        current_dir = str(Path.home())
        current_path = self._llm_path_edit.text()
        if current_path:
            current_dir = str(Path(current_path).parent)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select LLM Model File",
            current_dir,
            "Model files (*.gguf *.bin *.pt *.safetensors);;All files (*)",
        )
        if path:
            self._llm_path_edit.setText(path)

    # ------------------------------------------------------------------
    # yt-dlp update
    # ------------------------------------------------------------------

    def _update_ytdlp(self) -> None:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-U", "yt-dlp"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = (result.stdout + result.stderr).strip()
            if result.returncode == 0:
                QMessageBox.information(
                    self,
                    "yt-dlp Update",
                    f"yt-dlp updated successfully.\n\n{output[-800:]}",
                )
            else:
                QMessageBox.warning(
                    self,
                    "yt-dlp Update",
                    f"Update command returned exit code {result.returncode}."
                    f"\n\n{output[-800:]}",
                )
        except subprocess.TimeoutExpired:
            QMessageBox.warning(self, "yt-dlp Update", "Update timed out after 120 s.")
        except Exception as exc:
            QMessageBox.critical(self, "yt-dlp Update", f"Unexpected error:\n{exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("yt-summariser")
    app.setApplicationVersion("1.0.0")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
