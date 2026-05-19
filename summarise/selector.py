"""
selector.py — Greedy segment selection for the YouTube summariser.

Picks the highest-scoring segments until a target playback duration is filled,
then merges chronologically adjacent picks whose gap is within a configurable
threshold.  The result is a compact, non-overlapping list of
:class:`SelectedSegment` objects ready for display or further processing.
"""

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SelectionConfig:
    """Configuration for :func:`select_segments`."""

    target_duration: float  # seconds to fill
    merge_gap: float = 3.0  # merge segments within this many seconds of each other


@dataclass
class SelectedSegment:
    """A (possibly merged) output segment ready for display."""

    start: float
    end: float
    score: float
    full_text: str  # concatenated full_text of merged source segments
    source_badge: str  # badge of highest-scoring source segment
    source_indices: list  # indices into original scored_enriched list


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_segments(
    scored_enriched: list,
    config: SelectionConfig,
) -> List[SelectedSegment]:
    """Greedily select the most important segments up to *config.target_duration*.

    Each element of *scored_enriched* must expose:

    * ``.score`` — float ranking
    * ``.full_text`` — str (provided by :class:`~enricher.EnrichedSegment`)
    * ``.source_badge`` — str
    * Either ``.segment.start`` / ``.segment.end``, or ``.start`` / ``.end``
      directly.

    The function deduplicates overlapping spans (a segment entirely contained
    within an already-selected span is skipped) before sorting chronologically
    and merging nearby picks.

    Parameters
    ----------
    scored_enriched:
        Flat list of scored enriched segment objects.
    config:
        :class:`SelectionConfig` specifying the target duration and merge gap.

    Returns
    -------
    list[SelectedSegment]
        Merged, chronologically ordered selected segments.
    """
    if not scored_enriched:
        return []

    # ------------------------------------------------------------------
    # 1. Build index-tagged copies sorted by score descending
    # ------------------------------------------------------------------
    indexed = list(enumerate(scored_enriched))
    sorted_by_score = sorted(indexed, key=lambda t: t[1].score, reverse=True)

    # ------------------------------------------------------------------
    # 2. Greedy selection
    # ------------------------------------------------------------------
    accumulated_duration = 0.0
    selected_spans: List[tuple] = []  # (start, end) of already-accepted spans
    selected: List[tuple] = []  # (original_index, item)

    for orig_idx, item in sorted_by_score:
        start, end = _get_start_end(item)

        # Skip if fully contained within an already-selected span
        if any(s_start <= start and end <= s_end for s_start, s_end in selected_spans):
            continue

        selected.append((orig_idx, item))
        selected_spans.append((start, end))
        accumulated_duration += end - start

        if accumulated_duration >= config.target_duration:
            break

    # If we exhausted everything before hitting the target, that is fine —
    # we just use all segments.

    # ------------------------------------------------------------------
    # 3. Sort chronologically
    # ------------------------------------------------------------------
    selected.sort(key=lambda t: _get_start_end(t[1])[0])

    # ------------------------------------------------------------------
    # 4. Convert to SelectedSegment and merge
    # ------------------------------------------------------------------
    raw_segments = [
        SelectedSegment(
            start=_get_start_end(item)[0],
            end=_get_start_end(item)[1],
            score=item.score,
            full_text=item.full_text,
            source_badge=item.source_badge,
            source_indices=[orig_idx],
        )
        for orig_idx, item in selected
    ]

    return _merge_adjacent(raw_segments, config.merge_gap)


# ---------------------------------------------------------------------------
# Public utility
# ---------------------------------------------------------------------------


def total_duration(segments: List[SelectedSegment]) -> float:
    """Return the total playback time covered by *segments* in seconds."""
    return sum(s.end - s.start for s in segments)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_start_end(item) -> tuple:
    """Return (start, end) from an item that may use either access pattern.

    Supports objects where the timestamps live on a nested ``.segment``
    attribute (e.g. :class:`~scorer.ScoredSegment`) as well as objects that
    expose ``.start`` / ``.end`` directly (e.g. :class:`~enricher.EnrichedSegment`).
    """
    if hasattr(item, "segment"):
        return item.segment.start, item.segment.end
    return item.start, item.end


def _merge_adjacent(
    segments: List[SelectedSegment],
    merge_gap: float,
) -> List[SelectedSegment]:
    """Merge chronologically adjacent segments whose gap is <= *merge_gap*.

    When two segments are close enough to merge:

    * ``start`` — minimum of both starts
    * ``end`` — maximum of both ends
    * ``score`` — maximum of both scores
    * ``full_text`` — both texts joined with ``" [...] "``
    * ``source_badge`` — badge from the higher-scoring segment
    * ``source_indices`` — sorted union of both index lists

    The input list is not mutated; a new list is returned.

    Parameters
    ----------
    segments:
        Chronologically sorted :class:`SelectedSegment` objects.
    merge_gap:
        Maximum gap in seconds between consecutive segments that triggers a merge.

    Returns
    -------
    list[SelectedSegment]
        Merged segment list.
    """
    if not segments:
        return []

    merged: List[SelectedSegment] = []
    # Work with a shallow copy so we never mutate the caller's objects
    current = SelectedSegment(
        start=segments[0].start,
        end=segments[0].end,
        score=segments[0].score,
        full_text=segments[0].full_text,
        source_badge=segments[0].source_badge,
        source_indices=list(segments[0].source_indices),
    )

    for nxt in segments[1:]:
        gap = nxt.start - current.end

        if gap <= merge_gap:
            # Merge nxt into current
            if nxt.score > current.score:
                # nxt wins the badge
                badge = nxt.source_badge
            else:
                badge = current.source_badge

            current = SelectedSegment(
                start=min(current.start, nxt.start),
                end=max(current.end, nxt.end),
                score=max(current.score, nxt.score),
                full_text=current.full_text + " [...] " + nxt.full_text,
                source_badge=badge,
                source_indices=sorted(
                    set(current.source_indices) | set(nxt.source_indices)
                ),
            )
        else:
            merged.append(current)
            current = SelectedSegment(
                start=nxt.start,
                end=nxt.end,
                score=nxt.score,
                full_text=nxt.full_text,
                source_badge=nxt.source_badge,
                source_indices=list(nxt.source_indices),
            )

    merged.append(current)
    return merged
