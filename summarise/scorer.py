"""
scorer.py — Multi-signal segment scoring for the YouTube summariser.

Each segment receives four independent sub-scores that are min-max normalised
and then linearly combined according to :class:`ScoringWeights`.

Sub-scores
----------
tfidf    — TF-IDF mean weight; rewards information-dense segments.
semantic — Cosine similarity to the video title; rewards on-topic segments.
density  — Word-rate (words / second); rewards fast-paced segments.
energy   — RMS audio loudness; rewards energetic / emphatic segments.
"""

import re
import subprocess
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

from .transcript import Segment

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class ScoredSegment:
    segment: Segment
    score: float
    tfidf_score: float
    semantic_score: float
    density_score: float
    energy_score: float


@dataclass
class ScoringWeights:
    tfidf: float = 0.3
    semantic: float = 0.3
    density: float = 0.2
    energy: float = 0.2

    def validate(self) -> None:
        """Raise :exc:`ValueError` if the weights are invalid."""
        if any(w < 0 for w in [self.tfidf, self.semantic, self.density, self.energy]):
            raise ValueError("All weights must be >= 0")
        total = self.tfidf + self.semantic + self.density + self.energy
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_segments(
    segments: List[Segment],
    video_url: str,
    video_title: str,
    weights: ScoringWeights,
    video_path: Optional[str] = None,
) -> List[ScoredSegment]:
    """Score every segment using four signals and return :class:`ScoredSegment` objects.

    Parameters
    ----------
    segments:
        Ordered list of transcript segments to score.
    video_url:
        YouTube video URL (currently reserved for future use / provenance).
    video_title:
        Video title string used as the semantic reference query.
    weights:
        :class:`ScoringWeights` instance; must pass :meth:`~ScoringWeights.validate`.
    video_path:
        Local path to the video file for audio-energy analysis.
        Pass ``None`` to skip energy scoring (all energy scores will be 0.5
        after normalisation).

    Returns
    -------
    list[ScoredSegment]
        One entry per input segment, in the same order.
    """
    weights.validate()

    if not segments:
        return []

    # 1. Compute raw sub-scores
    raw_tfidf = _score_tfidf(segments)
    raw_semantic = _score_semantic(segments, video_title)
    raw_density = _score_density(segments)
    raw_energy = _score_energy(segments, video_path)

    # 2. Normalise each list independently
    norm_tfidf = _minmax_normalize(raw_tfidf)
    norm_semantic = _minmax_normalize(raw_semantic)
    norm_density = _minmax_normalize(raw_density)
    norm_energy = _minmax_normalize(raw_energy)

    # 3. Combine
    scored = []
    for i, seg in enumerate(segments):
        t = norm_tfidf[i]
        s = norm_semantic[i]
        d = norm_density[i]
        e = norm_energy[i]
        combined = (
            weights.tfidf * t
            + weights.semantic * s
            + weights.density * d
            + weights.energy * e
        )
        scored.append(
            ScoredSegment(
                segment=seg,
                score=combined,
                tfidf_score=t,
                semantic_score=s,
                density_score=d,
                energy_score=e,
            )
        )

    return scored


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _minmax_normalize(values: List[float]) -> List[float]:
    """Min-max normalise *values* to [0, 1].

    If all values are identical (including the degenerate single-element case),
    returns ``[0.5] * len(values)`` so that every segment is treated equally.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    span = hi - lo
    return [(v - lo) / span for v in values]


# ---------------------------------------------------------------------------
# Sub-score functions
# ---------------------------------------------------------------------------


def _score_tfidf(segments: List[Segment]) -> List[float]:
    """Compute a TF-IDF importance score for each segment.

    The score for a segment is the *mean* TF-IDF weight of its non-zero terms
    (i.e. the average importance of the words that actually appear in that
    segment, ignoring zero entries introduced by the shared vocabulary).

    Empty segments receive a score of 0.0.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

    corpus = [seg.text for seg in segments]

    # Guard: if every document is empty, return zeros immediately
    if not any(corpus):
        return [0.0] * len(segments)

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # fit_transform raises if the vocabulary ends up empty after stop-word removal
        return [0.0] * len(segments)

    scores: List[float] = []
    for row_idx in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(row_idx)
        data = row.data  # only non-zero values
        scores.append(float(data.mean()) if data.size > 0 else 0.0)

    return scores


def _score_semantic(segments: List[Segment], video_title: str) -> List[float]:
    """Compute cosine similarity between each segment and the video title.

    Uses the ``sentence-transformers`` library with the
    ``all-MiniLM-L6-v2`` model.  Returns ``[0.0] * len(segments)`` with a
    warning if the library is not installed.

    Empty segments receive a score of 0.0.
    """
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        from sentence_transformers import util as st_util
    except ImportError:
        warnings.warn(
            "sentence-transformers is not installed; semantic scoring will be "
            "skipped (all scores set to 0.0).  Install with: "
            "pip install sentence-transformers",
            ImportWarning,
            stacklevel=2,
        )
        return [0.0] * len(segments)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode the title once
    title_embedding = model.encode(video_title or "", convert_to_tensor=True)

    scores: List[float] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            scores.append(0.0)
            continue
        seg_embedding = model.encode(text, convert_to_tensor=True)
        similarity = st_util.cos_sim(seg_embedding, title_embedding)
        # cos_sim returns a 1×1 tensor; clamp to [0, 1] (similarity can be negative)
        scores.append(float(max(0.0, similarity.item())))

    return scores


def _score_density(segments: List[Segment]) -> List[float]:
    """Compute word-rate (words per second) for each segment.

    A higher rate indicates more information-dense speech.
    """
    scores: List[float] = []
    for seg in segments:
        duration = max(seg.end - seg.start, 0.001)
        word_count = len(seg.text.split())
        scores.append(word_count / duration)
    return scores


def _score_energy(
    segments: List[Segment],
    video_path: Optional[str],
) -> List[float]:
    """Estimate audio loudness for each segment using ffmpeg's volumedetect filter.

    Each segment's score is the *linear* mean volume derived from::

        ffmpeg -ss <start> -to <end> -i <video_path> -af volumedetect -f null -

    The ``mean_volume`` value (dB) reported in stderr is converted to linear
    amplitude via ``10 ** (db / 20)``.

    Returns ``[0.0] * len(segments)`` when *video_path* is ``None``.
    Segments shorter than 0.5 s, or segments where ffmpeg fails, receive 0.0.
    """
    if video_path is None:
        return [0.0] * len(segments)

    _mean_vol_re = re.compile(r"mean_volume:\s*(-?\d+\.?\d*)\s*dB")

    scores: List[float] = []
    for seg in segments:
        duration = seg.end - seg.start
        if duration < 0.5:
            scores.append(0.0)
            continue

        cmd = [
            "ffmpeg",
            "-ss",
            str(seg.start),
            "-to",
            str(seg.end),
            "-i",
            str(video_path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            # volumedetect writes to stderr
            match = _mean_vol_re.search(result.stderr)
            if match:
                db = float(match.group(1))
                scores.append(10 ** (db / 20))
            else:
                scores.append(0.0)
        except Exception:
            scores.append(0.0)

    return scores
