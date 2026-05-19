"""
synthesiser.py — Local LLM synthesis for the YouTube summariser.

Uses Mistral-7B-Instruct (Q4_K_M GGUF) via llama-cpp-python, auto-downloaded
from HuggingFace Hub on first use.  Falls back to a BART/TF-IDF extractive
pipeline when llama-cpp-python is not installed.

The LLM is instructed to:
  - Cite timestamps for every factual claim.
  - Signal uncertain or unexplained concepts with [[LOOKUP: term]] markers.
  - Only use information present in the transcript (no hallucination).

[[LOOKUP: X]] markers are NOT resolved here — that is references.py's job.
"""

import json
import logging
import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

from . import utils
from .enricher import EnrichedSegment
from .segmenter import Chapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level LLM singleton
# ---------------------------------------------------------------------------

_llm_instance = None  # Llama instance, created once per process


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class SynthesisResult:
    """Output of the synthesis step."""

    tldr: str  # 2-3 sentence summary of the entire video
    key_concepts: List[str]  # concept names extracted from transcript
    chapter_summaries: Dict[str, str]  # {"0": "summary", "1": "summary", ...}
    insights: List[str]  # cross-chapter connections and insights
    takeaways: List[str]  # actionable items
    lookup_terms: List[str]  # [[LOOKUP: X]] terms found, for references.py
    raw_output: str  # full LLM output for debugging


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def synthesise(
    enriched_segments: List[EnrichedSegment],
    chapters: List[Chapter],
    video_title: str,
    model_path: Optional[str] = None,
) -> SynthesisResult:
    """Generate a structured synthesis of the video using a local LLM.

    Parameters
    ----------
    enriched_segments:
        All enriched transcript segments (output of enricher.py).
    chapters:
        Detected chapters (output of segmenter.py).
    video_title:
        Human-readable title of the video.
    model_path:
        Path to a local GGUF file.  When ``None`` the model is auto-downloaded
        from HuggingFace Hub.

    Returns
    -------
    SynthesisResult
    """
    if not enriched_segments:
        return SynthesisResult(
            tldr="No transcript available.",
            key_concepts=[],
            chapter_summaries={},
            insights=[],
            takeaways=[],
            lookup_terms=[],
            raw_output="",
        )

    # ------------------------------------------------------------------
    # 1. Build transcript context string (≤ 6 000 chars)
    # ------------------------------------------------------------------
    transcript_ctx = _build_transcript_context(enriched_segments)

    # ------------------------------------------------------------------
    # 2. Build chapter context string
    # ------------------------------------------------------------------
    chapter_ctx = _build_chapter_context(chapters)

    # ------------------------------------------------------------------
    # 3. Build Mistral-instruct prompt
    # ------------------------------------------------------------------
    prompt = _build_prompt(transcript_ctx, chapter_ctx, video_title, chapters)

    # ------------------------------------------------------------------
    # 4. Generate
    # ------------------------------------------------------------------
    try:
        raw = _llm_generate(prompt, model_path)
    except Exception as exc:
        logger.warning("LLM generation failed (%s); using extractive fallback.", exc)
        return _bart_extractive_fallback(enriched_segments, chapters, video_title)

    # ------------------------------------------------------------------
    # 5. Parse JSON
    # ------------------------------------------------------------------
    parsed = _parse_json_response(raw)
    if not parsed:
        logger.warning("JSON parse failed; using extractive fallback.")
        return _bart_extractive_fallback(enriched_segments, chapters, video_title)

    # ------------------------------------------------------------------
    # 6. Extract [[LOOKUP: X]] terms from all string fields
    # ------------------------------------------------------------------
    all_text = json.dumps(parsed)
    lookup_terms = _extract_lookup_terms(all_text)

    # ------------------------------------------------------------------
    # 7. Normalise chapter_summaries keys to strings
    # ------------------------------------------------------------------
    raw_summaries = parsed.get("chapter_summaries", {})
    chapter_summaries: Dict[str, str] = {
        str(k): str(v) for k, v in raw_summaries.items()
    }

    return SynthesisResult(
        tldr=str(parsed.get("tldr", "")),
        key_concepts=list(parsed.get("key_concepts", [])),
        chapter_summaries=chapter_summaries,
        insights=list(parsed.get("insights", [])),
        takeaways=list(parsed.get("takeaways", [])),
        lookup_terms=lookup_terms,
        raw_output=raw,
    )


# ---------------------------------------------------------------------------
# Private — context builders
# ---------------------------------------------------------------------------


def _build_transcript_context(enriched_segments: List[EnrichedSegment]) -> str:
    """Return a ≤ 6 000-char timestamped transcript string.

    Segments are sorted by score where available (highest first) to keep the
    most informative content within the truncation window.  When no score
    attribute is present, chronological order is preserved.
    """

    # Sort by score descending if available, otherwise leave in time order
    def _score(seg: EnrichedSegment) -> float:
        return getattr(seg, "score", 0.0)

    has_scores = any(
        hasattr(s, "score") and getattr(s, "score", 0.0) != 0.0
        for s in enriched_segments
    )
    if has_scores:
        ordered = sorted(enriched_segments, key=_score, reverse=True)
    else:
        ordered = list(enriched_segments)

    lines: List[str] = []
    total = 0
    limit = 6000
    truncated = False

    for seg in ordered:
        ts = utils.format_timestamp(seg.start)
        line = f"[{ts}] {seg.full_text}"
        candidate = total + len(line) + 1  # +1 for newline
        if candidate > limit:
            truncated = True
            break
        lines.append(line)
        total = candidate

    # Restore chronological order inside the context window
    lines.sort()  # lexicographic sort on "[HH:MM:SS…]" prefix works correctly

    result = "\n".join(lines)
    if truncated:
        result += "\n[...TRUNCATED...]"
    return result


def _build_chapter_context(chapters: List[Chapter]) -> str:
    """Return a formatted chapter listing."""
    if not chapters:
        return "(no chapters detected)"

    lines: List[str] = []
    for ch in chapters:
        start_ts = utils.format_timestamp(ch.start)
        end_ts = utils.format_timestamp(ch.end)
        lines.append(f"Chapter {ch.index + 1} [{start_ts} - {end_ts}]: {ch.title}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private — prompt builder
# ---------------------------------------------------------------------------


def _build_prompt(
    transcript_ctx: str,
    chapter_ctx: str,
    video_title: str,
    chapters: List[Chapter],
) -> str:
    """Build a Mistral-instruct formatted prompt."""

    # Build the chapter_summaries skeleton so the model knows what keys to emit
    chapter_keys = ", ".join(f'"{ch.index}": "..."' for ch in chapters)
    if not chapter_keys:
        chapter_keys = '"0": "..."'

    system = (
        "You are an expert educational content analyser.\n"
        "You are given a timestamped transcript of an educational video.\n\n"
        "RULES:\n"
        "1. Every factual claim MUST end with a timestamp citation like [00:14:22]\n"
        "2. If a concept appears in the transcript but is not explained clearly,\n"
        "   write [[LOOKUP: concept name]] — do NOT invent a definition\n"
        "3. Only use information present in the transcript\n"
        "4. Be concise and precise\n\n"
        "OUTPUT FORMAT (respond with exactly this JSON structure):\n"
        "{\n"
        '  "tldr": "2-3 sentence summary of the entire video",\n'
        '  "key_concepts": ["concept1", "concept2"],\n'
        '  "chapter_summaries": {\n'
        f"    {chapter_keys}\n"
        "  },\n"
        '  "insights": [\n'
        '    "cross-chapter insight with citations [00:XX:XX] and [00:YY:YY]"\n'
        "  ],\n"
        '  "takeaways": [\n'
        '    "actionable takeaway from the video"\n'
        "  ]\n"
        "}"
    )

    user = (
        f"VIDEO TITLE: {video_title}\n\n"
        f"CHAPTERS:\n{chapter_ctx}\n\n"
        f"TRANSCRIPT:\n{transcript_ctx}\n\n"
        "Analyse the transcript above and respond with the JSON only."
    )

    # Mistral instruct format
    prompt = f"[INST] {system}\n\n{user} [/INST]"
    return prompt


# ---------------------------------------------------------------------------
# Private — LLM loading and generation
# ---------------------------------------------------------------------------


def _get_llm(model_path: Optional[str] = None):
    """Load and return the llama-cpp Llama singleton.

    Priority:
    1. Provided *model_path* (must exist on disk).
    2. Auto-download ``mistral-7b-instruct-v0.2.Q4_K_M.gguf`` from HF Hub.

    Raises ``ImportError`` if llama-cpp-python is not installed.
    Raises ``RuntimeError`` if no model file can be located.
    """
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    from llama_cpp import Llama  # noqa: PLC0415 — intentional lazy import

    # Resolve the model file path
    if model_path and os.path.isfile(model_path):
        resolved_path = model_path
    else:
        if model_path:
            warnings.warn(
                f"Model path {model_path!r} not found; auto-downloading from "
                "HuggingFace Hub.",
                RuntimeWarning,
                stacklevel=4,
            )
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        resolved_path = hf_hub_download(
            repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            cache_dir=None,  # uses default ~/.cache/huggingface
        )

    _llm_instance = Llama(
        model_path=resolved_path,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
        verbose=False,
    )
    return _llm_instance


def _llm_generate(prompt: str, model_path: Optional[str] = None) -> str:
    """Call the llama-cpp model and return the raw string response.

    Raises on any failure so the caller can decide to fall back.
    """
    llm = _get_llm(model_path)

    response = llm(
        prompt,
        max_tokens=1024,
        temperature=0.1,  # low temperature for factual / structured output
        stop=["[INST]"],
        echo=False,
    )

    choices = response.get("choices", [])
    if not choices:
        raise RuntimeError("llama-cpp returned no choices.")

    return choices[0].get("text", "").strip()


# ---------------------------------------------------------------------------
# Private — extractive fallback (no llama-cpp)
# ---------------------------------------------------------------------------


def _bart_extractive_fallback(
    enriched_segments: List[EnrichedSegment],
    chapters: List[Chapter],
    video_title: str,
) -> SynthesisResult:
    """Extractive fallback used when llama-cpp-python is not installed.

    * TL;DR — sumy LSA (3 sentences) or first 3 sentences of transcript.
    * Key concepts — KeyBERT top-8 or TF-IDF top-8 terms.
    * Chapter summaries — first sentence of each chapter.
    * Insights — empty (cross-chapter reasoning requires the LLM).
    * Takeaways — top-5 TF-IDF terms as plain strings.
    """
    full_text = " ".join(seg.full_text for seg in enriched_segments)

    # ------------------------------------------------------------------ TL;DR
    tldr = _sumy_tldr(full_text, n_sentences=3)

    # ----------------------------------------------------------- Key concepts
    key_concepts = _extract_key_concepts(full_text, n=8)

    # ----------------------------------------------- Chapter summaries
    chapter_summaries: Dict[str, str] = {}
    for ch in chapters:
        ch_segs = [
            enriched_segments[i] for i in ch.segments if i < len(enriched_segments)
        ]
        if not ch_segs:
            continue
        ch_text = " ".join(seg.full_text for seg in ch_segs)
        first_sentence = (ch_text.split(".")[0] + ".").strip() if ch_text else ""
        chapter_summaries[str(ch.index)] = first_sentence

    # ------------------------------------------------------- Takeaways
    takeaways = _tfidf_top_terms(full_text, n=5)

    return SynthesisResult(
        tldr=tldr,
        key_concepts=key_concepts,
        chapter_summaries=chapter_summaries,
        insights=[],
        takeaways=takeaways,
        lookup_terms=[],
        raw_output="[extractive fallback — llama-cpp-python not installed]",
    )


def _sumy_tldr(text: str, n_sentences: int = 3) -> str:
    """Return an LSA summary using sumy, or the first *n_sentences* of text."""
    try:
        from sumy.nlp.tokenizers import Tokenizer  # noqa: PLC0415
        from sumy.parsers.plaintext import PlaintextParser  # noqa: PLC0415
        from sumy.summarizers.lsa import LsaSummarizer  # noqa: PLC0415

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary_sentences = summarizer(parser.document, n_sentences)
        return " ".join(str(s) for s in summary_sentences)
    except Exception:
        # Hard fallback: first n_sentences split by period
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        return ". ".join(sentences[:n_sentences]) + ("." if sentences else "")


def _extract_key_concepts(text: str, n: int = 8) -> List[str]:
    """Extract top-*n* key concepts via KeyBERT or TF-IDF."""
    try:
        from keybert import KeyBERT  # noqa: PLC0415

        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=n,
        )
        return [kw for kw, _score in keywords]
    except Exception:
        return _tfidf_top_terms(text, n=n)


def _tfidf_top_terms(text: str, n: int = 8) -> List[str]:
    """Return top-*n* terms by TF-IDF score from a single document."""
    try:
        import numpy as np  # noqa: PLC0415
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

        vectorizer = TfidfVectorizer(stop_words="english", max_features=200)
        matrix = vectorizer.fit_transform([text])
        # Use numpy to work with the sparse matrix safely
        scores_array = np.asarray(matrix.todense())[0]  # type: ignore[union-attr]
        terms: List[str] = list(vectorizer.get_feature_names_out())
        top_indices = scores_array.argsort()[::-1][:n]
        return [terms[int(i)] for i in top_indices if scores_array[int(i)] > 0]
    except Exception:
        # Last resort: split and count unique words
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        freq: Dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        return sorted(freq, key=lambda k: freq[k], reverse=True)[:n]


# ---------------------------------------------------------------------------
# Private — utilities
# ---------------------------------------------------------------------------


def _extract_lookup_terms(text: str) -> List[str]:
    """Return a deduplicated list of X from all ``[[LOOKUP: X]]`` patterns in *text*."""
    matches = re.findall(r"\[\[LOOKUP:\s*([^\]]+?)\s*\]\]", text)
    seen: set = set()
    result: List[str] = []
    for m in matches:
        key = m.strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _parse_json_response(raw: str) -> dict:
    """Extract and parse the JSON object from a raw LLM response.

    Finds the first ``{`` and last ``}`` in *raw*, then applies progressive
    repair heuristics before giving up and returning ``{}``.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        logger.debug("No JSON object found in LLM response.")
        return {}

    candidate = raw[start : end + 1]

    # Attempt 1: verbatim
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Attempt 2: remove trailing commas before } or ]
    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 3: replace unescaped single quotes used as string delimiters
    # Only do this for values, not apostrophes inside words
    fixed2 = re.sub(r"(?<!\w)'([^']*)'(?!\w)", r'"\1"', fixed)
    try:
        return json.loads(fixed2)
    except json.JSONDecodeError:
        pass

    logger.debug("All JSON repair attempts failed for LLM output.")
    return {}
