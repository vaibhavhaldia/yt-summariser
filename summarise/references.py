"""
references.py — Wikipedia on-demand lookup for the YouTube summariser.

Called exclusively for [[LOOKUP: term]] markers produced by synthesiser.py.
Wikipedia is used as a dictionary — it never adds new topics or criticises
the speaker; it only clarifies terms the transcript itself left unexplained.

No external HTTP library is needed: only urllib from the standard library.
No API key or authentication is required for the Wikipedia REST API.
"""

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class WikiReference:
    """Result of a single Wikipedia lookup."""

    term: str  # the original [[LOOKUP: X]] term
    title: str  # Wikipedia article title (may differ from term)
    summary: str  # first 1-3 sentences extracted from the article
    url: str  # full desktop Wikipedia URL
    found: bool  # False if Wikipedia returned no article


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_lookups(lookup_terms: List[str]) -> Dict[str, WikiReference]:
    """Fetch a Wikipedia summary for each term in *lookup_terms*.

    Parameters
    ----------
    lookup_terms:
        List of term strings from :func:`~synthesiser._extract_lookup_terms`.

    Returns
    -------
    dict
        ``{term: WikiReference}`` mapping.  Every term receives an entry;
        failed lookups produce a :class:`WikiReference` with ``found=False``.
        Never raises.
    """
    results: Dict[str, WikiReference] = {}

    for i, term in enumerate(lookup_terms):
        if i > 0:
            time.sleep(0.2)  # gentle rate limiting — Wikipedia asks for this

        try:
            ref = _fetch_wiki(term)
        except Exception as exc:
            logger.warning("Unexpected error fetching %r: %s", term, exc)
            ref = WikiReference(term=term, title="", summary="", url="", found=False)

        results[term] = ref

    return results


def inject_references(text: str, references: Dict[str, WikiReference]) -> str:
    """Replace every ``[[LOOKUP: X]]`` marker in *text* with its resolved form.

    Replacement rules
    -----------------
    * ``found=True``:  ``[X — {summary} ({url})]``
    * ``found=False``: ``[X]``

    The marker is **always** replaced so no raw ``[[LOOKUP: …]]`` syntax leaks
    into the rendered report.

    Parameters
    ----------
    text:
        Any string that may contain ``[[LOOKUP: X]]`` markers.
    references:
        Output of :func:`resolve_lookups`.

    Returns
    -------
    str
        *text* with all markers replaced.
    """

    def _replace(match: re.Match) -> str:
        raw_term = match.group(1).strip()

        # Try exact match first, then case-insensitive scan
        ref = references.get(raw_term)
        if ref is None:
            for key, val in references.items():
                if key.lower() == raw_term.lower():
                    ref = val
                    break

        if ref is None:
            # Term was in the text but not looked up (shouldn't happen normally)
            return f"[{raw_term}]"

        if ref.found:
            return f"[{raw_term} — {ref.summary} ({ref.url})]"
        else:
            return f"[{raw_term}]"

    return re.sub(r"\[\[LOOKUP:\s*([^\]]+?)\s*\]\]", _replace, text)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fetch_wiki(term: str) -> WikiReference:
    """Fetch a Wikipedia summary for *term* via the REST API.

    Uses ``https://en.wikipedia.org/api/rest_v1/page/summary/{title}``.

    Parameters
    ----------
    term:
        The lookup term (e.g. ``"gradient descent"``).

    Returns
    -------
    WikiReference
        ``found=True`` when an article was located; ``found=False`` on 404
        or any error.
    """
    # Wikipedia REST API: spaces become underscores, rest is percent-encoded
    encoded = urllib.parse.quote(term.replace(" ", "_"), safe="")
    api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"

    req = urllib.request.Request(
        api_url,
        headers={
            "User-Agent": "yt-summariser/1.0 (educational video summariser; "
            "contact: open-source project)",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw_bytes = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            logger.debug("Wikipedia 404 for term %r", term)
        else:
            logger.warning("Wikipedia HTTP %d for term %r", exc.code, term)
        return WikiReference(term=term, title="", summary="", url="", found=False)
    except Exception as exc:
        logger.warning("Wikipedia request failed for %r: %s", term, exc)
        return WikiReference(term=term, title="", summary="", url="", found=False)

    try:
        data = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        logger.warning("Wikipedia JSON decode error for %r: %s", term, exc)
        return WikiReference(term=term, title="", summary="", url="", found=False)

    # Validate we got a real article (not a disambiguation or missing page)
    page_type = data.get("type", "")
    if page_type in ("disambiguation", "no-extract"):
        logger.debug("Wikipedia returned %r for term %r", page_type, term)
        return WikiReference(term=term, title="", summary="", url="", found=False)

    title = data.get("title", term)
    full_extract = data.get("extract", "")
    page_url = (
        data.get("content_urls", {}).get("desktop", {}).get("page", "")
        or f"https://en.wikipedia.org/wiki/{encoded}"
    )

    summary = _first_n_sentences(full_extract, n=3)

    if not summary:
        return WikiReference(
            term=term, title=title, summary="", url=page_url, found=False
        )

    return WikiReference(
        term=term,
        title=title,
        summary=summary,
        url=page_url,
        found=True,
    )


def _first_n_sentences(text: str, n: int = 3) -> str:
    """Return the first *n* sentences of *text*.

    Sentences are split on ``". "`` (period-space).  The final sentence
    fragment retains its trailing period if present.  Returns the full text
    when fewer than *n* sentence boundaries exist.
    """
    if not text:
        return ""

    # Split on ". " and reassemble up to n parts
    parts = text.split(". ")
    selected = parts[:n]

    if len(parts) > n:
        # Re-attach the period that was consumed by the split
        joined = ". ".join(selected) + "."
    else:
        joined = ". ".join(selected)
        # Preserve original trailing period if present
        if text.endswith(".") and not joined.endswith("."):
            joined += "."

    return joined.strip()
