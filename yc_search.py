#!/usr/bin/env python3
"""
yc_search.py

Real Y Combinator job search backed by the public Hacker News (Algolia) API,
parsing the latest "Ask HN: Who is hiring?" thread.

Exports a function:
    yc_search(terms: list[str], location: str | None) -> list[dict]

Each job dict has keys: id, source, title, company, location, description, url.

Also provides a CLI:
    python yc_search.py --terms python ml --location Remote --limit 25 --verbose

Notes
- This uses HN Algolia API endpoints that do not require authentication.
- We try to robustly parse comment text; content quality varies.
- If no location is provided, we return any result that matches the terms.
- If location is provided, we include only results where location keyword appears
  in the text (case-insensitive), or "remote" if location == "Remote".
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

ALGOLIA_SEARCH_URL = "https://hn.algolia.com/api/v1/search"
ALGOLIA_SEARCH_BY_DATE_URL = "https://hn.algolia.com/api/v1/search_by_date"
ALGOLIA_ITEM_URL = "https://hn.algolia.com/api/v1/items/{}"
HN_ITEM_URL = "https://news.ycombinator.com/item?id={}"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/119.0.0.0 Safari/537.36"
)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "application/json, */*;q=0.8",
})


def _http_get_json(url: str, params: Optional[dict] = None, retries: int = 2, timeout: float = 15.0) -> dict:
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = SESSION.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))
            else:
                raise
    # Should not reach here
    raise last_err  # type: ignore[misc]


def _strip_html(text: str) -> str:
    # Remove HTML tags and unescape entities
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    # Normalize whitespace
    return re.sub(r"\s+", " ", text).strip()


def _extract_company_and_title(text: str) -> tuple[str, str]:
    """Heuristic: use first sentence/line to derive company/title."""
    if not text:
        return "", ""
    first = text.split(". ")[0].splitlines()[0]
    # Try patterns like: "Company – Title" or "Company - Title"
    m = re.match(r"^\s*([^\-–:\u2013\u2014]+)[\-–:\u2013\u2014]\s*(.+)$", first)
    if m:
        company = m.group(1).strip()[:80]
        title = m.group(2).strip()[:120]
        return company, title
    # Try parentheses for location or role
    m2 = re.match(r"^\s*([^()]+)\s*\(([^)]+)\)\s*(.*)$", first)
    if m2:
        company = m2.group(1).strip()[:80]
        rest = (m2.group(3) or "").strip()
        return company, (rest or first)[:120]
    # Fallback: split by comma
    parts = [p.strip() for p in re.split(r",|\|", first) if p.strip()]
    if len(parts) >= 2:
        return parts[0][:80], parts[1][:120]
    # Default
    return "", first[:120]


def _extract_location(text: str) -> str:
    t = text.lower()
    if "remote" in t:
        return "Remote"
    # Simple city keywords (extend as needed)
    cities = [
        ("san francisco", "San Francisco"),
        ("sf", "San Francisco"),
        ("new york", "New York"),
        ("nyc", "New York"),
        ("seattle", "Seattle"),
        ("austin", "Austin"),
        ("boston", "Boston"),
        ("london", "London"),
        ("berlin", "Berlin"),
        ("toronto", "Toronto"),
    ]
    for key, proper in cities:
        if re.search(rf"\b{re.escape(key)}\b", t):
            return proper
    return ""


def _matches_terms(text: str, terms: List[str]) -> bool:
    if not terms:
        return True
    t = text.lower()
    return all(term.lower() in t for term in terms)


def _matches_location(text: str, location: Optional[str]) -> bool:
    if not location:
        return True
    loc = location.strip().lower()
    t = text.lower()
    if loc == "remote":
        return "remote" in t
    return loc in t


def _latest_who_is_hiring_story_id() -> Optional[int]:
    """Find the latest monthly 'Ask HN: Who is hiring?' thread id.

    We query Algolia's search_by_date endpoint (newest first) and filter by
    author 'whoishiring' and story tag, restricting to title matches. This
    avoids older results that a generic search can return.
    """
    # Newest first; use tags to filter by author and story
    params = {
        "query": "Ask HN: Who is hiring?",
        "tags": "story,author_whoishiring",
        "hitsPerPage": 10,
        "restrictSearchableAttributes": "title",
        "page": 0,
    }
    try:
        data = _http_get_json(ALGOLIA_SEARCH_BY_DATE_URL, params=params)
        hits = data.get("hits", [])
    except Exception:
        # Fallback: use generic search (may return older threads)
        data = _http_get_json(ALGOLIA_SEARCH_URL, params={
            "query": "Ask HN: Who is hiring?",
            "tags": "story",
            "hitsPerPage": 20,
            "restrictSearchableAttributes": "title",
            "page": 0,
        })
        hits = data.get("hits", [])

    pat = re.compile(r"ask hn:\s*who is hiring\?", re.IGNORECASE)
    for h in hits:
        title = h.get("title") or ""
        if pat.search(title):
            try:
                return int(h.get("objectID"))
            except Exception:
                continue
    # As a last resort, pick the most recent hit
    if hits:
        try:
            return int(hits[0].get("objectID"))
        except Exception:
            return None
    return None


def _fetch_thread_comments(story_id: int) -> List[dict]:
    data = _http_get_json(ALGOLIA_ITEM_URL.format(story_id))
    return data.get("children", []) or []


def yc_search(terms: List[str], location: Optional[str] = None, limit: int = 50, verbose: bool = False) -> List[Dict[str, Any]]:
    """Search Y Combinator (HN Who is hiring?) for job posts matching terms/location.

    Returns a list of job dictionaries with fields:
    id, source, title, company, location, description, url
    """
    try:
        story_id = _latest_who_is_hiring_story_id()
        if story_id is None:
            if verbose:
                print("[yc_search] No Who is hiring? thread found.", file=sys.stderr)
            return []
        if verbose:
            print(f"[yc_search] Using thread id: {story_id}", file=sys.stderr)
        comments = _fetch_thread_comments(story_id)
    except Exception as e:
        if verbose:
            print(f"[yc_search] Failed to retrieve thread: {e}", file=sys.stderr)
        return []

    results: List[Dict[str, Any]] = []
    for c in comments:
        if not c or c.get("type") != "comment":
            continue
        c_id = c.get("id")
        raw = c.get("text") or ""
        text = _strip_html(raw)
        if not text:
            continue
        if not _matches_terms(text, terms):
            continue
        if not _matches_location(text, location):
            continue

        company, title = _extract_company_and_title(text)
        loc = _extract_location(text)
        # If user specified a location, prefer that value
        if location:
            loc = location

        job = {
            "id": f"hn_{c_id}",
            "source": "ycombinator",
            "title": title or "Software Engineer",
            "company": company or (c.get("author") or "Unknown"),
            "location": loc or "",
            "description": text[:800],
            "url": HN_ITEM_URL.format(c_id),
        }
        results.append(job)
        if len(results) >= limit:
            break

    return results


def _main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Search Y Combinator (HN) Who is hiring? posts for jobs.")
    ap.add_argument("--terms", nargs="*", default=[], help="Search terms, e.g., python ml data")
    ap.add_argument("--location", default=None, help="Location keyword to match (e.g., Remote, San Francisco)")
    ap.add_argument("--limit", type=int, default=25, help="Max results to return")
    ap.add_argument("--verbose", action="store_true", help="Verbose diagnostics to stderr")
    args = ap.parse_args(argv)

    try:
        jobs = yc_search(args.terms, args.location, limit=args.limit, verbose=args.verbose)
        print(json.dumps(jobs, indent=2))
        return 0
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(_main())
