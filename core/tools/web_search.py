"""
Web search tool — DuckDuckGo search that stores results in memory.

Uses DuckDuckGo Instant Answer API (no API key required).
Falls back to HTML scraping if API returns nothing useful.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DDG_API_URL = "https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1&no_redirect=1"
DDG_HTML_URL = "https://html.duckduckgo.com/html/?q={query}"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class WebSearchResult:
    success: bool
    results: list[SearchResult]
    query: str
    error: str = ""
    stored_count: int = 0


class WebSearchTool:
    """Search DuckDuckGo and optionally store results as memories."""

    def __init__(
        self,
        db=None,
        writer=None,
        permissions: dict | None = None,
    ):
        self.db = db
        self.writer = writer
        self.permissions = permissions or {}

    async def execute(self, query: str, max_results: int = 5) -> WebSearchResult:
        """Perform search and optionally store results in memory."""
        if not query or not query.strip():
            return WebSearchResult(
                success=False, results=[], query=query, error="Empty query"
            )

        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None, self._search_duckduckgo, query, max_results
            )
        except Exception as e:
            logger.exception(f"Search failed for query: {query!r}")
            return WebSearchResult(
                success=False, results=[], query=query, error=str(e)
            )

        stored = 0
        if results and self.db is not None:
            stored = await self._store_results(query, results)

        return WebSearchResult(
            success=True,
            results=results,
            query=query,
            stored_count=stored,
        )

    def _search_duckduckgo(self, query: str, max_results: int) -> list[SearchResult]:
        """Try Instant Answer API, fall back to HTML parsing."""
        results = self._try_instant_answer(query, max_results)
        if not results:
            results = self._try_html_search(query, max_results)
        return results[:max_results]

    def _try_instant_answer(self, query: str, max_results: int) -> list[SearchResult]:
        """DuckDuckGo Instant Answer API."""
        url = DDG_API_URL.format(query=urllib.parse.quote_plus(query))
        results = []

        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.debug(f"DDG API failed: {e}")
            return []

        # Abstract (direct answer)
        abstract = data.get("AbstractText", "").strip()
        if abstract:
            results.append(SearchResult(
                title=data.get("Heading", query),
                url=data.get("AbstractURL", ""),
                snippet=abstract,
            ))

        # Related topics
        for topic in data.get("RelatedTopics", []):
            if len(results) >= max_results:
                break
            if not isinstance(topic, dict):
                continue
            # Skip category headers (they have "Topics" key)
            if "Topics" in topic:
                for subtopic in topic.get("Topics", []):
                    if not isinstance(subtopic, dict):
                        continue
                    text = subtopic.get("Text", "").strip()
                    furl = subtopic.get("FirstURL", "")
                    if text and furl:
                        # Title is usually before " - " in Text
                        parts = text.split(" - ", 1)
                        title = parts[0] if len(parts) > 1 else text[:60]
                        snippet = parts[1] if len(parts) > 1 else text
                        results.append(SearchResult(title=title, url=furl, snippet=snippet))
                        if len(results) >= max_results:
                            break
                continue

            text = topic.get("Text", "").strip()
            furl = topic.get("FirstURL", "")
            if text and furl:
                parts = text.split(" - ", 1)
                title = parts[0] if len(parts) > 1 else text[:60]
                snippet = parts[1] if len(parts) > 1 else text
                results.append(SearchResult(title=title, url=furl, snippet=snippet))

        return results

    def _try_html_search(self, query: str, max_results: int) -> list[SearchResult]:
        """Fallback: scrape DuckDuckGo HTML results."""
        url = DDG_HTML_URL.format(query=urllib.parse.quote_plus(query))
        results = []

        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            logger.debug(f"DDG HTML search failed: {e}")
            return []

        # Extract result links and snippets
        # Pattern for result titles
        title_pattern = re.compile(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', re.IGNORECASE
        )
        snippet_pattern = re.compile(
            r'class="result__snippet"[^>]*>(.+?)</(?:a|span|div)>', re.IGNORECASE | re.DOTALL
        )

        titles = title_pattern.findall(html)
        snippets = [s.strip() for s in snippet_pattern.findall(html)]

        for i, (href, title) in enumerate(titles[:max_results]):
            snippet = snippets[i] if i < len(snippets) else ""
            # Clean HTML tags from snippet
            snippet = re.sub(r"<[^>]+>", "", snippet).strip()
            # DDG uses redirect URLs; try to extract actual URL
            actual_url = href
            uddg_match = re.search(r"uddg=([^&]+)", href)
            if uddg_match:
                actual_url = urllib.parse.unquote(uddg_match.group(1))
            results.append(SearchResult(
                title=title.strip(),
                url=actual_url,
                snippet=snippet[:300],
            ))

        return results

    async def _store_results(self, query: str, results: list[SearchResult]) -> int:
        """Store search results as memories."""
        stored = 0
        for result in results[:3]:  # Only store top 3
            content = f"Web search result for '{query}': {result.title} — {result.snippet}"
            try:
                if hasattr(self.db, "insert_memory"):
                    self.db.insert_memory(
                        content=content,
                        category="search_result",
                        source=f"web:{result.url}",
                        importance=0.4,
                        confidence=0.6,
                    )
                    stored += 1
            except Exception as e:
                logger.warning(f"Failed to store search result: {e}")
        return stored

    def format_result(self, result: WebSearchResult) -> str:
        """Format results for LLM consumption."""
        if not result.success:
            return f"[SEARCH ERROR] {result.error}"

        if not result.results:
            return f'No results found for "{result.query}"'

        lines = [f'Search results for "{result.query}":\n']
        for i, r in enumerate(result.results, 1):
            lines.append(f"{i}. {r.title}")
            if r.url:
                lines.append(f"   {r.url}")
            if r.snippet:
                lines.append(f"   {r.snippet}")
            lines.append("")

        if result.stored_count:
            lines.append(f"(Stored {result.stored_count} result(s) in memory)")

        return "\n".join(lines).strip()
