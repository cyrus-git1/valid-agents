"""
Web scraping service with dual strategy: Scrapy primary, Firecrawl fallback.

Usage
-----
    from app.services.scraper_service import ScraperService

    svc = ScraperService()
    result = svc.scrape("https://example.com")
    print(result.total_pages, result.pages[0].title)

Also serves as the CLI entry point for Scrapy subprocess invocation:
    python app/services/scraper_service.py <url> <output_file.json>
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import dotenv
import scrapy
import trafilatura
from scrapy.crawler import CrawlerProcess
from scrapy.settings import Settings
from scrapy import signals
from scrapy.signalmanager import dispatcher

from app.models.domain.scraper import ScrapedPage, ScrapedSite

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

SCRAPY_TIMEOUT_SECONDS = 300


# ── Scrapy Spider ───────────────────────────────────────────────────────────


class SiteSpider(scrapy.Spider):
    name = "site"
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1.0,
        "AUTOTHROTTLE_MAX_DELAY": 10.0,
        "DOWNLOAD_TIMEOUT": 20,
        "RETRY_TIMES": 2,
        "CONCURRENT_REQUESTS": 8,
        "LOG_LEVEL": "INFO",
    }

    def __init__(self, start_url: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = [start_url]
        self.allowed_domains = [urlparse(start_url).netloc]

    def parse(self, response):
        html = response.text
        text = trafilatura.extract(html, include_comments=False, include_tables=False)
        if text:
            yield {
                "url": response.url,
                "title": response.css("title::text").get(),
                "text": text,
            }

        for href in response.css("a::attr(href)").getall():
            if href and not href.startswith(("tel:", "mailto:", "javascript:", "#", "data:")):
                try:
                    yield response.follow(href, callback=self.parse)
                except ValueError:
                    continue


# ── Page formatting helper ──────────────────────────────────────────────────


def _format_page(idx: int, url: str, title: str, text: str) -> ScrapedPage | None:
    """Format raw scraped data into a ScrapedPage. Returns None if empty."""
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if url:
        parts.append(f"URL: {url}")
    if text:
        parts.append(text)

    full_text = "\n\n".join(parts)
    if not full_text.strip():
        return None

    return ScrapedPage(page=idx, url=url or "", title=title or "", text=full_text)


# ── Scrapy in-process runner (called by subprocess) ────────────────────────


def _run_scrapy_in_process(url: str, output_file: str) -> None:
    """Run Scrapy spider in the current process and write results to output_file.

    This is called as a subprocess entry point because CrawlerProcess
    can only run once per process.
    """
    settings = Settings()
    settings.set("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    process = CrawlerProcess(settings)

    collected_items: List[dict] = []

    def item_scraped(item, response, spider):
        collected_items.append(dict(item))

    dispatcher.connect(item_scraped, signal=signals.item_scraped)

    process.crawl(SiteSpider, start_url=url)
    process.start()

    pages = []
    for idx, item in enumerate(collected_items, start=1):
        page = _format_page(idx, item.get("url", ""), item.get("title", ""), item.get("text", ""))
        if page:
            pages.append(page.model_dump())

    output_data = {
        "source_url": url,
        "scraped_at": datetime.now().isoformat(),
        "total_pages": len(pages),
        "pages": pages,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


# ── Service ─────────────────────────────────────────────────────────────────


class ScraperService:
    """Web scraper: Scrapy primary, Firecrawl fallback.

    Usage
    -----
        svc = ScraperService()
        result = svc.scrape("https://example.com")
        # result is a ScrapedSite with .pages, .total_pages, etc.
    """

    def __init__(
        self,
        firecrawl_api_key: str | None = None,
        timeout: int = SCRAPY_TIMEOUT_SECONDS,
    ):
        self.firecrawl_api_key = firecrawl_api_key or os.environ.get("FIRECRAWL_API_KEY")
        self.timeout = timeout

    def scrape(self, url: str) -> ScrapedSite:
        """Scrape a URL. Tries Scrapy first, falls back to Firecrawl.

        Returns a ScrapedSite with structured page data.
        Never raises — returns an empty ScrapedSite on total failure.
        """
        # Try Scrapy
        result = self._scrape_with_scrapy(url)
        if result and result.total_pages > 0:
            return result

        # Fallback to Firecrawl
        if self.firecrawl_api_key:
            logger.info("Scrapy returned no pages for %s, trying Firecrawl fallback", url)
            result = self._scrape_with_firecrawl(url)
            if result and result.total_pages > 0:
                return result

        logger.warning("All scraping strategies failed for %s", url)
        return ScrapedSite(
            source_url=url,
            scraped_at=datetime.now().isoformat(),
            total_pages=0,
            pages=[],
        )

    def _scrape_with_scrapy(self, url: str) -> ScrapedSite | None:
        """Run Scrapy spider as a subprocess. Returns None on failure."""
        script = Path(__file__).resolve()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name

        try:
            logger.info("Starting Scrapy scrape for %s (timeout: %ds)", url, self.timeout)

            subprocess.run(
                [sys.executable, str(script), url, out_path],
                timeout=self.timeout,
                capture_output=True,
                check=True,
            )

            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            pages = [ScrapedPage(**p) for p in data.get("pages", [])]
            result = ScrapedSite(
                source_url=url,
                scraped_at=data.get("scraped_at", ""),
                total_pages=len(pages),
                pages=pages,
            )

            logger.info("Scrapy scraped %d pages from %s", result.total_pages, url)
            return result

        except subprocess.TimeoutExpired:
            logger.warning("Scrapy timed out after %ds for %s", self.timeout, url)
            return None

        except subprocess.CalledProcessError as e:
            logger.warning(
                "Scrapy subprocess failed (exit %d) for %s: %s",
                e.returncode, url, (e.stderr or "")[:500],
            )
            return None

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning("Scrapy output unreadable for %s: %s", url, e)
            return None

        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def _scrape_with_firecrawl(self, url: str) -> ScrapedSite | None:
        """Crawl a site using the Firecrawl API."""
        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            logger.warning("firecrawl package not installed, cannot use Firecrawl fallback")
            return None

        if not self.firecrawl_api_key:
            logger.warning("FIRECRAWL_API_KEY not set, cannot use Firecrawl fallback")
            return None

        logger.info("Starting Firecrawl scrape for %s", url)

        app = FirecrawlApp(api_key=self.firecrawl_api_key)

        try:
            crawl_result = app.crawl(
                url,
                limit=50,
                scrape_options={"formats": ["markdown"]},
                poll_interval=5,
            )
            raw_pages = (
                crawl_result.data
                if hasattr(crawl_result, "data")
                else (crawl_result if isinstance(crawl_result, list) else [])
            )
        except Exception as e:
            logger.info("Firecrawl crawl failed (%s), trying single-page scrape", e)
            try:
                scrape_result = app.scrape(url, formats=["markdown"])
                raw_pages = [scrape_result]
            except Exception as e2:
                logger.warning("Firecrawl single-page scrape also failed: %s", e2)
                return None

        pages = []
        for idx, item in enumerate(raw_pages, start=1):
            markdown = (
                item.markdown if hasattr(item, "markdown")
                else (item.get("markdown") if isinstance(item, dict) else "")
            )
            metadata = (
                item.metadata if hasattr(item, "metadata")
                else (item.get("metadata", {}) if isinstance(item, dict) else {})
            )

            if isinstance(metadata, dict):
                page_title = metadata.get("title", "")
                page_url = metadata.get("sourceURL", metadata.get("url", ""))
            else:
                page_title = getattr(metadata, "title", "")
                page_url = getattr(metadata, "sourceURL", getattr(metadata, "url", ""))

            page = _format_page(idx, page_url, page_title, markdown or "")
            if page:
                pages.append(page)

        result = ScrapedSite(
            source_url=url,
            scraped_at=datetime.now().isoformat(),
            total_pages=len(pages),
            pages=pages,
        )

        logger.info("Firecrawl scraped %d pages from %s", result.total_pages, url)
        return result


# ── CLI entry point (subprocess target) ─────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    target_url = sys.argv[1] if len(sys.argv) > 1 else ""
    out_file = sys.argv[2] if len(sys.argv) > 2 else "scraped_data.json"

    if not target_url:
        print("Usage: python scraper_service.py <url> <output_file>", file=sys.stderr)
        sys.exit(1)

    _run_scrapy_in_process(target_url, out_file)
