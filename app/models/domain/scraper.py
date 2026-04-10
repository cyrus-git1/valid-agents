"""Domain models for web scraping results."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel


class ScrapedPage(BaseModel):
    """A single scraped page."""
    page: int
    url: str = ""
    title: str = ""
    text: str = ""


class ScrapedSite(BaseModel):
    """Result of scraping a website."""
    source_url: str
    scraped_at: str = ""
    total_pages: int = 0
    pages: List[ScrapedPage] = []

    def to_dict(self) -> dict:
        return self.model_dump()
