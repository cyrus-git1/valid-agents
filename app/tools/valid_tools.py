"""
Tools for the Valid sales chat agent.

Three tools:
  - search_valid: search Valid's internal KB
  - book_demo: collect lead info and store a demo booking
  - get_pricing: search KB specifically for pricing/plans/sales models
"""
from __future__ import annotations

import logging
import threading
import time as _time
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import tool

from app.core_client import CORE_API_URL, _outbound_headers, _TIMEOUT

logger = logging.getLogger(__name__)

# ── Demo bookings store (in-memory, replace with core API later) ──────────

_demo_bookings: List[Dict[str, Any]] = []
_demo_lock = threading.Lock()


def get_demo_bookings() -> List[Dict[str, Any]]:
    """Read-only access to stored demo bookings (for admin/API use)."""
    with _demo_lock:
        return list(_demo_bookings)


def create_valid_tools() -> list:
    """Build tools for the Valid sales chat agent."""

    @tool
    def search_valid(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search Valid's internal knowledge base.

        Use this to answer any question about Valid — the product,
        features, capabilities, team, customers, use cases, etc.
        Returns content chunks with similarity scores.
        Always cite the content you find in your response.
        """
        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.post(
                    f"{CORE_API_URL}/search/valid",
                    json={"query": query, "top_k": top_k, "node_types": ["Chunk"]},
                    headers=_outbound_headers(),
                )
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])
            return [
                {
                    "content": r.get("content", ""),
                    "similarity_score": r.get("similarity_score", 0.0),
                    "document_id": r.get("document_id"),
                    "chunk_index": r.get("chunk_index"),
                }
                for r in results
            ]
        except Exception as e:
            logger.warning("search_valid failed: %s", e)
            return [{"error": str(e)}]

    @tool
    def get_pricing(query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search Valid's KB specifically for pricing, plans, and sales models.

        Use this when the user asks about pricing, plans, costs, tiers,
        enterprise pricing, or sales models. Searches with pricing-focused
        queries for better results.
        """
        search_queries = [
            query or "pricing plans tiers cost",
            "sales model enterprise pricing subscription",
        ]
        all_results: List[Dict[str, Any]] = []
        seen_ids: set = set()

        for q in search_queries:
            try:
                with httpx.Client(timeout=_TIMEOUT) as client:
                    resp = client.post(
                        f"{CORE_API_URL}/search/valid",
                        json={"query": q, "top_k": 5, "node_types": ["Chunk"]},
                        headers=_outbound_headers(),
                    )
                    resp.raise_for_status()
                    data = resp.json()

                for r in data.get("results", []):
                    nid = r.get("node_id")
                    if nid and nid not in seen_ids:
                        seen_ids.add(nid)
                        all_results.append({
                            "content": r.get("content", ""),
                            "similarity_score": r.get("similarity_score", 0.0),
                            "document_id": r.get("document_id"),
                        })
            except Exception as e:
                logger.warning("get_pricing search failed: %s", e)

        if not all_results:
            return [{"message": "No pricing information found in the knowledge base."}]
        return all_results[:8]

    @tool
    def book_demo(
        name: str,
        email: str,
        company: Optional[str] = None,
        role: Optional[str] = None,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Book a demo with the Valid team.

        Collect the user's contact information to schedule a demo.
        Required: name and email. Optional: company, role, message.
        Call this when the user expresses interest in a demo, wants to
        talk to sales, or wants to get started with Valid.
        """
        if not name or not name.strip():
            return {"status": "missing_info", "message": "I need your name to book the demo. What's your name?"}
        if not email or not email.strip() or "@" not in email:
            return {"status": "missing_info", "message": "I need your email address to book the demo. What's your email?"}

        booking = {
            "name": name.strip(),
            "email": email.strip().lower(),
            "company": (company or "").strip() or None,
            "role": (role or "").strip() or None,
            "message": (message or "").strip() or None,
            "booked_at": _time.time(),
            "status": "pending",
        }

        with _demo_lock:
            _demo_bookings.append(booking)

        logger.info("Demo booked: %s (%s) from %s", booking["name"], booking["email"], booking.get("company"))

        return {
            "status": "booked",
            "message": (
                f"Great! I've submitted your demo request. Our team will reach out to "
                f"{booking['email']} shortly to schedule a time. "
                f"Is there anything else you'd like to know about Valid?"
            ),
        }

    return [search_valid, get_pricing, book_demo]
