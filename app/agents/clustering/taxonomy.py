"""
Static canonical taxonomy for respondent tag harmonization.

Each TAG_FIELD maps a canonical key → list of synonyms/variants. The
harmonizer (`tag_harmonizer.py`) does:
  1. Lowercased exact match across all synonyms → return canonical
  2. RapidFuzz fuzzy match (score_cutoff=85) → return canonical
  3. LLM fallback (cached) for genuinely unseen values

Adding a new survey-specific tag value just means appending to the
relevant synonym list. The taxonomy can grow without LLM calls.
"""
from __future__ import annotations

from typing import Dict, List


INDUSTRY: Dict[str, List[str]] = {
    "saas": ["saas", "software as a service", "b2b saas", "saas platform", "software"],
    "fintech": ["fintech", "financial technology", "payments", "banking", "finance"],
    "ecommerce": ["ecommerce", "e-commerce", "online retail", "d2c", "dtc", "shopify"],
    "healthtech": ["healthtech", "health tech", "healthcare", "digital health", "medtech"],
    "edtech": ["edtech", "education technology", "education", "elearning", "e-learning"],
    "marketplace": ["marketplace", "two-sided marketplace", "platform"],
    "media": ["media", "entertainment", "publishing", "streaming", "content"],
    "consumer_goods": ["cpg", "consumer goods", "consumer products", "fmcg"],
    "agency": ["agency", "consultancy", "consulting", "professional services"],
    "manufacturing": ["manufacturing", "industrial", "hardware"],
    "real_estate": ["real estate", "proptech", "property"],
    "travel": ["travel", "hospitality", "tourism", "travel tech"],
    "logistics": ["logistics", "supply chain", "shipping", "delivery"],
    "automotive": ["automotive", "auto", "mobility", "transportation"],
    "energy": ["energy", "cleantech", "climate tech", "utilities"],
    "gaming": ["gaming", "games", "videogames", "esports"],
    "developer_tools": ["devtools", "developer tools", "dev tools", "infrastructure"],
    "cybersecurity": ["cybersecurity", "security", "infosec"],
}

SENIORITY: Dict[str, List[str]] = {
    "junior": ["junior", "entry-level", "entry level", "associate", "intern", "trainee", "i1", "i2"],
    "mid": ["mid", "mid-level", "mid level", "intermediate", "ic", "individual contributor"],
    "senior": ["senior", "sr", "sr.", "lead", "principal", "staff"],
    "manager": ["manager", "mgr", "people manager", "team lead"],
    "director": ["director", "head of", "head", "dir"],
    "executive": ["executive", "vp", "svp", "evp", "c-suite", "cxo", "ceo", "cto", "cmo", "cpo", "coo", "cfo", "founder", "co-founder"],
}

COMPANY_SIZE: Dict[str, List[str]] = {
    "1-10": ["1-10", "1 to 10", "<10", "under 10", "tiny", "micro", "solo", "early-stage startup"],
    "11-50": ["11-50", "11 to 50", "10-50", "small", "smb", "small business", "startup"],
    "51-200": ["51-200", "51 to 200", "100", "growth-stage", "scaleup", "scale-up"],
    "201-1000": ["201-1000", "200-1000", "midmarket", "mid-market", "mid market", "medium"],
    "1000+": ["1000+", ">1000", "over 1000", "enterprise", "large", "fortune 500", "f500"],
}

REGION: Dict[str, List[str]] = {
    "north_america": ["north america", "na", "us", "usa", "united states", "canada", "us & canada"],
    "europe": ["europe", "eu", "emea", "uk", "united kingdom", "germany", "france"],
    "asia_pacific": ["apac", "asia pacific", "asia", "australia", "japan", "india", "china"],
    "latin_america": ["latam", "latin america", "south america", "brazil", "mexico"],
    "middle_east": ["middle east", "mena", "uae", "saudi arabia", "israel"],
    "africa": ["africa", "subsaharan africa", "south africa", "nigeria"],
    "global": ["global", "worldwide", "international", "multi-region"],
}

ROLE: Dict[str, List[str]] = {
    "engineer": ["engineer", "developer", "swe", "software engineer", "programmer"],
    "product_lead": ["product manager", "pm", "product lead", "product owner", "product"],
    "designer": ["designer", "ux", "ui", "ux/ui", "design", "product designer"],
    "marketing": ["marketing", "growth", "demand gen", "demand generation", "performance marketing"],
    "sales": ["sales", "ae", "account executive", "bdr", "sdr", "revenue"],
    "founder": ["founder", "co-founder", "ceo", "founder/ceo"],
    "operations": ["operations", "ops", "biz ops", "business operations", "rev ops"],
    "data": ["data", "analytics", "data scientist", "data analyst", "data engineer"],
    "research": ["research", "researcher", "user research", "market research", "ux researcher"],
    "finance": ["finance", "cfo", "controller", "accounting"],
    "hr": ["hr", "people", "people ops", "talent", "recruiting"],
    "support": ["support", "customer success", "cs", "customer support", "csm"],
    "executive": ["c-suite", "executive", "leadership", "vp"],
}


# Field → canonical-vocabulary lookup so harmonize_tag(field, value) can
# pick the right vocab.
TAG_FIELDS: Dict[str, Dict[str, List[str]]] = {
    "industry": INDUSTRY,
    "seniority": SENIORITY,
    "company_size": COMPANY_SIZE,
    "region": REGION,
    "role": ROLE,
}


def all_synonyms_for(field: str) -> List[str]:
    """Flat list of every synonym across canonicals for a field."""
    vocab = TAG_FIELDS.get(field) or {}
    out: List[str] = []
    for synonyms in vocab.values():
        out.extend(synonyms)
    return out


def synonym_to_canonical(field: str) -> Dict[str, str]:
    """Reverse-lookup map: synonym (lowercased) → canonical key."""
    vocab = TAG_FIELDS.get(field) or {}
    out: Dict[str, str] = {}
    for canonical, synonyms in vocab.items():
        for syn in synonyms:
            out[syn.lower().strip()] = canonical
    return out
