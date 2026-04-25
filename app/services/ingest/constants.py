"""Static ingest policy constants shared across ingest code."""
from __future__ import annotations

# MIME types accepted for uploaded ingest files.
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "text/vtt",
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "application/octet-stream",
}

# File extensions supported by the ingest pipeline.
ALLOWED_EXTENSIONS = {
    "pdf", "docx", "vtt", "xlsx", "xls",
    # Plain-text formats — chunked as raw text
    "txt", "md", "markdown", "csv", "json", "log",
}

# File types the ingest service can parse locally.
SUPPORTED_FILE_TYPES = ALLOWED_EXTENSIONS

# Plain-text extensions handled with a generic text chunker
TEXT_LIKE_EXTENSIONS = {"txt", "md", "markdown", "csv", "json", "log"}

# Inline payload types accepted by the ingest endpoint.
ALLOWED_INLINE_INGEST_TYPES = {"webvtt", "survey_results"}

# Canonical source type stored for each supported file extension.
EXT_TO_SOURCE_TYPE = {
    "pdf": "pdf",
    "docx": "docx",
    "vtt": "vtt",
    "xlsx": "xlsx",
    "xls": "xlsx",
    "webvtt": "webvtt",
    "survey_results": "survey_results",
    "txt": "text",
    "md": "markdown",
    "markdown": "markdown",
    "csv": "csv",
    "json": "json",
    "log": "text",
}

INLINE_SOURCE_TYPE = {
    "webvtt": "webvtt",
    "survey_results": "survey_results",
}
