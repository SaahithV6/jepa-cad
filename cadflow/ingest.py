"""Package-level ingestion helpers for CAD/CAE data.

This is a thin wrapper around `data.ingest` so callers can use the
`cadflow` namespace consistently.
"""

from __future__ import annotations

from data.ingest import IngestionResult, ingest_raw_sources, ingest_sources, ingest_verified_flywheel

__all__ = [
    "IngestionResult",
    "ingest_raw_sources",
    "ingest_sources",
    "ingest_verified_flywheel",
]
