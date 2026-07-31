"""Source validation and crawl/report helpers for public spaceflight datasets.

The validator is intentionally conservative:
- it classifies sources by access/rights and source type,
- it can optionally perform live HTTP reachability checks,
- and it emits a stable JSON-friendly report for downstream tooling.

The goal is not to prove a source is perfect; it is to decide whether it is
usable, reference-only, blocked, or needs manual review before ingestion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .datasets import DATASET_REGISTRY, DatasetSource, infer_information_mode


USABLE = "usable"
REFERENCE_ONLY = "reference-only"
BLOCKED = "blocked"
MANUAL_REVIEW = "manual-review"


_PUBLIC_HOST_HINTS = (
    "nasa.gov",
    "ntrs.nasa.gov",
    "data.nasa.gov",
    "science.nasa.gov",
    "esa.int",
    "cosmos.esa.int",
    "sci.esa.int",
    "jaxa.jp",
    "isro.gov.in",
    "isro.nic.in",
    "dlr.de",
    "cnes.fr",
    "cnsa.gov.cn",
    "gov.cn",
    "edu.cn",
    "spacejournal.cn",
    "journal.hep.com.cn",
    "jdse.bit.edu.cn",
    "hpe.com.cn",
    "lpre.de",
    "afit.edu",
    "dtic.mil",
    "apps.dtic.mil",
    "scholar.afit.edu",
    "space.jpl.nasa.gov",
    "github.com",
    "raw.githubusercontent.com",
    "huggingface.co",
    "github.io",
)

_REFERENCE_ONLY_HINTS = (
    "marketplace",
    "user uploads",
    "third-party",
    "vendor-specific",
    "reference-only",
    "license review",
    "author terms",
    "grabcad",
    "printables",
    "sketchfab",
    "makerworld",
)

_MANUAL_REVIEW_HINTS = (
    "unknown",
    "unclear",
    "review",
    "site terms",
    "archive terms",
    "other",
)

_SUPPORTED_SCHEMES = {"http", "https"}


@dataclass(frozen=True, slots=True)
class SourceValidationResult:
    source: DatasetSource
    information_mode: str
    source_kind: str
    status: str
    reachable: bool | None
    http_status: int | None
    content_type: str | None
    reasons: tuple[str, ...]
    checked_at: str
    live_check: bool
    training_eligible: bool
    reference_only: bool
    manual_review: bool
    blocked: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source"] = self.source.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class SourceValidationReport:
    checked_at: str
    live_check: bool
    results: tuple[SourceValidationResult, ...]

    @property
    def counts(self) -> dict[str, int]:
        counts = {USABLE: 0, REFERENCE_ONLY: 0, BLOCKED: 0, MANUAL_REVIEW: 0}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return counts

    @property
    def information_mode_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.information_mode] = counts.get(result.information_mode, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_at": self.checked_at,
            "live_check": self.live_check,
            "counts": self.counts,
            "information_mode_counts": self.information_mode_counts,
            "results": [result.to_dict() for result in self.results],
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_text(*parts: str) -> str:
    return " ".join(part for part in parts if part).lower()


def _classify_source_kind(source: DatasetSource) -> str:
    parsed = urlparse(source.url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    text = _normalise_text(source.key, source.title, source.notes, source.license, source.url)

    if "patent" in host or "patent" in path or "patents" in host:
        return "patent"
    if host in {"github.com", "raw.githubusercontent.com", "github.io"}:
        return "git-repo"
    if host.endswith("huggingface.co") or "/datasets/" in path:
        return "dataset-hub"
    if path.endswith(".pdf") or "pdf" in path:
        return "pdf"
    if any(token in text for token in ("3d model", "3d models", "obj", "stl", "step", "iges", "fbx", "print model")):
        return "cad-asset"
    if any(token in host for token in ("nasa.gov", "esa.int", "jaxa.jp", "isro.gov.in", "dlr.de", "cnes.fr", "gov.cn", "edu.cn", "dtic.mil")):
        return "institutional-web"
    return "html"


def _requires_manual_review(source: DatasetSource, source_kind: str) -> bool:
    text = _normalise_text(source.license, source.notes, source.url)
    if any(hint in text for hint in _MANUAL_REVIEW_HINTS):
        return True
    if source_kind == "cad-asset" and any(hint in text for hint in ("site terms", "terms", "archive terms")):
        return True
    return False


def _is_reference_only(source: DatasetSource) -> bool:
    text = _normalise_text(source.license, source.notes, source.url)
    return any(hint in text for hint in _REFERENCE_ONLY_HINTS)


def _is_public_or_open(source: DatasetSource) -> bool:
    text = _normalise_text(source.license, source.notes, source.url)
    return any(hint in text for hint in _PUBLIC_HOST_HINTS) or "public release" in text or "open data" in text or "open source" in text


def _live_fetch(url: str, timeout: float) -> tuple[bool, int | None, str | None, str | None, str | None]:
    request = Request(url, headers={"User-Agent": "jepa-cad-source-validator/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            headers = response.headers
            content_type = headers.get_content_type() if headers else None
            content_length = headers.get("Content-Length") if headers else None
            return True, getattr(response, "status", 200), content_type, content_length, None
    except HTTPError as exc:
        headers = exc.headers
        content_type = headers.get_content_type() if headers else None
        content_length = headers.get("Content-Length") if headers else None
        return False, int(exc.code), content_type, content_length, str(exc)
    except URLError as exc:
        return False, None, None, None, str(exc)
    except Exception as exc:  # noqa: BLE001 - report and continue
        return False, None, None, None, str(exc)


def validate_source(source: DatasetSource, *, live_check: bool = False, timeout: float = 10.0) -> SourceValidationResult:
    parsed = urlparse(source.url)
    checked_at = _utc_now()
    reasons: list[str] = []
    reachable: bool | None = None
    http_status: int | None = None
    content_type: str | None = None
    blocked = False

    if not parsed.scheme or parsed.scheme.lower() not in _SUPPORTED_SCHEMES:
        blocked = True
        reasons.append(f"unsupported URL scheme: {parsed.scheme or '<empty>'}")
    if not parsed.netloc:
        blocked = True
        reasons.append("missing URL host")

    source_kind = _classify_source_kind(source)
    reference_only = _is_reference_only(source)
    manual_review = _requires_manual_review(source, source_kind)
    public_or_open = _is_public_or_open(source)

    if live_check and not blocked:
        reachable, http_status, content_type, content_length, error = _live_fetch(source.url, timeout)
        if content_length:
            reasons.append(f"content-length={content_length}")
        if error:
            reasons.append(error)
        if not reachable:
            # A live failure is a hard block only if the source does not already
            # have a good reason to stay in the catalog.
            if public_or_open or source_kind in {"institutional-web", "pdf", "patent", "dataset-hub", "git-repo"}:
                reasons.append("live check failed")
                if source_kind in {"institutional-web", "pdf", "patent", "dataset-hub", "git-repo"}:
                    manual_review = True if not reference_only else manual_review
            else:
                blocked = True

    if blocked:
        status = BLOCKED
    elif reference_only:
        status = REFERENCE_ONLY
    elif manual_review:
        status = MANUAL_REVIEW
    elif public_or_open or source_kind in {"institutional-web", "pdf", "patent", "dataset-hub", "git-repo", "html"}:
        status = USABLE
    else:
        status = MANUAL_REVIEW

    if not reasons and source_kind == "institutional-web":
        reasons.append("institutional/public source")
    if reference_only:
        reasons.append("reference-only by rights/terms")
    if manual_review and not reference_only:
        reasons.append("manual review recommended")
    if status == USABLE and live_check and reachable is True:
        reasons.append("live check passed")

    training_eligible = status == USABLE and not reference_only and not blocked
    return SourceValidationResult(
        source=source,
        information_mode=source.information_mode or infer_information_mode(source),
        source_kind=source_kind,
        status=status,
        reachable=reachable,
        http_status=http_status,
        content_type=content_type,
        reasons=tuple(dict.fromkeys(reasons)),
        checked_at=checked_at,
        live_check=live_check,
        training_eligible=training_eligible,
        reference_only=reference_only,
        manual_review=manual_review,
        blocked=blocked,
    )


def validate_source_registry(
    *,
    live_check: bool = False,
    timeout: float = 10.0,
    limit: int | None = None,
    domains: Iterable[str] | None = None,
    keys: Iterable[str] | None = None,
) -> SourceValidationReport:
    selected = list(DATASET_REGISTRY.values())
    if domains is not None:
        wanted_domains = {domain.lower() for domain in domains}
        selected = [source for source in selected if source.domain.lower() in wanted_domains]
    if keys is not None:
        wanted_keys = {key.lower() for key in keys}
        selected = [source for source in selected if source.key.lower() in wanted_keys]
    if limit is not None:
        selected = selected[: max(0, limit)]

    results = tuple(validate_source(source, live_check=live_check, timeout=timeout) for source in selected)
    return SourceValidationReport(checked_at=_utc_now(), live_check=live_check, results=results)


def render_validation_report(report: SourceValidationReport, *, as_json: bool = False) -> str:
    if as_json:
        import json

        return json.dumps(report.to_dict(), indent=2)

    lines = [
        f"checked_at={report.checked_at}",
        f"live_check={report.live_check}",
        f"usable={report.counts.get(USABLE, 0)}",
        f"reference_only={report.counts.get(REFERENCE_ONLY, 0)}",
        f"manual_review={report.counts.get(MANUAL_REVIEW, 0)}",
        f"blocked={report.counts.get(BLOCKED, 0)}",
        "",
    ]
    for result in report.results:
        lines.append(f"[{result.status}] {result.source.key} :: {result.source.title}")
        lines.append(f"  kind={result.source_kind} url={result.source.url}")
        if result.reasons:
            lines.append(f"  reasons={'; '.join(result.reasons)}")
        if result.live_check:
            lines.append(f"  reachable={result.reachable} http_status={result.http_status} content_type={result.content_type}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
