"""Job manifests and provenance records for CAD/CAE runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_payload(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


@dataclass(frozen=True, slots=True)
class JobManifest:
    """A small, stable description of a CAD/CAE job."""

    name: str
    inputs: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    notes: str | None = None

    @property
    def fingerprint(self) -> str:
        payload = {
            "name": self.name,
            "inputs": self.inputs,
            "parameters": self.parameters,
            "tags": list(self.tags),
            "notes": self.notes,
        }
        return hashlib.sha256(_canonical_payload(payload).encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": self.inputs,
            "parameters": self.parameters,
            "tags": list(self.tags),
            "notes": self.notes,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JobManifest":
        return cls(
            name=str(payload["name"]),
            inputs=dict(payload.get("inputs", {})),
            parameters=dict(payload.get("parameters", {})),
            tags=tuple(payload.get("tags", ())),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Metadata describing where a manifest/run came from."""

    source: str
    manifest_fingerprint: str = ""
    created_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_manifest(
        cls,
        manifest: JobManifest,
        source: str,
        details: Mapping[str, Any] | None = None,
    ) -> "ProvenanceRecord":
        return cls(
            source=source,
            manifest_fingerprint=manifest.fingerprint,
            details=dict(details or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProvenanceRecord":
        return cls(
            source=str(payload["source"]),
            manifest_fingerprint=str(payload["manifest_fingerprint"]),
            created_at=str(payload.get("created_at", _utc_now())),
            details=dict(payload.get("details", {})),
        )


@dataclass(frozen=True, slots=True)
class RunRecord:
    """A manifest plus provenance and execution status."""

    manifest: JobManifest
    provenance: ProvenanceRecord
    status: str = "queued"
    solver_result: Mapping[str, Any] | None = None
    verification: Mapping[str, Any] | None = None

    @property
    def manifest_fingerprint(self) -> str:
        return self.manifest.fingerprint

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "manifest": self.manifest.to_dict(),
            "provenance": self.provenance.to_dict(),
            "status": self.status,
            "solver_result": dict(self.solver_result) if self.solver_result is not None else None,
            "verification": dict(self.verification) if self.verification is not None else None,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunRecord":
        return cls(
            manifest=JobManifest.from_dict(payload["manifest"]),
            provenance=ProvenanceRecord.from_dict(payload["provenance"]),
            status=str(payload["status"]),
            solver_result=payload.get("solver_result"),
            verification=payload.get("verification"),
        )
