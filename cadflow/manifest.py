"""Job manifests and provenance records for CAD/CAE runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping, Sequence


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
    artifacts: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> str:
        payload = {
            "name": self.name,
            "inputs": self.inputs,
            "parameters": self.parameters,
            "tags": list(self.tags),
            "notes": self.notes,
            "artifacts": list(self.artifacts),
        }
        return hashlib.sha256(_canonical_payload(payload).encode("utf-8")).hexdigest()[:16]

    def with_artifacts(self, artifacts: Sequence[str]) -> "JobManifest":
        return JobManifest(
            name=self.name,
            inputs=dict(self.inputs),
            parameters=dict(self.parameters),
            tags=self.tags,
            notes=self.notes,
            artifacts=tuple(str(a) for a in artifacts),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": self.inputs,
            "parameters": self.parameters,
            "tags": list(self.tags),
            "notes": self.notes,
            "artifacts": list(self.artifacts),
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
            artifacts=tuple(payload.get("artifacts", ())),
        )


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Metadata describing where a manifest/run came from."""

    source: str
    manifest_fingerprint: str = ""
    created_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)
    parent_fingerprints: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @classmethod
    def for_manifest(
        cls,
        manifest: JobManifest,
        source: str,
        details: Mapping[str, Any] | None = None,
        parent_fingerprints: Sequence[str] | None = None,
        artifact_refs: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
    ) -> "ProvenanceRecord":
        return cls(
            source=source,
            manifest_fingerprint=manifest.fingerprint,
            details=dict(details or {}),
            parent_fingerprints=tuple(parent_fingerprints or ()),
            artifact_refs=tuple(str(a) for a in (artifact_refs or manifest.artifacts)),
            tags=tuple(tags or manifest.tags),
        )

    @property
    def fingerprint(self) -> str:
        payload = {
            "source": self.source,
            "manifest_fingerprint": self.manifest_fingerprint,
            "details": self.details,
            "parent_fingerprints": list(self.parent_fingerprints),
            "artifact_refs": list(self.artifact_refs),
            "tags": list(self.tags),
        }
        return hashlib.sha256(_canonical_payload(payload).encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProvenanceRecord":
        return cls(
            source=str(payload["source"]),
            manifest_fingerprint=str(payload.get("manifest_fingerprint", "")),
            created_at=str(payload.get("created_at", _utc_now())),
            details=dict(payload.get("details", {})),
            parent_fingerprints=tuple(payload.get("parent_fingerprints", ())),
            artifact_refs=tuple(payload.get("artifact_refs", ())),
            tags=tuple(payload.get("tags", ())),
        )


@dataclass(frozen=True, slots=True)
class RunRecord:
    """A manifest plus provenance and execution status."""

    manifest: JobManifest
    provenance: ProvenanceRecord
    status: str = "queued"
    solver_result: Mapping[str, Any] | None = None
    verification: Mapping[str, Any] | None = None
    artifact_refs: tuple[str, ...] = ()

    @property
    def manifest_fingerprint(self) -> str:
        return self.manifest.fingerprint

    def with_status(self, status: str) -> "RunRecord":
        return RunRecord(
            manifest=self.manifest,
            provenance=self.provenance,
            status=status,
            solver_result=self.solver_result,
            verification=self.verification,
            artifact_refs=self.artifact_refs,
        )

    def with_results(
        self,
        *,
        status: str | None = None,
        solver_result: Mapping[str, Any] | None = None,
        verification: Mapping[str, Any] | None = None,
        artifact_refs: Sequence[str] | None = None,
    ) -> "RunRecord":
        return RunRecord(
            manifest=self.manifest,
            provenance=self.provenance,
            status=status if status is not None else self.status,
            solver_result=solver_result if solver_result is not None else self.solver_result,
            verification=verification if verification is not None else self.verification,
            artifact_refs=tuple(artifact_refs) if artifact_refs is not None else self.artifact_refs,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "provenance": self.provenance.to_dict(),
            "status": self.status,
            "solver_result": dict(self.solver_result) if self.solver_result is not None else None,
            "verification": dict(self.verification) if self.verification is not None else None,
            "artifact_refs": list(self.artifact_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunRecord":
        return cls(
            manifest=JobManifest.from_dict(payload["manifest"]),
            provenance=ProvenanceRecord.from_dict(payload["provenance"]),
            status=str(payload["status"]),
            solver_result=payload.get("solver_result"),
            verification=payload.get("verification"),
            artifact_refs=tuple(payload.get("artifact_refs", ())),
        )
