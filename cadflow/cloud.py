"""Cloud training planning for Modal / Fireworks style runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .datasets import DATASET_REGISTRY, DatasetSource, match_dataset_sources
from .manifest import JobManifest


@dataclass(frozen=True, slots=True)
class CloudTrainingPlan:
    family: str
    primary_provider: str
    secondary_provider: str | None
    project_manifest: str
    dataset_sources: tuple[DatasetSource, ...]
    preprocessing_steps: tuple[str, ...]
    training_steps: tuple[str, ...]
    evaluation_steps: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "primary_provider": self.primary_provider,
            "secondary_provider": self.secondary_provider,
            "project_manifest": self.project_manifest,
            "dataset_sources": [source.to_dict() for source in self.dataset_sources],
            "preprocessing_steps": list(self.preprocessing_steps),
            "training_steps": list(self.training_steps),
            "evaluation_steps": list(self.evaluation_steps),
            "notes": list(self.notes),
        }


_SPACE_HINTS = (
    "space",
    "spacecraft",
    "satellite",
    "rocket",
    "nozzle",
    "thruster",
    "propulsion",
    "orbiter",
    "probe",
    "module",
    "cassini",
    "hubble",
    "iss",
    "adapter",
)

_ASSEMBLY_HINTS = (
    "assembly",
    "joint",
    "mate",
    "fastener",
    "screw",
    "bolt",
    "nut",
    "washer",
    "bearing",
    "bracket",
    "mount",
    "constraint",
)


def _choose_providers(family: str) -> tuple[str, str | None]:
    primary = "Modal"
    secondary = "Fireworks" if family == "space" else "Fireworks"
    return primary, secondary


def _keywords_from_manifest(manifest: JobManifest) -> tuple[str, ...]:
    tokens: list[str] = [manifest.name, manifest.notes or ""]
    tokens.extend(str(tag) for tag in manifest.tags)
    tokens.extend(str(v) for v in manifest.parameters.values())
    for value in manifest.inputs.values():
        tokens.append(str(value))
    return tuple(tokens)


def _is_relevant(source: DatasetSource, keywords: Sequence[str]) -> bool:
    haystack = " ".join(keywords).lower()
    if source.domain == "space" and any(term in haystack for term in _SPACE_HINTS):
        return True
    if source.domain == "mechanical" and any(term in haystack for term in _ASSEMBLY_HINTS):
        return True
    if source.key in haystack:
        return True
    return False


def build_cloud_training_plan(
    manifest: JobManifest,
    *,
    family: str = "space",
    provider_preference: str | None = None,
    max_dataset_sources: int = 6,
) -> CloudTrainingPlan:
    """Translate a manifest into a cloud-friendly training/research plan."""

    keywords = _keywords_from_manifest(manifest)
    selected = match_dataset_sources(tuple(keywords))
    if family == "space":
        selected = [source for source in DATASET_REGISTRY.values() if _is_relevant(source, keywords)][: max_dataset_sources]
    if provider_preference is not None:
        primary = provider_preference
        secondary = "Fireworks" if provider_preference.lower() == "modal" else "Modal"
    else:
        primary, secondary = _choose_providers(family)

    selected = selected[:max_dataset_sources]
    dataset_sources = tuple(selected)

    dataset_lines = [
        f"stage {source.key} -> $DATA_ROOT/datasets/{source.key} ({source.url})"
        for source in dataset_sources
    ]
    preprocessing_steps = tuple(
        [
            "download and stage the selected public datasets",
            f"python -m cadflow.cli intake --project-root {manifest.inputs.get('project_root', '.')} --goal {manifest.notes or manifest.name} --out-dir artifacts/project_intake --family {family}",
            "python -m cadflow.cli ingest --raw-dir <staged datasets> --out-dir data/processed/space --format npz",
        ]
        + dataset_lines
    )
    training_steps = (
        f"python train.py --config configs/base.yaml --family {family} --data-source mixed --set data.data_dir=data/processed/space --set checkpoint.checkpoint_dir=checkpoints/{family}",
        "scale with torchrun or Modal workers when the local laptop is not the bottleneck",
        "use BF16 + activation checkpointing + sharded checkpoints for the larger family preset",
    )
    evaluation_steps = (
        "run eval/probe.py on candidate checkpoints",
        "promote only if the candidate improves on the baseline probe score",
        "record verified outputs in the flywheel registry before using them for the next training cycle",
    )
    notes = (
        "Modal is the better fit for dataset staging and distributed training jobs.",
        "Fireworks is best kept for inference, labeling, and lightweight evaluation if you want hosted compute around the loop.",
        "The first family should stay space-and-assembly focused; electronics remain out of scope for this iteration.",
    )
    return CloudTrainingPlan(
        family=family,
        primary_provider=primary,
        secondary_provider=secondary,
        project_manifest=manifest.fingerprint,
        dataset_sources=dataset_sources,
        preprocessing_steps=preprocessing_steps,
        training_steps=training_steps,
        evaluation_steps=evaluation_steps,
        notes=notes,
    )
