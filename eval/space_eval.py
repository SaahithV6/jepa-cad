"""Space-family checkpoint comparison utilities.

This is the explicit candidate-vs-baseline eval entrypoint for the space
family. It reuses the frozen-encoder probe machinery and returns a compact
comparison record that the CLI, flywheel, or external compute wrappers can use
for promotion decisions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import copy
import torch

from eval.probe import load_config, probe_checkpoint
from utils.config import apply_overrides


@dataclass(frozen=True, slots=True)
class SpaceEvalResult:
    candidate: str
    baseline: str | None
    data_source: str
    score_name: str
    candidate_score: float
    baseline_score: float | None
    improvement: float | None
    improvement_threshold: float
    improved: bool
    candidate_probe: dict[str, Any]
    baseline_probe: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "baseline": self.baseline,
            "data_source": self.data_source,
            "score_name": self.score_name,
            "candidate_score": self.candidate_score,
            "baseline_score": self.baseline_score,
            "improvement": self.improvement,
            "improvement_threshold": self.improvement_threshold,
            "improved": self.improved,
            "candidate_probe": self.candidate_probe,
            "baseline_probe": self.baseline_probe,
        }


def _load_cfg(path: str | Path, family: str | None = None) -> dict[str, Any]:
    cfg = load_config(path)
    if family is not None:
        from utils.config import load_yaml_with_family

        cfg = load_yaml_with_family(path, family=family)
    return cfg


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compare_checkpoints(
    cfg: dict[str, Any],
    candidate: str | Path,
    baseline: str | Path | None,
    data_source: str,
    *,
    threshold: float = 0.0,
    seed: int | None = None,
    device: torch.device | None = None,
    data_dir: str | Path | None = None,
) -> SpaceEvalResult:
    device = device or _device()
    probe_cfg = copy.deepcopy(cfg)
    if data_dir is not None:
        probe_cfg.setdefault("data", {})["data_dir"] = str(data_dir)
    else:
        probe_cfg.setdefault("data", {}).pop("data_dir", None)
    candidate_result = probe_checkpoint(probe_cfg, candidate, data_source, device, seed=seed, verbose=False)
    baseline_result = (
        probe_checkpoint(probe_cfg, baseline, data_source, device, seed=seed, verbose=False) if baseline is not None else None
    )

    candidate_score = float(candidate_result.score)
    baseline_score = float(baseline_result.score) if baseline_result is not None else None
    if baseline_score is None:
        improvement = None
        improved = True
    elif baseline_score <= 0:
        improvement = baseline_score - candidate_score
        improved = candidate_score < baseline_score
    else:
        improvement = (baseline_score - candidate_score) / baseline_score
        improved = candidate_score <= baseline_score * (1.0 - threshold)

    return SpaceEvalResult(
        candidate=str(candidate),
        baseline=str(baseline) if baseline is not None else None,
        data_source=data_source,
        score_name=candidate_result.score_name,
        candidate_score=candidate_score,
        baseline_score=baseline_score,
        improvement=improvement,
        improvement_threshold=threshold,
        improved=improved,
        candidate_probe=candidate_result.to_dict(),
        baseline_probe=baseline_result.to_dict() if baseline_result is not None else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare a candidate space-family checkpoint against a baseline")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--family", default=None, help="Optional config family overlay")
    parser.add_argument("--candidate", required=True, help="Path to candidate checkpoint")
    parser.add_argument("--baseline", default=None, help="Path to baseline checkpoint")
    parser.add_argument("--data-source", choices=["real", "synthetic", "mixed", "graph"], default="real")
    parser.add_argument("--threshold", type=float, default=0.0, help="Required fractional improvement over baseline")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-dir", default=None, help="Override data.data_dir for probing")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args(argv)

    cfg = _load_cfg(args.config, family=args.family)
    result = compare_checkpoints(
        cfg,
        args.candidate,
        args.baseline,
        args.data_source,
        threshold=args.threshold,
        seed=args.seed,
        data_dir=args.data_dir,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(
            f"candidate={result.candidate} baseline={result.baseline} data_source={result.data_source} "
            f"score_name={result.score_name} candidate_score={result.candidate_score:.6f} "
            f"baseline_score={result.baseline_score if result.baseline_score is not None else 'nan'} "
            f"improved={result.improved} improvement={result.improvement if result.improvement is not None else 'nan'}"
        )
    return 0 if result.improved or result.baseline is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
