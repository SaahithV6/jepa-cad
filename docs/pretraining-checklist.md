# JEPA / CAD / Spaceflight pre-training checklist

This repo now exposes the checklist as an executable flow.

## Primary command

```bash
python -m cadflow.cli preflight \
  --project-root /path/to/existing/project \
  --goal "reduce stress in an existing spacecraft bracket" \
  --family space \
  --material "Al 6061-T6" \
  --data-root data \
  --raw-dir /path/to/raw \
  --out-dir artifacts/preflight \
  --run-smoke \
  --json
```

## What it checks

- **Training goal / scope**: project intake creates a manifest from the goal and family.
- **Corpus plan**: cloud-plan selects space-relevant sources and preprocessing steps.
- **Graph wiring**: source registry graph schema and source graph export are materialized.
- **Data cleaning**: local data tree is scanned into corpus / file / sample / analogue summaries.
- **Training representation**: graph-backed dataset loading is validated from the corpus graph.
- **Baseline loader**: ingest -> corpus graph -> graph-backed dataset is exercised.
- **Evaluation before training**: probe / space-eval / flywheel promotion paths are available.
- **Tiny smoke pass**: optional `--run-smoke` runs ingest -> train against the selected goal.
- **Infrastructure readiness**: `doctor` validates solver runtime readiness.
- **Tracking / provenance**: manifests and provenance are produced deterministically.

## Supporting commands

- `python -m cadflow.cli doctor --json`
- `python -m cadflow.cli validate-sources --json`
- `python -m cadflow.cli graph-schema --json`
- `python -m cadflow.cli graph-export --json`
- `python -m cadflow.cli cloud-plan --manifest <manifest.json> --json`
- `python -m cadflow.cli space-eval --candidate <ckpt.pt> --baseline <baseline.pt> --family space --json`
- `python -m cadflow.cli e2e --raw-dir <raw> --out-dir <dataset> --max-steps 1`
- `python -m cadflow.cli loop --raw-dir <raw> --out-dir <loop> --flywheel <fw.jsonl>`

## Notes

- Space-family smoke runs default to the space-config field count used by the model.
- The preflight report is JSON-serializable, so it can be wired into cron jobs or CI gates.
- The graph-backed dataset path is intentional: the corpus graph is the bridge between provenance and training tensors.
