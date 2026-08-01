#!/usr/bin/env bash
# Build a portable train package that includes graph + files + physics_shards.
# Does NOT launch training.
#
# Layout:
#   artifacts/portable-train-package/
#     graph.json                      (symlink or copy of live TAO graph)
#     files/                          (bundle CAD/docs if present)
#     artifacts/physics_shards/       (symlink tree — path-stable for graph refs)
#     data/processed/nasa3d/          (symlink for nasa TensorShards)
#     MANIFEST.json
#
# Usage:
#   ./scripts/prepare_portable_train_package.sh
#   ./scripts/prepare_portable_train_package.sh /path/to/out

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${1:-artifacts/portable-train-package}"
BUNDLE="artifacts/jepa-train-bundle"
SHARDS="artifacts/physics_shards"
NASA="data/processed/nasa3d"

rm -rf "$OUT"
mkdir -p "$OUT/artifacts" "$OUT/data/processed"

if [[ -f "$BUNDLE/graph.json" ]]; then
  cp -L "$BUNDLE/graph.json" "$OUT/graph.json"
else
  echo "error: missing $BUNDLE/graph.json" >&2
  exit 1
fi

if [[ -d "$BUNDLE/files" ]]; then
  ln -sfn "$(cd "$BUNDLE/files" && pwd)" "$OUT/files"
fi

if [[ -d "$SHARDS" ]]; then
  ln -sfn "$(cd "$SHARDS" && pwd)" "$OUT/artifacts/physics_shards"
else
  echo "warn: missing $SHARDS" >&2
fi

if [[ -d "$NASA" ]]; then
  ln -sfn "$(cd "$NASA" && pwd)" "$OUT/data/processed/nasa3d"
fi

# Also expose repo-root-relative convenience links for resolvers that walk parents.
ln -sfn "$(pwd)/artifacts" "$OUT/_repo_artifacts" 2>/dev/null || true

python3 - <<PY
import json, os, time
from pathlib import Path
out = Path("$OUT")
shards = out / "artifacts" / "physics_shards"
nasa = out / "data" / "processed" / "nasa3d"
fea = len(list((shards / "fea").glob("*.npz"))) if (shards / "fea").exists() else 0
cfd = len(list((shards / "cfd").glob("*.npz"))) if (shards / "cfd").exists() else 0
nasa_n = len(list(nasa.glob("*.npz"))) if nasa.exists() else 0
manifest = {
  "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "out": str(out),
  "graph": str(out / "graph.json"),
  "physics_shards_fea_npz": fea,
  "physics_shards_cfd_npz": cfd,
  "nasa3d_npz": nasa_n,
  "has_files": (out / "files").exists(),
  "note": "Pass --graph-path \$OUT/graph.json and data.extra_search_roots=\$OUT for Modal/local.",
}
(out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

echo "portable package ready: $OUT"
