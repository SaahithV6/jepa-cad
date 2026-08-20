#!/usr/bin/env bash
# Build a VM-survivable offline export of TAO + train assets.
# Prefer hardlinks (near-zero extra disk); fall back to copy.
# Does NOT launch training / Modal.
#
# Usage:
#   ./scripts/export_offline_vm_snapshot.sh
#   ./scripts/export_offline_vm_snapshot.sh artifacts/offline-export
#   MAKE_TAR=1 ./scripts/export_offline_vm_snapshot.sh   # also write .tar.zst if zstd available

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${1:-artifacts/offline-export}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BUNDLE="artifacts/jepa-train-bundle"
SHARDS="artifacts/physics_shards"
NASA="data/processed/nasa3d"
NEO_BULK="artifacts/neo4j-tao-bulk"
PARITY="artifacts/neo4j_tao_parity.json"
SOLID="artifacts/tao_solidify_report.json"
COV="artifacts/corpus_full_coverage_report.json"
AUDIT="artifacts/corpus_coverage_audit.json"

link_or_copy() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [[ -d "$src" ]]; then
    mkdir -p "$dst"
    # Hardlink tree when same filesystem; else copy.
    if cp -al "$src/." "$dst/" 2>/dev/null; then
      return 0
    fi
    rm -rf "$dst"
    mkdir -p "$dst"
    cp -a "$src/." "$dst/"
  elif [[ -f "$src" ]]; then
    if ln "$src" "$dst" 2>/dev/null; then
      return 0
    fi
    cp -a "$src" "$dst"
  else
    echo "warn: missing $src" >&2
    return 1
  fi
}

rm -rf "$OUT"
mkdir -p "$OUT"/{bundle,physics_shards,nasa3d,neo4j-bulk,reports,configs,scripts_snapshot}

echo "== export stamp $STAMP =="

# Live graph + bundle files
if [[ -f "$BUNDLE/graph.json" ]]; then
  link_or_copy "$BUNDLE/graph.json" "$OUT/bundle/graph.json"
else
  echo "error: missing $BUNDLE/graph.json" >&2
  exit 1
fi
if [[ -d "$BUNDLE/files" ]]; then
  link_or_copy "$BUNDLE/files" "$OUT/bundle/files"
fi

# Physics + nasa shards
[[ -d "$SHARDS" ]] && link_or_copy "$SHARDS" "$OUT/physics_shards"
[[ -d "$NASA" ]] && link_or_copy "$NASA" "$OUT/nasa3d"

# Neo4j bulk CSVs (re-importable anywhere; prefer over runtime store)
if [[ -d "$NEO_BULK" ]]; then
  for f in nodes.csv relationships.csv import.stdout.log import-runtime.stdout.log; do
    [[ -e "$NEO_BULK/$f" ]] && link_or_copy "$NEO_BULK/$f" "$OUT/neo4j-bulk/$f"
  done
fi

# Reports
for f in "$PARITY" "$SOLID" "$COV" "$AUDIT" artifacts/portable-train-package/MANIFEST.json \
  artifacts/solver_case_tao_ingest_report.json artifacts/solver_case_tao_coverage.json \
  artifacts/text_cad_local_train/TRAIN_METRICS.json; do
  [[ -f "$f" ]] && link_or_copy "$f" "$OUT/reports/$(basename "$f")"
done

# Train configs + key scripts (code also on git; this is belt+suspenders)
for f in configs/base.yaml configs/families/space_24b.yaml; do
  [[ -f "$f" ]] && link_or_copy "$f" "$OUT/configs/$(basename "$f")"
done
for f in \
  scripts/prepare_portable_train_package.sh \
  scripts/export_offline_vm_snapshot.sh \
  scripts/tao_neo4j_bulk_import.py \
  scripts/solidify_tao_graph.py \
  scripts/corpus_full_coverage.py \
  scripts/launch_jepa24b_modal.sh
do
  [[ -f "$f" ]] && link_or_copy "$f" "$OUT/scripts_snapshot/$(basename "$f")"
done

GIT_HEAD="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
GIT_REMOTE="$(git remote get-url origin 2>/dev/null || echo unknown)"

# Counts
FEA=$(find "$OUT/physics_shards/fea" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
CFD=$(find "$OUT/physics_shards/cfd" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
NASA_N=$(find "$OUT/nasa3d" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
GRAPH_BYTES=$(wc -c <"$OUT/bundle/graph.json" | tr -d ' ')

python3 - <<PY
import json, time
from pathlib import Path
out = Path("$OUT")
manifest = {
  "generated_at": "$STAMP",
  "git_head": "$GIT_HEAD",
  "git_branch": "$GIT_BRANCH",
  "git_remote": "$GIT_REMOTE",
  "graph_bytes": int("$GRAPH_BYTES"),
  "physics_shards_fea_npz": int("$FEA"),
  "physics_shards_cfd_npz": int("$CFD"),
  "nasa3d_npz": int("$NASA_N"),
  "layout": {
    "bundle/graph.json": "live TAO training graph",
    "bundle/files/": "CAD/doc assets referenced by Samples",
    "physics_shards/": "FEA/CFD TensorShard NPZs",
    "nasa3d/": "NASA3D processed shards",
    "neo4j-bulk/": "CSV for neo4j-admin database import full",
    "reports/": "parity + coverage solidify reports",
    "configs/": "train yaml snapshots",
    "scripts_snapshot/": "key export/import scripts",
  },
  "train_hint": (
      "Pass graph-path=bundle/graph.json and "
      "data.extra_search_roots=<export_root> (with physics_shards + nasa3d under it "
      "or set roots to export_root/physics_shards parent + nasa parent)."
  ),
}
(out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

cat > "$OUT/RESTORE.md" <<EOF
# Offline VM export — restore

Generated: \`$STAMP\`
Git: \`$GIT_BRANCH\` @ \`$GIT_HEAD\`
Remote: \`$GIT_REMOTE\`

## What this is

Self-contained (hardlink or copy) snapshot of the densified TAO train graph,
physics shards, Neo4j bulk CSVs, and coverage/parity reports so the source VM
can shut down without losing train-critical bytes.

Code lives on GitHub (\`git pull\` at the recorded commit). This package is the
**data** side.

## Quick restore on a new machine

\`\`\`bash
# 1. Code
git clone "$GIT_REMOTE" jepa-cad
cd jepa-cad
git checkout $GIT_HEAD

# 2. Place export (example)
mkdir -p artifacts
# If you rsync'd/tarball'd this folder to artifacts/offline-export:
#   rsync -a /path/to/offline-export/ artifacts/offline-export/

# 3. Rehydrate live paths expected by scripts
mkdir -p artifacts/jepa-train-bundle artifacts/physics_shards data/processed
cp -a artifacts/offline-export/bundle/graph.json artifacts/jepa-train-bundle/graph.json
# files + shards: prefer rsync/hardlink from export
rsync -a artifacts/offline-export/bundle/files/ artifacts/jepa-train-bundle/files/ 2>/dev/null || true
rsync -a artifacts/offline-export/physics_shards/ artifacts/physics_shards/
rsync -a artifacts/offline-export/nasa3d/ data/processed/nasa3d/
rsync -a artifacts/offline-export/neo4j-bulk/ artifacts/neo4j-tao-bulk/
rsync -a artifacts/offline-export/reports/ artifacts/

# 4. Neo4j (optional): neo4j-admin database import full from neo4j-bulk CSVs
#    See scripts/tao_neo4j_bulk_import.py — needs raised inotify or project-local data dir.

# 5. Verify
sha256sum -c artifacts/offline-export/SHA256SUMS --ignore-missing | tail
python3 -c "import json; g=json.load(open('artifacts/jepa-train-bundle/graph.json')); print(len(g['nodes']), len(g['edges']))"
\`\`\`

## Train note (honest)

Current JEPA is latent-only (~77M) with hashed text bag and **no CAD decoder**.
Export preserves data for training; **text → physics-verified CAD assembly** still
needs semantic text + decoder work before Modal/full train is worth it.
EOF

# Checksums (content files; skip walking enormous identical hardlink trees slowly by focusing key paths)
{
  find "$OUT/bundle/graph.json" "$OUT/neo4j-bulk" "$OUT/reports" "$OUT/configs" "$OUT/scripts_snapshot" \
    "$OUT/MANIFEST.json" "$OUT/RESTORE.md" -type f 2>/dev/null
  find "$OUT/physics_shards" -type f -name '*.npz' 2>/dev/null | head -5
} | sort -u | while read -r f; do
  rel="${f#"$OUT"/}"
  (cd "$OUT" && sha256sum "$rel")
done > "$OUT/SHA256SUMS"

# Shard inventory counts (full checksum of all NPZ is slow; sample + count file)
find "$OUT/physics_shards" -type f -name '*.npz' 2>/dev/null | sort | sha256sum | awk '{print $1"  physics_shards_npz_path_list"}' >> "$OUT/SHA256SUMS"
find "$OUT/physics_shards" -type f -name '*.npz' 2>/dev/null | wc -l | awk '{print "count="$1}' > "$OUT/physics_shards.COUNT"
find "$OUT/bundle/files" -type f 2>/dev/null | wc -l | awk '{print "count="$1}' > "$OUT/bundle_files.COUNT"

# Optional compressed shippable archive
if [[ "${MAKE_TAR:-0}" == "1" ]]; then
  TAR="artifacts/offline-export-${STAMP}.tar"
  if command -v zstd >/dev/null 2>&1; then
    tar -C "$(dirname "$OUT")" -cf - "$(basename "$OUT")" | zstd -T0 -3 -o "${TAR}.zst"
    echo "wrote ${TAR}.zst"
  else
    tar -C "$(dirname "$OUT")" -cf "$TAR" "$(basename "$OUT")"
    echo "wrote $TAR"
  fi
fi

du -sh "$OUT" "$OUT"/* 2>/dev/null | head -40
echo "offline export ready: $OUT"
echo "restore doc: $OUT/RESTORE.md"
