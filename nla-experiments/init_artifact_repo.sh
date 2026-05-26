#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="${NLA_ARTIFACT_ROOT:-nla-artifacts}"

mkdir -p "$ARTIFACT_DIR/summeval/raw" "$ARTIFACT_DIR/slurm"

if [[ ! -d "$ARTIFACT_DIR/.git" ]]; then
  git init "$ARTIFACT_DIR"
fi

cat > "$ARTIFACT_DIR/.gitignore" <<'EOF'
# Heavy model/checkpoint artifacts.
*.bin
*.ckpt
*.pt
*.pth
*.safetensors
checkpoints/
models/
hf-cache/

# Runtime noise.
__pycache__/
*.pyc
.pytest_cache/
slurm/
*.log

# Keep curated small outputs.
!summeval/
!summeval/*.jsonl
!summeval/*.json
!summeval/*.parquet
!summeval/*.md
!summeval/raw/
!summeval/raw/*.json
EOF

if [[ ! -f "$ARTIFACT_DIR/README.md" ]]; then
  cat > "$ARTIFACT_DIR/README.md" <<'EOF'
# NLA Artifacts

Local Git repository for curated SummEval NLA experiment artifacts.

Large checkpoints, Hugging Face caches, full Slurm logs, and raw model files are
not tracked here. Keep small manifests, smoke Parquet files, verbalization JSONL
files, and concise reports.
EOF
fi

if [[ ! -f "$ARTIFACT_DIR/summeval/RUNS.md" ]]; then
  cat > "$ARTIFACT_DIR/summeval/RUNS.md" <<'EOF'
# SummEval NLA Runs

Append one short entry per meaningful local or cluster run. Include command,
commit hash from the parent `tesi` repo, artifact filenames, and whether the run
was a dry run, CPU fake-vector smoke, GPU smoke, or full v1 pass.
EOF
fi

echo "Artifact repo ready: $ARTIFACT_DIR"

