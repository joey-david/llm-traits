#!/usr/bin/env bash
# Push this checkout to a cluster and verify that what landed is what you sent.
#
#   scripts/deploy.sh                      # default host and path
#   scripts/deploy.sh myhost /path/to/dir
#
# The verification is not ceremony. A local shell can fail to parse a compound
# command and skip the rsync inside it while the rest of the line still looks
# like it ran; the next job then fails on a flag the remote copy has never heard
# of, several minutes of queue time later. Comparing a manifest of the source
# tree on both ends turns that into an immediate, local error.
set -euo pipefail

host="${1:-jean-zay}"
remote="${2:-/lustre/fswork/projects/rech/fas/uul94gf/llm-traits}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

excludes=(
  --exclude '.venv' --exclude '.git' --exclude 'runs' --exclude 'runs_*'
  --exclude '.hf_cache' --exclude '__pycache__' --exclude '.pytest_cache'
  --exclude '*.egg-info' --exclude '.mplcache' --exclude 'slurm_logs'
)

manifest() {
  # Hash of every tracked source file, order-independent of the filesystem.
  find src configs scripts tests pyproject.toml -type f \
    \( -name '*.py' -o -name '*.yaml' -o -name '*.sbatch' -o -name '*.sh' -o -name '*.toml' \) \
    | LC_ALL=C sort | xargs shasum | shasum | cut -d' ' -f1
}

local_hash="$(manifest)"
echo "local  $local_hash"

rsync -az --delete-after "${excludes[@]}" ./ "$host:$remote/"

remote_hash="$(ssh -o BatchMode=yes "$host" "bash -lc 'cd $remote && find src configs scripts tests pyproject.toml -type f \\( -name \"*.py\" -o -name \"*.yaml\" -o -name \"*.sbatch\" -o -name \"*.sh\" -o -name \"*.toml\" \\) | LC_ALL=C sort | xargs shasum | shasum | cut -d\" \" -f1'")"
echo "remote $remote_hash"

if [[ "$local_hash" != "$remote_hash" ]]; then
  echo "DEPLOY FAILED: the remote tree does not match the local one" >&2
  exit 1
fi
echo "deployed to $host:$remote"
