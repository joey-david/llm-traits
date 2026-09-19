#!/usr/bin/env bash
# Build the cross-trait panel from finished runs. No GPU needed: this only reads
# the vectors and the JSON the run jobs already wrote.
#
#   TAG=Qwen2.5-32B sbatch scripts/jean_zay_compare.sbatch
#SBATCH --job-name=lt-compare
#SBATCH --account=fas@cpu
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p slurm_logs
source scripts/jean_zay_env.sh

exec "$PYTHON" -m llm_traits.cli --run-root runs --tag "${TAG:?set TAG to the run tag}" \
  compare --highlight "${HIGHLIGHT:-pain}"
