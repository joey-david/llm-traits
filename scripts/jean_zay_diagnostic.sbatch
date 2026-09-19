#!/usr/bin/env bash
# Two questions the first battery raised, answered on the traits that matter.
#
# 1. The self-versus-user asymmetry came out negative for every trait including
#    pain, which is the wrong sign for the positive control. One candidate cause
#    is the read position: with the generation prompt appended, the final token
#    of a chat-formatted scenario is the assistant-turn header, identical in
#    every scenario. --read-at content reads the last token the user wrote.
#
# 2. The coefficient ladder produced no trait vocabulary at all for pain across
#    the paper's whole range. Either the effect needs a larger coefficient on
#    this model, or it is not there. A ladder out to +/-8 separates those.
#
#   sbatch scripts/jean_zay_diagnostic.sbatch
#SBATCH --job-name=lt-diag
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%A.out
#SBATCH --error=slurm_logs/%x-%A.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p slurm_logs
source scripts/jean_zay_env.sh

model="${MODEL:-huihui-ai/Qwen2.5-32B-Instruct-abliterated}"
traits="${TRAITS:-pain arousal hunger confusion}"

# shellcheck disable=SC2086
exec "$PYTHON" -m llm_traits.cli --run-root runs --tag "${TAG:-diag-content}" run $traits \
  --model "$model" --device cuda --dtype bfloat16 \
  --stages direction,geometry,steer,scenarios \
  --batch-size 8 \
  --read-at "${READ_AT:-content}" \
  --ladder -8 -4 -2 -1 0 1 2 4 6 8
