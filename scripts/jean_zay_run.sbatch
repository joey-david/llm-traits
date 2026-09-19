#!/usr/bin/env bash
# Run the pipeline for one or more traits on a single H100.
#
#   TRAITS="arousal" sbatch scripts/jean_zay_run.sbatch
#   TRAITS="arousal pain hunger" STAGES=direction,geometry,steer,scenarios,examples \
#     sbatch scripts/jean_zay_run.sbatch
#   STAGES=direction,geometry,steer,scenarios,button,examples \
#     sbatch scripts/jean_zay_run.sbatch   # everything, including the button task
#
# A 32B model in bfloat16 is about 64 GB of weights, which fits on one H100 with
# room for the activations at batch 8. Asking for more GPUs would not make it
# faster: nothing here is distributed, and the queue is shorter for one.
#
# The dev QoS caps a job at two hours and two jobs at a time, which is the right
# trade for a single trait; its queue is much shorter. A full battery needs the
# standard QoS, and sbatch command-line flags override the directives below:
#
#   sbatch --qos=qos_gpu_h100-t3 --time=12:00:00 scripts/jean_zay_run.sbatch
#SBATCH --job-name=lt-run
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
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
traits="${TRAITS:-arousal}"
stages="${STAGES:-direction,geometry,steer,scenarios,examples}"
tag="${TAG:-$(basename "$model")}"
batch="${BATCH:-8}"
# Where in a chat-formatted scenario to read: "header" is the final token with
# the generation prompt appended, which is the assistant-turn marker and the
# same tokens in every scenario; "content" is the last token the user wrote.
read_at="${READ_AT:-content}"
# The paper's ladder is -2 -1 0 .5 1 1.5 2 3. The wider default here exists
# because this model produces no trait vocabulary at all across that range for
# some traits, and the extra rungs tell "needs a bigger coefficient" apart from
# "the effect is not there". Collapsed continuations are filtered out, so the
# extremes cost nothing but time.
ladder="${LADDER:--4 -2 -1 0 1 2 3 4}"

echo "model    $model"
echo "traits   $traits"
echo "stages   $stages"
echo "tag      $tag"
echo "read-at  $read_at"
echo "ladder   $ladder"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
exec "$PYTHON" -m llm_traits.cli --run-root runs --tag "$tag" run $traits \
  --model "$model" \
  --stages "$stages" \
  --batch-size "$batch" \
  --read-at "$read_at" \
  --ladder $ladder \
  --device cuda \
  --dtype bfloat16
