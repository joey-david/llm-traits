#!/usr/bin/env bash
# Build the environment and pull the weights, on the one partition with network.
#
#   sbatch scripts/jean_zay_bootstrap.sbatch
#   MODEL=Qwen/Qwen2.5-32B sbatch scripts/jean_zay_bootstrap.sbatch
#
# The site module already provides a torch built for these H100s, so the venv is
# created with --system-site-packages over the module python and only the
# packages the module stack lacks are installed on top. Installing our own torch
# instead would replace a tuned build with a generic one.
#
# Only prepost reaches PyPI and the Hub. Login nodes have no route and no proxy,
# so an install submitted anywhere else fails on a connection timeout.
#SBATCH --job-name=lt-bootstrap
#SBATCH --account=fas@cpu
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p slurm_logs

# A batch job gets a non-login shell, which never sources /etc/profile.d, so it
# inherits no proxy and every fetch dies on a connect timeout. prodprox.idris.fr
# is the only way out to PyPI and the Hub.
if [[ -r /etc/profile.d/proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/proxy.sh
fi
if [[ -z "${https_proxy:-}" ]]; then
  echo "no https_proxy; this node cannot reach PyPI or the Hub" >&2
  exit 2
fi

export LLM_TRAITS_ONLINE=1
source scripts/jean_zay_env.sh
module_python="$PYTHON"

command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "uv not on PATH" >&2; exit 2; }

if [[ -x .venv/bin/python ]] && ! .venv/bin/python -c 'import torch' >/dev/null 2>&1; then
  mv .venv ".venv.unusable.$(date +%s)"
fi
if [[ ! -x .venv/bin/python ]]; then
  uv venv --system-site-packages --python "$module_python" .venv
fi

# --no-deps so the module's torch is left alone; the rest is filled in only
# where the module stack is actually missing something.
uv pip install --python .venv/bin/python --no-deps -e .
for package in numpy scipy scikit-learn matplotlib pyyaml tqdm huggingface_hub safetensors transformers packaging; do
  import_name="${package//-/_}"
  case "$package" in
    scikit-learn) import_name=sklearn ;;
    pyyaml) import_name=yaml ;;
  esac
  if ! .venv/bin/python -c "import $import_name" >/dev/null 2>&1; then
    echo "installing missing dependency: $package"
    uv pip install --python .venv/bin/python "$package"
  fi
done

.venv/bin/python - <<'PY'
import torch, transformers, sklearn, matplotlib, yaml
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
      f"transformers {transformers.__version__}  sklearn {sklearn.__version__}")
PY

model="${MODEL:-Qwen/Qwen2.5-32B}"
echo "prefetching $model into $HF_HOME"
.venv/bin/python - "$model" <<'PY'
import sys
from huggingface_hub import snapshot_download
name = sys.argv[1]
# allow_patterns keeps the consolidated/original copies some repos ship out of
# the cache; on a 32B checkpoint that is tens of gigabytes of quota.
path = snapshot_download(
    name,
    allow_patterns=["*.json", "*.txt", "*.model", "*.jinja", "*.safetensors"],
    ignore_patterns=["original/*", "consolidated*"],
)
print("cached at", path)
PY

echo "bootstrap complete"
