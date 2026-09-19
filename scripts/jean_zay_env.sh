#!/usr/bin/env bash
# Load the site's H100 stack and this checkout's overlay.
#
# Sourced by every batch script here, and safe to source by hand on a login
# node. It sets PYTHON, PYTHONPATH and HF_HOME and nothing else.

set -e

# Compute nodes do not carry /etc/profile.d/z_modules.sh, which login nodes do.
# Sourcing it unconditionally is what kills batch jobs inside two seconds, with
# an error no login-node test reproduces. Try the known init scripts in turn.
if ! command -v module >/dev/null 2>&1 && ! declare -F module >/dev/null 2>&1; then
  for module_init in \
    /etc/profile.d/z_modules.sh \
    /usr/share/lmod/lmod/init/bash \
    /opt/lmod/lmod/init/bash \
    /etc/profile.d/modules.sh
  do
    if [[ -r "$module_init" ]]; then
      # shellcheck disable=SC1090
      source "$module_init" && break
    fi
  done
fi
if ! command -v module >/dev/null 2>&1 && ! declare -F module >/dev/null 2>&1; then
  echo "no module system found on $(hostname); cannot load the H100 stack" >&2
  exit 2
fi

module purge
gpu_arch="${LLM_TRAITS_GPU_ARCH:-}"
if [[ -z "$gpu_arch" ]]; then
  case "${SLURM_JOB_CONSTRAINTS:-}" in
    *a100*) gpu_arch="a100" ;;
    *v100*) gpu_arch="v100" ;;
    *) gpu_arch="h100" ;;
  esac
fi
# V100 is the site default; A100 and H100 need an architecture module to pick
# builds for their newer compute capabilities.
if [[ "$gpu_arch" != "v100" ]]; then
  module load "arch/$gpu_arch"
fi
module load pytorch-gpu/py3/2.8.0

repo_root="${repo_root:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Tokens live in ~/.env as plain KEY=value lines. A batch job gets a non-login
# shell and inherits nothing, so sourcing it here is the only place HF_TOKEN
# arrives. Absent is fine: the default weights are ungated.
if [[ -r "$HOME/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOME/.env"
  set +a
fi

# Prefer the repo venv only if it can actually import torch. A venv that exists
# but is missing the stack silently shadows the working module python, and every
# job then dies on startup inside a second.
python_bin="$(command -v python)"
if [[ -x "${repo_root}/.venv/bin/python" ]] &&
   "${repo_root}/.venv/bin/python" -c 'import torch' >/dev/null 2>&1; then
  python_bin="${repo_root}/.venv/bin/python"
fi
export PYTHON="$python_bin"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

# A 32B checkpoint is about 64 GB. WORK has the quota for it and survives the
# SCRATCH purge; the repo-local fallback is for a checkout outside WORK.
export HF_HOME="${LLM_TRAITS_HF_HOME:-${WORK:+${WORK}/hf_cache}}"
export HF_HOME="${HF_HOME:-${repo_root}/.hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${repo_root}/.mplcache}"

# Compute nodes have no route out. Anything that reaches the Hub has to run on
# prepost, so GPU jobs are offline by construction and fail loudly rather than
# hanging on a connect timeout if a weight file is missing.
if [[ "${LLM_TRAITS_ONLINE:-0}" != 1 ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export DATASETS_OFFLINE=1
else
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE DATASETS_OFFLINE
fi
