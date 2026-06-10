#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-vlnce}"
HABITAT_LAB_DIR="${HABITAT_LAB_DIR:-"$HOME/src/habitat-lab-v0.1.7"}"

if ! command -v conda >/dev/null 2>&1; then
  printf 'conda is required. Install Miniconda or Mambaforge in WSL first.\n' >&2
  exit 1
fi

conda create -n "$ENV_NAME" python=3.6 -y

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless -y
python -m pip --default-timeout=120 install "pip==21.3.1" "setuptools<60" wheel

if [[ ! -d "$HABITAT_LAB_DIR/.git" ]]; then
  mkdir -p "$(dirname "$HABITAT_LAB_DIR")"
  git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git "$HABITAT_LAB_DIR"
fi

cd "$HABITAT_LAB_DIR"
python -m pip --default-timeout=120 install -r requirements.txt
python -m pip --default-timeout=120 install -r habitat_baselines/rl/requirements.txt
python -m pip --default-timeout=120 install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all

cd "$ROOT/third_party/vln-ce"
python -m pip --default-timeout=120 install -r requirements.txt

printf 'VLN-CE environment is ready: conda activate %s\n' "$ENV_NAME"
