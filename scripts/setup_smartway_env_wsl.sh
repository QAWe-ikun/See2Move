#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-smartway}"
HABITAT_LAB_DIR="${HABITAT_LAB_DIR:-"$HOME/src/habitat-lab-v0.1.7-smartway"}"
HABITAT_SIM_TARBALL="${HABITAT_SIM_TARBALL:-}"

if ! command -v conda >/dev/null 2>&1; then
  printf 'conda is required. Install Miniconda or Mambaforge in WSL first.\n' >&2
  exit 1
fi

conda create -n "$ENV_NAME" python=3.8.20 -y

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

if [[ -n "$HABITAT_SIM_TARBALL" ]]; then
  conda install "$HABITAT_SIM_TARBALL" -y
else
  printf 'Skipping habitat-sim install. Set HABITAT_SIM_TARBALL to the SmartWay Habitat-Sim 0.1.7 tarball path, then rerun or install manually.\n'
fi

if [[ ! -d "$HABITAT_LAB_DIR/.git" ]]; then
  mkdir -p "$(dirname "$HABITAT_LAB_DIR")"
  git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git "$HABITAT_LAB_DIR"
fi

cd "$HABITAT_LAB_DIR"
python setup.py develop --all

cd "$ROOT/third_party/smartway-code"
python -m pip install -r requirements.txt
pip install webdataset openai tenacity timm fairscale

if [[ ! -d recognize-anything/.git ]]; then
  git clone https://github.com/xinyu1205/recognize-anything.git
fi

printf 'SmartWay environment is ready: conda activate %s\n' "$ENV_NAME"
