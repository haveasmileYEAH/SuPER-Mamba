#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SuPER-Mamba Environment Setup
#
# Verified environment:
#   Python          3.10.20
#   PyTorch         2.5.0+cu124
#   CUDA Toolkit    12.4
#   causal-conv1d   1.5.0.post8
#   mamba-ssm       2.2.4
#
# Usage:
#   bash setup.sh
#   bash setup.sh --skip-deps
# ============================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMB_DIR="$ROOT/third_party/lamb"

LAMB_REPO="https://github.com/GATECH-EIC/LAMB.git"
LAMB_COMMIT="b3ce2a9476b46fb1e28b5572a39b82e7da3dc519"

SKIP_DEPS=0

case "${1:-}" in
    "")
        ;;
    --skip-deps)
        SKIP_DEPS=1
        ;;
    *)
        echo "Usage: bash setup.sh [--skip-deps]"
        exit 1
        ;;
esac

echo "================================================="
echo "SuPER-Mamba Setup"
echo "================================================="

# ------------------------------------------------------------
# Step 1: Check Python environment
# ------------------------------------------------------------

echo "[1/7] Checking Python environment"

if [[ -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: Activate an isolated Conda or venv environment."
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
    echo "ERROR: Do not install SuPER-Mamba into Conda base."
    exit 1
fi

python - <<'PY'
import sys

print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)

assert sys.version_info[:2] == (3, 10), (
    "SuPER-Mamba requires Python 3.10"
)
PY

# ------------------------------------------------------------
# Step 2: Install PyTorch CUDA 12.4
# ------------------------------------------------------------

echo "[2/7] Checking PyTorch"

if [[ "$SKIP_DEPS" -eq 0 ]]; then

    if ! python - <<'PY'
import torch
assert torch.__version__ == "2.5.0+cu124"
assert torch.version.cuda == "12.4"
PY
    then
        python -m pip install \
            "torch==2.5.0+cu124" \
            --index-url https://download.pytorch.org/whl/cu124
    fi

    # --------------------------------------------------------
    # Step 3: Install regular dependencies
    # --------------------------------------------------------

    echo "[3/7] Installing requirements"

    python -m pip install -r "$ROOT/requirements.txt"

    # --------------------------------------------------------
    # Step 4: Install CUDA extensions separately
    # --------------------------------------------------------

    echo "[4/7] Installing CUDA extensions"

    python -m pip install \
        "causal-conv1d==1.5.0.post8" \
        --no-build-isolation \
        --no-deps

    python -m pip install \
        "mamba-ssm==2.2.4" \
        --no-build-isolation \
        --no-deps

    python -m pip check

else
    echo "[3/7] Skipping regular dependencies"
    echo "[4/7] Skipping CUDA extension installation"
fi

# ------------------------------------------------------------
# Step 5: Download pinned LAMB
# ------------------------------------------------------------

echo "[5/7] Preparing LAMB"

mkdir -p "$ROOT/third_party"

if [[ ! -d "$LAMB_DIR/.git" ]]; then

    if [[ -e "$LAMB_DIR" ]]; then
        echo "ERROR: LAMB destination already exists:"
        echo "$LAMB_DIR"
        exit 1
    fi

    git clone \
        --filter=blob:none \
        --depth 1 \
        --no-checkout \
        "$LAMB_REPO" \
        "$LAMB_DIR"
fi

CURRENT_COMMIT="$(git -C "$LAMB_DIR" rev-parse HEAD)"

if [[ "$CURRENT_COMMIT" != "$LAMB_COMMIT" ]]; then

    if [[ -n "$(git -C "$LAMB_DIR" status --porcelain)" ]]; then
        echo "ERROR: Existing LAMB directory has local changes."
        exit 1
    fi

    git -C "$LAMB_DIR" fetch \
        --depth 1 origin "$LAMB_COMMIT"
fi

if [[ "$(git -C "$LAMB_DIR" rev-parse HEAD)" != "$LAMB_COMMIT" ]]; then
    git -C "$LAMB_DIR" \
        -c advice.detachedHead=false \
        checkout --detach "$LAMB_COMMIT"
elif [[ ! -f "$LAMB_DIR/mamba_ssm/mapping/__init__.py" ]]; then
    git -C "$LAMB_DIR" \
        -c advice.detachedHead=false \
        checkout --detach "$LAMB_COMMIT"
fi

ACTUAL_COMMIT="$(git -C "$LAMB_DIR" rev-parse HEAD)"

if [[ "$ACTUAL_COMMIT" != "$LAMB_COMMIT" ]]; then
    echo "ERROR: LAMB commit mismatch."
    exit 1
fi

echo "LAMB commit: $ACTUAL_COMMIT"

# ------------------------------------------------------------
# Step 6: Check Global Channel assets
# ------------------------------------------------------------

echo "[6/7] Checking Global Channel assets"

TENSOR_DIR="$LAMB_DIR/remapping_configs/precomputed_tensors"

for filename in \
    mamba2-1.3b_global_channel_indexes.pth \
    mamba2-1.3b_per_channel_decays.pth
do
    FILE="$TENSOR_DIR/$filename"

    if [[ ! -s "$FILE" ]]; then
        echo "ERROR: Missing LAMB tensor:"
        echo "$FILE"
        exit 1
    fi

    echo "FOUND: $filename"
done

# ------------------------------------------------------------
# Step 7: Verify installation and project imports
# ------------------------------------------------------------

echo "[7/7] Validating installation"

export LAMB_ROOT="$LAMB_DIR"
export PYTHONPATH="$LAMB_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export SUPER_MAMBA_ROOT="$ROOT"

# LAMB uses relative paths for precomputed tensors.
cd "$LAMB_DIR"

python - <<'PY'
import json
import os
import sys
from pathlib import Path

import torch
import transformers
import datasets

root = Path(os.environ["SUPER_MAMBA_ROOT"])
lamb = Path(os.environ["LAMB_ROOT"])

from mamba_ssm.mapping import load_dt_remapping_from_config
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from tasks.longbench import scorer_e

import mamba_ssm

assert Path(mamba_ssm.__file__).resolve().is_relative_to(lamb)

sys.path.insert(0, str(root / "scripts"))

import run_longbench_e as runner

for module in (
    runner,
    runner.six,
    runner.dynamic,
    runner.base,
):
    assert Path(module.__file__).resolve().is_relative_to(root)

selection = json.loads(
    (root / "configs/category_frozen_full_selection.json").read_text()
)

assert len(selection["datasets"]) == 13

config = json.loads(
    (lamb / "remapping_configs/mamba2-1.3b_topk1024_pk9.json")
    .read_text()
)

for key in ("global_channel_indexes", "per_channel_decays"):
    path = lamb / config[key]
    tensor = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )
    assert hasattr(tensor, "__len__"), key
    assert len(tensor) > 0, key
    assert isinstance(tensor[0], torch.Tensor), key

    print(
        f"{key}: "
        f"type={type(tensor).__name__}, "
        f"layers={len(tensor)}, "
        f"first_layer_shape={tuple(tensor[0].shape)}"
    )

assert torch.__version__ == "2.5.0+cu124"
assert torch.cuda.is_available()

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)
print("Mamba source:", mamba_ssm.__file__)
print("CUDA:", torch.version.cuda)
print("Frozen datasets:", len(selection["datasets"]))
print("Global Channel tensors: PASS")
print("Project module imports: PASS")
print("INSTALLATION VALIDATION: PASS")
PY

echo "================================================="
echo "SuPER-Mamba Setup Complete"
echo "================================================="
