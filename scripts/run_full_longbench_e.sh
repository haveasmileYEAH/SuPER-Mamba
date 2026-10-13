#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAMB_ROOT="${LAMB_ROOT:-$ROOT/third_party/lamb}"
RUNNER="$ROOT/scripts/run_longbench_e.py"

WORKERS=1
DEVICE=0
OUT="$ROOT/results/mamba2_full"
PREFLIGHT_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output-root)
            OUT="$2"
            shift 2
            ;;
        --preflight-only)
            PREFLIGHT_ONLY=1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: Workers must be a positive integer."
    exit 1
fi

LAMB_ROOT="$(realpath "$LAMB_ROOT")"
OUT="$(realpath -m "$OUT")"

if [[ ! -f "$LAMB_ROOT/mamba_ssm/mapping/__init__.py" ]]; then
    echo "ERROR: LAMB missing. Run bash setup.sh first."
    exit 1
fi

export LAMB_ROOT
export PYTHONPATH="$LAMB_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# LAMB resolves precomputed tensor paths relative to CWD.
cd "$LAMB_ROOT"

echo "========== PREFLIGHT =========="

python - "$ROOT" <<'PY'
import sys
import json
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "scripts"))

import run_longbench_e as runner

selection = json.loads(runner.SELECTION_PATH.read_text())

assert len(runner.DATASETS) == 13
assert set(selection["datasets"]) == set(runner.DATASETS)

for module in (runner, runner.six, runner.dynamic, runner.base):
    assert Path(module.__file__).resolve().is_relative_to(root)

runner.six.preflight()
runner.dynamic.baseline_equivalence_check("cuda:0")

print("PREFLIGHT: PASS")
PY

if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
    echo "PREFLIGHT ONLY: COMPLETE"
    exit 0
fi

mkdir -p "$OUT/logs"

# Prevent concurrent launches using the same output directory.
exec 9>"$OUT/.run.lock"
flock -n 9 || {
    echo "ERROR: Another run is using this output directory."
    exit 1
}

echo "========== MAMBA2 FULL LONGBENCH-E =========="
echo "Workers: $WORKERS"
echo "GPU: $DEVICE"
echo "Output: $OUT"

if [[ "$WORKERS" -gt 1 ]]; then
    echo "WARNING: Multiple workers may cause CUDA OOM."
fi

PIDS=()

for ((i=0; i<WORKERS; i++)); do
    python -u "$RUNNER" \
        --device "$DEVICE" \
        --worker-id "$i" \
        --num-workers "$WORKERS" \
        --output-root "$OUT" \
        --logit-chunk-size 64 \
        > "$OUT/logs/worker_${i}.log" 2>&1 &

    PIDS+=("$!")
    echo "Worker $i launched: PID ${PIDS[$i]}"
done

FAILED=0

for ((i=0; i<WORKERS; i++)); do
    if wait "${PIDS[$i]}"; then
        echo "Worker $i: COMPLETE"
    else
        echo "Worker $i: FAILED"
        tail -n 20 "$OUT/logs/worker_${i}.log"
        FAILED=1
    fi
done

if [[ "$FAILED" -ne 0 ]]; then
    echo "ERROR: Worker failure. Saved samples can be reused."
    exit 1
fi

COUNT=$(find "$OUT" -type f -path '*/samples/*.json' | wc -l)

echo "Samples: $COUNT / 3668"

if [[ "$COUNT" -ne 3668 ]]; then
    echo "ERROR: Incomplete evaluation."
    exit 1
fi

echo "========== AGGREGATION =========="

python -u "$RUNNER" \
    --output-root "$OUT" \
    --aggregate-only \
    | tee "$OUT/logs/aggregate.log"

python - "$OUT/full_summary.csv" <<'PY'
import csv
import sys
from statistics import mean

with open(sys.argv[1], newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 13
assert sum(int(r["n"]) for r in rows) == 3668

macro = mean(float(r["final"]) for r in rows)

print("=" * 55)
print("SuPER-Mamba / Mamba2 LongBench-E")
print("Datasets: 13")
print("Samples: 3668")
print(f"Macro Average: {macro:.6f}")
print("Paper reference: 21.56")
print("=" * 55)
PY

echo "FULL EVALUATION COMPLETE"
