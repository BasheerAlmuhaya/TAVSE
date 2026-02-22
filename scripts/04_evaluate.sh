#!/bin/bash
# ─────────────────────────────────────────────────────────────
# TAVSE: Model Evaluation
#
# Evaluates a trained model on the test set, computing PESQ,
# STOI, SI-SNR, SI-SNRi, and SDR. Saves results as JSON.
#
# Usage:
#   sbatch scripts/04_evaluate.sh audio_only
#   sbatch scripts/04_evaluate.sh audio_rgb
#   sbatch scripts/04_evaluate.sh audio_thermal
#   sbatch scripts/04_evaluate.sh audio_rgb_thermal
#
# With comparison:
#   sbatch scripts/04_evaluate.sh audio_thermal audio_rgb
# ─────────────────────────────────────────────────────────────

# ── Resolve project directory ─────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Load .env ─────────────────────────────────────────────────
if [ -f "${PROJECT_DIR}/.env" ]; then
    set -a; source "${PROJECT_DIR}/.env"; set +a
else
    echo "ERROR: ${PROJECT_DIR}/.env not found. Copy .env.example to .env."; exit 1
fi

# ── SLURM directives ─────────────────────────────────────────
#SBATCH --job-name=tavse-eval
#SBATCH --partition=${SLURM_GPU_PARTITION:-gpu}
#SBATCH --gres=${SLURM_GPU_GRES:-gpu:1}
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=${TAVSE_DATA_ROOT}/logs/eval_%j.out
#SBATCH --error=${TAVSE_DATA_ROOT}/logs/eval_%j.err

set -euo pipefail

# ── Parse arguments ───────────────────────────────────────────
EXPERIMENT=${1:-audio_only}
COMPARE_WITH=${2:-}

# ── Paths ─────────────────────────────────────────────────────
SCRATCH_BASE="${TAVSE_DATA_ROOT:?Set TAVSE_DATA_ROOT in .env}"
CONFIG_FILE="${PROJECT_DIR}/configs/${EXPERIMENT}.yaml"
CKPT_DIR="${SCRATCH_BASE}/checkpoints/${EXPERIMENT}"
EVAL_DIR="${CKPT_DIR}/eval"

# Find best checkpoint (highest SI-SNR in filename)
CHECKPOINT=$(ls -t ${CKPT_DIR}/ckpt_*.pt 2>/dev/null | head -1)
if [ -z "${CHECKPOINT}" ]; then
    CHECKPOINT="${CKPT_DIR}/latest.pt"
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: No checkpoint found in ${CKPT_DIR}/"
    exit 1
fi

# ── Environment ───────────────────────────────────────────────
export HF_HOME="${HF_HOME:-${SCRATCH_BASE}/.hf_cache}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export TAVSE_DATA_ROOT="${SCRATCH_BASE}"

# ── Storage check ─────────────────────────────────────────────
echo "=== Storage Check ==="
echo "Home: $(du -sh ~ 2>/dev/null | cut -f1)"
echo "Data root: $(du -sh ${SCRATCH_BASE} 2>/dev/null | cut -f1)"
echo "====================="

cd "${PROJECT_DIR}"
source activate tavse 2>/dev/null || conda activate tavse 2>/dev/null || true

echo ""
echo "=================================================="
echo "TAVSE Evaluation"
echo "  Experiment:  ${EXPERIMENT}"
echo "  Config:      ${CONFIG_FILE}"
echo "  Checkpoint:  ${CHECKPOINT}"
echo "  Output:      ${EVAL_DIR}/"
echo "  Compare:     ${COMPARE_WITH:-none}"
echo "=================================================="
echo ""

# ── Build comparison flag ─────────────────────────────────────
COMPARE_FLAG=""
if [ -n "${COMPARE_WITH}" ]; then
    COMPARE_METRICS="${SCRATCH_BASE}/checkpoints/${COMPARE_WITH}/eval/eval_metrics.json"
    if [ -f "${COMPARE_METRICS}" ]; then
        COMPARE_FLAG="--compare-with ${COMPARE_METRICS}"
    else
        echo "[WARN] Comparison metrics not found: ${COMPARE_METRICS}"
    fi
fi

# ── Evaluate ──────────────────────────────────────────────────
python -m src.training.evaluate \
    --config "${CONFIG_FILE}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${EVAL_DIR}" \
    --save-samples 5 \
    ${COMPARE_FLAG}

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "=== Evaluation Complete ==="
echo "Results: ${EVAL_DIR}/eval_metrics.json"
echo "Samples: ${EVAL_DIR}/samples/"
echo ""
if [ -f "${EVAL_DIR}/eval_metrics.json" ]; then
    echo "Metrics:"
    python3 -c "
import json
with open('${EVAL_DIR}/eval_metrics.json') as f:
    d = json.load(f)
for k, v in d['statistics'].items():
    print(f\"  {k:>8s}: {v['mean']:.3f} ± {v['std']:.3f}\")
"
fi
echo "==========================="
