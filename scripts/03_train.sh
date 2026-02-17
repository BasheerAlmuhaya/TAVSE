#!/bin/bash
#SBATCH --job-name=tavse-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/scratch/users/40741008/tavse/logs/train_%j.out
#SBATCH --error=/mnt/scratch/users/40741008/tavse/logs/train_%j.err

# ─────────────────────────────────────────────────────────────
# TAVSE: Model Training
#
# Trains one of the four model variants (A, A+R, A+T, A+R+T).
# Uses mixed precision, cosine LR schedule, early stopping.
#
# Usage:
#   sbatch scripts/03_train.sh audio_only           # Audio-only baseline
#   sbatch scripts/03_train.sh audio_rgb            # Audio + RGB
#   sbatch scripts/03_train.sh audio_thermal        # Audio + Thermal
#   sbatch scripts/03_train.sh audio_rgb_thermal    # Full trimodal
#
# Resume:
#   sbatch scripts/03_train.sh audio_rgb --resume
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Parse arguments ───────────────────────────────────────────
EXPERIMENT=${1:-audio_only}
RESUME=""
shift || true
for arg in "$@"; do
    [ "$arg" = "--resume" ] && RESUME="--resume"
done

# ── Paths ─────────────────────────────────────────────────────
SCRATCH_BASE=/mnt/scratch/users/40741008/tavse
PROJECT_DIR=~/my_projects/AVTSE/TAVSE
CONFIG_FILE=${PROJECT_DIR}/configs/${EXPERIMENT}.yaml

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config not found: ${CONFIG_FILE}"
    echo "Available: $(ls ${PROJECT_DIR}/configs/*.yaml 2>/dev/null | xargs -n1 basename)"
    exit 1
fi

# ── Environment ───────────────────────────────────────────────
export HF_HOME=${SCRATCH_BASE}/.hf_cache
export PYTHONPATH=${PROJECT_DIR}:${PYTHONPATH:-}

# Ensure log/checkpoint dirs exist
mkdir -p ${SCRATCH_BASE}/{checkpoints/${EXPERIMENT},logs/${EXPERIMENT}}

# ── Storage check ─────────────────────────────────────────────
echo "=== Storage Check ==="
echo "Home: $(du -sh ~ 2>/dev/null | cut -f1) / 50 GB quota"
quota -s 2>/dev/null | tail -1 || true
echo "Scratch: $(du -sh $SCRATCH_BASE 2>/dev/null | cut -f1) (no quota)"
echo "====================="

# ── GPU info ──────────────────────────────────────────────────
echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo "================"

cd ${PROJECT_DIR}
source activate tavse 2>/dev/null || conda activate tavse 2>/dev/null || true

echo ""
echo "=================================================="
echo "TAVSE Training"
echo "  Experiment: ${EXPERIMENT}"
echo "  Config:     ${CONFIG_FILE}"
echo "  Resume:     ${RESUME:-no}"
echo "  Node:       $(hostname)"
echo "  GPU:        ${CUDA_VISIBLE_DEVICES:-all}"
echo "  Checkpoint: ${SCRATCH_BASE}/checkpoints/${EXPERIMENT}/"
echo "  Logs:       ${SCRATCH_BASE}/logs/${EXPERIMENT}/"
echo "=================================================="
echo ""

# ── Optional: copy LMDBs to /tmp for faster I/O ──────────────
# Uncomment below if Lustre I/O becomes a bottleneck:
#
# echo "Staging data to /tmp..."
# mkdir -p /tmp/tavse_data/processed
# cp -r ${SCRATCH_BASE}/processed/rgb_mouth.lmdb /tmp/tavse_data/processed/ 2>/dev/null || true
# cp -r ${SCRATCH_BASE}/processed/thermal_mouth.lmdb /tmp/tavse_data/processed/ 2>/dev/null || true
# OVERRIDE='{"data": {"rgb_lmdb_path": "/tmp/tavse_data/processed/rgb_mouth.lmdb", "thermal_lmdb_path": "/tmp/tavse_data/processed/thermal_mouth.lmdb"}}'
# ADD_OVERRIDE="--override '${OVERRIDE}'"

# ── Train ─────────────────────────────────────────────────────
python -m src.training.train \
    --config ${CONFIG_FILE} \
    ${RESUME}

# ── Final storage check ──────────────────────────────────────
echo ""
echo "=== Final Storage Check ==="
echo "Checkpoints: $(du -sh ${SCRATCH_BASE}/checkpoints/${EXPERIMENT}/ 2>/dev/null | cut -f1)"
echo "Logs: $(du -sh ${SCRATCH_BASE}/logs/${EXPERIMENT}/ 2>/dev/null | cut -f1)"
echo "==========================="

echo ""
echo "Training complete: ${EXPERIMENT}"
echo "Checkpoints: ${SCRATCH_BASE}/checkpoints/${EXPERIMENT}/"
echo "TensorBoard: tensorboard --logdir ${SCRATCH_BASE}/logs/${EXPERIMENT}/"
