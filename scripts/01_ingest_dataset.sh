#!/bin/bash
#SBATCH --job-name=tavse-ingest
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/scratch/users/40741008/tavse/logs/ingest_%j.out
#SBATCH --error=/mnt/scratch/users/40741008/tavse/logs/ingest_%j.err

# ─────────────────────────────────────────────────────────────
# TAVSE: Dataset Ingestion Pipeline
#
# Downloads SpeakingFaces subjects from HuggingFace, extracts
# mouth ROIs, builds LMDB databases, and generates manifests.
#
# Usage:
#   sbatch scripts/01_ingest_dataset.sh              # All 142 subjects
#   sbatch scripts/01_ingest_dataset.sh 1 50         # Subjects 1-50
#   sbatch scripts/01_ingest_dataset.sh --resume     # Resume interrupted
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────
SCRATCH_BASE=/mnt/scratch/users/40741008/tavse
PROJECT_DIR=~/my_projects/AVTSE/TAVSE

# ── Redirect HuggingFace cache to scratch ─────────────────────
export HF_HOME=${SCRATCH_BASE}/.hf_cache
export TRANSFORMERS_CACHE=${HF_HOME}

# ── Create directories ────────────────────────────────────────
mkdir -p ${SCRATCH_BASE}/{staging,processed/{rgb_mouth.lmdb,thermal_mouth.lmdb,audio_16k,noise,metadata},logs}

# ── Storage check ─────────────────────────────────────────────
echo "=== Storage Check ==="
echo "Home: $(du -sh ~ 2>/dev/null | cut -f1) / 50 GB quota"
quota -s 2>/dev/null | tail -1 || true
echo "Scratch: $(du -sh $SCRATCH_BASE 2>/dev/null | cut -f1) (no quota)"
echo "====================="

# ── Load environment ──────────────────────────────────────────
cd ${PROJECT_DIR}

# Activate conda environment (adjust name as needed)
source activate tavse 2>/dev/null || conda activate tavse 2>/dev/null || true

# ── Parse arguments ───────────────────────────────────────────
START_SUB=${1:-1}
END_SUB=${2:-142}
RESUME_FLAG=""

# Check if --resume was passed
for arg in "$@"; do
    if [ "$arg" = "--resume" ]; then
        RESUME_FLAG="--resume"
    fi
done

echo ""
echo "=================================================="
echo "TAVSE Dataset Ingestion"
echo "  Subjects: ${START_SUB} to ${END_SUB}"
echo "  Resume: ${RESUME_FLAG:-no}"
echo "  HF_HOME: ${HF_HOME}"
echo "  Output: ${SCRATCH_BASE}/processed/"
echo "=================================================="
echo ""

# ── Run ingestion ─────────────────────────────────────────────
python -m src.data.ingest_pipeline \
    --subjects ${START_SUB} ${END_SUB} \
    --rgb-map-size 15000000000 \
    --thr-map-size 8000000000 \
    ${RESUME_FLAG}

# ── Final storage check ──────────────────────────────────────
echo ""
echo "=== Final Storage Check ==="
echo "Scratch processed: $(du -sh ${SCRATCH_BASE}/processed/ 2>/dev/null | cut -f1)"
echo "  RGB LMDB:    $(du -sh ${SCRATCH_BASE}/processed/rgb_mouth.lmdb/ 2>/dev/null | cut -f1)"
echo "  THR LMDB:    $(du -sh ${SCRATCH_BASE}/processed/thermal_mouth.lmdb/ 2>/dev/null | cut -f1)"
echo "  Audio 16k:   $(du -sh ${SCRATCH_BASE}/processed/audio_16k/ 2>/dev/null | cut -f1)"
echo "  Manifests:"
for f in ${SCRATCH_BASE}/processed/metadata/*_manifest.csv; do
    [ -f "$f" ] && echo "    $(basename $f): $(wc -l < $f) lines"
done
echo "==========================="

echo ""
echo "Ingestion complete!"
