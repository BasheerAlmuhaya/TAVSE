#!/bin/bash
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
#
# ── SLURM directives (MUST appear before any executable line) ─
# Override from command line:  sbatch --partition=long --time=48:00:00 scripts/01_ingest_dataset.sh
#SBATCH --job-name=tavse-ingest
#SBATCH --partition=nodes
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/ingest_%j.out
#SBATCH --error=logs/ingest_%j.err

# ── Resolve project directory ─────────────────────────────────
# SLURM copies scripts to /var/spool/slurmd/, so BASH_SOURCE won't work.
# Use SLURM_SUBMIT_DIR (set to cwd where sbatch was called) when available.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

# ── Load .env (user-specific paths) ──────────────────────────
if [ -f "${PROJECT_DIR}/.env" ]; then
    set -a
    source "${PROJECT_DIR}/.env"
    set +a
else
    echo "ERROR: ${PROJECT_DIR}/.env not found."
    echo "Copy .env.example to .env and configure your paths."
    exit 1
fi

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────
SCRATCH_BASE="${TAVSE_DATA_ROOT:?Set TAVSE_DATA_ROOT in .env}"

# ── Redirect HuggingFace cache to data root ───────────────────
export HF_HOME="${HF_HOME:-${SCRATCH_BASE}/.hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}"

# ── Create directories ────────────────────────────────────────
mkdir -p "${SCRATCH_BASE}"/{staging,processed/{rgb_mouth.lmdb,thermal_mouth.lmdb,audio_16k,noise,metadata},logs}

# ── Storage check ─────────────────────────────────────────────
echo "=== Storage Check ==="
echo "Home: $(du -sh ~ 2>/dev/null | cut -f1)"
quota -s 2>/dev/null | tail -1 || true
echo "Data root: $(du -sh ${SCRATCH_BASE} 2>/dev/null | cut -f1)"
echo "====================="

# ── Load environment ──────────────────────────────────────────
cd "${PROJECT_DIR}"

# Activate conda (non-interactive SLURM shells need explicit init)
CONDA_SH="${CONDA_EXE:?Set CONDA_EXE in .env (run: echo \$CONDA_EXE)}"
source "${CONDA_SH%/*}/../etc/profile.d/conda.sh"
conda activate "${TAVSE_CONDA_ENV:-tavse}"

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
