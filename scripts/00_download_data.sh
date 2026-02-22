#!/bin/bash
# ─────────────────────────────────────────────────────────────
# TAVSE: Download Dataset (Login Node Only)
#
# HPC compute nodes typically have NO internet access.
# Run this script on the LOGIN node to pre-download all
# SpeakingFaces zips from HuggingFace, then run the
# processing step (01_ingest_dataset.sh) via sbatch.
#
# Usage:
#   bash scripts/00_download_data.sh              # All 142 subjects
#   bash scripts/00_download_data.sh 1 50         # Subjects 1-50
#   bash scripts/00_download_data.sh --check      # Check what's downloaded
#
# Prerequisites:
#   - HuggingFace account with access to ISSAI/SpeakingFaces
#   - huggingface-cli login  (or set HF_TOKEN in .env)
#
# After this completes, submit the processing job:
#   sbatch scripts/01_ingest_dataset.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Resolve project directory ─────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Load .env ─────────────────────────────────────────────────
if [ -f "${PROJECT_DIR}/.env" ]; then
    set -a; source "${PROJECT_DIR}/.env"; set +a
else
    echo "ERROR: ${PROJECT_DIR}/.env not found. Copy .env.example to .env."
    exit 1
fi

# ── Activate conda ────────────────────────────────────────────
CONDA_SH="${CONDA_EXE:?Set CONDA_EXE in .env (run: echo \$CONDA_EXE)}"
source "${CONDA_SH%/*}/../etc/profile.d/conda.sh"
conda activate "${TAVSE_CONDA_ENV:-tavse}"

# ── Paths ─────────────────────────────────────────────────────
SCRATCH_BASE="${TAVSE_DATA_ROOT:?Set TAVSE_DATA_ROOT in .env}"
STAGING_DIR="${SCRATCH_BASE}/staging"
HF_CACHE="${HF_HOME:-${SCRATCH_BASE}/.hf_cache}"

export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"

mkdir -p "${STAGING_DIR}"

# ── Parse arguments ───────────────────────────────────────────
START_SUB=1
END_SUB=142
CHECK_ONLY=false

for arg in "$@"; do
    if [ "$arg" = "--check" ]; then
        CHECK_ONLY=true
    fi
done

if [ "$CHECK_ONLY" = false ] && [ $# -ge 2 ]; then
    START_SUB=${1}
    END_SUB=${2}
fi

# ── Check mode: report download status ────────────────────────
if [ "$CHECK_ONLY" = true ]; then
    echo "=== Download Status ==="
    downloaded=0
    missing=0
    for i in $(seq 1 142); do
        f="${STAGING_DIR}/sub_${i}_ia.zip"
        if [ -f "$f" ]; then
            downloaded=$((downloaded + 1))
        else
            missing=$((missing + 1))
        fi
    done
    echo "  Downloaded: ${downloaded}/142"
    echo "  Missing:    ${missing}/142"
    echo "  Location:   ${STAGING_DIR}/"
    if [ "$downloaded" -gt 0 ]; then
        echo "  Disk usage: $(du -sh ${STAGING_DIR}/ 2>/dev/null | cut -f1)"
    fi
    echo "======================="
    exit 0
fi

# ── Verify internet access ───────────────────────────────────
echo "Checking internet access..."
if ! curl -sI --connect-timeout 10 https://huggingface.co >/dev/null 2>&1; then
    echo "ERROR: Cannot reach huggingface.co"
    echo "This script must be run on the LOGIN node (not via sbatch)."
    echo ""
    echo "If you are on the login node, check your network/proxy settings."
    exit 1
fi
echo "  OK — huggingface.co reachable"

# ── Verify HuggingFace auth ──────────────────────────────────
echo "Checking HuggingFace authentication..."

# Support HF_TOKEN from .env or environment
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
    echo "  Using HF_TOKEN from environment/.env"
fi

HF_AUTH_OK=false
if python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    HF_AUTH_OK=true
    echo "  OK — logged in to HuggingFace"
fi

if [ "$HF_AUTH_OK" = false ]; then
    echo ""
    echo "Not logged in to HuggingFace."
    echo "The ISSAI/SpeakingFaces dataset requires authentication."
    echo ""
    echo "Options:"
    echo "  1) Run: huggingface-cli login"
    echo "  2) Set HF_TOKEN=hf_... in your .env file"
    echo "     (Get your token at https://huggingface.co/settings/tokens)"
    echo ""
    read -p "Would you like to log in now? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "Cannot proceed without HuggingFace authentication."
        exit 1
    fi
    huggingface-cli login
    # Verify login succeeded
    if ! python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
        echo "ERROR: Login failed. Please try again."
        exit 1
    fi
    echo "  OK — logged in to HuggingFace"
fi

# ── Download ──────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "TAVSE Dataset Download"
echo "  Subjects:  ${START_SUB} to ${END_SUB}"
echo "  Staging:   ${STAGING_DIR}/"
echo "  HF Cache:  ${HF_CACHE}/"
echo "=================================================="
echo ""

FAILED=0
SKIPPED=0
DOWNLOADED=0

for sub_id in $(seq ${START_SUB} ${END_SUB}); do
    zip_file="sub_${sub_id}_ia.zip"
    zip_path="${STAGING_DIR}/${zip_file}"

    if [ -f "${zip_path}" ]; then
        echo "[Skip] Subject ${sub_id}: already downloaded"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "[Download] Subject ${sub_id}/${END_SUB}..."
    if python -c "
from huggingface_hub import hf_hub_download
import os
hf_hub_download(
    repo_id='ISSAI/SpeakingFaces',
    filename='${zip_file}',
    repo_type='dataset',
    cache_dir='${HF_CACHE}',
    local_dir='${STAGING_DIR}',
)
print('  OK')
" 2>&1; then
        DOWNLOADED=$((DOWNLOADED + 1))
    else
        echo "  [FAILED] Subject ${sub_id}"
        FAILED=$((FAILED + 1))
    fi
done

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "=== Download Summary ==="
echo "  Downloaded: ${DOWNLOADED}"
echo "  Skipped:    ${SKIPPED} (already present)"
echo "  Failed:     ${FAILED}"
echo "  Location:   ${STAGING_DIR}/"
echo "  Disk usage: $(du -sh ${STAGING_DIR}/ 2>/dev/null | cut -f1)"
echo "========================"

if [ ${FAILED} -gt 0 ]; then
    echo ""
    echo "WARNING: ${FAILED} subjects failed to download."
    echo "Re-run this script to retry failed subjects."
    exit 1
fi

echo ""
echo "Download complete! Now submit the processing job:"
echo "  sbatch scripts/01_ingest_dataset.sh"
