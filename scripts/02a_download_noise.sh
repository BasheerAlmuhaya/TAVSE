#!/bin/bash
# ─────────────────────────────────────────────────────────────
# TAVSE: Download DEMAND Noise Corpus (Login Node Only)
#
# HPC compute nodes typically have NO internet access.
# Run this script on the LOGIN node to pre-download the
# DEMAND noise corpus, then run the processing step
# (02b_prepare_noise.sh) via sbatch.
#
# Usage:
#   bash scripts/02a_download_noise.sh
#   bash scripts/02a_download_noise.sh --check
#
# After this completes, submit the processing job:
#   sbatch scripts/02b_prepare_noise.sh
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

# ── Paths ─────────────────────────────────────────────────────
SCRATCH_BASE="${TAVSE_DATA_ROOT:?Set TAVSE_DATA_ROOT in .env}"
STAGING_DIR="${SCRATCH_BASE}/staging"
DEMAND_URL="https://zenodo.org/record/1227121/files/demand.zip"
DEMAND_ZIP="${STAGING_DIR}/demand.zip"

mkdir -p "${STAGING_DIR}"

# ── Check mode ────────────────────────────────────────────────
if [ "${1:-}" = "--check" ]; then
    echo "=== DEMAND Noise Download Status ==="
    if [ -f "${DEMAND_ZIP}" ] && [ -s "${DEMAND_ZIP}" ]; then
        echo "  Status:   Downloaded"
        echo "  File:     ${DEMAND_ZIP}"
        echo "  Size:     $(du -h "${DEMAND_ZIP}" | cut -f1)"
    elif [ -f "${DEMAND_ZIP}" ]; then
        echo "  Status:   CORRUPT (0-byte file)"
        echo "  Action:   Re-run this script to re-download"
    else
        echo "  Status:   Not downloaded"
    fi
    echo "====================================="
    exit 0
fi

# ── Verify internet access ───────────────────────────────────
echo "Checking internet access..."
if ! curl -sI --connect-timeout 10 https://zenodo.org >/dev/null 2>&1; then
    echo "ERROR: Cannot reach zenodo.org"
    echo "This script must be run on the LOGIN node (not via sbatch)."
    echo ""
    echo "If you are on the login node, check your network/proxy settings."
    exit 1
fi
echo "  OK — zenodo.org reachable"

# ── Clean up any corrupt file from previous attempt ───────────
if [ -f "${DEMAND_ZIP}" ] && [ ! -s "${DEMAND_ZIP}" ]; then
    echo "[WARN] Removing empty/corrupt ${DEMAND_ZIP} from previous attempt"
    rm -f "${DEMAND_ZIP}"
fi

# ── Download ──────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "TAVSE Noise Corpus Download"
echo "  Source:  DEMAND (Zenodo)"
echo "  Output:  ${DEMAND_ZIP}"
echo "=================================================="
echo ""

if [ -f "${DEMAND_ZIP}" ] && [ -s "${DEMAND_ZIP}" ]; then
    echo "[Skip] DEMAND corpus already downloaded"
    echo "  File: ${DEMAND_ZIP}"
    echo "  Size: $(du -h "${DEMAND_ZIP}" | cut -f1)"
else
    echo "Downloading DEMAND noise corpus (~1.6 GB)..."
    wget --progress=bar:force --timeout=120 --tries=5 \
         -O "${DEMAND_ZIP}" "${DEMAND_URL}" || {
        rm -f "${DEMAND_ZIP}"
        echo ""
        echo "[ERROR] Download failed."
        echo "  You can manually download from: ${DEMAND_URL}"
        echo "  Place the file at: ${DEMAND_ZIP}"
        exit 1
    }

    # Verify downloaded file
    if [ ! -s "${DEMAND_ZIP}" ]; then
        rm -f "${DEMAND_ZIP}"
        echo "[ERROR] Downloaded file is empty. Network issue or invalid URL."
        exit 1
    fi
fi

# ── Verify zip integrity ─────────────────────────────────────
echo ""
echo "Verifying zip integrity..."
if unzip -tq "${DEMAND_ZIP}" >/dev/null 2>&1; then
    echo "  OK — zip file is valid"
else
    echo "[ERROR] Zip file is corrupt. Removing and please re-download."
    rm -f "${DEMAND_ZIP}"
    exit 1
fi

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "=== Download Summary ==="
echo "  File:      ${DEMAND_ZIP}"
echo "  Size:      $(du -h "${DEMAND_ZIP}" | cut -f1)"
echo "  Status:    OK"
echo "========================"
echo ""
echo "Download complete! Now submit the noise processing job:"
echo "  sbatch scripts/02b_prepare_noise.sh"
