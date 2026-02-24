#!/bin/bash
# ─────────────────────────────────────────────────────────────
# TAVSE: Download DEMAND Noise Corpus (Login Node Only)
#
# HPC compute nodes typically have NO internet access.
# Run this script on the LOGIN node to pre-download the
# DEMAND noise corpus from Zenodo, then run the processing
# step (02b_prepare_noise.sh) via sbatch.
#
# The DEMAND dataset is stored as individual per-noise-type
# zip files on Zenodo (there is no single demand.zip).
# We download the 16kHz versions directly (no resampling needed).
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
STAGING_DIR="${SCRATCH_BASE}/staging/demand"
ZENODO_RECORD="1227121"
ZENODO_API="https://zenodo.org/api/records/${ZENODO_RECORD}/files"

mkdir -p "${STAGING_DIR}"

# ── DEMAND noise types ────────────────────────────────────────
# 18 noise environments from DEMAND corpus on Zenodo.
# 17 have native 16kHz versions; SCAFE only has 48kHz.
NOISE_FILES=(
    DKITCHEN_16k.zip
    DLIVING_16k.zip
    DWASHING_16k.zip
    NFIELD_16k.zip
    NPARK_16k.zip
    NRIVER_16k.zip
    OOFFICE_16k.zip
    OHALLWAY_16k.zip
    OMEETING_16k.zip
    PCAFETER_16k.zip
    PRESTO_16k.zip
    PSTATION_16k.zip
    SCAFE_48k.zip
    SPSQUARE_16k.zip
    STRAFFIC_16k.zip
    TBUS_16k.zip
    TCAR_16k.zip
    TMETRO_16k.zip
)
TOTAL_FILES=${#NOISE_FILES[@]}

# ── Check mode ────────────────────────────────────────────────
if [ "${1:-}" = "--check" ]; then
    echo "=== DEMAND Noise Download Status ==="
    downloaded=0
    missing=0
    for f in "${NOISE_FILES[@]}"; do
        if [ -f "${STAGING_DIR}/${f}" ] && [ -s "${STAGING_DIR}/${f}" ]; then
            downloaded=$((downloaded + 1))
        else
            missing=$((missing + 1))
            echo "  Missing: ${f}"
        fi
    done
    echo ""
    echo "  Downloaded: ${downloaded}/${TOTAL_FILES}"
    echo "  Missing:    ${missing}/${TOTAL_FILES}"
    echo "  Location:   ${STAGING_DIR}/"
    if [ "${downloaded}" -gt 0 ]; then
        echo "  Disk usage: $(du -sh "${STAGING_DIR}/" 2>/dev/null | cut -f1)"
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

# ── Download ──────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "TAVSE DEMAND Noise Corpus Download"
echo "  Source:  Zenodo record ${ZENODO_RECORD}"
echo "  Files:   ${TOTAL_FILES} noise environment zips (16kHz)"
echo "  Output:  ${STAGING_DIR}/"
echo "=================================================="
echo ""

FAILED=0
SKIPPED=0
DOWNLOADED=0

for zip_file in "${NOISE_FILES[@]}"; do
    zip_path="${STAGING_DIR}/${zip_file}"
    download_url="${ZENODO_API}/${zip_file}/content"

    # Skip if already downloaded
    if [ -f "${zip_path}" ] && [ -s "${zip_path}" ]; then
        echo "[Skip] ${zip_file}: already downloaded"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Remove any corrupt/empty file from previous attempt
    if [ -f "${zip_path}" ] && [ ! -s "${zip_path}" ]; then
        rm -f "${zip_path}"
    fi

    echo "[Download] ${zip_file} ..."
    if wget --progress=bar:force --timeout=120 --tries=3 \
            -O "${zip_path}" "${download_url}" 2>&1; then
        # Verify non-empty
        if [ -s "${zip_path}" ]; then
            DOWNLOADED=$((DOWNLOADED + 1))
        else
            echo "  [WARN] Downloaded file is empty, removing"
            rm -f "${zip_path}"
            FAILED=$((FAILED + 1))
        fi
    else
        rm -f "${zip_path}"
        echo "  [FAILED] ${zip_file}"
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
echo "  Disk usage: $(du -sh "${STAGING_DIR}/" 2>/dev/null | cut -f1)"
echo "========================"

if [ ${FAILED} -gt 0 ]; then
    echo ""
    echo "WARNING: ${FAILED} files failed to download."
    echo "Re-run this script to retry failed downloads."
    exit 1
fi

echo ""
echo "Download complete! Now submit the noise processing job:"
echo "  sbatch scripts/02b_prepare_noise.sh"
