#!/bin/bash
# ─────────────────────────────────────────────────────────────
# TAVSE: Noise Corpus Processing (Compute Node via sbatch)
#
# Extracts and resamples the pre-downloaded DEMAND noise corpus.
# Run the download step FIRST on the login node:
#   bash scripts/02a_download_noise.sh
#
# Usage:
#   sbatch scripts/02b_prepare_noise.sh
# ─────────────────────────────────────────────────────────────
#
# ── SLURM directives (MUST appear before any executable line) ─
# Override from command line:  sbatch --partition=short scripts/02b_prepare_noise.sh
#SBATCH --job-name=tavse-noise
#SBATCH --partition=nodes
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/noise_%j.out
#SBATCH --error=logs/noise_%j.err

# ── Resolve project directory ─────────────────────────────────
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

# ── Load .env ─────────────────────────────────────────────────
if [ -f "${PROJECT_DIR}/.env" ]; then
    set -a; source "${PROJECT_DIR}/.env"; set +a
else
    echo "ERROR: ${PROJECT_DIR}/.env not found. Copy .env.example to .env."; exit 1
fi

set -euo pipefail

SCRATCH_BASE="${TAVSE_DATA_ROOT:?Set TAVSE_DATA_ROOT in .env}"
NOISE_DIR="${SCRATCH_BASE}/processed/noise"
STAGING_DIR="${SCRATCH_BASE}/staging"
DEMAND_ZIP="${STAGING_DIR}/demand.zip"

mkdir -p "${NOISE_DIR}" "${STAGING_DIR}"

# ── Storage check ─────────────────────────────────────────────
echo "=== Storage Check ==="
echo "Home: $(du -sh ~ 2>/dev/null | cut -f1)"
echo "Data root: $(du -sh ${SCRATCH_BASE} 2>/dev/null | cut -f1)"
echo "====================="

cd "${PROJECT_DIR}"

# Activate conda (non-interactive SLURM shells need explicit init)
CONDA_SH="${CONDA_EXE:?Set CONDA_EXE in .env (run: echo \$CONDA_EXE)}"
source "${CONDA_SH%/*}/../etc/profile.d/conda.sh"
conda activate "${TAVSE_CONDA_ENV:-tavse}"

echo ""
echo "=================================================="
echo "TAVSE Noise Corpus Processing"
echo "  Source: ${DEMAND_ZIP}"
echo "  Output: ${NOISE_DIR}/"
echo "  Target SR: 16000 Hz"
echo "=================================================="
echo ""

# ── Check if already processed ────────────────────────────────
if [ -n "$(ls -A ${NOISE_DIR} 2>/dev/null)" ]; then
    echo "Noise directory already contains files. Skipping processing."
    echo "Files: $(ls ${NOISE_DIR}/*.wav 2>/dev/null | wc -l) WAV files"
    echo "Size:  $(du -sh ${NOISE_DIR}/ 2>/dev/null | cut -f1)"
    echo ""
    echo "To reprocess, remove ${NOISE_DIR}/ and re-run."
    exit 0
fi

# ── Verify the zip exists and is valid ────────────────────────
if [ ! -f "${DEMAND_ZIP}" ]; then
    echo "ERROR: ${DEMAND_ZIP} not found."
    echo ""
    echo "You must download the DEMAND corpus first on the login node:"
    echo "  bash scripts/02a_download_noise.sh"
    exit 1
fi

if [ ! -s "${DEMAND_ZIP}" ]; then
    echo "ERROR: ${DEMAND_ZIP} is empty (0 bytes)."
    echo "Delete it and re-download on the login node:"
    echo "  rm -f ${DEMAND_ZIP}"
    echo "  bash scripts/02a_download_noise.sh"
    exit 1
fi

# ── Extract and resample ─────────────────────────────────────
echo "Extracting DEMAND corpus..."
DEMAND_EXTRACT=${STAGING_DIR}/demand
mkdir -p ${DEMAND_EXTRACT}
unzip -qo ${DEMAND_ZIP} -d ${DEMAND_EXTRACT}

echo "Resampling noise files to 16kHz..."
python3 -c "
import os
import torchaudio
from pathlib import Path
from tqdm import tqdm

src_dir = Path('${DEMAND_EXTRACT}')
dst_dir = Path('${NOISE_DIR}')
target_sr = 16000
count = 0

# Find all WAV files (DEMAND stores as ch01.wav in subfolders)
wav_files = list(src_dir.rglob('*.wav'))
print(f'Found {len(wav_files)} WAV files')

for wav_path in tqdm(wav_files, desc='Resampling'):
    try:
        waveform, sr = torchaudio.load(str(wav_path))
        # Take first channel only (mono)
        waveform = waveform[0:1]

        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)

        # Name: noiseType_channel.wav
        parent = wav_path.parent.name
        out_name = f'{parent}_{wav_path.stem}.wav'
        out_path = dst_dir / out_name
        torchaudio.save(str(out_path), waveform, target_sr)
        count += 1
    except Exception as e:
        print(f'  [WARN] Skipping {wav_path}: {e}')

print(f'\nResampled {count} noise files to {dst_dir}/')
"

# Cleanup
echo "Cleaning up staging..."
rm -rf ${DEMAND_ZIP} ${STAGING_DIR}/demand

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "=== Noise Corpus Summary ==="
echo "Location: ${NOISE_DIR}/"
echo "Files: $(ls ${NOISE_DIR}/*.wav 2>/dev/null | wc -l) WAV files"
echo "Size: $(du -sh ${NOISE_DIR}/ 2>/dev/null | cut -f1)"
echo "==========================="
echo ""
echo "Noise preparation complete!"
