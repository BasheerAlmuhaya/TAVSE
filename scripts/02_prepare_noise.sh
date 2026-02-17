#!/bin/bash
#SBATCH --job-name=tavse-noise
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/scratch/users/40741008/tavse/logs/noise_%j.out
#SBATCH --error=/mnt/scratch/users/40741008/tavse/logs/noise_%j.err

# ─────────────────────────────────────────────────────────────
# TAVSE: Noise Corpus Preparation
#
# Downloads the DEMAND noise corpus, resamples to 16kHz, and
# stores on scratch for on-the-fly noise mixing during training.
#
# Usage:
#   sbatch scripts/02_prepare_noise.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

SCRATCH_BASE=/mnt/scratch/users/40741008/tavse
NOISE_DIR=${SCRATCH_BASE}/processed/noise
STAGING_DIR=${SCRATCH_BASE}/staging
PROJECT_DIR=~/my_projects/AVTSE/TAVSE

mkdir -p ${NOISE_DIR} ${STAGING_DIR}

# ── Storage check ─────────────────────────────────────────────
echo "=== Storage Check ==="
echo "Home: $(du -sh ~ 2>/dev/null | cut -f1) / 50 GB quota"
echo "Scratch: $(du -sh $SCRATCH_BASE 2>/dev/null | cut -f1) (no quota)"
echo "====================="

cd ${PROJECT_DIR}
source activate tavse 2>/dev/null || conda activate tavse 2>/dev/null || true

echo ""
echo "=================================================="
echo "TAVSE Noise Corpus Preparation"
echo "  Source: DEMAND dataset"
echo "  Output: ${NOISE_DIR}/"
echo "  Target SR: 16000 Hz"
echo "=================================================="
echo ""

# ── Download DEMAND noise corpus ─────────────────────────────
# DEMAND (Diverse Environments Multichannel Acoustic Noise Database)
# Each noise type directory contains ch01.wav (mono channel)
DEMAND_URL="https://zenodo.org/record/1227121/files/demand.zip"
DEMAND_ZIP=${STAGING_DIR}/demand.zip

if [ ! -f "${DEMAND_ZIP}" ] && [ -z "$(ls -A ${NOISE_DIR} 2>/dev/null)" ]; then
    echo "Downloading DEMAND noise corpus..."
    wget -q --show-progress -O ${DEMAND_ZIP} ${DEMAND_URL} || {
        echo "[INFO] Auto-download failed. Please download DEMAND manually:"
        echo "  1. Go to: https://zenodo.org/record/1227121"
        echo "  2. Download demand.zip"
        echo "  3. Place it in: ${STAGING_DIR}/"
        echo "  4. Re-run this script"
        echo ""
        echo "Alternative: manually place WAV files in ${NOISE_DIR}/"
        echo "  - Any 16kHz mono WAV files will be used as noise sources"
        echo "  - Longer files (>30s) are better for random cropping"
        exit 1
    }
fi

# ── Extract and resample ─────────────────────────────────────
if [ -f "${DEMAND_ZIP}" ]; then
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
fi

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "=== Noise Corpus Summary ==="
echo "Location: ${NOISE_DIR}/"
echo "Files: $(ls ${NOISE_DIR}/*.wav 2>/dev/null | wc -l) WAV files"
echo "Size: $(du -sh ${NOISE_DIR}/ 2>/dev/null | cut -f1)"
echo "==========================="
echo ""
echo "Noise preparation complete!"
