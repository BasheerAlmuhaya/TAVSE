#!/bin/bash
# ─────────────────────────────────────────────────────────────
# TAVSE: Noise Corpus Processing (Compute Node via sbatch)
#
# Extracts the pre-downloaded DEMAND noise corpus zip files
# and prepares mono 16kHz WAV files for on-the-fly mixing.
#
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
STAGING_DIR="${SCRATCH_BASE}/staging/demand"

mkdir -p "${NOISE_DIR}"

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
echo "  Source: ${STAGING_DIR}/"
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

# ── Verify staging zips exist ─────────────────────────────────
ZIP_COUNT=$(ls "${STAGING_DIR}"/*.zip 2>/dev/null | wc -l)
if [ "${ZIP_COUNT}" -eq 0 ]; then
    echo "ERROR: No zip files found in ${STAGING_DIR}/"
    echo ""
    echo "You must download the DEMAND corpus first on the login node:"
    echo "  bash scripts/02a_download_noise.sh"
    exit 1
fi
echo "Found ${ZIP_COUNT} noise zip files in staging."

# ── Extract and process ──────────────────────────────────────
TEMP_EXTRACT="${SCRATCH_BASE}/staging/demand_extract"
mkdir -p "${TEMP_EXTRACT}"

echo "Extracting and processing noise files..."
python3 -c "
import os
import torchaudio
from pathlib import Path
from tqdm import tqdm
import zipfile

staging_dir = Path('${STAGING_DIR}')
extract_dir = Path('${TEMP_EXTRACT}')
dst_dir = Path('${NOISE_DIR}')
target_sr = 16000
count = 0
errors = 0

# Process each zip file
zip_files = sorted(staging_dir.glob('*.zip'))
print(f'Processing {len(zip_files)} noise zip files...')
print()

for zf_path in zip_files:
    noise_type = zf_path.stem  # e.g. DKITCHEN_16k or SCAFE_48k
    needs_resample = '48k' in noise_type
    base_name = noise_type.replace('_16k', '').replace('_48k', '')

    print(f'  {noise_type}...')

    # Extract
    try:
        with zipfile.ZipFile(str(zf_path), 'r') as zf:
            zf.extractall(str(extract_dir))
    except zipfile.BadZipFile:
        print(f'    [ERROR] Bad zip file: {zf_path.name}')
        errors += 1
        continue

    # Find extracted WAV files for this noise type
    # DEMAND stores multi-channel WAVs as ch01.wav .. ch16.wav in subfolders
    wav_files = list(extract_dir.rglob('*.wav'))
    if not wav_files:
        print(f'    [WARN] No WAV files found')
        continue

    for wav_path in wav_files:
        try:
            waveform, sr = torchaudio.load(str(wav_path))
            # Take first channel only (mono)
            waveform = waveform[0:1]

            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)

            # Name: NOISEENV_channel.wav  (e.g. DKITCHEN_ch01.wav)
            parent = wav_path.parent.name
            out_name = f'{parent}_{wav_path.stem}.wav'
            out_path = dst_dir / out_name
            torchaudio.save(str(out_path), waveform, target_sr)
            count += 1
        except Exception as e:
            print(f'    [WARN] Skipping {wav_path.name}: {e}')
            errors += 1

    # Clean up extracted files for this zip
    import shutil
    for item in extract_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

print(f'')
print(f'Processed {count} noise files to {dst_dir}/')
if errors > 0:
    print(f'Warnings/errors: {errors}')
"

# Cleanup
echo ""
echo "Cleaning up staging..."
rm -rf "${TEMP_EXTRACT}"
rm -rf "${STAGING_DIR}"

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "=== Noise Corpus Summary ==="
echo "Location: ${NOISE_DIR}/"
echo "Files: $(ls ${NOISE_DIR}/*.wav 2>/dev/null | wc -l) WAV files"
echo "Size: $(du -sh ${NOISE_DIR}/ 2>/dev/null | cut -f1)"
echo "==========================="
echo ""
echo "Noise preparation complete!"
