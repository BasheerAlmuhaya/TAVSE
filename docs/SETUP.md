# TAVSE Setup Guide

This guide covers environment setup, data preparation, and project configuration for the TAVSE (Thermal Audio-Visual Speech Enhancement) project. It is written for HPC clusters with SLURM but also works on local machines.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Environment Setup](#environment-setup)
- [Storage Architecture](#storage-architecture)
- [Data Preparation](#data-preparation)
  - [1. Initialize Data Directory](#1-initialize-data-directory)
  - [2. Clear Home Quota (HPC only)](#2-clear-home-quota-hpc-only)
  - [3. Ingest SpeakingFaces Dataset](#3-ingest-speakingfaces-dataset)
  - [4. Prepare Noise Corpus](#4-prepare-noise-corpus)
- [Verification](#verification)

---

## Prerequisites

- SLURM job scheduler (for HPC) **or** a local machine with a GPU
- HuggingFace account (for downloading ISSAI/SpeakingFaces dataset)
- `conda` or `mamba` available

## Environment Configuration

All user-specific paths are set through a `.env` file at the project root.
Scripts automatically source this file, so **no hardcoded paths exist in the codebase**.

```bash
# From the project root
cp .env.example .env
```

Open `.env` and set the two required paths:

```dotenv
# Root directory for ALL TAVSE data (LMDBs, audio, noise, checkpoints, logs).
# Should be on a high-capacity, fast filesystem (scratch, local SSD, etc.).
TAVSE_DATA_ROOT=/path/to/your/tavse/data

# Root directory for the TAVSE source code (the directory containing .env).
TAVSE_PROJECT_DIR=/path/to/TAVSE
```

See `.env.example` for the full list of optional variables (SLURM partitions, GPU flags, etc.).

## Environment Setup

### 1. Create Conda Environment

```bash
# On the login node (or locally)
conda create -n tavse python=3.10 -y
conda activate tavse

# Install PyTorch (adjust CUDA version for your system)
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
cd "$TAVSE_PROJECT_DIR"
pip install -r requirements.txt
```

### 2. Configure HuggingFace

The ISSAI/SpeakingFaces dataset is gated and requires authentication:

```bash
# Login to HuggingFace (one-time setup)
huggingface-cli login

# Alternatively, set HF_TOKEN in .env:
# Get your token at https://huggingface.co/settings/tokens
# Then add to .env:  HF_TOKEN=hf_...
```

> **Important:** Also accept the dataset terms at https://huggingface.co/datasets/ISSAI/SpeakingFaces
>
> The HF cache is automatically redirected to `$TAVSE_DATA_ROOT/.hf_cache` via `.env`, keeping it off your home directory.

### 3. Set PYTHONPATH

```bash
echo "export PYTHONPATH=\$TAVSE_PROJECT_DIR:\${PYTHONPATH:-}" >> ~/.bashrc
source ~/.bashrc
```

---

## Storage Architecture

TAVSE uses a **three-tier storage strategy** to work within typical HPC constraints:

| Tier | Path | Typical Capacity | Contents |
|------|------|----------|----------|
| **Home** (NFS) | `$TAVSE_PROJECT_DIR` | ~50 GB (quota-limited) | Code, configs, conda env |
| **Scratch** (fast filesystem) | `$TAVSE_DATA_ROOT` | Large, often no quota | ALL data, checkpoints, logs |
| **Local /tmp** | `/tmp/tavse_data/` | Varies (~100–200 GB) | Optional: LMDB cache for faster I/O |

> **Key rule:** Never write data, checkpoints, or logs to your home directory. Everything goes to `$TAVSE_DATA_ROOT`.

### Data Directory Structure

```
$TAVSE_DATA_ROOT/
├── .hf_cache/                    # HuggingFace downloads
├── staging/                      # Temporary zip downloads
├── processed/
│   ├── rgb_mouth.lmdb/          # RGB mouth ROIs (~6.2 GB)
│   ├── thermal_mouth.lmdb/      # Thermal mouth ROIs (~3.1 GB)
│   ├── audio_16k/               # Resampled WAVs (~1.8 GB)
│   ├── noise/                   # DEMAND noise corpus (~2 GB)
│   └── metadata/
│       ├── train_manifest.csv
│       ├── val_manifest.csv
│       ├── test_manifest.csv
│       └── ingest_state.json    # Resume tracking
├── checkpoints/                 # Model checkpoints
│   ├── audio_only/
│   ├── audio_rgb/
│   ├── audio_thermal/
│   └── audio_rgb_thermal/
└── logs/                        # TensorBoard + SLURM logs
```

---

## Data Preparation

> **Note:** All scripts below auto-source the `.env` file from the project root,
> so `$TAVSE_DATA_ROOT` and `$TAVSE_PROJECT_DIR` are available inside every script.

### 1. Initialize Data Directory

```bash
# Source your .env (scripts do this automatically, but useful interactively)
source "$(dirname "$0")/../.env" 2>/dev/null || source .env

mkdir -p "${TAVSE_DATA_ROOT}"/{staging,.hf_cache}
mkdir -p "${TAVSE_DATA_ROOT}"/processed/{audio_16k,noise,metadata}
mkdir -p "${TAVSE_DATA_ROOT}"/{checkpoints,logs}

# Verify write access
dd if=/dev/zero of="${TAVSE_DATA_ROOT}/write_test" bs=1M count=10 oflag=direct 2>/dev/null && \
    rm "${TAVSE_DATA_ROOT}/write_test" && echo "Data root write OK"
```

### 2. Clear Home Quota (HPC only)

If your home directory is near quota:

```bash
# Check current usage
quota -s

# Remove HuggingFace cache from home (IMPORTANT!)
rm -rf ~/.cache/huggingface/hub/ ~/.cache/huggingface/xet/

# Verify space reclaimed
du -sh ~
```

### 3. Ingest SpeakingFaces Dataset

The ingestion pipeline extracts 96×96 mouth ROIs using MediaPipe face detection, builds LMDB databases, resamples audio to 16kHz, and generates train/val/test manifests.

> **Two-step process:** HPC compute nodes typically have **no internet access**.
> You must download data on the login node first, then process on compute nodes.

**Step A: Download from HuggingFace** (run on login node, has internet)

```bash
cd "$TAVSE_PROJECT_DIR"

# Download all 142 subjects (~900 GB total, ~6.4 GB per subject)
bash scripts/00_download_data.sh

# Or download a smaller subset for testing
bash scripts/00_download_data.sh 1 5

# Check download status
bash scripts/00_download_data.sh --check
```

The download script:
- Verifies internet access and HuggingFace authentication
- Downloads `sub_{id}_ia.zip` to `$TAVSE_DATA_ROOT/staging/`
- Skips already-downloaded subjects (safe to re-run)
- Guides you through HF login if needed

**Step B: Process on compute node** (submit via sbatch)

```bash
# Process all downloaded subjects
sbatch scripts/01_ingest_dataset.sh

# Process a subset
sbatch scripts/01_ingest_dataset.sh 1 5

# Resume interrupted processing
sbatch scripts/01_ingest_dataset.sh --resume
```

**What processing does:**
1. Opens each `sub_{id}_ia.zip` from `$TAVSE_DATA_ROOT/staging/`
2. Detects face landmarks on RGB frames → computes mouth bounding box
3. Crops 96×96 mouth ROIs for both RGB and thermal frames
4. Writes ROIs to LMDB (JPEG for RGB, PNG for thermal)
5. Resamples audio from 44.1kHz → 16kHz
6. Deletes the zip after processing to reclaim space
7. Saves progress to `ingest_state.json` for resume support

**Speaker split:**
- Train: subjects 1-100 (100 speakers)
- Validation: subjects 101-114 (14 speakers)
- Test: subjects 115-142 (28 speakers)

### 4. Prepare Noise Corpus

Downloads and resamples the DEMAND noise corpus:

```bash
sbatch scripts/02_prepare_noise.sh
```

If automatic download fails, manually download from [Zenodo](https://zenodo.org/record/1227121) and place the zip in `$TAVSE_DATA_ROOT/staging/`.

---

## Verification

After ingestion, verify the processed data:

```bash
source .env

# Check LMDB sizes
du -sh "$TAVSE_DATA_ROOT/processed/rgb_mouth.lmdb/"
du -sh "$TAVSE_DATA_ROOT/processed/thermal_mouth.lmdb/"

# Check manifests
for f in "$TAVSE_DATA_ROOT"/processed/metadata/*_manifest.csv; do
    echo "$(basename "$f"): $(wc -l < "$f") entries"
done

# Check audio files
ls "$TAVSE_DATA_ROOT/processed/audio_16k/" | wc -l

# Quick sanity check: verify LMDB reads work
python3 -c "
import os, lmdb
data_root = os.environ['TAVSE_DATA_ROOT']
env = lmdb.open(os.path.join(data_root, 'processed/rgb_mouth.lmdb'), readonly=True)
with env.begin() as txn:
    cursor = txn.cursor()
    cursor.first()
    key, val = cursor.item()
    print(f'First key: {key.decode()}, value size: {len(val)} bytes')
    print(f'Total entries: {env.stat()[\"entries\"]}')
env.close()
"

# Verify noise files
ls "$TAVSE_DATA_ROOT/processed/noise/"*.wav | wc -l
```

---

## Troubleshooting

### SSL handshake timeout / download fails on compute node
Compute nodes on most HPC clusters have no outbound internet access. Always run downloads on the **login node**:
```bash
bash scripts/00_download_data.sh    # on login node
sbatch scripts/01_ingest_dataset.sh # processing happens on compute node
```

### HuggingFace authentication error
```bash
huggingface-cli login
# Or set HF_TOKEN in .env
```
Also ensure you've accepted the dataset terms at https://huggingface.co/datasets/ISSAI/SpeakingFaces.

### Home quota exceeded
```bash
quota -s                    # Check current usage
du -sh ~/.cache/huggingface # Should be empty (redirected via HF_HOME)
du -sh ~/.conda             # Conda env (~5 GB)
du -sh ~/.vscode-server     # VS Code (~2.6 GB)
```

### Ingestion crashes mid-way
The pipeline saves state to `ingest_state.json`. Just re-run with `--resume`:
```bash
sbatch scripts/01_ingest_dataset.sh --resume
```

### No face detected for some subjects
Face detection may fail for extreme head positions (positions 7-9). These are logged as warnings and skipped. The model trains on available data.

### LMDB "map full" error
Increase map size in the ingestion script:
```bash
sbatch scripts/01_ingest_dataset.sh --rgb-map-size 20000000000
```
