# TAVSE Setup Guide

This guide covers environment setup, data preparation, and project configuration for the TAVSE (Thermal Audio-Visual Speech Enhancement) project on the ENUCC cluster.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Storage Architecture](#storage-architecture)
- [Data Preparation](#data-preparation)
  - [1. Initialize Scratch Space](#1-initialize-scratch-space)
  - [2. Clear Home Quota](#2-clear-home-quota)
  - [3. Ingest SpeakingFaces Dataset](#3-ingest-speakingfaces-dataset)
  - [4. Prepare Noise Corpus](#4-prepare-noise-corpus)
- [Verification](#verification)

---

## Prerequisites

- Access to the ENUCC cluster with SLURM job scheduler
- HuggingFace account (for downloading ISSAI/SpeakingFaces dataset)
- `conda` or `mamba` available on the cluster

## Environment Setup

### 1. Create Conda Environment

```bash
# On the login node
conda create -n tavse python=3.10 -y
conda activate tavse

# Install PyTorch (adjust CUDA version for your cluster)
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
cd ~/my_projects/AVTSE/TAVSE
pip install -r requirements.txt
```

### 2. Configure HuggingFace

```bash
# Login to HuggingFace (needed for ISSAI/SpeakingFaces access)
huggingface-cli login

# CRITICAL: Redirect HF cache to scratch (not home!)
echo 'export HF_HOME=/mnt/scratch/users/40741008/tavse/.hf_cache' >> ~/.bashrc
source ~/.bashrc
```

### 3. Set PYTHONPATH

```bash
echo 'export PYTHONPATH=~/my_projects/AVTSE/TAVSE:${PYTHONPATH:-}' >> ~/.bashrc
source ~/.bashrc
```

---

## Storage Architecture

TAVSE uses a **three-tier storage strategy** to work within the cluster's constraints:

| Tier | Path | Capacity | Contents |
|------|------|----------|----------|
| **Home** (NFS) | `~/my_projects/AVTSE/TAVSE/` | 50 GB quota | Code, configs, conda env |
| **Scratch** (Lustre) | `/mnt/scratch/users/40741008/tavse/` | ~67 TB free, no quota | ALL data, checkpoints, logs |
| **Local /tmp** | `/tmp/tavse_data/` | ~175 GB | Optional: LMDB cache for faster I/O |

**Key rule:** Never write data, checkpoints, or logs to home. Everything goes to scratch.

### Scratch Directory Structure

```
/mnt/scratch/users/40741008/tavse/
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

### 1. Initialize Scratch Space

```bash
SCRATCH_BASE=/mnt/scratch/users/40741008/tavse

mkdir -p ${SCRATCH_BASE}/{staging,.hf_cache}
mkdir -p ${SCRATCH_BASE}/processed/{audio_16k,noise,metadata}
mkdir -p ${SCRATCH_BASE}/{checkpoints,logs}

# Verify write access
dd if=/dev/zero of=${SCRATCH_BASE}/write_test bs=1M count=10 oflag=direct 2>/dev/null && \
    rm ${SCRATCH_BASE}/write_test && echo "Scratch write OK"
```

### 2. Clear Home Quota

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

The ingestion pipeline downloads each subject from HuggingFace, extracts 96×96 mouth ROIs using MediaPipe face detection, builds LMDB databases, resamples audio to 16kHz, and generates train/val/test manifests.

**Option A: Full ingestion as SLURM job** (~24 hours for all 142 subjects)

```bash
cd ~/my_projects/AVTSE/TAVSE
sbatch scripts/01_ingest_dataset.sh
```

**Option B: Partial ingestion** (for testing)

```bash
# Process only subjects 1-5
sbatch scripts/01_ingest_dataset.sh 1 5
```

**Option C: Resume interrupted ingestion**

```bash
sbatch scripts/01_ingest_dataset.sh --resume
```

**What it does:**
1. Downloads `sub_{id}_ia.zip` to scratch staging (~6.4 GB per subject)
2. Opens zip in streaming mode (no full extraction)
3. Detects face landmarks on RGB frames → computes mouth bounding box
4. Crops 96×96 mouth ROIs for both RGB and thermal frames
5. Writes ROIs to LMDB (JPEG for RGB, PNG for thermal)
6. Resamples audio from 44.1kHz → 16kHz
7. Deletes the zip after processing
8. Saves progress to `ingest_state.json` for resume support

**Speaker split:**
- Train: subjects 1-100 (100 speakers)
- Validation: subjects 101-114 (14 speakers)
- Test: subjects 115-142 (28 speakers)

### 4. Prepare Noise Corpus

Downloads and resamples the DEMAND noise corpus:

```bash
sbatch scripts/02_prepare_noise.sh
```

If automatic download fails, manually download from [Zenodo](https://zenodo.org/record/1227121) and place the zip in the staging directory.

---

## Verification

After ingestion, verify the processed data:

```bash
# Check LMDB sizes
du -sh /mnt/scratch/users/40741008/tavse/processed/rgb_mouth.lmdb/
du -sh /mnt/scratch/users/40741008/tavse/processed/thermal_mouth.lmdb/

# Check manifests
for f in /mnt/scratch/users/40741008/tavse/processed/metadata/*_manifest.csv; do
    echo "$(basename $f): $(wc -l < $f) entries"
done

# Check audio files
ls /mnt/scratch/users/40741008/tavse/processed/audio_16k/ | wc -l

# Quick sanity check: verify LMDB reads work
python3 -c "
import lmdb
env = lmdb.open('/mnt/scratch/users/40741008/tavse/processed/rgb_mouth.lmdb', readonly=True)
with env.begin() as txn:
    cursor = txn.cursor()
    cursor.first()
    key, val = cursor.item()
    print(f'First key: {key.decode()}, value size: {len(val)} bytes')
    print(f'Total entries: {env.stat()[\"entries\"]}')
env.close()
"

# Verify noise files
ls /mnt/scratch/users/40741008/tavse/processed/noise/*.wav | wc -l
```

---

## Troubleshooting

### Home quota exceeded
```bash
quota -s                    # Check current usage
du -sh ~/.cache/huggingface # Should be empty (redirected to scratch)
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
