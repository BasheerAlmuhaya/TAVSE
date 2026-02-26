# TAVSE — Thermal Audio-Visual Speech Enhancement

A PyTorch framework for comparing **Audio-only**, **Audio+RGB**, **Audio+Thermal**, and **Audio+RGB+Thermal** speech enhancement on the [ISSAI SpeakingFaces](https://doi.org/10.3390/data6120132) dataset.

## Key Idea

Thermal imaging captures sub-surface articulatory cues (lip heat, nasal airflow) invisible to RGB cameras. We hypothesize that thermal visual features provide complementary information for speech enhancement, especially under low-SNR conditions where audio is severely degraded.

## Architecture

```
Noisy STFT ──→ [Audio U-Net Encoder] ──→ Bottleneck
                                              │
RGB Mouth  ──→ [ResNet-18 + BiGRU] ──→ ──→ [Cross-Attention Fusion]
Thermal    ──→ [ResNet-18 + BiGRU] ──→ ──↗        │
                                          [Audio U-Net Decoder] ──→ Mask M
                                                   │
                                          Enhanced = M ⊙ Noisy → iSTFT → Waveform
```

Four model variants share >90% of parameters, differing only in which visual branch is active.

| Model | Modalities | Params |
|-------|-----------|--------|
| **A**     | Audio only            | ~5.2M  |
| **A+R**   | Audio + RGB           | ~8.8M  |
| **A+T**   | Audio + Thermal       | ~8.5M  |
| **A+R+T** | Audio + RGB + Thermal | ~12.1M |

## Project Structure

```
TAVSE/
├── src/
│   ├── data/           # Dataset, ingestion, noise mixing, transforms
│   ├── models/         # Audio encoder, visual encoder, fusion, unified model
│   ├── training/       # Training loop, evaluation, loss functions
│   └── utils/          # Config management, evaluation metrics
├── configs/            # YAML configs for each experiment
├── scripts/            # SLURM job scripts (download, ingest, train, evaluate)
├── docs/               # Setup and training documentation
└── requirements.txt    # Python dependencies
```

## Quick Start

```bash
# 1. Setup environment
conda create -n tavse python=3.10 -y && conda activate tavse
pip install torch torchaudio torchvision  # add --index-url for specific CUDA version
pip install -r requirements.txt

# 2. Configure local paths (REQUIRED — sets data root, HF cache, etc.)
cp .env.example .env
# Edit .env — set TAVSE_DATA_ROOT, TAVSE_PROJECT_DIR, CONDA_EXE, etc.

# 3. Login to HuggingFace (dataset is gated)
huggingface-cli login

# 4. Download data on LOGIN node (compute nodes have no internet)
bash scripts/00_download_data.sh          # all 142 subjects
bash scripts/00_download_data.sh 1 5      # or a small subset for testing

# 5. Download noise corpus on LOGIN node
bash scripts/02a_download_noise.sh

# 6. Process data on COMPUTE node via SLURM
sbatch scripts/01_ingest_dataset.sh
sbatch scripts/02b_prepare_noise.sh

# 7. Train models
sbatch scripts/03_train.sh audio_only
sbatch scripts/03_train.sh audio_rgb
sbatch scripts/03_train.sh audio_thermal
sbatch scripts/03_train.sh audio_rgb_thermal

# 8. Evaluate
sbatch scripts/04_evaluate.sh audio_only
sbatch scripts/04_evaluate.sh audio_rgb
sbatch scripts/04_evaluate.sh audio_thermal
sbatch scripts/04_evaluate.sh audio_rgb_thermal
```

> **GPU Acceleration:** On Ampere+ GPUs (A100, RTX 30xx, etc.), training automatically enables TF32 and BF16 for ~2x speedup. See `.env.example` for configuration options.

### SLURM Node Selection

On shared HPC clusters, a GPU node may have residual processes from other users consuming GPU memory. If you encounter `CUDA out of memory` errors despite having a large GPU (e.g., A100 80GB), check whether another process is hogging the GPU.

**Diagnose** — inspect the error log for lines like:  
```
Process XXXXXX has 72.72 GiB memory in use
```
This means a rogue process is occupying most of the GPU. The training script logs GPU memory status at startup to help detect this.

**Solutions:**

```bash
# Exclude a specific node known to have issues
sbatch --exclude=gpu01 scripts/03_train.sh audio_rgb_thermal

# Target a specific clean node
sbatch --nodelist=gpu02 scripts/03_train.sh audio_rgb_thermal

# Request exclusive node access (no GPU sharing)
sbatch --exclusive scripts/03_train.sh audio_rgb_thermal

# Check available nodes and their state before submitting
sinfo -p gpu -N --format="%N %P %T %G %m %e"
```

> **Note:** `nvidia-smi` only works on GPU compute nodes, not on the login node. To inspect GPUs interactively, use:
> ```bash
> srun --partition=gpu --gres=gpu:1 --pty bash -c "nvidia-smi"
> ```

## Documentation

- **[docs/SETUP.md](docs/SETUP.md)** — Environment setup, storage architecture, data ingestion
- **[docs/TRAINING.md](docs/TRAINING.md)** — Training workflow, monitoring, evaluation, ablations

## Dataset

[ISSAI SpeakingFaces](https://github.com/IS2AI/SpeakingFaces) — 142 subjects with synchronized RGB, thermal, and audio recordings. Speaker-disjoint split: 100 train / 14 val / 28 test.

## Evaluation Metrics

| Metric | Library | Description |
|--------|---------|-------------|
| PESQ   | `pesq`  | Perceptual speech quality |
| STOI   | `pystoi`| Short-time objective intelligibility |
| SI-SNR | custom  | Scale-invariant signal-to-noise ratio |
| SDR    | `mir_eval` | Signal-to-distortion ratio |

## License

Research use only. The SpeakingFaces dataset has its own [license terms](https://github.com/IS2AI/SpeakingFaces).

