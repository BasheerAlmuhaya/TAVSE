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
├── scripts/            # SLURM job scripts (ingest, train, evaluate)
├── docs/               # Setup and training documentation
└── requirements.txt    # Python dependencies
```

## Quick Start

```bash
# 1. Setup environment
conda create -n tavse python=3.10 -y && conda activate tavse
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 2. Prepare data (see docs/SETUP.md for full details)
sbatch scripts/01_ingest_dataset.sh
sbatch scripts/02_prepare_noise.sh

# 3. Train models
sbatch scripts/03_train.sh audio_only
sbatch scripts/03_train.sh audio_rgb
sbatch scripts/03_train.sh audio_thermal
sbatch scripts/03_train.sh audio_rgb_thermal

# 4. Evaluate
sbatch scripts/04_evaluate.sh audio_only
sbatch scripts/04_evaluate.sh audio_rgb_thermal
```

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

