# TAVSE Training Guide

Step-by-step guide for training, evaluating, and comparing all four TAVSE model variants.

## Table of Contents

- [Overview](#overview)
- [Training Workflow](#training-workflow)
- [Running Experiments](#running-experiments)
  - [Step 1: Audio-Only Baseline](#step-1-audio-only-baseline-model-a)
  - [Step 2: Audio + RGB](#step-2-audio--rgb-model-ar)
  - [Step 3: Audio + Thermal](#step-3-audio--thermal-model-at)
  - [Step 4: Full Trimodal](#step-4-full-trimodal-model-art)
- [Monitoring Training](#monitoring-training)
- [Evaluation](#evaluation)
- [Statistical Comparison](#statistical-comparison)
- [Ablation Studies](#ablation-studies)
- [Configuration Reference](#configuration-reference)
- [Tips & Troubleshooting](#tips--troubleshooting)

---

## Overview

TAVSE trains **four model variants** to systematically compare visual modalities for speech enhancement:

| Model | Config | Active Modalities | Approx. Params |
|-------|--------|-------------------|----------------|
| **A**     | `audio_only.yaml`        | audio             | ~5.2M  |
| **A+R**   | `audio_rgb.yaml`         | audio, rgb        | ~8.8M  |
| **A+T**   | `audio_thermal.yaml`     | audio, thermal    | ~8.5M  |
| **A+R+T** | `audio_rgb_thermal.yaml` | audio, rgb, thermal | ~12.1M |

All variants share the same audio U-Net encoder, differ only in which visual encoder(s) are active, and use the same training recipe for fair comparison.

### Architecture Summary

```
Noisy STFT Mag ──→ [Audio U-Net Encoder] ──→ Bottleneck z_a
                                                    │
RGB Frames ──→ [ResNet-18 + BiGRU] ──→ z_rgb ──→ [Cross-Attn Fusion] ──→ Fused z
Thermal Frames ──→ [ResNet-18 + BiGRU] ──→ z_thr ──↗        │
                                                              ↓
                                              [Audio U-Net Decoder] ──→ Sigmoid Mask M
                                                              ↓
                                              Enhanced Ŝ = M ⊙ Y_noisy
                                                              ↓
                                              iSTFT ──→ Enhanced Waveform
```

---

## Training Workflow

**Prerequisite:** Complete all steps in [SETUP.md](SETUP.md) first (environment, data ingestion, noise corpus).

### Recommended Experiment Order

```
1. Train Audio-Only (A)     → establishes baseline
2. Train Audio+RGB (A+R)    → measures RGB visual contribution
3. Train Audio+Thermal (A+T) → measures thermal visual contribution (key novelty)
4. Train Audio+RGB+Thermal (A+R+T) → measures multimodal fusion ceiling
```

Each model trains independently — you can submit all four as SLURM jobs simultaneously if GPU resources allow.

---

## Running Experiments

### Step 1: Audio-Only Baseline (Model A)

```bash
cd ~/my_projects/AVTSE/TAVSE

# Submit training job
sbatch scripts/03_train.sh audio_only
```

This trains for up to 100 epochs with early stopping (patience=10 on validation SI-SNR). Expected time: ~12-24 hours depending on GPU.

### Step 2: Audio + RGB (Model A+R)

```bash
sbatch scripts/03_train.sh audio_rgb
```

### Step 3: Audio + Thermal (Model A+T)

```bash
sbatch scripts/03_train.sh audio_thermal
```

### Step 4: Full Trimodal (Model A+R+T)

```bash
sbatch scripts/03_train.sh audio_rgb_thermal
```

### Resume a Crashed/Preempted Job

```bash
sbatch scripts/03_train.sh audio_rgb --resume
```

This loads the latest checkpoint and continues from where it left off.

### Manual Training (Interactive)

For debugging or development, you can run training interactively:

```bash
# Request GPU node
srun --partition=gpu --gres=gpu:1 --mem=64G --cpus-per-task=8 --time=02:00:00 --pty bash

# Activate environment
conda activate tavse
export PYTHONPATH=~/my_projects/AVTSE/TAVSE:${PYTHONPATH:-}
export HF_HOME=/mnt/scratch/users/40741008/tavse/.hf_cache

# Run training
python -m src.training.train --config configs/audio_only.yaml
```

### Override Config Parameters

Pass JSON overrides without editing YAML files:

```bash
# Reduce batch size for debugging
python -m src.training.train \
    --config configs/audio_rgb.yaml \
    --override '{"training": {"batch_size": 4, "max_epochs": 5}}'
```

---

## Monitoring Training

### TensorBoard

```bash
# From login node (port forward to your local machine)
tensorboard --logdir /mnt/scratch/users/40741008/tavse/logs/ --port 6006

# Or for a specific experiment
tensorboard --logdir /mnt/scratch/users/40741008/tavse/logs/audio_rgb/ --port 6006
```

Then open `http://localhost:6006` in your browser (with SSH port forwarding).

### SLURM Job Status

```bash
squeue -u $(whoami)                    # Running jobs
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS  # Completed job details
tail -f /mnt/scratch/users/40741008/tavse/logs/train_JOBID.out  # Live output
```

### Key Metrics to Watch

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| Train loss | Steadily decreasing | Stuck or oscillating wildly |
| Val SI-SNR | Increasing, >5 dB after 10 epochs | Flat or decreasing |
| Train-Val gap | <3 dB | >5 dB (overfitting) |
| Learning rate | Following cosine schedule | Stuck at 0 |

### Quick Sanity Check

Before a full training run, verify the model works:

```bash
python -c "
import torch
from src.utils.config import load_config
from src.models.tavse_model import TAVSEModel

cfg = load_config('configs/audio_rgb_thermal.yaml')
model = TAVSEModel.from_config(cfg)
params = model.get_num_params()
print('Parameters:', {k: f'{v:,}' for k, v in params.items()})

# Test forward pass
B = 2
noisy_mag = torch.randn(B, 1, 257, 251)
noisy_phase = torch.randn(B, 1, 257, 251)
rgb = torch.randn(B, 70, 3, 96, 96)
thr = torch.randn(B, 70, 1, 96, 96)
out = model(noisy_mag, noisy_phase, rgb, thr)
print('Output shapes:', {k: v.shape for k, v in out.items()})
print('Forward pass OK!')
"
```

---

## Evaluation

After training completes, evaluate each model on the test set:

```bash
# Evaluate each model
sbatch scripts/04_evaluate.sh audio_only
sbatch scripts/04_evaluate.sh audio_rgb
sbatch scripts/04_evaluate.sh audio_thermal
sbatch scripts/04_evaluate.sh audio_rgb_thermal
```

### What Evaluation Computes

| Metric | Range | Higher/Lower Better |
|--------|-------|---------------------|
| **PESQ** | [-0.5, 4.5] | Higher |
| **STOI** | [0, 1] | Higher |
| **SI-SNR** | dB | Higher |
| **SI-SNRi** | dB | Higher |
| **SDR** | dB | Higher |

Results are saved to:
```
/mnt/scratch/users/40741008/tavse/checkpoints/{experiment}/eval/
├── eval_metrics.json       # Summary statistics (mean, std, 95% CI)
├── eval_raw_metrics.json   # Per-utterance metrics (for statistical tests)
└── samples/
    ├── sample0000_clean.wav
    ├── sample0000_noisy.wav
    ├── sample0000_enhanced.wav
    └── ...
```

### Manual Evaluation

```bash
python -m src.training.evaluate \
    --config configs/audio_thermal.yaml \
    --checkpoint /mnt/scratch/users/40741008/tavse/checkpoints/audio_thermal/latest.pt \
    --output-dir /mnt/scratch/users/40741008/tavse/checkpoints/audio_thermal/eval \
    --save-samples 10
```

---

## Statistical Comparison

Compare two models using paired Wilcoxon signed-rank tests:

```bash
# Compare A+T vs A+R (key hypothesis test)
sbatch scripts/04_evaluate.sh audio_thermal audio_rgb
```

This performs a Wilcoxon test with Bonferroni correction (p < 0.0017 for significance across 6 pairwise comparisons).

### Manual Comparison

```python
import json
from src.training.evaluate import paired_wilcoxon_test

# Load raw metrics for two models
with open('.../audio_thermal/eval/eval_raw_metrics.json') as f:
    thermal_metrics = json.load(f)
with open('.../audio_rgb/eval/eval_raw_metrics.json') as f:
    rgb_metrics = json.load(f)

# Test for SI-SNR difference
result = paired_wilcoxon_test(thermal_metrics, rgb_metrics, 'si_snr')
print(f"Mean diff: {result['mean_diff']:+.3f} dB, p = {result['p_value']:.4e}")
print(f"Significant: {result['significant']}")
```

---

## Ablation Studies

After the main experiments, run ablation studies as described in the plan:

### 1. Modality Dropout (Priority: High)

Train the trimodal model with random modality dropout during training:

```bash
python -m src.training.train \
    --config configs/audio_rgb_thermal.yaml \
    --override '{"experiment_name": "art_modality_dropout", "model": {"modality_dropout": 0.2}}'
```

Then test with each modality combination at inference time.

### 2. Fusion Method Comparison (Priority: High)

Modify the fusion module and retrain:
- Cross-attention (default)
- Concatenation + MLP
- FiLM conditioning

### 3. Gate Analysis (Priority: High)

After training A+R+T, extract gate values:

```python
import torch
from src.models.tavse_model import TAVSEModel

model = TAVSEModel.from_checkpoint('...', cfg)
model.eval()

# Hook into fusion module to capture gate values
gate_values = []
def hook_fn(module, input, output):
    # Access intermediate gate activations
    gate_values.append(module.trimodal_gate.gate_proj[0].weight.data)

model.fusion.register_forward_hook(hook_fn)
```

---

## Configuration Reference

### Key Config Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.active_modalities` | ["audio"] | Which modalities to use |
| `training.batch_size` | 16 | Per-GPU batch size |
| `training.lr` | 3e-4 | Peak learning rate |
| `training.max_epochs` | 100 | Maximum training epochs |
| `training.patience` | 10 | Early stopping patience |
| `training.use_amp` | true | Mixed precision training |
| `training.grad_accum_steps` | 1 | Gradient accumulation steps |
| `audio.segment_seconds` | 2.5 | Audio segment length |
| `visual.roi_size` | 96 | Mouth ROI size (pixels) |

### Adjusting for GPU Memory

If you get OOM errors:

```yaml
# Option 1: Reduce batch size
training:
  batch_size: 8
  grad_accum_steps: 2  # Effective batch = 8 * 2 = 16

# Option 2: Disable mixed precision (uses more VRAM but avoids FP16 issues)
training:
  use_amp: false
  batch_size: 8
```

---

## Tips & Troubleshooting

### Training is too slow

1. **Check data loading:** If GPU utilization is low (<80%), I/O is the bottleneck.
   - Increase `num_workers` (up to number of CPUs - 2)
   - Copy LMDBs to `/tmp` at job start (uncomment section in `03_train.sh`)

2. **Check GPU utilization:**
   ```bash
   watch -n 1 nvidia-smi
   ```

### Out of GPU memory

1. Reduce `batch_size` and increase `grad_accum_steps` proportionally
2. Enable gradient checkpointing (add to model code)
3. Use `torch.utils.checkpoint.checkpoint()` for visual encoder

### Checkpoints filling scratch

Only top-3 checkpoints are kept per experiment. To manually clean:
```bash
ls -lhS /mnt/scratch/users/40741008/tavse/checkpoints/*/
```

### Reproducibility

All experiments use `seed: 42` by default. To verify stability, re-run one experiment with a different seed:

```bash
python -m src.training.train \
    --config configs/audio_only.yaml \
    --override '{"training": {"seed": 123}, "experiment_name": "audio_only_seed123"}'
```

Expected variance: ±0.1 dB SI-SNR between seeds.

### View All Results

```bash
# Quick summary of all evaluated models
for dir in /mnt/scratch/users/40741008/tavse/checkpoints/*/eval; do
    if [ -f "$dir/eval_metrics.json" ]; then
        exp=$(basename $(dirname $dir))
        sisnr=$(python3 -c "import json; d=json.load(open('$dir/eval_metrics.json')); print(f\"{d['statistics']['si_snr']['mean']:.2f}\")")
        pesq=$(python3 -c "import json; d=json.load(open('$dir/eval_metrics.json')); print(f\"{d['statistics']['pesq']['mean']:.2f}\")")
        echo "  ${exp}: SI-SNR=${sisnr} dB, PESQ=${pesq}"
    fi
done
```
