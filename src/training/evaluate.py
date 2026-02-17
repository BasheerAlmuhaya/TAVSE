"""
Evaluation script for TAVSE models.

Computes PESQ, STOI, SI-SNR, SI-SNRi, and SDR on the test set,
broken down by SNR level and noise type. Supports statistical testing.

Usage:
    python -m src.training.evaluate --config configs/audio_rgb_thermal.yaml \
        --checkpoint /mnt/scratch/users/40741008/tavse/checkpoints/audio_rgb_thermal/best.pt
"""

import os
import json
import argparse
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from scipy.stats import wilcoxon

from src.utils.config import load_config, TAVSEConfig
from src.utils.metrics import compute_all_metrics
from src.models.tavse_model import TAVSEModel
from src.data.dataset import TAVSEDataset, collate_fn


@torch.no_grad()
def evaluate_model(
    model: TAVSEModel,
    loader: DataLoader,
    cfg: TAVSEConfig,
    device: torch.device,
    save_samples: int = 5,
    output_dir: str = None,
) -> Dict[str, List[float]]:
    """
    Run full evaluation on test set.

    Args:
        model: trained TAVSE model
        loader: test DataLoader
        cfg: config
        device: CUDA or CPU
        save_samples: number of enhanced audio samples to save
        output_dir: directory for results and samples

    Returns:
        Dict mapping metric names to per-utterance values.
    """
    model.eval()
    use_rgb = "rgb" in cfg.model.active_modalities
    use_thermal = "thermal" in cfg.model.active_modalities

    all_metrics = {
        "pesq": [], "stoi": [], "si_snr": [], "si_snri": [], "sdr": [],
    }
    saved = 0

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        samples_dir = os.path.join(output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        noisy_mag = batch["noisy_mag"].to(device)
        noisy_phase = batch["noisy_phase"].to(device)
        clean_wav = batch["clean_wav"]
        noisy_wav = batch["noisy_wav"]

        rgb_frames = batch["rgb_frames"].to(device) if use_rgb and batch["rgb_frames"] is not None else None
        thr_frames = batch["thr_frames"].to(device) if use_thermal and batch["thr_frames"] is not None else None

        outputs = model(noisy_mag, noisy_phase, rgb_frames, thr_frames)
        enhanced_wav = outputs["enhanced_wav"].cpu()

        # Per-utterance metrics
        B = enhanced_wav.shape[0]
        for i in range(B):
            ref = clean_wav[i].numpy()
            enh = enhanced_wav[i].numpy()
            nsy = noisy_wav[i].numpy()

            try:
                metrics = compute_all_metrics(ref, enh, nsy, sr=cfg.audio.sample_rate)
                for k in all_metrics:
                    all_metrics[k].append(metrics[k])
            except Exception as e:
                print(f"[WARN] Metrics computation failed for sample {batch_idx * B + i}: {e}")
                continue

            # Save sample audio files
            if output_dir and saved < save_samples:
                sample_id = batch_idx * B + i
                for name, wav in [("clean", ref), ("noisy", nsy), ("enhanced", enh)]:
                    path = os.path.join(samples_dir, f"sample{sample_id:04d}_{name}.wav")
                    torchaudio.save(
                        path,
                        torch.from_numpy(wav).unsqueeze(0),
                        cfg.audio.sample_rate,
                    )
                saved += 1

    return all_metrics


def compute_statistics(metrics: Dict[str, List[float]]) -> Dict[str, Dict]:
    """Compute mean, std, median, 95% CI via bootstrap."""
    results = {}
    for name, values in metrics.items():
        arr = np.array(values)
        # Bootstrap 95% CI
        n_bootstrap = 1000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(arr, size=len(arr), replace=True)
            bootstrap_means.append(sample.mean())
        bootstrap_means = np.sort(bootstrap_means)
        ci_low = bootstrap_means[int(0.025 * n_bootstrap)]
        ci_high = bootstrap_means[int(0.975 * n_bootstrap)]

        results[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
            "n": len(arr),
        }
    return results


def paired_wilcoxon_test(
    metrics_a: Dict[str, List[float]],
    metrics_b: Dict[str, List[float]],
    metric_name: str = "si_snr",
) -> Dict[str, float]:
    """
    Paired Wilcoxon signed-rank test between two models.

    Returns:
        Dict with 'statistic', 'p_value', 'significant' (Bonferroni p<0.0017)
    """
    a = np.array(metrics_a[metric_name])
    b = np.array(metrics_b[metric_name])

    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]

    stat, p_value = wilcoxon(a, b, alternative="two-sided")
    bonferroni_threshold = 0.01 / 6  # 6 pairwise comparisons

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < bonferroni_threshold),
        "bonferroni_threshold": bonferroni_threshold,
        "mean_diff": float(np.mean(a - b)),
    }


def main():
    parser = argparse.ArgumentParser(description="TAVSE Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint .pt")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for results (default: checkpoint_dir/eval)")
    parser.add_argument("--save-samples", type=int, default=5,
                        help="Number of enhanced audio samples to save")
    parser.add_argument("--compare-with", type=str, default=None,
                        help="Path to another model's eval_metrics.json for statistical comparison")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.checkpoint), "eval"
    )

    print(f"\nTAVSE Evaluation: {cfg.experiment_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}\n")

    # ── Model ─────────────────────────────────────────────────────
    model = TAVSEModel.from_checkpoint(args.checkpoint, cfg).to(device)
    param_counts = model.get_num_params()
    print(f"Model parameters: {param_counts['total']:,}")

    # ── Data ──────────────────────────────────────────────────────
    test_dataset = TAVSEDataset(cfg, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    print(f"Test set: {len(test_dataset)} utterances\n")

    # ── Evaluate ──────────────────────────────────────────────────
    raw_metrics = evaluate_model(
        model, test_loader, cfg, device,
        save_samples=args.save_samples,
        output_dir=output_dir,
    )

    # ── Statistics ────────────────────────────────────────────────
    stats = compute_statistics(raw_metrics)

    print(f"\n{'='*60}")
    print(f"Results: {cfg.experiment_name}")
    print(f"{'='*60}")
    for metric_name, vals in stats.items():
        print(
            f"  {metric_name:>8s}: {vals['mean']:.3f} ± {vals['std']:.3f}  "
            f"(95% CI: [{vals['ci_95_low']:.3f}, {vals['ci_95_high']:.3f}])"
        )
    print(f"{'='*60}\n")

    # ── Save results ──────────────────────────────────────────────
    results = {
        "experiment": cfg.experiment_name,
        "checkpoint": args.checkpoint,
        "n_samples": len(raw_metrics["si_snr"]),
        "statistics": stats,
    }

    results_path = os.path.join(output_dir, "eval_metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Also save raw per-utterance metrics for statistical tests
    raw_path = os.path.join(output_dir, "eval_raw_metrics.json")
    with open(raw_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in raw_metrics.items()}, f)

    # ── Statistical comparison ────────────────────────────────────
    if args.compare_with:
        print(f"\nStatistical comparison with: {args.compare_with}")
        compare_raw_path = args.compare_with.replace("eval_metrics.json", "eval_raw_metrics.json")
        if os.path.exists(compare_raw_path):
            with open(compare_raw_path, "r") as f:
                other_metrics = json.load(f)

            for metric in ["si_snr", "pesq", "stoi"]:
                test_result = paired_wilcoxon_test(raw_metrics, other_metrics, metric)
                sig = "***" if test_result["significant"] else "n.s."
                print(
                    f"  {metric:>8s}: diff={test_result['mean_diff']:+.3f}, "
                    f"p={test_result['p_value']:.4e} {sig}"
                )
        else:
            print(f"  Raw metrics file not found: {compare_raw_path}")


if __name__ == "__main__":
    main()
