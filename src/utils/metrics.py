"""
Speech enhancement evaluation metrics.

Computes PESQ, STOI, SI-SNR, and SDR between clean and enhanced signals.
All functions expect 1D numpy arrays at 16 kHz.
"""

import numpy as np
from typing import Dict


def si_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR) in dB.

    Args:
        reference: clean signal, shape (T,)
        estimate: enhanced signal, shape (T,)

    Returns:
        SI-SNR in dB (higher is better).
    """
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)

    dot = np.sum(reference * estimate)
    s_target = dot * reference / (np.sum(reference ** 2) + 1e-8)
    e_noise = estimate - s_target

    si_snr_val = 10 * np.log10(
        np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-8) + 1e-8
    )
    return float(si_snr_val)


def si_snr_improvement(reference: np.ndarray, estimate: np.ndarray,
                       noisy: np.ndarray) -> float:
    """SI-SNR improvement: SI-SNR(enhanced) - SI-SNR(noisy)."""
    return si_snr(reference, estimate) - si_snr(reference, noisy)


def compute_pesq(reference: np.ndarray, estimate: np.ndarray,
                 sr: int = 16000) -> float:
    """
    Perceptual Evaluation of Speech Quality (wideband).

    Returns:
        PESQ MOS-LQO score, range [-0.5, 4.5].
    """
    from pesq import pesq as pesq_fn
    return float(pesq_fn(sr, reference, estimate, "wb"))


def compute_stoi(reference: np.ndarray, estimate: np.ndarray,
                 sr: int = 16000) -> float:
    """
    Short-Time Objective Intelligibility.

    Returns:
        STOI score, range [0, 1].
    """
    from pystoi import stoi
    return float(stoi(reference, estimate, sr, extended=False))


def compute_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Signal-to-Distortion Ratio using mir_eval.

    Returns:
        SDR in dB.
    """
    from mir_eval.separation import bss_eval_sources
    reference = reference.reshape(1, -1)
    estimate = estimate.reshape(1, -1)
    sdr, _, _, _ = bss_eval_sources(reference, estimate, compute_permutation=False)
    return float(sdr[0])


def compute_all_metrics(reference: np.ndarray, estimate: np.ndarray,
                        noisy: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        reference: clean signal (T,)
        estimate: enhanced signal (T,)
        noisy: noisy input signal (T,)
        sr: sample rate (must be 16000)

    Returns:
        Dict with keys: pesq, stoi, si_snr, si_snri, sdr
    """
    # Ensure same length
    min_len = min(len(reference), len(estimate), len(noisy))
    reference = reference[:min_len]
    estimate = estimate[:min_len]
    noisy = noisy[:min_len]

    return {
        "pesq": compute_pesq(reference, estimate, sr),
        "stoi": compute_stoi(reference, estimate, sr),
        "si_snr": si_snr(reference, estimate),
        "si_snri": si_snr_improvement(reference, estimate, noisy),
        "sdr": compute_sdr(reference, estimate),
    }
