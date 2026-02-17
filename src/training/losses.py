"""
Loss functions for TAVSE speech enhancement.

Combines SI-SNR (time-domain) with L1 magnitude (frequency-domain) loss.
"""

import torch
import torch.nn as nn


class SISNRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio loss (negated for minimization).

    L_SI-SNR = -10 * log10( ||s_target||^2 / ||e_noise||^2 )

    where:
        s_target = <ŝ, s> * s / ||s||^2
        e_noise  = ŝ - s_target
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Args:
            estimate:  (B, T) enhanced waveform
            reference: (B, T) clean waveform

        Returns:
            Scalar loss (lower is better).
        """
        # Zero-mean normalization
        estimate = estimate - estimate.mean(dim=-1, keepdim=True)
        reference = reference - reference.mean(dim=-1, keepdim=True)

        # s_target = <ŝ, s> * s / ||s||^2
        dot = torch.sum(estimate * reference, dim=-1, keepdim=True)
        s_target = dot * reference / (torch.sum(reference ** 2, dim=-1, keepdim=True) + self.eps)
        e_noise = estimate - s_target

        si_snr = 10 * torch.log10(
            torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + self.eps) + self.eps
        )

        # Negate for minimization
        return -si_snr.mean()


class MagnitudeLoss(nn.Module):
    """L1 loss on STFT magnitude spectrograms."""

    def forward(self, estimate_mag: torch.Tensor, reference_mag: torch.Tensor) -> torch.Tensor:
        """
        Args:
            estimate_mag:  (B, 1, F, T) enhanced magnitude
            reference_mag: (B, 1, F, T) clean magnitude

        Returns:
            Scalar L1 loss.
        """
        return nn.functional.l1_loss(estimate_mag, reference_mag)


class TAVSELoss(nn.Module):
    """
    Combined loss: L = L_SI-SNR + lambda_mag * L_magnitude

    Default: lambda_mag = 0.1 (auxiliary magnitude loss stabilizes early training).
    """

    def __init__(self, si_snr_weight: float = 1.0, mag_weight: float = 0.1):
        super().__init__()
        self.si_snr_loss = SISNRLoss()
        self.mag_loss = MagnitudeLoss()
        self.si_snr_weight = si_snr_weight
        self.mag_weight = mag_weight

    def forward(
        self,
        enhanced_wav: torch.Tensor,
        clean_wav: torch.Tensor,
        enhanced_mag: torch.Tensor,
        clean_mag: torch.Tensor,
    ) -> dict:
        """
        Args:
            enhanced_wav: (B, T) enhanced waveform from iSTFT
            clean_wav:    (B, T) clean reference waveform
            enhanced_mag: (B, 1, F, T) enhanced STFT magnitude
            clean_mag:    (B, 1, F, T) clean STFT magnitude

        Returns:
            dict with 'total', 'si_snr', 'magnitude' loss values.
        """
        l_sisnr = self.si_snr_loss(enhanced_wav, clean_wav)
        l_mag = self.mag_loss(enhanced_mag, clean_mag)

        total = self.si_snr_weight * l_sisnr + self.mag_weight * l_mag

        return {
            "total": total,
            "si_snr": l_sisnr.detach(),
            "magnitude": l_mag.detach(),
        }
