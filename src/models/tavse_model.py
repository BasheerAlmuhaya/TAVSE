"""
TAVSE Model: Unified Thermal Audio-Visual Speech Enhancement.

A single model class that supports all four configurations:
    - Audio-only (A)
    - Audio + RGB (A+R)
    - Audio + Thermal (A+T)
    - Audio + RGB + Thermal (A+R+T)

Controlled by `active_modalities` in the config.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from src.models.audio_encoder import AudioEncoder
from src.models.visual_encoder import VisualEncoder
from src.models.fusion import FusionModule
from src.utils.config import TAVSEConfig


class TAVSEModel(nn.Module):
    """
    Thermal Audio-Visual Speech Enhancement model.

    Architecture:
        1. Audio U-Net encoder → bottleneck z_a
        2. Visual encoder(s) → z_rgb, z_thr (if active)
        3. Fusion module → z_fused
        4. Audio U-Net decoder → sigmoid mask M
        5. Enhanced STFT = M ⊙ Y_noisy
    """

    def __init__(self, cfg: TAVSEConfig):
        super().__init__()
        self.cfg = cfg
        self.active_modalities = cfg.model.active_modalities

        # ── Audio encoder (always active) ─────────────────────────
        self.audio_encoder = AudioEncoder(
            channels=cfg.model.audio_channels,
        )

        # ── Visual encoders (conditional) ─────────────────────────
        self.rgb_encoder: Optional[VisualEncoder] = None
        self.thr_encoder: Optional[VisualEncoder] = None

        if "rgb" in self.active_modalities:
            self.rgb_encoder = VisualEncoder(
                in_channels=cfg.visual.rgb_channels,
                feat_dim=cfg.model.visual_feat_dim,
                gru_hidden=cfg.model.gru_hidden,
                gru_layers=cfg.model.gru_layers,
                n_visual_frames=cfg.visual.n_frames_per_segment,
                n_audio_frames=cfg.audio.n_stft_frames,
            )

        if "thermal" in self.active_modalities:
            self.thr_encoder = VisualEncoder(
                in_channels=cfg.visual.thermal_channels,
                feat_dim=cfg.model.visual_feat_dim,
                gru_hidden=cfg.model.gru_hidden,
                gru_layers=cfg.model.gru_layers,
                n_visual_frames=cfg.visual.n_frames_per_segment,
                n_audio_frames=cfg.audio.n_stft_frames,
            )

        # ── Fusion module ─────────────────────────────────────────
        self.fusion = FusionModule(
            embed_dim=cfg.model.fusion_dim,
            num_heads=cfg.model.fusion_heads,
            dropout=cfg.model.fusion_dropout,
            alpha_init=cfg.model.fusion_alpha_init,
        )

        # ── STFT params for reconstruction ────────────────────────
        self.n_fft = cfg.audio.n_fft
        self.hop_length = cfg.audio.hop_length
        self.win_length = cfg.audio.win_length
        self.register_buffer(
            "window", torch.hann_window(cfg.audio.win_length)
        )

    def forward(
        self,
        noisy_mag: torch.Tensor,
        noisy_phase: torch.Tensor,
        rgb_frames: Optional[torch.Tensor] = None,
        thr_frames: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            noisy_mag:   (B, 1, F, T) — noisy STFT magnitude
            noisy_phase: (B, 1, F, T) — noisy STFT phase
            rgb_frames:  (B, N, 3, H, W) or None
            thr_frames:  (B, N, 1, H, W) or None

        Returns:
            dict with keys:
                mask:         (B, 1, F, T) — estimated ratio mask
                enhanced_mag: (B, 1, F, T) — masked magnitude
                enhanced_wav: (B, S)       — reconstructed waveform
        """
        target_shape = noisy_mag.shape[2:]  # (F, T)

        # ── Audio encoder ─────────────────────────────────────────
        bottleneck, skips = self.audio_encoder.encode(noisy_mag)

        # ── Visual encoders (conditional) ─────────────────────────
        z_rgb = None
        z_thr = None

        if self.rgb_encoder is not None and rgb_frames is not None:
            z_rgb = self.rgb_encoder(rgb_frames)  # (B, T_audio, D)

        if self.thr_encoder is not None and thr_frames is not None:
            z_thr = self.thr_encoder(thr_frames)  # (B, T_audio, D)

        # ── Fusion ────────────────────────────────────────────────
        fused_bottleneck = self.fusion(bottleneck, z_rgb, z_thr)

        # ── Audio decoder ─────────────────────────────────────────
        mask = self.audio_encoder.decode(fused_bottleneck, skips, target_shape)

        # ── Apply mask and reconstruct ────────────────────────────
        enhanced_mag = mask * noisy_mag

        # Reconstruct waveform using noisy phase
        enhanced_complex = enhanced_mag.squeeze(1) * torch.exp(
            1j * noisy_phase.squeeze(1)
        )
        enhanced_wav = torch.istft(
            enhanced_complex,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )

        return {
            "mask": mask,
            "enhanced_mag": enhanced_mag,
            "enhanced_wav": enhanced_wav,
        }

    def get_num_params(self) -> Dict[str, int]:
        """Count parameters per component."""
        def _count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {"audio_encoder": _count(self.audio_encoder)}
        if self.rgb_encoder is not None:
            counts["rgb_encoder"] = _count(self.rgb_encoder)
        if self.thr_encoder is not None:
            counts["thr_encoder"] = _count(self.thr_encoder)
        counts["fusion"] = _count(self.fusion)
        counts["total"] = sum(counts.values())
        return counts

    @classmethod
    def from_config(cls, cfg: TAVSEConfig) -> "TAVSEModel":
        """Create model from config."""
        return cls(cfg)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, cfg: TAVSEConfig) -> "TAVSEModel":
        """Load model from checkpoint."""
        model = cls(cfg)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        return model
