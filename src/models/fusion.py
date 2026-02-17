"""
Fusion Module: Cross-Modal Attention for audio-visual feature fusion.

Supports:
    - Bimodal fusion (Audio + one visual modality)
    - Trimodal fusion (Audio + RGB + Thermal) with learned gating
"""

import torch
import torch.nn as nn
from typing import Optional


class CrossModalAttention(nn.Module):
    """
    Cross-attention between audio (query) and visual (key/value) features.

    Audio queries attend to visual features, producing a residual that is
    added to the audio representation with a learnable scaling factor alpha.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm_q = nn.LayerNorm(embed_dim)
        self.layer_norm_kv = nn.LayerNorm(embed_dim)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio_features:  (B, T, D) — audio bottleneck flattened temporally
            visual_features: (B, T, D) — visual features upsampled to audio time

        Returns:
            fused: (B, T, D) — audio + alpha * cross_attn(audio, visual)
        """
        q = self.layer_norm_q(audio_features)
        kv = self.layer_norm_kv(visual_features)

        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        fused = audio_features + self.alpha * attn_out
        return fused


class TrimodalGate(nn.Module):
    """
    Learned gating mechanism for trimodal (Audio + RGB + Thermal) fusion.

    Computes modality-specific gate values from the concatenated
    audio, RGB, and thermal representations. The gate values are
    interpretable — they show how much each visual modality contributes.

    Gate equation:
        g_rgb, g_thr = split(sigmoid(W @ [z_a; z_rgb; z_thr]))

    Fused output:
        z_fused = z_a + g_rgb * CrossAttn(z_a, z_rgb) + g_thr * CrossAttn(z_a, z_thr)
    """

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.Sigmoid(),
        )
        self.embed_dim = embed_dim

    def forward(
        self,
        audio_feat: torch.Tensor,
        rgb_attn_out: torch.Tensor,
        thr_attn_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio_feat:     (B, T, D) — raw audio features
            rgb_attn_out:   (B, T, D) — cross-attention output for RGB
            thr_attn_out:   (B, T, D) — cross-attention output for thermal

        Returns:
            fused: (B, T, D) — gated fusion of all three modalities
        """
        # Compute gates from all three modalities
        combined = torch.cat([audio_feat, rgb_attn_out, thr_attn_out], dim=-1)
        gates = self.gate_proj(combined)  # (B, T, 2*D)
        g_rgb = gates[..., :self.embed_dim]  # (B, T, D)
        g_thr = gates[..., self.embed_dim:]  # (B, T, D)

        fused = audio_feat + g_rgb * rgb_attn_out + g_thr * thr_attn_out
        return fused


class FusionModule(nn.Module):
    """
    Modality-aware fusion module.

    Handles three cases:
        1. Audio-only:  bypass (identity)
        2. Bimodal:     CrossModalAttention with learnable alpha
        3. Trimodal:    Two CrossModalAttention + TrimodalGate

    The bottleneck from the audio encoder is reshaped to (B, T_flat, D),
    fused with visual features, then reshaped back to spatial dims.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Bimodal/trimodal cross-attention modules
        self.rgb_cross_attn = CrossModalAttention(
            embed_dim, num_heads, dropout, alpha_init
        )
        self.thr_cross_attn = CrossModalAttention(
            embed_dim, num_heads, dropout, alpha_init
        )

        # Trimodal gating
        self.trimodal_gate = TrimodalGate(embed_dim)

    def forward(
        self,
        audio_bottleneck: torch.Tensor,
        rgb_features: Optional[torch.Tensor] = None,
        thr_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse audio bottleneck with available visual features.

        Args:
            audio_bottleneck: (B, C, H, W) — from audio encoder
            rgb_features:     (B, T, D)     — from RGB visual encoder, or None
            thr_features:     (B, T, D)     — from thermal visual encoder, or None

        Returns:
            fused_bottleneck: (B, C, H, W) — same shape as input
        """
        B, C, H, W = audio_bottleneck.shape

        # ── Audio-only: bypass ────────────────────────────────────
        if rgb_features is None and thr_features is None:
            return audio_bottleneck

        # ── Reshape audio to sequence ─────────────────────────────
        # (B, 512, H, W) → (B, H*W, 512)
        audio_seq = audio_bottleneck.reshape(B, C, H * W).permute(0, 2, 1)

        # Visual features need to be matched to H*W temporal length
        T_audio = H * W

        if rgb_features is not None:
            # Interpolate visual features to match audio spatial dims
            rgb_feat = nn.functional.interpolate(
                rgb_features.permute(0, 2, 1),  # (B, D, T_vis)
                size=T_audio,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)  # (B, T_audio, D)

        if thr_features is not None:
            thr_feat = nn.functional.interpolate(
                thr_features.permute(0, 2, 1),
                size=T_audio,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)

        # ── Trimodal fusion ───────────────────────────────────────
        if rgb_features is not None and thr_features is not None:
            rgb_attn = self.rgb_cross_attn(audio_seq, rgb_feat) - audio_seq
            thr_attn = self.thr_cross_attn(audio_seq, thr_feat) - audio_seq
            fused_seq = self.trimodal_gate(audio_seq, rgb_attn, thr_attn)

        # ── Bimodal: Audio + RGB ──────────────────────────────────
        elif rgb_features is not None:
            fused_seq = self.rgb_cross_attn(audio_seq, rgb_feat)

        # ── Bimodal: Audio + Thermal ──────────────────────────────
        else:
            fused_seq = self.thr_cross_attn(audio_seq, thr_feat)

        # ── Reshape back to spatial ───────────────────────────────
        fused = fused_seq.permute(0, 2, 1).reshape(B, C, H, W)
        return fused
