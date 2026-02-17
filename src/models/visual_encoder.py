"""
Visual Encoder: ResNet-18-small + Bidirectional GRU for temporal aggregation.

Processes a sequence of mouth ROI frames and outputs a temporally-aligned
feature sequence that can be fused with the audio encoder bottleneck.

Two variants:
    - RGB encoder:     in_channels=3, ~3.6M params
    - Thermal encoder: in_channels=1, ~3.3M params
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class VisualEncoder(nn.Module):
    """
    ResNet-18-based visual encoder with BiGRU temporal aggregation.

    Architecture:
        1. Spatial: ResNet-18 (first 4 blocks, no FC) → GAP → (B*N, 512)
        2. Temporal: 2-layer BiGRU (hidden=256) → projection → (B, N, 512)
        3. Upsampling: 1D transposed conv to map N visual frames to T audio frames

    Input:  (B, N, C, H, W) — sequence of N mouth ROI frames
    Output: (B, T, 512)     — temporally-upsampled visual features
    """

    def __init__(
        self,
        in_channels: int = 3,
        feat_dim: int = 512,
        gru_hidden: int = 256,
        gru_layers: int = 2,
        n_visual_frames: int = 70,
        n_audio_frames: int = 251,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_dim = feat_dim
        self.n_visual_frames = n_visual_frames
        self.n_audio_frames = n_audio_frames

        # ── Spatial backbone: ResNet-18 (modified) ────────────────
        resnet = models.resnet18(weights=None)

        # Modify first conv for arbitrary input channels and small input (96×96)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False
        )  # stride=1 instead of 2 for 96×96 input
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # stride 2

        self.layer1 = resnet.layer1  # 64 → 64
        self.layer2 = resnet.layer2  # 64 → 128
        self.layer3 = resnet.layer3  # 128 → 256
        self.layer4 = resnet.layer4  # 256 → 512

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ── Temporal aggregation: BiGRU ───────────────────────────
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0,
        )
        # BiGRU output: 2 * gru_hidden → project to feat_dim
        self.temporal_proj = nn.Linear(2 * gru_hidden, feat_dim)

        # ── Temporal upsampling: N visual frames → T audio frames ─
        # Using 1D transposed conv + interpolation
        # 70 → ~280 via stride 4, then interpolate to exact T
        self.temporal_upsample = nn.Sequential(
            nn.ConvTranspose1d(feat_dim, feat_dim, kernel_size=4, stride=4, padding=0),
            nn.ReLU(inplace=True),
        )

    def _extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract per-frame spatial features.

        Args:
            x: (B*N, C, H, W)
        Returns:
            features: (B*N, 512)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)  # (B*N, 512)
        return x

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Full visual encoding pipeline.

        Args:
            frames: (B, N, C, H, W) — N mouth ROI frames

        Returns:
            features: (B, T_audio, feat_dim) — temporally upsampled
        """
        B, N, C, H, W = frames.shape

        # ── Spatial: process all frames at once ───────────────────
        x = frames.reshape(B * N, C, H, W)
        x = self._extract_spatial_features(x)  # (B*N, 512)
        x = x.reshape(B, N, -1)                # (B, N, 512)

        # ── Temporal: BiGRU ───────────────────────────────────────
        x, _ = self.gru(x)                     # (B, N, 2*hidden)
        x = self.temporal_proj(x)               # (B, N, feat_dim)

        # ── Upsample to audio temporal resolution ─────────────────
        x = x.permute(0, 2, 1)                 # (B, feat_dim, N)
        x = self.temporal_upsample(x)           # (B, feat_dim, ~4*N)
        # Interpolate to exact audio frame count
        x = nn.functional.interpolate(
            x, size=self.n_audio_frames, mode="linear", align_corners=False
        )                                       # (B, feat_dim, T_audio)
        x = x.permute(0, 2, 1)                 # (B, T_audio, feat_dim)

        return x
