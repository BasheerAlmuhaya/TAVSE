"""
Audio Encoder: Convolutional U-Net for STFT magnitude mask prediction.

Architecture:
    Encoder: 5 Conv2d blocks (1→32→64→128→256→512) with stride 2
    Bottleneck: 512 × 9 × 8 (for 257×251 input)
    Decoder: 5 TransposedConv2d blocks with skip connections
    Output: Sigmoid mask (1 × F × T)
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → LeakyReLU"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 2, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    """TransposedConv2d → BatchNorm → ReLU (with skip connection via concat)"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 2, padding: int = 1, output_padding: int = 1):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        # Handle size mismatch from non-power-of-2 dimensions
        if x.shape != skip.shape:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return x


class AudioEncoder(nn.Module):
    """
    U-Net audio encoder/decoder for STFT magnitude masking.

    Input:  (B, 1, F, T) — noisy magnitude spectrogram
    Output: (B, 1, F, T) — sigmoid mask

    The bottleneck representation z_a ∈ (B, 512, H', W') is accessible
    via encode() for fusion with visual features.
    """

    def __init__(self, channels: Optional[List[int]] = None):
        super().__init__()
        if channels is None:
            channels = [1, 32, 64, 128, 256, 512]

        # ── Encoder ───────────────────────────────────────────────
        self.enc1 = ConvBlock(channels[0], channels[1])  # 1→32
        self.enc2 = ConvBlock(channels[1], channels[2])  # 32→64
        self.enc3 = ConvBlock(channels[2], channels[3])  # 64→128
        self.enc4 = ConvBlock(channels[3], channels[4])  # 128→256
        self.enc5 = ConvBlock(channels[4], channels[5])  # 256→512

        # ── Decoder (input channels = current + skip) ─────────────
        self.dec5 = DeconvBlock(channels[5], channels[4])          # 512→256
        self.dec4 = DeconvBlock(channels[4] * 2, channels[3])     # 512→128
        self.dec3 = DeconvBlock(channels[3] * 2, channels[2])     # 256→64
        self.dec2 = DeconvBlock(channels[2] * 2, channels[1])     # 128→32
        # Final layer: takes concat of dec2 output + enc1 skip
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(channels[1] * 2, channels[0], 3, 2, 1, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        """
        Encoder forward pass.

        Args:
            x: (B, 1, F, T) noisy magnitude spectrogram

        Returns:
            bottleneck: (B, 512, H', W')
            skips: list of encoder feature maps for skip connections
        """
        e1 = self.enc1(x)   # (B, 32, F/2, T/2)
        e2 = self.enc2(e1)  # (B, 64, F/4, T/4)
        e3 = self.enc3(e2)  # (B, 128, F/8, T/8)
        e4 = self.enc4(e3)  # (B, 256, F/16, T/16)
        e5 = self.enc5(e4)  # (B, 512, F/32, T/32)

        return e5, [e1, e2, e3, e4]

    def decode(self, bottleneck: torch.Tensor, skips: list,
               target_shape: tuple) -> torch.Tensor:
        """
        Decoder forward pass with skip connections.

        Args:
            bottleneck: (B, 512, H', W') — possibly fused with visual
            skips: [e1, e2, e3, e4] from encoder
            target_shape: (F, T) of original input for final size matching

        Returns:
            mask: (B, 1, F, T) sigmoid mask
        """
        e1, e2, e3, e4 = skips

        d5 = self.dec5(bottleneck, e4)  # (B, 512, ...)
        d4 = self.dec4(d5, e3)          # (B, 256, ...)
        d3 = self.dec3(d4, e2)          # (B, 128, ...)
        d2 = self.dec2(d3, e1)          # (B, 64, ...)
        mask = self.dec1(d2)            # (B, 1, ...)

        # Ensure output matches input shape exactly
        if mask.shape[2:] != target_shape:
            mask = nn.functional.interpolate(
                mask, size=target_shape, mode="bilinear", align_corners=False
            )
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full U-Net forward pass (audio-only mode).

        Args:
            x: (B, 1, F, T) noisy magnitude
        Returns:
            mask: (B, 1, F, T) estimated mask
        """
        target_shape = x.shape[2:]
        bottleneck, skips = self.encode(x)
        mask = self.decode(bottleneck, skips, target_shape)
        return mask
