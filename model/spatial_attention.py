"""
Spatial Attention temporal fusion module.

Source: VideoMatt paper (Li et al., CVPR Workshop 2023), Section 4.2, Spatial Attention^gamma (Eq. 8):
    F'_{n+1} = SpatialAttn(Conv(Concat(F_n, F_{n+1})))

The module fuses feature maps from two consecutive frames using self-attention,
giving the model short-term pixel-level correspondence between frames.
This complements ConvGRU's long-term memory with precise frame-to-frame alignment.

Applied at the 1/8 and 1/4 decoder scales only (not full-res — too expensive).
"""

import torch
from torch import nn, Tensor
from typing import Optional


class SpatialAttention(nn.Module):
    """
    Spatial Attention^gamma from VideoMatt (Eq. 8).

    Given feature maps F_prev (previous frame) and F_curr (current frame):
        1. Concatenate along channel dim
        2. 1x1 conv to reduce back to original channel count
        3. Self-attention: Q, K, V projections -> softmax -> weighted sum
        4. Residual add to F_curr

    Args:
        channels: Number of channels in the feature maps.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Step 2: reduce concatenated (2*C) back to C
        self.fusion_conv = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)

        # Step 3: self-attention projections
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1, bias=False)
        self.key   = nn.Conv2d(channels, channels // 8, kernel_size=1, bias=False)
        self.value = nn.Conv2d(channels, channels,      kernel_size=1, bias=False)

        # Learnable scale for residual — starts at 0 so it begins as identity
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f_prev: Optional[Tensor], f_curr: Tensor) -> Tensor:
        """
        Args:
            f_prev: Feature map from previous frame [B, C, H, W].
                    Pass None on the first frame — returns f_curr unchanged.
            f_curr: Feature map from current frame  [B, C, H, W].

        Returns:
            Updated feature map for current frame   [B, C, H, W].
        """
        if f_prev is None:
            return f_curr

        B, C, H, W = f_curr.shape

        # 1. Fuse prev + curr
        fused = self.fusion_conv(torch.cat([f_prev, f_curr], dim=1))  # [B, C, H, W]

        # 2. Q, K, V  — shaped [B, num_heads=1, HW, head_dim] for scaled_dot_product_attention
        Q = self.query(fused).view(B, 1, -1, H * W).permute(0, 1, 3, 2)  # [B, 1, HW, C//8]
        K = self.key(fused).view(B, 1, -1, H * W).permute(0, 1, 3, 2)    # [B, 1, HW, C//8]
        V = self.value(fused).view(B, 1, -1, H * W).permute(0, 1, 3, 2)  # [B, 1, HW, C]

        # 3+4. Memory-efficient attention (FlashAttention — never materialises HW x HW matrix)
        out = nn.functional.scaled_dot_product_attention(Q, K, V)  # [B, 1, HW, C]
        out = out.squeeze(1).permute(0, 2, 1).view(B, C, H, W)

        # 5. Residual
        return f_curr + self.gamma * out
