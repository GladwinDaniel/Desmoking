"""
MambaPFAN — PFAN Generator with Mamba SSM replacing Sea_Attention
================================================================
A lightweight hybrid CNN-Mamba generator for surgical smoke removal.

Architecture:
    Input(3) → 1×1 Conv(64) → ConvNeXt Block ×2 → MambaViT Stage → 1×1 Conv(3) → Tanh

The key difference from original PFAN:
    - Sea_Attention (axial attention, O(n^1.5)) is replaced with
      MambaVisionBlock (selective SSM, O(n) linear complexity)
    - Multi-directional scanning (4-way) provides full 2D global context
    - Everything else (ConvNeXt blocks, channel attention, residuals) is preserved
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import math

from models.mamba_block import MambaVisionBlock


# Reuse existing components from the original PFAN
class ChannelGate(nn.Module):
    """Channel attention via avg+max pool → MLP → sigmoid (from original PFAN)"""
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.GELU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pool = F.avg_pool2d(x, (x.size(2), x.size(3)))
            elif pool_type == 'max':
                pool = F.max_pool2d(x, (x.size(2), x.size(3)))
            else:
                continue
            channel_att_raw = self.mlp(pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MambaSwinBlock(nn.Module):
    """
    Replaces the original SwinBlock.
    Uses MambaVisionBlock instead of Sea_Attention.
    Keeps LeFF as the feed-forward network.
    """
    def __init__(self, dim, mlp_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        from models.networks import LeFF

        self.attention_block = Residual(PreNorm(dim, MambaVisionBlock(
            dim=dim, d_state=d_state, d_conv=d_conv, expand=expand
        )))
        self.mlp_block = Residual(PreNorm(dim, LeFF(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    """Patch merging / embedding layer (from original PFAN, unchanged)"""
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x)
        x = x.view(b, -1, new_h, new_w)
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class MambaStageModule(nn.Module):
    """
    Replaces the original StageModule.
    Uses MambaSwinBlock pairs instead of SwinBlock pairs.
    """
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2.'

        self.patch_partition = PatchMerging(
            in_channels=in_channels,
            out_channels=hidden_dimension,
            downscaling_factor=downscaling_factor
        )

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                MambaSwinBlock(dim=hidden_dimension, mlp_dim=hidden_dimension * 4,
                              d_state=d_state, d_conv=d_conv, expand=expand),
                MambaSwinBlock(dim=hidden_dimension, mlp_dim=hidden_dimension * 4,
                              d_state=d_state, d_conv=d_conv, expand=expand),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for block1, block2 in self.layers:
            x = block1(x)
            x = block2(x)
        return x.permute(0, 3, 1, 2)


class MambaViTs(nn.Module):
    """
    Replaces the original ViTs module.
    Uses MambaStageModule + ChannelGate for global feature extraction.
    """
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.stage1 = MambaStageModule(
            in_channels=in_channels,
            hidden_dimension=hidden_dimension,
            layers=layers,
            downscaling_factor=downscaling_factor,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.channel_att = ChannelGate(hidden_dimension, reduction_ratio=16, pool_types=['avg', 'max'])

    def forward(self, x):
        out = self.stage1(x)
        out = self.channel_att(out)
        return out + x


class MambaPFAN(nn.Module):
    """
    Mamba-PFAN: Lightweight CNN-Mamba hybrid generator for smoke removal.
    
    Architecture:
        1. Input projection: 1×1 Conv(3→ngf) + BN + LeakyReLU
        2. Local features: ConvNeXt Block × 2 (multi-scale depthwise conv)
        3. Global features: MambaViTs (selective SSM with 4-way scanning)
        4. Output: 1×1 Conv(ngf→3) + Tanh
    
    Args:
        input_nc: Input channels (3 for RGB)
        output_nc: Output channels (3 for RGB)
        ngf: Base feature dimension (default: 64)
        hidden_dim: Hidden dimension for Mamba stage
        layers: Number of Mamba blocks (list, uses index 2)
        d_state: SSM state dimension
        d_conv: Causal convolution width
        expand: SSM expansion factor
        norm_layer_1: Normalization layer for input projection
    """
    def __init__(self, *, input_nc, output_nc, ngf, hidden_dim,
                 layers, d_state=16, d_conv=4, expand=2,
                 downscaling_factors=(1, 1, 1, 1),
                 norm_layer_1='batch'):
        super().__init__()
        from models.networks import Block

        if type(norm_layer_1) == functools.partial:
            use_bias = norm_layer_1.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer_1 == nn.InstanceNorm2d

        # 1. Input projection
        model_1 = [
            nn.Conv2d(input_nc, ngf, 1, 1, 0),
            norm_layer_1(ngf),
            nn.LeakyReLU(0.05)
        ]
        self.model_1 = nn.Sequential(*model_1)

        # 2. ConvNeXt blocks (local features — kept from original PFAN)
        self.convnext1 = Block(ngf)
        self.convnext2 = Block(ngf)

        # 3. Mamba ViT stage (global features — NEW: replaces Sea_Attention)
        self.mamba_vit = MambaViTs(
            in_channels=hidden_dim,
            hidden_dimension=hidden_dim,
            layers=layers[2],
            downscaling_factor=downscaling_factors[2],
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # 4. Output projection
        self.model_3 = nn.Conv2d(hidden_dim, output_nc, 1, 1, 0)
        self.model_3_1 = nn.Tanh()

    def forward(self, img):
        x = self.model_1(img)

        x1 = self.convnext1(x)
        x1 = self.convnext2(x1)
        x2 = self.mamba_vit(x1) + x  # global + skip connection
        x3 = self.model_3(x2)
        x3 = self.model_3_1(x3)

        return x3
