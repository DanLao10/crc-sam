"""
LoRA (Low-Rank Adaptation) module for efficient fine-tuning of vision transformers.

Based on: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2022)
Adapted for CRC-SAM: SAM-based multi-modal segmentation of colorectal cancer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Type


class LoRALinear(nn.Module):
    """
    LoRA layer for linear projections.

    Implements: W' = W + BA, where:
    - W is the frozen original weight matrix (D x D)
    - A is a low-rank matrix (r x D), r << D
    - B is a low-rank matrix (D x r)
    - Only A and B are trained

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank (r), controls the bottleneck dimension
        alpha: LoRA scaling factor (alpha/rank is the actual scaling)
        dropout: Dropout probability for LoRA path
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices A and B
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Optional dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize LoRA weights following the original paper."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the LoRA delta: x @ (B @ A)^T * scaling

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            LoRA delta of shape (..., out_features)
        """
        delta = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return delta * self.scaling


class LoRAAttention(nn.Module):
    """
    Multi-head Attention block with LoRA adaptation for Q, K, V projections.

    Wraps the original Attention module and adds LoRA adapters to the
    query, key, and value projections while keeping the original weights frozen.

    Args:
        base_attention: The original frozen Attention module
        rank: LoRA rank for Q, K, V adapters
        alpha: LoRA scaling factor
        dropout: Dropout for LoRA paths
        enable_lora_q: Enable LoRA for query projection
        enable_lora_k: Enable LoRA for key projection
        enable_lora_v: Enable LoRA for value projection
    """

    def __init__(
        self,
        base_attention: nn.Module,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        enable_lora_q: bool = True,
        enable_lora_k: bool = True,
        enable_lora_v: bool = True,
    ) -> None:
        super().__init__()
        self.base_attention = base_attention

        dim = base_attention.qkv.in_features

        # Freeze the base attention parameters
        for param in self.base_attention.parameters():
            param.requires_grad = False

        self.enable_lora_q = enable_lora_q
        self.enable_lora_k = enable_lora_k
        self.enable_lora_v = enable_lora_v

        if enable_lora_q:
            self.lora_q = LoRALinear(dim, dim, rank=rank, alpha=alpha, dropout=dropout)
        if enable_lora_k:
            self.lora_k = LoRALinear(dim, dim, rank=rank, alpha=alpha, dropout=dropout)
        if enable_lora_v:
            self.lora_v = LoRALinear(dim, dim, rank=rank, alpha=alpha, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA-enhanced attention.

        The original QKV computation is done via the frozen base attention,
        then LoRA deltas are added to enhance the projections.
        """
        B, H, W, _ = x.shape

        # Get base QKV (frozen)
        qkv = self.base_attention.qkv(x)
        qkv = qkv.reshape(B, H * W, 3, self.base_attention.num_heads, -1).permute(2, 0, 3, 1, 4)

        # Apply LoRA deltas
        if self.enable_lora_q:
            delta_q = self.lora_q(x)
            delta_q = delta_q.reshape(B, H * W, self.base_attention.num_heads, -1).permute(0, 2, 1, 3)
            qkv[0] = qkv[0] + delta_q

        if self.enable_lora_k:
            delta_k = self.lora_k(x)
            delta_k = delta_k.reshape(B, H * W, self.base_attention.num_heads, -1).permute(0, 2, 1, 3)
            qkv[1] = qkv[1] + delta_k

        if self.enable_lora_v:
            delta_v = self.lora_v(x)
            delta_v = delta_v.reshape(B, H * W, self.base_attention.num_heads, -1).permute(0, 2, 1, 3)
            qkv[2] = qkv[2] + delta_v

        # Continue with attention computation
        q, k, v = qkv.reshape(3, B * self.base_attention.num_heads, H * W, -1).unbind(0)

        attn = (q * self.base_attention.scale) @ k.transpose(-2, -1)

        if self.base_attention.use_rel_pos:
            from .image_encoder import add_decomposed_rel_pos
            attn = add_decomposed_rel_pos(
                attn, q,
                self.base_attention.rel_pos_h,
                self.base_attention.rel_pos_w,
                (H, W), (H, W)
            )

        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.base_attention.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.base_attention.proj(x)

        return x


def inject_lora_into_encoder(
    image_encoder: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
    enable_lora_q: bool = True,
    enable_lora_k: bool = True,
    enable_lora_v: bool = True,
) -> nn.Module:
    """
    Inject LoRA adapters into all attention blocks of the image encoder.

    This function:
    1. Freezes all parameters of the image encoder
    2. Wraps each Attention module with LoRAAttention
    3. Returns the modified encoder with only LoRA parameters trainable

    Args:
        image_encoder: The SAM image encoder (ImageEncoderViT)
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout for LoRA paths
        enable_lora_q/k/v: Which projections to apply LoRA to

    Returns:
        Modified image encoder with LoRA layers
    """
    # Freeze all parameters in the encoder
    for param in image_encoder.parameters():
        param.requires_grad = False

    # Inject LoRA into each transformer block
    for i, block in enumerate(image_encoder.blocks):
        lora_attn = LoRAAttention(
            base_attention=block.attn,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            enable_lora_q=enable_lora_q,
            enable_lora_k=enable_lora_k,
            enable_lora_v=enable_lora_v,
        )
        block.attn = lora_attn

    return image_encoder


def get_lora_parameters(model: nn.Module):
    """
    Get only the LoRA parameters from a model for optimization.

    Args:
        model: Model containing LoRA layers

    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
            lora_params.append(param)
    return lora_params


def count_lora_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable LoRA parameters vs total parameters.

    Args:
        model: Model containing LoRA layers

    Returns:
        Tuple of (lora_params, total_params)
    """
    lora_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'lora_' in name:
            lora_params += param.numel()

    return lora_params, total_params
