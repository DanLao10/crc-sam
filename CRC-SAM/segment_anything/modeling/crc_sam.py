"""
CRC-SAM: SAM-based Multi-Modal Segmentation of Colorectal Cancer

A unified framework for colorectal cancer segmentation across colonoscopy,
CT, and histopathology images. Built upon MedSAM with LoRA adaptation for
efficient domain transfer.

Reference:
    "CRC-SAM: SAM-Based Multi-Modal Segmentation and Quantification
    of Colorectal Cancer in CT, Colonoscopy, and Histology Images"
    Daniel Z. Lao, Quncai Zou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Any

from .lora import inject_lora_into_encoder, get_lora_parameters, count_lora_parameters


class CRCSAM(nn.Module):
    """
    CRC-SAM model for automatic colorectal cancer segmentation.

    Key features:
    1. Frozen image encoder with LoRA adaptation layers
    2. Fine-tunable mask decoder
    3. Learnable default prompt embedding for prompt-free inference
    4. Multi-modal support (CT, colonoscopy, histopathology)

    Args:
        image_encoder: SAM image encoder (ViT-B/H)
        mask_decoder: SAM mask decoder
        prompt_encoder: SAM prompt encoder
        lora_rank: Rank for LoRA adaptation (default: 4)
        lora_alpha: LoRA scaling factor (default: 4.0)
        lora_dropout: Dropout for LoRA layers (default: 0.0)
        freeze_prompt_encoder: Whether to freeze prompt encoder (default: True)
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        mask_decoder: nn.Module,
        prompt_encoder: nn.Module,
        lora_rank: int = 4,
        lora_alpha: float = 4.0,
        lora_dropout: float = 0.0,
        freeze_prompt_encoder: bool = True,
    ) -> None:
        super().__init__()

        # Inject LoRA into the image encoder
        self.image_encoder = inject_lora_into_encoder(
            image_encoder,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            enable_lora_q=True,
            enable_lora_k=True,
            enable_lora_v=True,
        )

        # Mask decoder is fine-tunable
        self.mask_decoder = mask_decoder

        # Prompt encoder
        self.prompt_encoder = prompt_encoder
        if freeze_prompt_encoder:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False

        # Learnable default prompt embedding for automatic (prompt-free) inference
        prompt_embed_dim = prompt_encoder.embed_dim
        self.default_sparse_embedding = nn.Parameter(
            torch.zeros(1, 2, prompt_embed_dim)  # Shape matches box prompt: (1, 2, 256)
        )
        nn.init.normal_(self.default_sparse_embedding, std=0.02)

        # Whether to use automatic (prompt-free) mode
        self.automatic_mode = True

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get all trainable parameters for optimization."""
        params = []

        # LoRA parameters
        params.extend(get_lora_parameters(self.image_encoder))

        # Mask decoder parameters
        for param in self.mask_decoder.parameters():
            if param.requires_grad:
                params.append(param)

        # Default prompt embedding
        params.append(self.default_sparse_embedding)

        return params

    def print_trainable_parameters(self) -> None:
        """Print information about trainable parameters."""
        lora_params, encoder_total = count_lora_parameters(self.image_encoder)
        decoder_params = sum(p.numel() for p in self.mask_decoder.parameters() if p.requires_grad)
        prompt_params = self.default_sparse_embedding.numel()

        total_trainable = lora_params + decoder_params + prompt_params
        total_params = encoder_total + sum(p.numel() for p in self.mask_decoder.parameters())

        print("=" * 50)
        print("CRC-SAM Trainable Parameters Summary")
        print("=" * 50)
        print(f"LoRA parameters (encoder):     {lora_params:,}")
        print(f"Mask decoder parameters:       {decoder_params:,}")
        print(f"Default prompt parameters:     {prompt_params:,}")
        print("-" * 50)
        print(f"Total trainable:               {total_trainable:,}")
        print(f"Total parameters:              {total_params:,}")
        print(f"Trainable ratio:               {100 * total_trainable / total_params:.2f}%")
        print("=" * 50)

    def forward(
        self,
        image: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        masks: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for CRC-SAM.

        Args:
            image: Input image tensor of shape (B, 3, H, W)
            boxes: Optional bounding boxes of shape (B, 4) or (B, N, 4)
            points: Optional tuple of (point_coords, point_labels)
            masks: Optional mask inputs
            multimask_output: Whether to output multiple masks

        Returns:
            Tuple of (masks, iou_predictions)
            - masks: (B, 1, H, W) or (B, 3, H, W) if multimask_output
            - iou_predictions: (B, 1) or (B, 3)
        """
        B = image.shape[0]
        device = image.device

        # Encode image with LoRA-enhanced encoder
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        # Get prompt embeddings
        if boxes is not None or points is not None:
            # Use provided prompts
            if boxes is not None:
                box_torch = torch.as_tensor(boxes, dtype=torch.float32, device=device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]  # (B, 1, 4)
            else:
                box_torch = None

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=box_torch,
                masks=masks,
            )
        else:
            # Automatic mode: use learned default embeddings
            sparse_embeddings = self.default_sparse_embedding.expand(B, -1, -1)
            dense_embeddings = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
            dense_embeddings = dense_embeddings.expand(
                B, -1,
                self.prompt_encoder.image_embedding_size[0],
                self.prompt_encoder.image_embedding_size[1]
            )

        # Decode masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale masks to original resolution
        masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        return masks, iou_predictions

    def forward_automatic(
        self,
        image: torch.Tensor,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Automatic (prompt-free) forward pass.

        Uses the learned default prompt embedding instead of box/point prompts.
        This enables fully automatic segmentation without user interaction.

        Args:
            image: Input image tensor of shape (B, 3, H, W)
            multimask_output: Whether to output multiple masks

        Returns:
            Tuple of (masks, iou_predictions)
        """
        return self.forward(
            image=image,
            boxes=None,
            points=None,
            masks=None,
            multimask_output=multimask_output,
        )

    def forward_with_boxes(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with bounding box prompts.

        Args:
            image: Input image tensor of shape (B, 3, H, W)
            boxes: Bounding boxes of shape (B, 4) - format: [x_min, y_min, x_max, y_max]
            multimask_output: Whether to output multiple masks

        Returns:
            Tuple of (masks, iou_predictions)
        """
        return self.forward(
            image=image,
            boxes=boxes,
            points=None,
            masks=None,
            multimask_output=multimask_output,
        )


def build_crc_sam(
    sam_model: nn.Module,
    lora_rank: int = 4,
    lora_alpha: float = 4.0,
    lora_dropout: float = 0.0,
    freeze_prompt_encoder: bool = True,
) -> CRCSAM:
    """
    Build a CRC-SAM model from a pretrained SAM/MedSAM model.

    Args:
        sam_model: Pretrained SAM or MedSAM model
        lora_rank: LoRA rank (default: 4, try 8 for more capacity)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout for LoRA layers
        freeze_prompt_encoder: Whether to freeze prompt encoder

    Returns:
        CRC-SAM model ready for fine-tuning
    """
    return CRCSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        freeze_prompt_encoder=freeze_prompt_encoder,
    )
