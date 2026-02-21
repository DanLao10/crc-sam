# -*- coding: utf-8 -*-
"""
CRC-SAM Inference Script

Automatic (prompt-free) colorectal cancer segmentation across:
- CT images
- Colonoscopy images
- Histopathology images

Usage:
    # Single image inference
    python inference.py -i path/to/image.png -o output/ -chk work_dir/CRC-SAM/crc_sam_best.pth

    # Batch inference on folder
    python inference.py -i path/to/images/ -o output/ -chk work_dir/CRC-SAM/crc_sam_best.pth

    # With bounding box prompt (optional)
    python inference.py -i image.png -o output/ -chk model.pth --box "[100,100,400,400]"

    # Evaluate on NPY dataset with ground truth
    python inference.py -i data/npy/colonoscopy -o output/ -chk model.pth --evaluate
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from skimage import io, transform

from segment_anything import sam_model_registry
from segment_anything.modeling import build_crc_sam

join = os.path.join


def show_mask(mask, ax, random_color=False):
    """Overlay segmentation mask on image."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])  # Dodger blue
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    """Draw bounding box on image."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="lime", facecolor=(0, 0, 0, 0), lw=2)
    )


def preprocess_image(image_path: str, target_size: int = 1024):
    """
    Load and preprocess image for CRC-SAM inference.

    Args:
        image_path: Path to input image
        target_size: Target image size (default 1024 for SAM)

    Returns:
        img_tensor: Preprocessed image tensor (1, 3, H, W)
        img_original: Original image for visualization
        original_size: Original (H, W) for mask resizing
    """
    img_np = io.imread(image_path)

    # Handle grayscale images
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    elif img_np.shape[2] == 4:  # RGBA
        img_3c = img_np[:, :, :3]
    else:
        img_3c = img_np

    original_size = img_3c.shape[:2]  # (H, W)

    # Resize to target size
    img_resized = transform.resize(
        img_3c,
        (target_size, target_size),
        order=3,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)

    # Normalize to [0, 1]
    img_normalized = (img_resized - img_resized.min()) / np.clip(
        img_resized.max() - img_resized.min(), a_min=1e-8, a_max=None
    )

    # Convert to tensor (B, C, H, W)
    img_tensor = (
        torch.tensor(img_normalized)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
    )

    return img_tensor, img_3c, original_size


def postprocess_mask(
    mask_tensor: torch.Tensor,
    original_size: tuple,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Postprocess predicted mask to original image size.

    Args:
        mask_tensor: Predicted mask tensor (B, 1, H, W)
        original_size: Original (H, W) of input image
        threshold: Binarization threshold

    Returns:
        Binary mask at original resolution
    """
    mask_prob = torch.sigmoid(mask_tensor)
    mask_resized = F.interpolate(
        mask_prob,
        size=original_size,
        mode="bilinear",
        align_corners=False,
    )
    mask_binary = (mask_resized.squeeze().cpu().numpy() > threshold).astype(np.uint8)
    return mask_binary


@torch.no_grad()
def inference_single(
    model,
    image_path: str,
    output_path: str,
    device: torch.device,
    box: list = None,
    save_visualization: bool = True,
    threshold: float = 0.5,
):
    """
    Run inference on a single image.

    Args:
        model: CRC-SAM model
        image_path: Path to input image
        output_path: Directory to save outputs
        device: Torch device
        box: Optional bounding box [x_min, y_min, x_max, y_max]
        save_visualization: Whether to save visualization
        threshold: Segmentation threshold
    """
    img_tensor, img_original, original_size = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)

    if box is not None:
        # Prompted inference with bounding box
        H, W = original_size
        box_1024 = np.array(box) / np.array([W, H, W, H]) * 1024
        box_tensor = torch.tensor(box_1024).float().unsqueeze(0).to(device)
        masks, iou_pred = model.forward_with_boxes(img_tensor, box_tensor)
    else:
        # Automatic (prompt-free) inference
        masks, iou_pred = model.forward_automatic(img_tensor)

    mask_binary = postprocess_mask(masks, original_size, threshold)

    os.makedirs(output_path, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save binary mask
    mask_save_path = join(output_path, f"{base_name}_mask.png")
    io.imsave(mask_save_path, mask_binary * 255, check_contrast=False)

    # Save visualization
    if save_visualization:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_original)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(mask_binary, cmap="gray")
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")

        axes[2].imshow(img_original)
        show_mask(mask_binary, axes[2])
        if box is not None:
            show_box(box, axes[2])
        axes[2].set_title(f"Segmentation (IoU: {iou_pred.item():.3f})")
        axes[2].axis("off")

        plt.tight_layout()
        viz_path = join(output_path, f"{base_name}_visualization.png")
        plt.savefig(viz_path, dpi=150, bbox_inches="tight")
        plt.close()

    return mask_binary, iou_pred.item()


def inference_batch(
    model,
    input_dir: str,
    output_path: str,
    device: torch.device,
    save_visualization: bool = True,
    threshold: float = 0.5,
):
    """
    Run inference on a directory of images.

    Args:
        model: CRC-SAM model
        input_dir: Directory containing input images
        output_path: Directory to save outputs
        device: Torch device
        save_visualization: Whether to save visualizations
        threshold: Segmentation threshold
    """
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(join(input_dir, ext)))
        image_files.extend(glob.glob(join(input_dir, ext.upper())))

    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images in {input_dir}")

    results = []
    for image_path in tqdm(image_files, desc="Processing"):
        mask, iou = inference_single(
            model=model,
            image_path=image_path,
            output_path=output_path,
            device=device,
            box=None,
            save_visualization=save_visualization,
            threshold=threshold,
        )
        results.append({
            "image": os.path.basename(image_path),
            "iou_pred": iou,
        })

    with open(join(output_path, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    avg_iou = np.mean([r["iou_pred"] for r in results])
    print(f"\nInference complete! Average predicted IoU: {avg_iou:.4f}")
    print(f"Results saved to: {output_path}")

    return results


def compute_segmentation_metrics(pred: np.ndarray, gt: np.ndarray):
    """
    Compute comprehensive segmentation evaluation metrics.

    Args:
        pred: Binary prediction mask
        gt: Binary ground truth mask

    Returns:
        Dictionary with Dice, IoU, Precision, Recall, Specificity, F1, Accuracy
    """
    pred_flat = pred.flatten().astype(bool)
    gt_flat = gt.flatten().astype(bool)

    TP = np.sum(pred_flat & gt_flat)
    TN = np.sum(~pred_flat & ~gt_flat)
    FP = np.sum(pred_flat & ~gt_flat)
    FN = np.sum(~pred_flat & gt_flat)

    eps = 1e-8

    dice = (2 * TP) / (2 * TP + FP + FN + eps)
    iou = TP / (TP + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    specificity = TN / (TN + FP + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
    }


def evaluate_on_dataset(
    model,
    data_path: str,
    output_path: str,
    device: torch.device,
    threshold: float = 0.5,
):
    """
    Evaluate CRC-SAM on a preprocessed NPY dataset with ground truth.

    Args:
        model: CRC-SAM model
        data_path: Path to npy data folder (with gts/ and imgs/ subfolders)
        output_path: Directory to save evaluation results
        device: Torch device
        threshold: Segmentation threshold

    Returns:
        Dictionary with evaluation metrics
    """
    gt_path = join(data_path, "gts")
    img_path = join(data_path, "imgs")

    gt_files = sorted(glob.glob(join(gt_path, "*.npy")))

    all_metrics = {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "f1": [],
        "accuracy": [],
    }
    sample_results = []

    print(f"Evaluating on {len(gt_files)} samples...")

    for gt_file in tqdm(gt_files):
        img_name = os.path.basename(gt_file)
        img_file = join(img_path, img_name)

        if not os.path.exists(img_file):
            continue

        img = np.load(img_file, allow_pickle=True)
        gt = np.load(gt_file, allow_pickle=True)

        img_tensor = (
            torch.tensor(img)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            masks, _ = model.forward_automatic(img_tensor)

        mask_prob = torch.sigmoid(masks)
        mask_binary = (mask_prob > threshold).float()

        gt_tensor = (
            torch.tensor(gt > 0)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
        gt_resized = F.interpolate(
            gt_tensor,
            size=mask_binary.shape[-2:],
            mode="nearest",
        )

        pred_np = mask_binary.squeeze().cpu().numpy()
        gt_np = gt_resized.squeeze().cpu().numpy()

        metrics = compute_segmentation_metrics(pred_np, gt_np)

        sample_results.append({"sample": img_name, **metrics})

        for key in all_metrics.keys():
            all_metrics[key].append(metrics[key])

    # Compute statistics
    results = {
        "num_samples": len(sample_results),
        "threshold": threshold,
    }

    metric_names = {
        "dice": "Dice (DSC)",
        "iou": "IoU (Jaccard)",
        "precision": "Precision",
        "recall": "Recall (Sensitivity)",
        "specificity": "Specificity",
        "f1": "F1 Score",
        "accuracy": "Accuracy",
    }

    for key, display_name in metric_names.items():
        values = np.array(all_metrics[key])
        results[f"mean_{key}"] = float(np.mean(values))
        results[f"std_{key}"] = float(np.std(values))
        results[f"median_{key}"] = float(np.median(values))
        results[f"min_{key}"] = float(np.min(values))
        results[f"max_{key}"] = float(np.max(values))

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of samples: {results['num_samples']}")
    print(f"Threshold: {results['threshold']}")
    print("-" * 60)
    print(f"{'Metric':<25} {'Mean +/- Std':<20} {'Median':<10}")
    print("-" * 60)

    for key, display_name in metric_names.items():
        mean_val = results[f"mean_{key}"]
        std_val = results[f"std_{key}"]
        median_val = results[f"median_{key}"]
        print(f"{display_name:<25} {mean_val:.4f} +/- {std_val:.4f}      {median_val:.4f}")

    print("=" * 60)

    # Save results
    os.makedirs(output_path, exist_ok=True)

    with open(join(output_path, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(join(output_path, "per_sample_metrics.json"), "w") as f:
        json.dump(sample_results, f, indent=2)

    # Precision-recall analysis
    pr_analysis = generate_precision_recall_analysis(all_metrics)
    results["precision_recall_analysis"] = pr_analysis

    with open(join(output_path, "evaluation_results_complete.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


def generate_precision_recall_analysis(all_metrics: dict) -> dict:
    """Generate precision-recall analysis summary."""
    precision_vals = np.array(all_metrics["precision"])
    recall_vals = np.array(all_metrics["recall"])
    f1_vals = np.array(all_metrics["f1"])

    if len(precision_vals) > 1:
        pr_correlation = float(np.corrcoef(precision_vals, recall_vals)[0, 1])
    else:
        pr_correlation = 0.0

    high_precision_low_recall = np.sum((precision_vals > 0.8) & (recall_vals < 0.6))
    high_recall_low_precision = np.sum((recall_vals > 0.8) & (precision_vals < 0.6))
    balanced_cases = np.sum((np.abs(precision_vals - recall_vals) < 0.1))

    analysis = {
        "precision_recall_correlation": pr_correlation,
        "high_precision_low_recall_count": int(high_precision_low_recall),
        "high_recall_low_precision_count": int(high_recall_low_precision),
        "balanced_precision_recall_count": int(balanced_cases),
        "precision_quartiles": {
            "q1": float(np.percentile(precision_vals, 25)),
            "q2": float(np.percentile(precision_vals, 50)),
            "q3": float(np.percentile(precision_vals, 75)),
        },
        "recall_quartiles": {
            "q1": float(np.percentile(recall_vals, 25)),
            "q2": float(np.percentile(recall_vals, 50)),
            "q3": float(np.percentile(recall_vals, 75)),
        },
        "f1_quartiles": {
            "q1": float(np.percentile(f1_vals, 25)),
            "q2": float(np.percentile(f1_vals, 50)),
            "q3": float(np.percentile(f1_vals, 75)),
        },
    }

    return analysis


def parse_args():
    parser = argparse.ArgumentParser(description="CRC-SAM Inference")

    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to input image or directory of images",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./output",
        help="Output directory for segmentation results",
    )
    parser.add_argument(
        "-chk", "--checkpoint",
        type=str,
        required=True,
        help="Path to trained CRC-SAM checkpoint",
    )
    parser.add_argument(
        "--medsam_checkpoint",
        type=str,
        default="work_dir/MedSAM/medsam_vit_b.pth",
        help="Path to base MedSAM checkpoint (for architecture)",
    )
    parser.add_argument(
        "--box",
        type=str,
        default=None,
        help="Optional bounding box prompt: '[x_min,y_min,x_max,y_max]'",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Segmentation threshold (default: 0.5)",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="LoRA rank (must match training)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="LoRA alpha (must match training)",
    )
    parser.add_argument(
        "--no_visualization",
        action="store_true",
        help="Skip saving visualizations",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate on NPY dataset with ground truth",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load base SAM model
    print(f"Loading base model architecture...")
    sam_model = sam_model_registry["vit_b"](checkpoint=args.medsam_checkpoint)

    # Build CRC-SAM
    print(f"Building CRC-SAM with LoRA rank={args.lora_rank}")
    crc_sam_model = build_crc_sam(
        sam_model=sam_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    ).to(device)

    # Load trained checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    crc_sam_model.load_state_dict(checkpoint["model"])
    crc_sam_model.eval()
    print("Model loaded successfully!")
    crc_sam_model.print_trainable_parameters()

    # Parse box if provided
    box = None
    if args.box is not None:
        box = [int(x) for x in args.box.strip("[]").split(",")]
        print(f"Using bounding box prompt: {box}")

    # Run inference
    if args.evaluate:
        evaluate_on_dataset(
            model=crc_sam_model,
            data_path=args.input,
            output_path=args.output,
            device=device,
            threshold=args.threshold,
        )
    elif os.path.isdir(args.input):
        inference_batch(
            model=crc_sam_model,
            input_dir=args.input,
            output_path=args.output,
            device=device,
            save_visualization=not args.no_visualization,
            threshold=args.threshold,
        )
    else:
        mask, iou = inference_single(
            model=crc_sam_model,
            image_path=args.input,
            output_path=args.output,
            device=device,
            box=box,
            save_visualization=not args.no_visualization,
            threshold=args.threshold,
        )
        print(f"Inference complete! Predicted IoU: {iou:.4f}")
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
