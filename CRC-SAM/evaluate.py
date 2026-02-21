# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation Script for CRC-SAM

Evaluates trained CRC-SAM models with multiple metrics:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU / Jaccard)
- Precision
- Recall (Sensitivity)
- Specificity
- F1 Score
- Accuracy
- Hausdorff Distance 95 (HD95) - optional
- Average Surface Distance (ASD) - optional

Usage:
    # Basic evaluation
    python evaluate.py -i data/npy/test -chk work_dir/CRC-SAM/crc_sam_best.pth -o results/

    # With surface metrics (slower but more comprehensive)
    python evaluate.py -i data/npy/test -chk work_dir/CRC-SAM/crc_sam_best.pth -o results/ --surface_metrics

    # Multi-modal evaluation
    python evaluate.py -i data/npy/colonoscopy data/npy/CT data/npy/histology -chk model.pth -o results/
"""

import numpy as np
import os
import glob
import argparse
import json
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from segment_anything import sam_model_registry
from segment_anything.modeling import build_crc_sam
from utils.evaluation_metrics import SegmentationEvaluator

join = os.path.join


class EvalDataset(Dataset):
    """Dataset for evaluation."""

    def __init__(self, data_root: str, modality: str = "unknown"):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.modality = modality

        self.gt_path_files = sorted(glob.glob(join(self.gt_path, "*.npy")))
        print(f"[{modality}] Found {len(self.gt_path_files)} samples")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )

        if len(img_1024.shape) == 2:
            img_1024 = np.repeat(img_1024[:, :, None], 3, axis=-1)

        img_1024 = np.transpose(img_1024, (2, 0, 1))

        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )
        gt2D = np.uint8(gt > 0)

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D).long(),
            img_name,
            self.modality,
        )


@torch.no_grad()
def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    include_surface_metrics: bool = False,
    save_predictions: bool = False,
    output_path: str = None,
):
    """
    Evaluate model with comprehensive metrics.

    Args:
        model: CRC-SAM model
        dataloader: Evaluation dataloader
        device: Torch device
        threshold: Binarization threshold
        include_surface_metrics: Whether to compute HD95 and ASD
        save_predictions: Whether to save prediction masks
        output_path: Path to save predictions

    Returns:
        Evaluation results dictionary
    """
    model.eval()
    evaluator = SegmentationEvaluator(include_surface_metrics=include_surface_metrics)

    all_predictions = []
    all_ground_truths = []
    all_names = []
    all_modalities = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images, gts, names, modalities = batch
        images = images.to(device)

        masks, _ = model.forward_automatic(images)

        pred_probs = torch.sigmoid(masks)
        pred_binary = (pred_probs > threshold).float()

        if pred_binary.shape[-2:] != gts.shape[-2:]:
            pred_binary = F.interpolate(
                pred_binary,
                size=gts.shape[-2:],
                mode="nearest",
            )

        batch_size = pred_binary.shape[0]
        for i in range(batch_size):
            pred_np = pred_binary[i].squeeze().cpu().numpy()
            gt_np = gts[i].numpy()

            all_predictions.append(pred_np)
            all_ground_truths.append(gt_np)
            all_names.append(names[i])
            all_modalities.append(modalities[i])

            if save_predictions and output_path:
                pred_save_path = join(output_path, "predictions")
                os.makedirs(pred_save_path, exist_ok=True)
                np.save(join(pred_save_path, names[i]), pred_np)

    # Compute comprehensive metrics
    results = evaluator.evaluate_batch(
        predictions=all_predictions,
        ground_truths=all_ground_truths,
        sample_names=all_names,
    )

    # Add modality information
    for i, sample in enumerate(results["per_sample"]):
        sample["modality"] = all_modalities[i]

    # Per-modality metrics
    modality_metrics = {}
    unique_modalities = list(set(all_modalities))

    for mod in unique_modalities:
        mod_indices = [i for i, m in enumerate(all_modalities) if m == mod]
        mod_preds = [all_predictions[i] for i in mod_indices]
        mod_gts = [all_ground_truths[i] for i in mod_indices]
        mod_names = [all_names[i] for i in mod_indices]

        mod_results = evaluator.evaluate_batch(mod_preds, mod_gts, mod_names)
        modality_metrics[mod] = mod_results["aggregated"]

    results["per_modality"] = modality_metrics

    return results


def print_detailed_results(results: dict):
    """Print detailed evaluation results."""
    agg = results["aggregated"]

    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total samples: {agg['num_samples']}")
    print("-" * 80)

    metrics_display = [
        ("Dice (DSC)", "dice"),
        ("IoU (Jaccard)", "iou"),
        ("Precision", "precision"),
        ("Recall (Sensitivity)", "recall"),
        ("Specificity", "specificity"),
        ("F1 Score", "f1"),
        ("Accuracy", "accuracy"),
    ]

    if "mean_hd95" in agg:
        metrics_display.extend([
            ("HD95", "hd95"),
            ("ASD", "asd"),
        ])

    print(f"\n{'Metric':<25} {'Mean +/- Std':<20} {'Median':<10} {'Min':<10} {'Max':<10}")
    print("-" * 80)

    for display_name, key in metrics_display:
        mean_val = agg.get(f"mean_{key}", float('nan'))
        std_val = agg.get(f"std_{key}", float('nan'))
        median_val = agg.get(f"median_{key}", float('nan'))
        min_val = agg.get(f"min_{key}", float('nan'))
        max_val = agg.get(f"max_{key}", float('nan'))

        if np.isfinite(mean_val):
            print(f"{display_name:<25} {mean_val:.4f} +/- {std_val:.4f}    "
                  f"{median_val:.4f}     {min_val:.4f}     {max_val:.4f}")

    # Precision-Recall Analysis
    print("\n" + "-" * 80)
    print("PRECISION-RECALL ANALYSIS")
    print("-" * 80)

    pr_corr = agg.get("precision_recall_correlation", float('nan'))
    if np.isfinite(pr_corr):
        print(f"Precision-Recall Correlation: {pr_corr:.4f}")
        if pr_corr > 0.5:
            print("  -> High positive correlation: Precision and recall tend to vary together")
        elif pr_corr < -0.5:
            print("  -> High negative correlation: Trade-off between precision and recall")
        else:
            print("  -> Low correlation: Precision and recall are relatively independent")

    # Per-modality results
    if "per_modality" in results and len(results["per_modality"]) > 1:
        print("\n" + "-" * 80)
        print("PER-MODALITY RESULTS")
        print("-" * 80)

        for mod, mod_agg in results["per_modality"].items():
            print(f"\n{mod.upper()} (n={mod_agg['num_samples']}):")
            print(f"  Dice:      {mod_agg['mean_dice']:.4f} +/- {mod_agg['std_dice']:.4f}")
            print(f"  IoU:       {mod_agg['mean_iou']:.4f} +/- {mod_agg['std_iou']:.4f}")
            print(f"  Precision: {mod_agg['mean_precision']:.4f} +/- {mod_agg['std_precision']:.4f}")
            print(f"  Recall:    {mod_agg['mean_recall']:.4f} +/- {mod_agg['std_recall']:.4f}")

    print("\n" + "=" * 80)


def generate_latex_table(results: dict) -> str:
    """Generate LaTeX table for paper inclusion."""
    agg = results["aggregated"]

    latex = """
\\begin{table}[h]
\\centering
\\caption{Comprehensive Evaluation Results}
\\label{tab:eval_results}
\\begin{tabular}{lcccc}
\\toprule
Metric & Mean & Std & Median & Range \\\\
\\midrule
"""

    metrics = [
        ("Dice (DSC)", "dice"),
        ("IoU (Jaccard)", "iou"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("Specificity", "specificity"),
        ("F1 Score", "f1"),
        ("Accuracy", "accuracy"),
    ]

    for display_name, key in metrics:
        mean_val = agg.get(f"mean_{key}", 0)
        std_val = agg.get(f"std_{key}", 0)
        median_val = agg.get(f"median_{key}", 0)
        min_val = agg.get(f"min_{key}", 0)
        max_val = agg.get(f"max_{key}", 0)

        latex += f"{display_name} & {mean_val:.4f} & {std_val:.4f} & {median_val:.4f} & [{min_val:.4f}, {max_val:.4f}] \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    return latex


def parse_args():
    parser = argparse.ArgumentParser(description="CRC-SAM Comprehensive Evaluation")

    parser.add_argument(
        "-i", "--data_paths",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to test npy folders with gts/ and imgs/ subfolders",
    )
    parser.add_argument(
        "-chk", "--checkpoint",
        type=str,
        required=True,
        help="Path to trained CRC-SAM checkpoint",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--medsam_checkpoint",
        type=str,
        default="work_dir/MedSAM/medsam_vit_b.pth",
        help="Path to base MedSAM checkpoint (for architecture)",
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
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--surface_metrics",
        action="store_true",
        help="Compute surface-based metrics (HD95, ASD) - slower",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save prediction masks",
    )
    parser.add_argument(
        "--generate_latex",
        action="store_true",
        help="Generate LaTeX table for paper",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for evaluation",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = join(args.output, f"eval_{timestamp}")
    os.makedirs(output_path, exist_ok=True)

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

    # Setup datasets
    datasets = []
    for path in args.data_paths:
        modality = os.path.basename(path.rstrip("/"))
        ds = EvalDataset(data_root=path, modality=modality)
        datasets.append(ds)

    if len(datasets) > 1:
        eval_dataset = ConcatDataset(datasets)
    else:
        eval_dataset = datasets[0]

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    print(f"\nTotal evaluation samples: {len(eval_dataset)}")
    print(f"Surface metrics: {'Enabled' if args.surface_metrics else 'Disabled'}")

    # Run evaluation
    print("\n" + "=" * 80)
    print("Starting comprehensive evaluation...")
    print("=" * 80)

    results = evaluate_model(
        model=crc_sam_model,
        dataloader=eval_dataloader,
        device=device,
        threshold=args.threshold,
        include_surface_metrics=args.surface_metrics,
        save_predictions=args.save_predictions,
        output_path=output_path,
    )

    print_detailed_results(results)

    # Save results
    with open(join(output_path, "evaluation_summary.json"), "w") as f:
        json.dump(results["aggregated"], f, indent=2)

    with open(join(output_path, "per_sample_metrics.json"), "w") as f:
        json.dump(results["per_sample"], f, indent=2)

    if "per_modality" in results:
        with open(join(output_path, "per_modality_metrics.json"), "w") as f:
            json.dump(results["per_modality"], f, indent=2)

    with open(join(output_path, "complete_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if args.generate_latex:
        latex_table = generate_latex_table(results)
        with open(join(output_path, "results_table.tex"), "w") as f:
            f.write(latex_table)
        print(f"\nLaTeX table saved to: {join(output_path, 'results_table.tex')}")

    print(f"\nAll results saved to: {output_path}")
    print("Files generated:")
    print("  - evaluation_summary.json (aggregated metrics)")
    print("  - per_sample_metrics.json (per-sample details)")
    print("  - per_modality_metrics.json (per-modality breakdown)")
    print("  - complete_results.json (all data)")
    if args.generate_latex:
        print("  - results_table.tex (LaTeX table for paper)")


if __name__ == "__main__":
    main()
