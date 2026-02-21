"""
CRC-SAM Training Script

Train CRC-SAM for colorectal cancer segmentation across multiple modalities:
- CT (MSD-Colon dataset)
- Colonoscopy (CVC-ClinicDB dataset)
- Histopathology (EBHI-Seg dataset)

Key features:
- Frozen image encoder with LoRA adaptation
- Fine-tunable mask decoder
- Learnable default prompt for automatic segmentation
- Multi-modal training support

Usage:
    # Single modality training
    python train.py -i data/npy/colonoscopy -task_name CRC-SAM-colonoscopy

    # Multi-modal training
    python train.py -i data/npy/colonoscopy data/npy/CT data/npy/histology_ebhi -task_name CRC-SAM-multimodal

    # With custom LoRA rank
    python train.py -i data/npy/colonoscopy -task_name CRC-SAM-colonoscopy --lora_rank 8
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
import argparse
from datetime import datetime
from tqdm import tqdm
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import monai
from segment_anything import sam_model_registry
from segment_anything.modeling import build_crc_sam
from utils.evaluation_metrics import SegmentationEvaluator

join = os.path.join

# Set seeds for reproducibility
torch.manual_seed(2023)
torch.cuda.empty_cache()

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"


def show_mask(mask, ax, random_color=False):
    """Visualize segmentation mask."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    """Visualize bounding box."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


@torch.no_grad()
def validate_with_metrics(
    model,
    val_dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    use_automatic_mode: bool = True,
) -> dict:
    """
    Validate model with comprehensive evaluation metrics.

    Returns:
        Dictionary with all validation metrics (dice, iou, precision, recall, etc.)
    """
    model.eval()
    evaluator = SegmentationEvaluator(include_surface_metrics=False)

    all_metrics = {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "f1": [],
        "accuracy": [],
    }

    for batch in tqdm(val_dataloader, desc="Validating"):
        image, gt2D, boxes, names, modalities = batch
        image, gt2D = image.to(device), gt2D.to(device)

        if use_automatic_mode:
            pred_masks, _ = model.forward_automatic(image)
        else:
            boxes_np = boxes.detach().cpu().numpy()
            pred_masks, _ = model.forward_with_boxes(image, boxes_np)

        pred_binary = (torch.sigmoid(pred_masks) > threshold).float()

        batch_size = pred_binary.shape[0]
        for i in range(batch_size):
            pred_np = pred_binary[i].squeeze().cpu().numpy()
            gt_np = gt2D[i].squeeze().cpu().numpy()

            metrics = evaluator.compute_all_metrics(pred_np, gt_np)

            for key in all_metrics.keys():
                all_metrics[key].append(metrics[key])

    model.train()

    results = {}
    for key, values in all_metrics.items():
        values = np.array(values)
        results[f"val_{key}_mean"] = float(np.mean(values))
        results[f"val_{key}_std"] = float(np.std(values))

    precision_vals = np.array(all_metrics["precision"])
    recall_vals = np.array(all_metrics["recall"])
    if len(precision_vals) > 1:
        results["val_precision_recall_corr"] = float(
            np.corrcoef(precision_vals, recall_vals)[0, 1]
        )

    return results


class CRCDataset(Dataset):
    """
    Dataset for CRC-SAM training.

    Supports both prompted (with bounding box) and automatic (prompt-free) training.

    Args:
        data_root: Path to npy data folder (with gts/ and imgs/ subfolders)
        bbox_shift: Random perturbation for bounding box augmentation
        modality: Modality name for logging (e.g., 'colonoscopy', 'CT', 'histology')
        use_automatic_mode: If True, don't provide bounding boxes during training
    """

    def __init__(
        self,
        data_root: str,
        bbox_shift: int = 20,
        modality: str = "unknown",
        use_automatic_mode: bool = False,
    ):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.bbox_shift = bbox_shift
        self.modality = modality
        self.use_automatic_mode = use_automatic_mode

        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )

        # Filter to only files with corresponding images
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]

        print(f"[{modality}] Number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # Load image (1024, 1024, 3), normalized to [0, 1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )

        # Convert shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))

        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "Image should be normalized to [0, 1]"

        # Load ground truth
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )

        # Handle multi-label masks
        label_ids = np.unique(gt)[1:]  # Exclude background (0)
        if len(label_ids) == 0:
            gt2D = np.zeros_like(gt, dtype=np.uint8)
        else:
            gt2D = np.uint8(gt == random.choice(label_ids.tolist()))

        assert np.max(gt2D) <= 1 and np.min(gt2D) >= 0, "Ground truth should be 0 or 1"

        # Compute bounding box from ground truth
        if not self.use_automatic_mode and np.sum(gt2D) > 0:
            y_indices, x_indices = np.where(gt2D > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            H, W = gt2D.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            bboxes = np.array([x_min, y_min, x_max, y_max])
        else:
            bboxes = np.array([0, 0, 0, 0])

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
            self.modality,
        )


class MultiModalDataset(Dataset):
    """
    Combined dataset for multi-modal training.

    Wraps multiple modality-specific datasets and samples uniformly from all.
    """

    def __init__(self, datasets: list):
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, index):
        for i, cumsum in enumerate(self.cumulative_sizes):
            if index < cumsum:
                if i == 0:
                    local_index = index
                else:
                    local_index = index - self.cumulative_sizes[i - 1]
                return self.datasets[i][local_index]
        raise IndexError("Index out of range")


def parse_args():
    parser = argparse.ArgumentParser(description="CRC-SAM Training")

    # Data paths
    parser.add_argument(
        "-i", "--tr_npy_paths",
        type=str,
        nargs="+",
        default=["data/npy/colonoscopy"],
        help="Path(s) to training npy folders; each should have gts/ and imgs/ subfolders",
    )

    # Model configuration
    parser.add_argument("-task_name", type=str, default="CRC-SAM")
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument(
        "-checkpoint",
        type=str,
        default="work_dir/MedSAM/medsam_vit_b.pth",
        help="Path to MedSAM checkpoint",
    )
    parser.add_argument("-work_dir", type=str, default="./work_dir")

    # LoRA configuration
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="LoRA rank (r). Higher = more capacity but more params. Try 4, 8, or 16.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="LoRA scaling factor (alpha). Usually set equal to rank.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="Dropout for LoRA layers",
    )

    # Training configuration
    parser.add_argument("-num_epochs", type=int, default=200)
    parser.add_argument("-batch_size", type=int, default=4)
    parser.add_argument("-num_workers", type=int, default=4)
    parser.add_argument(
        "-lr", type=float, default=1e-4,
        help="Learning rate for LoRA and decoder parameters",
    )
    parser.add_argument(
        "-weight_decay", type=float, default=0.01,
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--use_automatic_mode",
        action="store_true",
        help="Train for automatic (prompt-free) segmentation",
    )
    parser.add_argument(
        "--use_mixed_mode",
        action="store_true",
        help="Alternate between automatic and prompted training",
    )
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--use_wandb", action="store_true", help="Use W&B for logging")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Validation configuration
    parser.add_argument(
        "--val_npy_paths",
        type=str,
        nargs="*",
        default=None,
        help="Path(s) to validation npy folders for comprehensive metric evaluation",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=10,
        help="Run validation every N epochs",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup output directory
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
    os.makedirs(model_save_path, exist_ok=True)

    # Copy training script for reference
    shutil.copyfile(
        __file__,
        join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    # Load base SAM/MedSAM model
    print(f"Loading checkpoint from: {args.checkpoint}")
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)

    # Build CRC-SAM model with LoRA
    print(f"\nBuilding CRC-SAM with LoRA rank={args.lora_rank}, alpha={args.lora_alpha}")
    crc_sam_model = build_crc_sam(
        sam_model=sam_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        freeze_prompt_encoder=True,
    ).to(device)

    # Print parameter summary
    crc_sam_model.print_trainable_parameters()

    # Setup datasets
    datasets = []
    for path in args.tr_npy_paths:
        modality = os.path.basename(path.rstrip("/"))
        ds = CRCDataset(
            data_root=path,
            modality=modality,
            use_automatic_mode=args.use_automatic_mode,
        )
        datasets.append(ds)

    if len(datasets) > 1:
        train_dataset = MultiModalDataset(datasets)
        print(f"\nMulti-modal training with {len(datasets)} modalities")
    else:
        train_dataset = datasets[0]

    print(f"Total training samples: {len(train_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Setup validation dataloader if validation paths provided
    val_dataloader = None
    if args.val_npy_paths:
        val_datasets = []
        for path in args.val_npy_paths:
            modality = os.path.basename(path.rstrip("/"))
            val_ds = CRCDataset(
                data_root=path,
                modality=modality,
                use_automatic_mode=args.use_automatic_mode,
                bbox_shift=0,
            )
            val_datasets.append(val_ds)

        if len(val_datasets) > 1:
            val_dataset = MultiModalDataset(val_datasets)
        else:
            val_dataset = val_datasets[0]

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print(f"Validation samples: {len(val_dataset)}")

    # Setup optimizer with only trainable parameters
    trainable_params = crc_sam_model.get_trainable_parameters()
    print(f"Number of trainable parameter groups: {len(trainable_params)}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )

    # Loss functions
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        crc_sam_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

    # Setup AMP
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # Setup W&B
    if args.use_wandb:
        import wandb
        wandb.login()
        wandb.init(
            project=args.task_name,
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "data_paths": args.tr_npy_paths,
                "model_type": args.model_type,
                "num_epochs": args.num_epochs,
            },
        )

    # Training loop
    crc_sam_model.train()
    losses = []
    best_loss = float("inf")

    print(f"\nStarting training for {args.num_epochs} epochs...")
    print("=" * 50)

    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0
        epoch_dice_loss = 0
        epoch_ce_loss = 0

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            image, gt2D, boxes, names, modalities = batch
            image, gt2D = image.to(device), gt2D.to(device)

            optimizer.zero_grad()

            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    if args.use_automatic_mode:
                        pred_masks, _ = crc_sam_model.forward_automatic(image)
                    elif args.use_mixed_mode and random.random() < 0.5:
                        pred_masks, _ = crc_sam_model.forward_automatic(image)
                    else:
                        boxes_np = boxes.detach().cpu().numpy()
                        pred_masks, _ = crc_sam_model.forward_with_boxes(image, boxes_np)

                    dice = seg_loss(pred_masks, gt2D)
                    ce = ce_loss(pred_masks, gt2D.float())
                    loss = dice + ce

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.use_automatic_mode:
                    pred_masks, _ = crc_sam_model.forward_automatic(image)
                elif args.use_mixed_mode and random.random() < 0.5:
                    pred_masks, _ = crc_sam_model.forward_automatic(image)
                else:
                    boxes_np = boxes.detach().cpu().numpy()
                    pred_masks, _ = crc_sam_model.forward_with_boxes(image, boxes_np)

                dice = seg_loss(pred_masks, gt2D)
                ce = ce_loss(pred_masks, gt2D.float())
                loss = dice + ce

                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            epoch_dice_loss += dice.item()
            epoch_ce_loss += ce.item()

        scheduler.step()

        # Compute average losses
        num_steps = len(train_dataloader)
        epoch_loss /= num_steps
        epoch_dice_loss /= num_steps
        epoch_ce_loss /= num_steps
        losses.append(epoch_loss)

        # Logging
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch} | Loss: {epoch_loss:.4f} | "
            f"Dice: {epoch_dice_loss:.4f} | CE: {epoch_ce_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if args.use_wandb:
            wandb.log({
                "epoch_loss": epoch_loss,
                "dice_loss": epoch_dice_loss,
                "ce_loss": epoch_ce_loss,
                "learning_rate": current_lr,
            })

        # Run validation with comprehensive metrics
        val_metrics = None
        if val_dataloader is not None and (epoch + 1) % args.val_interval == 0:
            print(f"  Running validation with comprehensive metrics...")
            val_metrics = validate_with_metrics(
                model=crc_sam_model,
                val_dataloader=val_dataloader,
                device=device,
                threshold=0.5,
                use_automatic_mode=args.use_automatic_mode or args.use_mixed_mode,
            )

            print(f"  Validation Results:")
            print(f"    Dice: {val_metrics['val_dice_mean']:.4f} +/- {val_metrics['val_dice_std']:.4f}")
            print(f"    IoU:  {val_metrics['val_iou_mean']:.4f} +/- {val_metrics['val_iou_std']:.4f}")
            print(f"    Precision: {val_metrics['val_precision_mean']:.4f} +/- {val_metrics['val_precision_std']:.4f}")
            print(f"    Recall:    {val_metrics['val_recall_mean']:.4f} +/- {val_metrics['val_recall_std']:.4f}")
            print(f"    F1:        {val_metrics['val_f1_mean']:.4f} +/- {val_metrics['val_f1_std']:.4f}")

            if args.use_wandb:
                wandb.log(val_metrics)

        # Save latest checkpoint
        checkpoint = {
            "model": crc_sam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "loss": epoch_loss,
            "args": vars(args),
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, join(model_save_path, "crc_sam_latest.pth"))

        # Save best checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, join(model_save_path, "crc_sam_best.pth"))
            print(f"  -> New best model saved (loss: {best_loss:.4f})")

        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(
                checkpoint,
                join(model_save_path, f"crc_sam_epoch{epoch+1}.pth")
            )

        # Plot and save loss curve
        if (epoch + 1) % 10 == 0:
            plt.figure(figsize=(10, 5))
            plt.plot(losses)
            plt.title("CRC-SAM Training Loss (Dice + CE)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.savefig(join(model_save_path, "training_loss.png"), dpi=150)
            plt.close()

    print("\n" + "=" * 50)
    print(f"Training complete! Best loss: {best_loss:.4f}")
    print(f"Model saved to: {model_save_path}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
