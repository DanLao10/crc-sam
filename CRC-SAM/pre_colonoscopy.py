# -*- coding: utf-8 -*-
"""
Preprocess colonoscopy images (CVC-ClinicDB) for CRC-SAM training.

Converts PNG images + binary masks into per-image NPY files at 1024x1024.

Dataset: CVC-ClinicDB
    https://polyp.grand-challenge.org/CVCClinicDB/

Usage:
    python pre_colonoscopy.py \\
        --img_path data/CVC-ClinicDB/images \\
        --gt_path  data/CVC-ClinicDB/masks \\
        --npy_path data/npy/colonoscopy

Output structure:
    data/npy/colonoscopy/
        imgs/   # (1024, 1024, 3) float64 normalised to [0,1]
        gts/    # (1024, 1024) uint8 binary masks
"""

import numpy as np
import matplotlib.image as mpimg
import os
import argparse
from skimage import transform
from tqdm import tqdm
import cc3d

join = os.path.join

IMAGE_SIZE = 1024
VOXEL_NUM_THRE_2D = 100


def preprocess_colonoscopy(args):
    img_path = args.img_path
    gt_path = args.gt_path
    npy_path = args.npy_path
    train_ratio = args.train_ratio

    os.makedirs(join(npy_path, "gts"), exist_ok=True)
    os.makedirs(join(npy_path, "imgs"), exist_ok=True)

    names = sorted(os.listdir(gt_path))
    print(f"Total mask files: {len(names)}")

    # Keep only names whose corresponding image exists
    names = [
        n for n in names
        if os.path.exists(join(img_path, n.replace(args.gt_suffix, args.img_suffix)))
    ]
    print(f"After sanity check: {len(names)} files")

    tr_num = int(len(names) * train_ratio + 0.5)
    subset = names[:tr_num]
    print(f"Processing {len(subset)} images (train split, {train_ratio*100:.0f}%)")

    for name in tqdm(subset):
        image_name = name.replace(args.gt_suffix, args.img_suffix)
        gt_data = mpimg.imread(join(gt_path, name))

        # Handle multi-channel masks
        if gt_data.ndim == 3:
            gt_i = np.copy(gt_data[:, :, 0].astype(np.uint8))
        else:
            gt_i = np.copy(gt_data.astype(np.uint8))

        # Remove small objects
        gt_clean = cc3d.dust(
            gt_i, threshold=VOXEL_NUM_THRE_2D, connectivity=8, in_place=True
        )

        if gt_clean.max() == 0:
            continue

        # Load and resize image
        image_data = mpimg.imread(join(img_path, image_name))
        img_resized = transform.resize(
            image_data, (IMAGE_SIZE, IMAGE_SIZE),
            order=3, preserve_range=True, anti_aliasing=True,
        )
        img_norm = (img_resized - img_resized.min()) / np.clip(
            img_resized.max() - img_resized.min(), a_min=1e-8, a_max=None
        )

        # Resize ground truth
        gt_resized = transform.resize(
            gt_clean, (IMAGE_SIZE, IMAGE_SIZE),
            order=0, preserve_range=True, anti_aliasing=False,
        )
        gt_resized = np.uint8(gt_resized)

        assert img_norm.shape[:2] == gt_resized.shape

        stem = name.split(args.gt_suffix)[0]
        np.save(join(npy_path, "imgs", stem + ".npy"), img_norm)
        np.save(join(npy_path, "gts", stem + ".npy"), gt_resized)

    print("Done! Output directory:", npy_path)


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess CVC-ClinicDB for CRC-SAM")
    p.add_argument("--img_path", type=str, default="data/CVC-ClinicDB/images")
    p.add_argument("--gt_path", type=str, default="data/CVC-ClinicDB/masks")
    p.add_argument("--npy_path", type=str, default="data/npy/colonoscopy")
    p.add_argument("--img_suffix", type=str, default=".png")
    p.add_argument("--gt_suffix", type=str, default=".png")
    p.add_argument("--train_ratio", type=float, default=0.8,
                    help="Fraction of data for training (default: 0.8)")
    return p.parse_args()


if __name__ == "__main__":
    preprocess_colonoscopy(parse_args())
