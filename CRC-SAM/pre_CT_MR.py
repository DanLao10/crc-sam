# -*- coding: utf-8 -*-
"""
Preprocess CT volumes (MSD-Colon / Decathlon) for CRC-SAM training.

Converts NIfTI CT volumes + segmentation masks into per-slice NPY files
at 1024x1024 resolution with CT windowing (level=40, width=400).

Dataset: Medical Segmentation Decathlon - Task10_Colon
    http://medicaldecathlon.com/

Usage:
    python pre_CT_MR.py \\
        --nii_path data/Task10_Colon/imagesTr \\
        --gt_path  data/Task10_Colon/labelsTr \\
        --npy_path data/npy/CT

Output structure:
    data/npy/CT/
        imgs/   # (1024, 1024, 3) float64 normalised to [0,1]
        gts/    # (1024, 1024) uint8 label masks
"""

import numpy as np
import SimpleITK as sitk
import os
import argparse
from skimage import transform
from tqdm import tqdm
import cc3d

join = os.path.join

# ---------- default CT windowing (abdomen) ----------
WINDOW_LEVEL = 40
WINDOW_WIDTH = 400

IMAGE_SIZE = 1024
VOXEL_NUM_THRE_2D = 100
VOXEL_NUM_THRE_3D = 1000


def preprocess_ct(args):
    nii_path = args.nii_path
    gt_path = args.gt_path
    npy_path = args.npy_path
    prefix = args.prefix
    img_suffix = args.img_suffix
    gt_suffix = args.gt_suffix
    num_train = args.num_train

    os.makedirs(join(npy_path, "gts"), exist_ok=True)
    os.makedirs(join(npy_path, "imgs"), exist_ok=True)

    names = sorted(os.listdir(gt_path))
    print(f"Total ground-truth files: {len(names)}")

    # Keep only names whose corresponding image exists
    names = [
        n for n in names
        if os.path.exists(join(nii_path, n.split(gt_suffix)[0] + img_suffix))
    ]
    print(f"After sanity check: {len(names)} files")

    remove_label_ids = [12]  # e.g. duodenum (hard to box)

    subset = names[:num_train] if num_train else names
    print(f"Processing {len(subset)} volumes for training split")

    for name in tqdm(subset):
        image_name = name.split(gt_suffix)[0] + img_suffix
        gt_sitk = sitk.ReadImage(join(gt_path, name))
        gt_data = np.uint8(sitk.GetArrayFromImage(gt_sitk))

        # Remove excluded labels
        for rid in remove_label_ids:
            gt_data[gt_data == rid] = 0

        # Remove small 3-D objects
        gt_data = cc3d.dust(
            gt_data, threshold=VOXEL_NUM_THRE_3D, connectivity=26, in_place=True
        )
        # Remove small 2-D objects per slice
        for s in range(gt_data.shape[0]):
            gt_data[s] = cc3d.dust(
                gt_data[s], threshold=VOXEL_NUM_THRE_2D, connectivity=8, in_place=True
            )

        z_index = np.unique(np.where(gt_data > 0)[0])
        if len(z_index) == 0:
            continue

        gt_roi = gt_data[z_index]

        # Load and window the CT image
        img_sitk = sitk.ReadImage(join(nii_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)

        lo = WINDOW_LEVEL - WINDOW_WIDTH / 2
        hi = WINDOW_LEVEL + WINDOW_WIDTH / 2
        image_pre = np.clip(image_data, lo, hi)
        image_pre = (image_pre - image_pre.min()) / (image_pre.max() - image_pre.min()) * 255.0
        image_pre = np.uint8(image_pre)
        img_roi = image_pre[z_index]

        # Save per-slice npy
        case_id = name.split(gt_suffix)[0]
        for i in range(img_roi.shape[0]):
            img_i = img_roi[i]
            img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
            img_resized = transform.resize(
                img_3c, (IMAGE_SIZE, IMAGE_SIZE),
                order=3, preserve_range=True, anti_aliasing=True,
            )
            img_norm = (img_resized - img_resized.min()) / np.clip(
                img_resized.max() - img_resized.min(), a_min=1e-8, a_max=None
            )

            gt_i = gt_roi[i]
            gt_resized = transform.resize(
                gt_i, (IMAGE_SIZE, IMAGE_SIZE),
                order=0, preserve_range=True, anti_aliasing=False,
            )
            gt_resized = np.uint8(gt_resized)

            slug = f"{prefix}{case_id}-{str(i).zfill(3)}"
            np.save(join(npy_path, "imgs", slug + ".npy"), img_norm)
            np.save(join(npy_path, "gts", slug + ".npy"), gt_resized)

    print("Done! Output directory:", npy_path)


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess CT volumes for CRC-SAM")
    p.add_argument("--nii_path", type=str, default="data/Task10_Colon/imagesTr",
                    help="Directory of NIfTI images")
    p.add_argument("--gt_path", type=str, default="data/Task10_Colon/labelsTr",
                    help="Directory of NIfTI ground truths")
    p.add_argument("--npy_path", type=str, default="data/npy/CT",
                    help="Output directory for npy files")
    p.add_argument("--prefix", type=str, default="CT_Colon_",
                    help="Filename prefix")
    p.add_argument("--img_suffix", type=str, default="_0000.nii.gz")
    p.add_argument("--gt_suffix", type=str, default=".nii.gz")
    p.add_argument("--num_train", type=int, default=None,
                    help="Number of volumes for training (default: all)")
    return p.parse_args()


if __name__ == "__main__":
    preprocess_ct(parse_args())
