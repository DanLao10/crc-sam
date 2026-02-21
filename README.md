# CRC-SAM: SAM-Based Multi-Modal Segmentation of Colorectal Cancer

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of:

> **CRC-SAM: SAM-Based Multi-Modal Segmentation and Quantification of Colorectal Cancer in CT, Colonoscopy, and Histology Images**
>
> Daniel Z. Lao

CRC-SAM adapts the Segment Anything Model (SAM) for **automatic, prompt-free** colorectal cancer segmentation across three imaging modalities—**CT**, **colonoscopy**, and **histopathology**—using Low-Rank Adaptation (LoRA) and a learnable default prompt embedding.

<p align="center">
  <img src="assets/architecture.png" width="85%" alt="CRC-SAM Architecture">
</p>

## Highlights

- **Parameter-efficient fine-tuning**: Only 4.57% of SAM's parameters are trainable (LoRA + decoder), keeping the powerful pretrained encoder largely intact.
- **Prompt-free inference**: A learnable default sparse embedding replaces manual box/point prompts, enabling fully automatic segmentation.
- **Multi-modal**: A single architecture handles CT volumes, colonoscopy frames, and histopathology slides.
- **Strong results**: mDSC of **90.55%** (CT), **95.30%** (colonoscopy), **83.90%** (histopathology).

## Architecture

| Component | Parameters | Trainable | % of Total |
|-----------|-----------|-----------|-----------|
| Image Encoder (ViT-B) | 89.4M | Frozen | 0% |
| LoRA (rank=4, Q/K/V) | 221K | ✓ | 0.24% |
| Mask Decoder | 4.06M | ✓ | 4.33% |
| Default Prompt Embedding | 512 | ✓ | <0.01% |
| **Total** | **93.7M** | **4.28M** | **4.57%** |

## Installation

```bash
git clone https://github.com/<your-org>/CRC-SAM.git
cd CRC-SAM
pip install -e .
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA (recommended for training)

Core dependencies are installed automatically via `pip install -e .`. For the full list see [requirements.txt](requirements.txt).

### Pre-trained Weights

Download the MedSAM ViT-B checkpoint (the base model that CRC-SAM fine-tunes):

```bash
mkdir -p work_dir/MedSAM
# Download medsam_vit_b.pth from https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN
# and place it in work_dir/MedSAM/
```

## Dataset Preparation

CRC-SAM is evaluated on three public datasets:

| Dataset | Modality | Samples | Train/Val | Link |
|---------|----------|---------|-----------|------|
| MSD-Colon (Task 10) | CT | 126 cases | 99/27 | [medicaldecathlon.com](http://medicaldecathlon.com/) |
| CVC-ClinicDB | Colonoscopy | 612 images | 490/122 | [grand-challenge.org](https://polyp.grand-challenge.org/CVCClinicDB/) |
| EBHI-Seg (adenocarcinoma) | Histopathology | 795 images | 636/159 | [figshare.com](https://figshare.com/articles/dataset/EBHI-SEG/21540159/1) |

### Preprocessing

Each preprocessing script converts raw images and masks into 1024×1024 NPY files:

```bash
# CT (MSD-Colon)
python pre_CT_MR.py \
    --nii_path data/Task10_Colon/imagesTr \
    --gt_path  data/Task10_Colon/labelsTr \
    --npy_path data/npy/CT

# Colonoscopy (CVC-ClinicDB)
python pre_colonoscopy.py \
    --img_path data/CVC-ClinicDB/images \
    --gt_path  data/CVC-ClinicDB/masks \
    --npy_path data/npy/colonoscopy

# Histopathology (EBHI-Seg)
python pre_histology_ebhi.py \
    --img_path data/EBHI-SEG/adenocarcinoma/images \
    --gt_path  data/EBHI-SEG/adenocarcinoma/masks \
    --npy_path data/npy/histology_ebhi
```

The output NPY directory structure for each modality:
```
data/npy/<modality>/
├── imgs/    # (1024, 1024, 3) float64, normalised to [0, 1]
└── gts/     # (1024, 1024) uint8, label masks
```

## Training

Train CRC-SAM on a single modality:

```bash
python train.py \
    -i data/npy/colonoscopy \
    -task_name CRC-SAM-colonoscopy \
    -checkpoint work_dir/MedSAM/medsam_vit_b.pth \
    --use_automatic_mode \
    --use_amp \
    -num_epochs 200 \
    -batch_size 4 \
    -lr 1e-4
```

Multi-modal training:

```bash
python train.py \
    -i data/npy/colonoscopy data/npy/CT data/npy/histology_ebhi \
    -task_name CRC-SAM-multimodal \
    -checkpoint work_dir/MedSAM/medsam_vit_b.pth \
    --use_automatic_mode \
    --use_amp \
    -num_epochs 200
```

### Key training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lora_rank` | 4 | LoRA rank (try 4, 8, 16) |
| `--lora_alpha` | 4.0 | LoRA scaling factor |
| `--use_automatic_mode` | False | Train for prompt-free segmentation |
| `--use_mixed_mode` | False | Alternate prompted / automatic |
| `--use_amp` | False | Mixed-precision training |
| `--use_wandb` | False | Log to Weights & Biases |
| `--val_npy_paths` | None | Validation data paths |
| `--val_interval` | 10 | Validate every N epochs |

Checkpoints are saved to `work_dir/<task_name>-<timestamp>/`.

## Inference

### Single image

```bash
python inference.py \
    -i path/to/image.png \
    -o output/ \
    -chk work_dir/CRC-SAM/crc_sam_best.pth
```

### Batch inference

```bash
python inference.py \
    -i path/to/images/ \
    -o output/ \
    -chk work_dir/CRC-SAM/crc_sam_best.pth
```

### With bounding box prompt

```bash
python inference.py \
    -i image.png \
    -o output/ \
    -chk work_dir/CRC-SAM/crc_sam_best.pth \
    --box "[100,100,400,400]"
```

### Evaluate on NPY dataset

```bash
python inference.py \
    -i data/npy/colonoscopy \
    -o output/ \
    -chk work_dir/CRC-SAM/crc_sam_best.pth \
    --evaluate
```

## Comprehensive Evaluation

Run detailed evaluation with multiple metrics (Dice, IoU, Precision, Recall, Specificity, F1, Accuracy, and optionally HD95/ASD):

```bash
python evaluate.py \
    -i data/npy/colonoscopy data/npy/CT data/npy/histology_ebhi \
    -chk work_dir/CRC-SAM/crc_sam_best.pth \
    -o evaluation_results/ \
    --surface_metrics \
    --generate_latex
```

## Results

Performance on the three CRC datasets (automatic / prompt-free mode):

| Modality | mDSC (%) | IoU (%) | Precision (%) | Recall (%) |
|----------|----------|---------|---------------|------------|
| CT (MSD-Colon) | **90.55** | — | — | — |
| Colonoscopy (CVC-ClinicDB) | **95.30** | — | — | — |
| Histopathology (EBHI-Seg) | **83.90** | — | — | — |

> Detailed per-modality results are generated by `evaluate.py`.

## Project Structure

```
CRC-SAM/
├── segment_anything/          # SAM model with LoRA extensions
│   ├── modeling/
│   │   ├── crc_sam.py         # CRC-SAM wrapper (LoRA + prompt-free)
│   │   ├── lora.py            # LoRA injection for ViT encoder
│   │   ├── sam.py             # Base SAM model
│   │   ├── image_encoder.py   # ViT-B/L/H image encoder
│   │   ├── mask_decoder.py    # Mask decoder
│   │   ├── prompt_encoder.py  # Prompt encoder
│   │   ├── transformer.py     # Two-way transformer
│   │   └── common.py          # Shared layers
│   ├── build_sam.py           # Model registry & builders
│   ├── predictor.py           # Interactive predictor
│   └── utils/                 # Transforms, AMG utilities
├── utils/
│   └── evaluation_metrics.py  # SegmentationEvaluator class
├── train.py                   # Training script
├── inference.py               # Inference (single, batch, evaluate)
├── evaluate.py                # Comprehensive multi-metric evaluation
├── pre_CT_MR.py               # CT preprocessing (MSD-Colon)
├── pre_colonoscopy.py         # Colonoscopy preprocessing (CVC-ClinicDB)
├── pre_histology_ebhi.py      # Histopathology preprocessing (EBHI-Seg)
├── setup.py                   # Package installation
├── requirements.txt           # Python dependencies
└── LICENSE                    # Apache 2.0
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{lao2025crcsam,
  title={CRC-SAM: SAM-Based Multi-Modal Segmentation and Quantification of Colorectal Cancer in CT, Colonoscopy, and Histology Images},
  author={Lao, Daniel Z.},
  year={2025}
}
```

## Acknowledgements

- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI
- [MedSAM](https://github.com/bowang-lab/MedSAM) by Jun Ma et al. (Nature Communications, 2024)
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
- [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/)
- [EBHI-Seg](https://figshare.com/articles/dataset/EBHI-SEG/21540159/1)

## License

This project is licensed under the [Apache License 2.0](LICENSE).
