from setuptools import setup, find_packages

setup(
    name="crc-sam",
    version="1.0.0",
    description="CRC-SAM: SAM-Based Multi-Modal Segmentation of Colorectal Cancer",
    author="Daniel Z. Lao",
    python_requires=">=3.9",
    packages=find_packages(exclude=["data", "work_dir", "assets", "*.egg-info"]),
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy>=1.24",
        "matplotlib",
        "scikit-image",
        "SimpleITK",
        "nibabel",
        "tqdm",
        "scipy",
        "opencv-python",
        "monai",
        "connected-components-3d",
    ],
    extras_require={
        "wandb": ["wandb"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
