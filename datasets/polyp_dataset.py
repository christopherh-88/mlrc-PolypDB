"""
PyTorch Dataset for PolypDB segmentation training.
Handles variable image sizes, mask loading, and the paper's augmentation set:
random rotation, horizontal flip, vertical flip, coarse dropout.
"""
import json
import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).parent.parent
# Allow override via env var so the same code works on Kaggle
DATA = Path(os.environ.get("POLYPDB_DATA_ROOT", str(ROOT / "data" / "modality")))
COCO = Path(os.environ.get("POLYPDB_COCO_ROOT", str(ROOT / "coco_labels")))

# Paper augmentation set (Table 2 / Section 3 of PolypDB)
TRAIN_TRANSFORMS = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=90, p=0.5),
    A.CoarseDropout(
        num_holes_range=(1, 8),
        hole_height_range=(16, 64),
        hole_width_range=(16, 64),
        fill=0,
        p=0.3,
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

VAL_TRANSFORMS = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class PolypDataset(Dataset):
    """
    Args:
        modality: one of BLI, FICE, LCI, NBI, WLI
        split: train | val | test
        transforms: albumentations Compose or None (uses defaults above)
    """
    def __init__(self, modality: str, split: str, transforms=None):
        assert modality in ["BLI", "FICE", "LCI", "NBI", "WLI"]
        assert split in ["train", "val", "test"]
        self.modality = modality
        self.split = split
        self.img_dir  = DATA / modality / "images"
        self.mask_dir = DATA / modality / "masks"

        json_path = COCO / f"{modality.lower()}_{split}.json"
        with open(json_path) as f:
            coco = json.load(f)

        # Only keep samples where both image and mask exist on disk
        self.samples = []
        for entry in coco["images"]:
            fn   = entry["file_name"]
            stem = Path(fn).stem
            img_path  = self.img_dir  / fn
            mask_path = self.mask_dir / (stem + ".png")
            if img_path.exists() and mask_path.exists():
                self.samples.append((img_path, mask_path))

        if transforms is not None:
            self.transforms = transforms
        elif split == "train":
            self.transforms = TRAIN_TRANSFORMS
        else:
            self.transforms = VAL_TRANSFORMS

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        augmented = self.transforms(image=image, mask=mask)
        image = torch.from_numpy(augmented["image"].transpose(2, 0, 1)).float()
        mask  = torch.from_numpy(augmented["mask"]).unsqueeze(0).float()

        return image, mask

    def __repr__(self):
        return (f"PolypDataset(modality={self.modality}, split={self.split}, "
                f"n={len(self.samples)})")


def get_dataloaders(modality: str, batch_size: int = 8, num_workers: int = 4):
    """Returns train, val, test DataLoaders for a given modality."""
    import torch
    from torch.utils.data import DataLoader
    pin = torch.cuda.is_available()  # pin_memory unsupported on MPS
    import platform
    # spawn required on macOS; fork is fine on Linux (Kaggle/CUDA)
    mp_ctx = "spawn" if (num_workers > 0 and platform.system() == "Darwin") else None
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = PolypDataset(modality, split)
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=(num_workers > 0),
            multiprocessing_context=mp_ctx,
            drop_last=(split == "train"),
        )
    return loaders["train"], loaders["val"], loaders["test"]
