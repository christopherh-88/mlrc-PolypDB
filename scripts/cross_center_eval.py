"""
Cross-center generalization: evaluate the WLI-trained U-Net on images
from each medical center (Simula, Karolinska, BKAI).

The WLI training set is dominated by Simula images (~2558/2870 = 89%).
This tests whether the model generalizes to Karolinska (Sweden, 30 imgs)
and BKAI/Vietnam (1000 WLI imgs).

Usage: conda run -n polypdb python scripts/cross_center_eval.py
"""
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from datasets.polyp_dataset import VAL_TRANSFORMS
from utils.metrics import RunningMetrics
import segmentation_models_pytorch as smp

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = ROOT / "results" / "unet_WLI" / "best_model.pth"
CENTER_DIR = ROOT / "data" / "center"

CENTERS = {
    "Simula":     ["WLI", "NBI"],
    "Karolinska": ["WLI"],
    "BKAI":       ["WLI", "BLI", "FICE", "LCI"],
}


class CenterDataset(Dataset):
    def __init__(self, center, modality):
        img_dir  = CENTER_DIR / center / modality / "images"
        mask_dir = CENTER_DIR / center / modality / "masks"
        self.samples = [
            (f, mask_dir / (f.stem + ".png"))
            for f in sorted(img_dir.glob("*.jpg"))
            if (mask_dir / (f.stem + ".png")).exists()
        ]
        self.transforms = VAL_TRANSFORMS

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img  = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        aug  = self.transforms(image=img, mask=mask)
        x = torch.from_numpy(aug["image"].transpose(2, 0, 1)).float()
        m = torch.from_numpy(aug["mask"]).unsqueeze(0).float()
        return x, m


def load_model():
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    return model.to(DEVICE).eval()


def evaluate(model, ds):
    metrics = RunningMetrics()
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    with torch.no_grad():
        for imgs, masks in dl:
            pred = model(imgs.to(DEVICE))
            metrics.update(pred.cpu(), masks)
    return metrics.mean()


def main():
    if not WEIGHTS.exists():
        print(f"Weights not found: {WEIGHTS}")
        return

    print("Loading WLI-trained U-Net...")
    model = load_model()

    print(f"\n{'='*68}")
    print("CROSS-CENTER GENERALIZATION — U-Net trained on WLI (89% Simula)")
    print(f"{'='*68}")
    print(f"  {'Center':<14} {'Modality':<8} {'N':>5} {'mDSC':>8} {'mIoU':>8} {'Recall':>8}")
    print("  " + "-" * 60)

    for center, mods in CENTERS.items():
        for mod in mods:
            ds = CenterDataset(center, mod)
            if len(ds) == 0:
                continue
            m = evaluate(model, ds)
            note = " ← dominant in train" if center == "Simula" and mod == "WLI" else ""
            print(f"  {center:<14} {mod:<8} {len(ds):>5} {m['dice']:>8.4f} "
                  f"{m['iou']:>8.4f} {m['recall']:>8.4f}{note}")

    print(f"\n{'='*68}")
    print("Clinical note: Karolinska (n=30) and BKAI reflect performance on")
    print("underrepresented centers. Low n means high variance — interpret")
    print("with caution.")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
