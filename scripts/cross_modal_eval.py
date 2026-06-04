"""
Cross-modality generalization: load the WLI-trained U-Net and evaluate
it on BLI, FICE, LCI, NBI test sets without any retraining.

Tests whether a model trained on the most common clinical modality (WLI)
generalizes to rarer imaging modalities.

Usage: conda run -n polypdb python scripts/cross_modal_eval.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from datasets.polyp_dataset import PolypDataset, VAL_TRANSFORMS
from utils.metrics import RunningMetrics
import segmentation_models_pytorch as smp

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = ROOT / "results" / "unet_WLI" / "best_model.pth"
MODALITIES = ["WLI", "BLI", "FICE", "LCI", "NBI"]


def load_model():
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    return model.to(DEVICE).eval()


def evaluate(model, modality, split="test"):
    ds = PolypDataset(modality, split, transforms=VAL_TRANSFORMS)
    metrics = RunningMetrics()
    with torch.no_grad():
        from torch.utils.data import DataLoader
        dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        for imgs, masks in dl:
            imgs = imgs.to(DEVICE)
            pred = model(imgs)
            metrics.update(pred.cpu(), masks)
    return metrics.mean(), len(ds)


def main():
    if not WEIGHTS.exists():
        print(f"Weights not found: {WEIGHTS}")
        return

    print("Loading WLI-trained U-Net...")
    model = load_model()

    print(f"\n{'='*70}")
    print("CROSS-MODALITY GENERALIZATION — U-Net trained on WLI only")
    print(f"{'='*70}")
    print(f"  {'Modality':<10} {'N':>5} {'mDSC':>8} {'mIoU':>8} {'Recall':>8} {'Prec':>8}  Note")
    print("  " + "-" * 65)

    wli_dsc = None
    for mod in MODALITIES:
        m, n = evaluate(model, mod)
        note = "← trained on this" if mod == "WLI" else ""
        if mod == "WLI":
            wli_dsc = m["dice"]
        drop = f"  Δ{m['dice']-wli_dsc:+.3f}" if wli_dsc and mod != "WLI" else ""
        print(f"  {mod:<10} {n:>5} {m['dice']:>8.4f} {m['iou']:>8.4f} "
              f"{m['recall']:>8.4f} {m['precision']:>8.4f}  {note}{drop}")

    print(f"\n{'='*70}")
    print("Clinical note: performance drops on BLI/FICE/LCI/NBI indicate the")
    print("model relies on WLI-specific color/texture features that do not")
    print("transfer across imaging modalities.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
