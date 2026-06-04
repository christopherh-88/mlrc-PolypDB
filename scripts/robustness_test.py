"""
Robustness testing: apply image degradations to WLI test set and measure
metric drop vs. clean baseline.

Degradations tested:
  blur          - Gaussian blur (kernel 15)
  brightness    - +80 brightness shift
  contrast      - 0.4x contrast reduction
  jpeg          - JPEG quality 10
  lowres        - 4x downsample then upsample
  flare         - random white ellipse (simulated specular reflection)

Usage: conda run -n polypdb python scripts/robustness_test.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from datasets.polyp_dataset import PolypDataset, VAL_TRANSFORMS
from utils.metrics import RunningMetrics
import segmentation_models_pytorch as smp

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = ROOT / "results" / "unet_WLI" / "best_model.pth"


def load_model():
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    return model.to(DEVICE).eval()


# ── degradation functions ────────────────────────────────────────────────────
def apply_blur(img):
    return cv2.GaussianBlur(img, (15, 15), 0)

def apply_brightness(img):
    return np.clip(img.astype(np.int32) + 80, 0, 255).astype(np.uint8)

def apply_contrast(img):
    return np.clip((img.astype(np.float32) - 128) * 0.4 + 128, 0, 255).astype(np.uint8)

def apply_jpeg(img):
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 10])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def apply_lowres(img):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_flare(img):
    out = img.copy()
    h, w = out.shape[:2]
    cx, cy = np.random.randint(w // 4, 3 * w // 4), np.random.randint(h // 4, 3 * h // 4)
    axes = (np.random.randint(20, 60), np.random.randint(20, 60))
    cv2.ellipse(out, (cx, cy), axes, 0, 0, 360, (255, 255, 255), -1)
    return out

DEGRADATIONS = {
    "clean":      None,
    "blur":       apply_blur,
    "brightness": apply_brightness,
    "contrast":   apply_contrast,
    "jpeg":       apply_jpeg,
    "lowres":     apply_lowres,
    "flare":      apply_flare,
}


def evaluate_with_degradation(model, ds, deg_fn):
    metrics = RunningMetrics()
    with torch.no_grad():
        for img_path, mask_path in ds.samples:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if deg_fn is not None:
                img = deg_fn(img)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
            aug = VAL_TRANSFORMS(image=img, mask=mask)
            x = torch.from_numpy(aug["image"].transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
            m = torch.from_numpy(aug["mask"]).unsqueeze(0).unsqueeze(0)
            pred = model(x)
            metrics.update(pred.cpu(), m)
    return metrics.mean()


def main():
    if not WEIGHTS.exists():
        print(f"Weights not found at {WEIGHTS}")
        print("Run train_segmentation.py first.")
        return

    print("Loading model...")
    model = load_model()
    ds = PolypDataset("WLI", "test", transforms=VAL_TRANSFORMS)
    print(f"Test samples: {len(ds)}\n")

    results = {}
    for name, fn in DEGRADATIONS.items():
        print(f"  Evaluating: {name}...")
        results[name] = evaluate_with_degradation(model, ds, fn)

    clean = results["clean"]

    print(f"\n{'='*72}")
    print("ROBUSTNESS TEST RESULTS — U-Net WLI")
    print(f"{'='*72}")
    print(f"{'Degradation':<14} {'mDSC':>7} {'mIoU':>7} {'Recall':>7} {'Prec':>7}  {'ΔDSC':>7}")
    print("-" * 72)
    for name, m in results.items():
        delta = m["dice"] - clean["dice"] if name != "clean" else 0.0
        print(f"  {name:<12} {m['dice']:>7.4f} {m['iou']:>7.4f} "
              f"{m['recall']:>7.4f} {m['precision']:>7.4f}  {delta:>+7.4f}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
