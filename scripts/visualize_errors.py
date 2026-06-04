"""
Visualize segmentation failure cases: saves a grid of the worst predictions
from the WLI test set with error type labels.

Error taxonomy:
  missed_small    - polyp present but DSC < 0.1 (completely missed)
  underseg        - 0.1 <= DSC < 0.5 (boundary under-segmentation)
  overseg         - precision < 0.3 (over-segmentation into mucosa)
  good            - DSC >= 0.8

Saves: error_analysis_WLI.png

Usage: conda run -n polypdb python scripts/visualize_errors.py
"""
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from datasets.polyp_dataset import PolypDataset, VAL_TRANSFORMS
from utils.metrics import compute_all
import segmentation_models_pytorch as smp

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = ROOT / "results" / "unet_WLI" / "best_model.pth"


def load_model():
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    return model.to(DEVICE).eval()


def classify_error(m):
    if m["dice"] >= 0.8:
        return "good"
    elif m["dice"] < 0.1:
        return "missed_small"
    elif m["precision"] < 0.3:
        return "overseg"
    else:
        return "underseg"


def main():
    if not WEIGHTS.exists():
        print(f"Weights not found: {WEIGHTS}")
        return

    model = load_model()
    ds = PolypDataset("WLI", "test", transforms=VAL_TRANSFORMS)

    cases = {"missed_small": [], "underseg": [], "overseg": [], "good": []}

    print(f"Scanning {len(ds)} test samples...")
    with torch.no_grad():
        for img_path, mask_path in ds.samples:
            img  = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
            aug  = VAL_TRANSFORMS(image=img, mask=mask)
            x = torch.from_numpy(aug["image"].transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
            m_gt = torch.from_numpy(aug["mask"]).unsqueeze(0).unsqueeze(0)
            pred = model(x)
            metrics = compute_all(pred[0].cpu(), m_gt[0])
            etype = classify_error(metrics)

            if len(cases[etype]) < 3:
                cases[etype].append({
                    "img": img,
                    "gt":  mask,
                    "pred": (torch.sigmoid(pred[0, 0]).cpu().numpy() > 0.5).astype(np.float32),
                    "metrics": metrics,
                    "fname": img_path.name,
                })

    # ── plot ─────────────────────────────────────────────────────────────────
    error_types = ["missed_small", "underseg", "overseg", "good"]
    colors = {"missed_small": "red", "underseg": "orange", "overseg": "purple", "good": "green"}
    labels_map = {
        "missed_small": "Missed (DSC<0.1)",
        "underseg":     "Under-seg (DSC<0.5)",
        "overseg":      "Over-seg (Prec<0.3)",
        "good":         "Good (DSC≥0.8)",
    }

    fig, axes = plt.subplots(len(error_types), 9, figsize=(27, 12))
    fig.suptitle("PolypDB U-Net WLI — Error Taxonomy", fontsize=14, fontweight="bold")

    col_titles = ["Image", "GT", "Pred", "Image", "GT", "Pred", "Image", "GT", "Pred"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=8)

    for row, etype in enumerate(error_types):
        axes[row, 0].set_ylabel(labels_map[etype], fontsize=8,
                                color=colors[etype], fontweight="bold")
        for col_idx, case in enumerate(cases[etype][:3]):
            base = col_idx * 3
            axes[row, base].imshow(case["img"])
            axes[row, base + 1].imshow(case["gt"], cmap="Reds", vmin=0, vmax=1)
            axes[row, base + 2].imshow(case["pred"], cmap="Blues", vmin=0, vmax=1)
            m = case["metrics"]
            axes[row, base + 2].set_xlabel(
                f"DSC={m['dice']:.2f} IoU={m['iou']:.2f}", fontsize=6)

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    out = ROOT / "error_analysis_WLI.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved → {out}")

    # ── taxonomy summary ──────────────────────────────────────────────────────
    print("\nError taxonomy counts:")
    for etype, case_list in cases.items():
        print(f"  {labels_map[etype]}: {len(case_list)} shown (up to 3 per category)")


if __name__ == "__main__":
    main()
