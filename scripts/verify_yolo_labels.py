"""
Visual verification of YOLO labels: plots 10 random images with bounding boxes
from the YOLO label files to confirm conversion is correct.

Usage: conda run -n polypdb python scripts/verify_yolo_labels.py [--modality WLI] [--split train]
"""
import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--modality", default="WLI")
parser.add_argument("--split", default="train")
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

img_dir   = ROOT / "yolo_labels" / args.modality / args.split / "images"
label_dir = ROOT / "yolo_labels" / args.modality / args.split / "labels"

img_files = sorted(img_dir.glob("*.jpg"))
random.seed(args.seed)
sample = random.sample(img_files, min(args.n, len(img_files)))

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle(f"YOLO label verification — {args.modality} {args.split}", fontsize=13)
axes = axes.flatten()

for ax, img_path in zip(axes, sample):
    img_bgr = cv2.imread(str(img_path.resolve()))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    ax.imshow(img_rgb)
    label_path = label_dir / (img_path.stem + ".txt")
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            cls, xc, yc, wn, hn = map(float, line.split())
            x = (xc - wn / 2) * w
            y = (yc - hn / 2) * h
            bw, bh = wn * w, hn * h
            rect = patches.Rectangle((x, y), bw, bh,
                                      linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
    ax.axis("off")
    ax.set_title(img_path.name[:16], fontsize=7)

plt.tight_layout()
out = ROOT / f"yolo_verify_{args.modality}_{args.split}.png"
plt.savefig(out, dpi=120)
print(f"Saved → {out}")
