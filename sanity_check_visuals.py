"""
Visual sanity check: display 10 images with segmentation masks and
10 images with bounding boxes side-by-side. Saves output to sanity_check.png.

Run AFTER verify_data.py passes (images must be on disk).
Usage: conda run -n polypdb python sanity_check_visuals.py [--modality WLI]
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "modality"
COCO = ROOT / "coco_labels"

parser = argparse.ArgumentParser()
parser.add_argument("--modality", default="WLI", choices=["BLI", "FICE", "LCI", "NBI", "WLI"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n", type=int, default=10, help="samples per panel")
args = parser.parse_args()

random.seed(args.seed)
mod = args.modality
img_dir  = DATA / mod / "images"
mask_dir = DATA / mod / "annotations"

# Load all split JSONs for this modality and merge
all_images, all_annots = [], []
for split in ["train", "val", "test"]:
    p = COCO / f"{mod.lower()}_{split}.json"
    with open(p) as f:
        d = json.load(f)
    offset = len(all_images)
    all_images.extend(d["images"])
    for ann in d["annotations"]:
        a = dict(ann)
        a["image_id"] = a["image_id"] + offset
        all_annots.append(a)

id_to_fname = {img["id"]: img["file_name"] for img in all_images}
id_to_annots = {}
for ann in all_annots:
    id_to_annots.setdefault(ann["image_id"], []).append(ann)

# Filter to images that exist on disk
valid_ids = [img["id"] for img in all_images if (img_dir / img["file_name"]).exists()]
if not valid_ids:
    print(f"No images found in {img_dir}. Download the dataset first.")
    raise SystemExit(1)

sample_ids = random.sample(valid_ids, min(args.n, len(valid_ids)))

fig, axes = plt.subplots(2, len(sample_ids), figsize=(3 * len(sample_ids), 7))
fig.suptitle(f"PolypDB {mod} — top: masks, bottom: bounding boxes", fontsize=13)

for col, img_id in enumerate(sample_ids):
    fname = id_to_fname[img_id]
    img_bgr = cv2.imread(str(img_dir / fname))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # --- top row: segmentation mask overlay ---
    mask_path = mask_dir / fname
    ax_top = axes[0, col]
    ax_top.imshow(img_rgb)
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
        overlay[mask > 0] = [0, 255, 0, 120]
        ax_top.imshow(overlay)
    ax_top.axis("off")
    ax_top.set_title(fname[:12], fontsize=6)

    # --- bottom row: bounding boxes ---
    ax_bot = axes[1, col]
    ax_bot.imshow(img_rgb)
    for ann in id_to_annots.get(img_id, []):
        x, y, bw, bh = ann["bbox"]
        rect = patches.Rectangle((x, y), bw, bh,
                                  linewidth=1.5, edgecolor="red", facecolor="none")
        ax_bot.add_patch(rect)
    ax_bot.axis("off")

plt.tight_layout()
out = ROOT / f"sanity_check_{mod}.png"
plt.savefig(out, dpi=120)
print(f"Saved → {out}")
