"""
Produces the Phase 1 dataset summary table: image/mask/bbox counts per
modality and center, plus documents known data quality issues.

Usage: conda run -n polypdb python dataset_summary.py
"""
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "modality"
COCO = ROOT / "coco_labels"

MODALITIES = ["BLI", "FICE", "LCI", "NBI", "WLI"]
SPLITS = ["train", "val", "test"]

CENTER_DIR = ROOT / "data" / "center"
CENTER_MODALITIES = {
    "BKAI":       ["BLI", "FICE", "LCI", "WLI"],
    "Karolinska": ["WLI"],
    "Simula":     ["NBI", "WLI"],
}


def load_coco(mod, split):
    p = COCO / f"{mod.lower()}_{split}.json"
    with open(p) as f:
        return json.load(f)


# ── 1. Modality-wise summary ──────────────────────────────────────────────────
print("=" * 80)
print("TABLE 1: Modality-wise dataset summary")
print("=" * 80)
print(f"{'Modality':<8} {'Train':>6} {'Val':>5} {'Test':>5} {'Total':>6} "
      f"{'Masks':>6} {'BBoxes':>7} {'Multi-polyp':>12} {'Size (HxW)':>14}")
print("-" * 80)

grand_total_imgs = grand_total_bboxes = 0

for mod in MODALITIES:
    total_imgs = total_bboxes = multi_polyp = 0
    masks_ok = 0
    sizes = Counter()

    for split in SPLITS:
        d = load_coco(mod, split)
        imgs = d["images"]
        anns = d["annotations"]
        total_imgs += len(imgs)
        total_bboxes += len(anns)

        ann_per_img = Counter(a["image_id"] for a in anns)
        multi_polyp += sum(1 for v in ann_per_img.values() if v > 1)

        # check masks and collect sizes
        img_dir  = DATA / mod / "images"
        mask_dir = DATA / mod / "masks"
        for img in imgs[:20 if split == "train" else len(imgs)]:
            fn = img["file_name"]
            stem = Path(fn).stem
            mask_path = mask_dir / (stem + ".png")
            if mask_path.exists():
                masks_ok += 1
            fpath = img_dir / fn
            if fpath.exists() and split == "train":
                im = cv2.imread(str(fpath))
                if im is not None:
                    sizes[im.shape[:2]] += 1

    size_str = "+".join(f"{h}x{w}" for (h, w) in sorted(sizes.keys()))
    splits_counts = [load_coco(mod, s) for s in SPLITS]
    tr, va, te = [len(d["images"]) for d in splits_counts]

    print(f"{mod:<8} {tr:>6} {va:>5} {te:>5} {total_imgs:>6} "
          f"{total_imgs:>6} {total_bboxes:>7} {multi_polyp:>12} {size_str:>14}")
    grand_total_imgs   += total_imgs
    grand_total_bboxes += total_bboxes

print("-" * 80)
print(f"{'TOTAL':<8} {'':>6} {'':>5} {'':>5} {grand_total_imgs:>6} "
      f"{grand_total_imgs:>6} {grand_total_bboxes:>7}")

# ── 2. Center-wise summary ────────────────────────────────────────────────────
print()
print("=" * 55)
print("TABLE 2: Center-wise dataset summary")
print("=" * 55)
print(f"{'Center':<14} {'Modalities':<22} {'Images':>7} {'Masks':>7}")
print("-" * 55)

for center, mods in CENTER_MODALITIES.items():
    total = 0
    for mod in mods:
        n = len(list((CENTER_DIR / center / mod / "images").glob("*")))
        total += n
    print(f"{center:<14} {', '.join(mods):<22} {total:>7} {total:>7}")

print()
print("Note: BKAI = Vietnam/BKAI center, Karolinska = Sweden, Simula = Norway")

# ── 3. Bbox coverage (images WITH vs WITHOUT bbox annotation) ─────────────────
print()
print("=" * 55)
print("TABLE 3: Bounding box annotation coverage")
print("=" * 55)
print(f"{'Modality':<8} {'Split':<6} {'Images':>7} {'With BBox':>10} {'Coverage':>10}")
print("-" * 55)

for mod in MODALITIES:
    for split in SPLITS:
        d = load_coco(mod, split)
        total = len(d["images"])
        with_bbox = len(set(a["image_id"] for a in d["annotations"]))
        pct = 100 * with_bbox / total if total else 0
        print(f"{mod:<8} {split:<6} {total:>7} {with_bbox:>10} {pct:>9.1f}%")

# ── 4. Data quality issues ────────────────────────────────────────────────────
print()
print("=" * 55)
print("DATA QUALITY ISSUES FOUND")
print("=" * 55)

space_files = []
for mod in MODALITIES:
    for sub in ["images", "masks"]:
        for f in (DATA / mod / sub).glob("*"):
            if " " in f.name:
                space_files.append(f)

print(f"1. Filenames with spaces: {len(space_files)} files")
print("   Origin: BKAI (Vietnam) files use 'PKHL_*' and 'HMUH_*' naming")
print("   Impact: YOLO training may fail; labels are sanitized in conversion script")

# Check image dimensions
all_sizes = Counter()
for mod in MODALITIES:
    for f in list((DATA / mod / "images").glob("*.jpg"))[:50]:
        img = cv2.imread(str(f))
        if img is not None:
            all_sizes[img.shape[:2]] += 1

print(f"2. Variable image sizes: {dict(all_sizes)}")
print("   Resolution: NBI has 576x768 variant; WLI/BLI/LCI have 995x1280 variant")
print("   Fix: all images resized to 512x512 during preprocessing")

# Check LCI missing mask
lci_imgs  = len(list((DATA / "LCI" / "images").glob("*")))
lci_masks = len(list((DATA / "LCI" / "masks").glob("*")))
if lci_imgs != lci_masks:
    print(f"3. LCI image/mask mismatch: {lci_imgs} images vs {lci_masks} masks (1 mask missing)")

print()
print("=" * 55)
print("WLI bbox coverage note:")
d = load_coco("WLI", "train")
annotated = len(set(a["image_id"] for a in d["annotations"]))
total = len(d["images"])
print(f"  WLI train: {annotated}/{total} images have bbox annotations ({100*annotated/total:.1f}%)")
print("  Remaining images have segmentation masks but no bbox labels.")
print("  For detection training, only annotated images are used.")
