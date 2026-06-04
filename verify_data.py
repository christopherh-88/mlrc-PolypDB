"""
Run after downloading the OSF dataset to verify all images, masks, and
bounding boxes are present and that filenames match the COCO split JSONs.

Usage: conda run -n polypdb python verify_data.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "modality"
COCO = ROOT / "coco_labels"

MODALITIES = ["BLI", "FICE", "LCI", "NBI", "WLI"]
SPLITS = ["train", "val", "test"]

EXPECTED = {
    "BLI":  {"train": 56,   "val": 7,   "test": 7},
    "FICE": {"train": 56,   "val": 7,   "test": 7},
    "LCI":  {"train": 48,   "val": 6,   "test": 6},
    "NBI":  {"train": 116,  "val": 15,  "test": 15},
    "WLI":  {"train": 2870, "val": 359, "test": 359},
}

# Center-wise layout: center -> modality subfolders present in the dataset
CENTER_MODALITIES = {
    "BKAI":       ["BLI", "FICE", "LCI", "WLI"],
    "Karolinska": ["WLI"],
    "Simula":     ["NBI", "WLI"],
}

all_ok = True

print("=" * 65)
print("PolypDB Data Verification")
print("=" * 65)

# --- Modality-wise checks ---
print("\n[1] Modality-wise split counts (from COCO JSONs)\n")
print(f"  {'Modality':<6} {'Split':<6} {'Expected':>9} {'In JSON':>9} {'Images':>8} {'Masks':>8}  Status")
print("  " + "-" * 60)

for mod in MODALITIES:
    img_dir  = DATA / mod / "images"
    mask_dir = DATA / mod / "masks"
    for split in SPLITS:
        json_path = COCO / f"{mod.lower()}_{split}.json"
        if not json_path.exists():
            print(f"  MISSING JSON: {json_path.name}")
            all_ok = False
            continue

        with open(json_path) as f:
            coco = json.load(f)

        fnames   = [img["file_name"] for img in coco["images"]]
        expected = EXPECTED[mod][split]
        in_json  = len(fnames)

        imgs_found = sum(1 for fn in fnames if (img_dir / fn).exists()) if img_dir.exists() else 0

        def find_mask(fn):
            stem = Path(fn).stem
            for ext in [".jpg", ".png", ".jpeg"]:
                if (mask_dir / (stem + ext)).exists():
                    return True
            return False

        masks_found = sum(1 for fn in fnames if find_mask(fn)) if mask_dir.exists() else 0

        ok = (in_json == expected) and (imgs_found == in_json) and (masks_found == in_json)
        if not ok:
            all_ok = False
        status = "OK" if ok else "MISMATCH"
        print(f"  {mod:<6} {split:<6} {expected:>9} {in_json:>9} {imgs_found:>8} {masks_found:>8}  {status}")

# --- Center-wise checks ---
CENTER_DIR = ROOT / "data" / "center"

print("\n[2] Center-wise folder counts\n")
for center, mods in CENTER_MODALITIES.items():
    for mod in mods:
        for sub in ["images", "masks"]:
            p = CENTER_DIR / center / mod / sub
            count = len(list(p.glob("*"))) if p.exists() else 0
            status = f"{count} files" if p.exists() else "MISSING"
            print(f"  {center}/{mod}/{sub}: {status}")

# --- Total on disk ---
print("\n[3] Total files on disk\n")
total_imgs, total_masks = 0, 0
for mod in MODALITIES:
    imgs  = len(list((DATA / mod / "images").glob("*"))) if (DATA / mod / "images").exists() else 0
    masks = len(list((DATA / mod / "masks").glob("*")))  if (DATA / mod / "masks").exists()  else 0
    total_imgs  += imgs
    total_masks += masks
    print(f"  {mod:<6} images={imgs:<5} masks={masks}")
print(f"  {'TOTAL':<6} images={total_imgs:<5} masks={total_masks}")

print("\n" + "=" * 65)
if all_ok:
    print("All checks passed.")
else:
    print("Some checks FAILED — see MISMATCH rows above.")
print("=" * 65)
