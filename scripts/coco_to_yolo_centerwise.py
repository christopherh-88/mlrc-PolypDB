"""
Create center-wise YOLO label directories and YAML configs.
Matches center images against the modality COCO JSONs by filename.

Output:
  yolo_labels/center/<CENTER>/<MOD>/images/  <- symlinks
  yolo_labels/center/<CENTER>/<MOD>/labels/  <- YOLO .txt files
  configs/<center>_detection.yaml

Usage: conda run -n polypdb python scripts/coco_to_yolo_centerwise.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_CENTER = ROOT / "data" / "center"
COCO = ROOT / "coco_labels"
OUT = ROOT / "yolo_labels" / "center"

CENTERS = {
    "BKAI":       ["BLI", "FICE", "LCI", "WLI"],
    "Karolinska": ["WLI"],
    "Simula":     ["NBI", "WLI"],
}
MODALITIES = ["BLI", "FICE", "LCI", "NBI", "WLI"]


def build_bbox_lookup():
    """Build filename -> {anns, w, h} from all modality COCO JSONs."""
    lookup = {}
    for mod in MODALITIES:
        for split in ["train", "val", "test"]:
            d = json.load(open(COCO / f"{mod.lower()}_{split}.json"))
            id_to_img = {i["id"]: i for i in d["images"]}
            id_to_anns = {}
            for a in d["annotations"]:
                id_to_anns.setdefault(a["image_id"], []).append(a)
            for img_id, info in id_to_img.items():
                fn = info["file_name"]
                lookup[fn] = {
                    "anns": id_to_anns.get(img_id, []),
                    "w": info["width"],
                    "h": info["height"],
                    "split": split,
                }
    return lookup


def coco_to_yolo(bbox, w, h):
    x, y, bw, bh = bbox
    return (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h


def sanitize(fn: str) -> str:
    return fn.replace(" ", "_")


def main():
    bbox_lookup = build_bbox_lookup()

    for center, mods in CENTERS.items():
        print(f"\n[{center}]")
        written = skipped = 0

        for mod in mods:
            img_dir = DATA_CENTER / center / mod / "images"
            if not img_dir.exists():
                continue

            for img_path in sorted(img_dir.glob("*.jpg")):
                fn = img_path.name
                entry = bbox_lookup.get(fn)
                if entry is None or not entry["anns"]:
                    skipped += 1
                    continue

                split = entry["split"]
                safe_fn = sanitize(fn)

                out_img_dir   = OUT / center / mod / split / "images"
                out_label_dir = OUT / center / mod / split / "labels"
                out_img_dir.mkdir(parents=True, exist_ok=True)
                out_label_dir.mkdir(parents=True, exist_ok=True)

                dst = out_img_dir / safe_fn
                if not dst.exists():
                    dst.symlink_to(img_path.resolve())

                label_file = out_label_dir / (Path(safe_fn).stem + ".txt")
                with open(label_file, "w") as f:
                    for ann in entry["anns"]:
                        xc, yc, wn, hn = coco_to_yolo(ann["bbox"], entry["w"], entry["h"])
                        f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
                written += 1

        print(f"  written={written}, skipped (no bbox)={skipped}")

        # Write YAML — point val/test to the first available modality with val data
        primary_mod = mods[0]
        mod_path = (OUT / center / primary_mod).resolve()
        yaml_path = ROOT / "configs" / f"{center.lower()}_detection.yaml"
        yaml_path.write_text(f"""# YOLOv8 center-wise config for {center}
# Modalities: {', '.join(mods)}
# Use for center-wise generalization experiments

path: {(OUT / center).resolve()}

# Default: train/val/test on same center (primary modality = {primary_mod})
train: {primary_mod}/train/images
val:   {primary_mod}/val/images
test:  {primary_mod}/test/images

nc: 1
names: ['polyp']
""")
        print(f"  Wrote configs/{center.lower()}_detection.yaml")

    print("\nDone.")


if __name__ == "__main__":
    main()
