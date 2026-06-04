"""
Convert PolypDB COCO-format bounding box labels to YOLO format.

COCO bbox:  [x_topleft, y_topleft, width, height]  (absolute pixels)
YOLO label: <class> <x_center> <y_center> <w> <h>  (normalized 0-1)

Skips images with no bbox annotations (they exist in segmentation but not detection).
Sanitizes filenames with spaces by replacing space with underscore in label filenames
and writing a symlink/rename mapping for the images.

Output structure:
  yolo_labels/<modality>/
      train/images/   <- symlinks or copies
      train/labels/   <- .txt YOLO label files
      val/images/
      val/labels/
      test/images/
      test/labels/

Usage: conda run -n polypdb python scripts/coco_to_yolo.py [--modality WLI]
"""
import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "modality"
COCO = ROOT / "coco_labels"
OUT  = ROOT / "yolo_labels"

MODALITIES = ["BLI", "FICE", "LCI", "NBI", "WLI"]


def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_c = (x + w / 2) / img_w
    y_c = (y + h / 2) / img_h
    w_n = w / img_w
    h_n = h / img_h
    return x_c, y_c, w_n, h_n


def sanitize(filename: str) -> str:
    return filename.replace(" ", "_")


def convert_modality(modality: str):
    img_dir = DATA / modality / "images"

    for split in ["train", "val", "test"]:
        json_path = COCO / f"{modality.lower()}_{split}.json"
        with open(json_path) as f:
            coco = json.load(f)

        id_to_img = {img["id"]: img for img in coco["images"]}
        id_to_anns = {}
        for ann in coco["annotations"]:
            id_to_anns.setdefault(ann["image_id"], []).append(ann)

        out_imgs   = OUT / modality / split / "images"
        out_labels = OUT / modality / split / "labels"
        out_imgs.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        skipped = 0
        written = 0
        for img_id, img_info in id_to_img.items():
            anns = id_to_anns.get(img_id, [])
            if not anns:
                skipped += 1
                continue

            fn   = img_info["file_name"]
            w, h = img_info["width"], img_info["height"]
            src  = img_dir / fn

            if not src.exists():
                continue

            # Sanitize filename for YOLO compatibility
            safe_fn = sanitize(fn)
            dst_img = out_imgs / safe_fn

            # Symlink image (avoids duplicating GBs of data)
            if not dst_img.exists():
                dst_img.symlink_to(src.resolve())

            # Write YOLO label file
            label_fn = Path(safe_fn).stem + ".txt"
            with open(out_labels / label_fn, "w") as f:
                for ann in anns:
                    xc, yc, wn, hn = coco_to_yolo_bbox(ann["bbox"], w, h)
                    f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
            written += 1

        print(f"  {modality}/{split}: {written} labeled, {skipped} skipped (no bbox)")


def write_yaml(modality: str):
    mod_dir = (OUT / modality).resolve()
    yaml_path = ROOT / "configs" / f"{modality.lower()}_detection.yaml"
    yaml_path.parent.mkdir(exist_ok=True)
    content = f"""# YOLOv8 dataset config for PolypDB {modality} modality
path: {mod_dir}
train: train/images
val:   val/images
test:  test/images

nc: 1
names: ['polyp']
"""
    yaml_path.write_text(content)
    print(f"  Wrote {yaml_path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", default="all",
                        choices=MODALITIES + ["all"])
    args = parser.parse_args()

    mods = MODALITIES if args.modality == "all" else [args.modality]

    print("Converting COCO → YOLO format\n")
    for mod in mods:
        print(f"[{mod}]")
        convert_modality(mod)
        write_yaml(mod)
        print()

    # Also write a combined all-modality YAML
    print("Writing combined YAML...")
    all_yaml = ROOT / "configs" / "all_modalities_detection.yaml"
    # combined = WLI (largest); reference per-modality YAMLs for cross-modal eval
    all_yaml.write_text(f"""# YOLOv8 combined all-modality config (WLI-dominant)
# For cross-modality eval, use the per-modality configs in configs/
path: {(OUT / 'WLI').resolve()}
train: train/images
val:   val/images
test:  test/images

nc: 1
names: ['polyp']
""")
    print(f"  Wrote {all_yaml.relative_to(ROOT)}")
    print("\nDone. Verify labels with: conda run -n polypdb python scripts/verify_yolo_labels.py")


if __name__ == "__main__":
    main()
