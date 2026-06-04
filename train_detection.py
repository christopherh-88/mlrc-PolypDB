"""
Train YOLOv8 detection on PolypDB.

Run scripts/coco_to_yolo.py first to generate yolo_labels/.

Usage:
  conda run -n polypdb python train_detection.py --modality WLI
  conda run -n polypdb python train_detection.py --modality WLI --model yolov8m

Results are saved to results/yolov8_<modality>/
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--modality", default="WLI",
                    choices=["BLI", "FICE", "LCI", "NBI", "WLI"])
parser.add_argument("--model",   default="yolov8s",
                    choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l"])
parser.add_argument("--epochs",     type=int,   default=100)
parser.add_argument("--batch_size", type=int,   default=8)
parser.add_argument("--imgsz",      type=int,   default=512)
parser.add_argument("--seed",       type=int,   default=42)
args = parser.parse_args()

ROOT    = Path(__file__).parent
out_dir = ROOT / "results" / f"yolov8_{args.modality}"
out_dir.mkdir(parents=True, exist_ok=True)

yaml_path = ROOT / "configs" / f"{args.modality.lower()}_detection.yaml"
if not yaml_path.exists():
    raise FileNotFoundError(f"Missing {yaml_path}. Run scripts/coco_to_yolo.py first.")

# Check yolo_labels exist
label_dir = ROOT / "yolo_labels" / args.modality / "train" / "labels"
if not label_dir.exists():
    raise FileNotFoundError(
        f"Missing {label_dir}. Run: conda run -n polypdb python scripts/coco_to_yolo.py"
    )

# device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "0"
else:
    device = "cpu"
print(f"Device: {device}")

model = YOLO(f"{args.model}.pt")

model_params = sum(p.numel() for p in model.model.parameters())
model_size_mb = sum(p.numel() * 4 for p in model.model.parameters()) / 1e6
print(f"Model: {args.model}  |  Params: {model_params/1e6:.1f}M  |  Size: {model_size_mb:.1f} MB")

# ── train ─────────────────────────────────────────────────────────────────────
train_start = time.time()
train_results = model.train(
    data=str(yaml_path),
    epochs=args.epochs,
    batch=args.batch_size,
    imgsz=args.imgsz,
    device=device,
    seed=args.seed,
    project=str(ROOT / "runs" / "detect"),
    name=f"{args.model}_{args.modality}",
    exist_ok=True,
    verbose=True,
    amp=(device != "mps"),  # AMP unsupported on MPS; fine on CUDA
    # match paper augmentation where possible
    flipud=0.5,
    fliplr=0.5,
    degrees=90,
    # disable mosaic for medical images (patches are misleading)
    mosaic=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
)
total_train_time = time.time() - train_start

# ── evaluate on test set ──────────────────────────────────────────────────────
print("\nEvaluating on test set...")
best_weights = ROOT / "runs" / "detect" / f"{args.model}_{args.modality}" / "weights" / "best.pt"
model_best = YOLO(str(best_weights))

t0 = time.perf_counter()
test_results = model_best.val(
    data=str(yaml_path),
    split="test",
    device=device,
    imgsz=args.imgsz,
    verbose=True,
)
inference_time_total = time.perf_counter() - t0

# count test images
test_img_dir = ROOT / "yolo_labels" / args.modality / "test" / "images"
n_test = len(list(test_img_dir.glob("*.jpg")))
avg_inference_ms = 1000 * inference_time_total / max(n_test, 1)

# ── collect metrics ───────────────────────────────────────────────────────────
box = test_results.box
results_dict = {
    "model":          args.model,
    "modality":       args.modality,
    "epochs":         args.epochs,
    "batch_size":     args.batch_size,
    "imgsz":          args.imgsz,
    "seed":           args.seed,
    "device":         device,
    "model_size_mb":  round(model_size_mb, 2),
    "total_train_time_min": round(total_train_time / 60, 2),
    "avg_inference_ms_per_image": round(avg_inference_ms, 3),
    "test": {
        "mAP50":    round(float(box.map50),  4),
        "mAP50_95": round(float(box.map),    4),
        "mAP75":    round(float(getattr(box, "map75", box.map50)), 4),
        "Precision": round(float(box.mp),    4),
        "Recall":    round(float(box.mr),    4),
    },
}

with open(out_dir / "results.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print(f"\n{'='*60}")
print(f"Model: {args.model}  |  Modality: {args.modality}")
print(f"Device: {device}  |  Size: {model_size_mb:.1f} MB")
print(f"Train time: {total_train_time/60:.1f} min")
print(f"Inference:  {avg_inference_ms:.1f} ms/image")
print(f"{'─'*60}")
print(f"mAP50:     {float(box.map50):.4f}")
print(f"mAP50-95:  {float(box.map):.4f}")
print(f"mAP75:     {float(getattr(box, 'map75', box.map50)):.4f}")
print(f"Precision: {float(box.mp):.4f}")
print(f"Recall:    {float(box.mr):.4f}")
print(f"{'='*60}")
print(f"Results saved → {out_dir}/results.json")
