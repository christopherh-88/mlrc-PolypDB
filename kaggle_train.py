"""
Self-contained Kaggle training script for PolypDB reproduction.

Dataset setup on Kaggle:
  1. Create a new Kaggle dataset named "polypdb-modality"
  2. Upload the contents of data/modality/ (BLI/ FICE/ LCI/ NBI/ WLI/)
  3. Add dataset to this notebook: /kaggle/input/polypdb-modality/

Then run this script as a Kaggle notebook (GPU T4 x2 recommended).
Results saved to /kaggle/working/results/ — download after completion.

Expected runtime on Kaggle T4 (~8h GPU budget):
  Setup + install:        ~10 min
  U-Net 100 epochs:       ~60 min
  DeepLabV3+ 100 epochs:  ~75 min
  YOLOv8s 100 epochs:     ~40 min
  Total:                  ~3 hrs  (well within 8h15m limit)
"""

import os
import subprocess
import sys
from pathlib import Path

# ── Kaggle paths ──────────────────────────────────────────────────────────────
KAGGLE_DATA   = Path("/kaggle/input/polypdb-modality")
KAGGLE_WORK   = Path("/kaggle/working")
REPO          = KAGGLE_WORK / "mlrc-PolypDB"
COCO_DIR      = REPO / "coco_labels"
RESULTS_DIR   = KAGGLE_WORK / "results"

def run(cmd, **kwargs):
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kwargs)

# ── 1. Clone repo ─────────────────────────────────────────────────────────────
if not REPO.exists():
    run("git clone https://github.com/christopherh-88/mlrc-PolypDB.git /kaggle/working/mlrc-PolypDB")

os.chdir(REPO)
sys.path.insert(0, str(REPO))

# ── 2. Install packages ───────────────────────────────────────────────────────
run("pip install -q segmentation-models-pytorch albumentations pycocotools torchmetrics ultralytics")

# ── 3. Point dataset class at Kaggle input ────────────────────────────────────
os.environ["POLYPDB_DATA_ROOT"] = str(KAGGLE_DATA)
os.environ["POLYPDB_COCO_ROOT"] = str(COCO_DIR)

# ── 4. Verify data accessible ─────────────────────────────────────────────────
from datasets.polyp_dataset import PolypDataset
for mod in ["WLI"]:
    for split in ["train", "val", "test"]:
        ds = PolypDataset(mod, split)
        print(f"{mod} {split}: {len(ds)} samples")

# ── 5. Generate YOLO labels ───────────────────────────────────────────────────
# Update YAML to use absolute Kaggle paths before converting
run(f"python scripts/coco_to_yolo.py --modality WLI")

import json
YOLO_DIR = REPO / "yolo_labels" / "WLI"
for split in ["train", "val", "test"]:
    yaml_content = f"""path: {YOLO_DIR}
train: train/images
val:   val/images
test:  test/images
nc: 1
names: ['polyp']
"""
    (REPO / "configs" / "wli_detection.yaml").write_text(yaml_content)

# ── 6. Train U-Net ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TRAINING: U-Net WLI")
print("="*60)
run("python train_segmentation.py --model unet --modality WLI --epochs 100 --batch_size 16 --num_workers 2")

# ── 7. Train DeepLabV3+ ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("TRAINING: DeepLabV3+ WLI")
print("="*60)
run("python train_segmentation.py --model deeplabv3plus --modality WLI --epochs 100 --batch_size 16 --num_workers 2")

# ── 8. Train YOLOv8 ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TRAINING: YOLOv8s WLI")
print("="*60)
run("python train_detection.py --modality WLI --epochs 100 --batch_size 16")

# ── 9. Collect results ────────────────────────────────────────────────────────
import shutil
RESULTS_DIR.mkdir(exist_ok=True)

for run_dir in (REPO / "results").glob("*"):
    if run_dir.is_dir():
        shutil.copytree(run_dir, RESULTS_DIR / run_dir.name, dirs_exist_ok=True)

# Copy YOLO weights
yolo_weights = REPO / "runs" / "detect" / "yolov8s_WLI" / "weights" / "best.pt"
if yolo_weights.exists():
    shutil.copy(yolo_weights, RESULTS_DIR / "yolov8s_WLI_best.pt")

print("\n" + "="*60)
print("ALL TRAINING COMPLETE")
print(f"Results in: {RESULTS_DIR}")
print("="*60)

run(f"python print_results.py")
