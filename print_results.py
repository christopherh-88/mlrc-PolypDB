"""
Print a comparison table of all completed experiments vs. PolypDB paper values.

Usage: conda run -n polypdb python print_results.py
"""
import json
from pathlib import Path

ROOT    = Path(__file__).parent
RESULTS = ROOT / "results"

# PolypDB paper benchmark values (WLI, from Table in arxiv 2409.00045)
# Segmentation: mDSC / mIoU / Recall / Precision / F2
# Detection:    mAP50 / mAP50-95 / mAP75 / Precision / Recall
PAPER = {
    # segmentation — WLI test set values from Table 3
    "unet_WLI":         {"mDSC": 0.7810, "mIoU": 0.6910, "Recall": 0.8050, "Precision": 0.8180, "F2": 0.7980},
    "deeplabv3plus_WLI":{"mDSC": 0.8120, "mIoU": 0.7210, "Recall": 0.8310, "Precision": 0.8460, "F2": 0.8220},
    # detection — WLI test set values from Table 4
    "yolov8_WLI":       {"mAP50": 0.831, "mAP50_95": 0.534, "mAP75": 0.601, "Precision": 0.822, "Recall": 0.785},
}

SEG_KEYS = ["mDSC", "mIoU", "Recall", "Precision", "F2"]
DET_KEYS = ["mAP50", "mAP50_95", "mAP75", "Precision", "Recall"]

def load(path):
    with open(path) as f:
        return json.load(f)

found_any = False

print("=" * 75)
print("SEGMENTATION RESULTS")
print("=" * 75)
print(f"{'Model':<22} {'Source':<8} {'mDSC':>7} {'mIoU':>7} {'Recall':>7} {'Prec':>7} {'F2':>7}")
print("-" * 75)

for run_dir in sorted(RESULTS.glob("*_*")):
    rfile = run_dir / "results.json"
    if not rfile.exists():
        continue
    r = load(rfile)
    if "mAP50" in r.get("test", {}):
        continue  # detection result, skip here
    t = r["test"]
    key = f"{r['model']}_{r['modality']}"
    ours_label = f"{r['model']} ({r['modality']})"
    found_any = True

    print(f"  {ours_label:<20} {'ours':<8} "
          f"{t['mDSC']:>7.4f} {t['mIoU']:>7.4f} {t['Recall']:>7.4f} "
          f"{t['Precision']:>7.4f} {t['F2']:>7.4f}")

    if key in PAPER:
        p = PAPER[key]
        print(f"  {'':20} {'paper':<8} "
              f"{p['mDSC']:>7.4f} {p['mIoU']:>7.4f} {p['Recall']:>7.4f} "
              f"{p['Precision']:>7.4f} {p['F2']:>7.4f}")
        print()

if not found_any:
    print("  (no segmentation results yet)")

print()
print("=" * 75)
print("DETECTION RESULTS")
print("=" * 75)
print(f"{'Model':<22} {'Source':<8} {'mAP50':>7} {'mAP75':>7} {'mAP50-95':>9} {'Prec':>7} {'Rec':>7}")
print("-" * 75)

found_det = False
for run_dir in sorted(RESULTS.glob("yolov8_*")):
    rfile = run_dir / "results.json"
    if not rfile.exists():
        continue
    r = load(rfile)
    t = r.get("test", {})
    if "mAP50" not in t:
        continue
    found_det = True
    label = f"{r['model']} ({r['modality']})"

    print(f"  {label:<20} {'ours':<8} "
          f"{t['mAP50']:>7.4f} {t['mAP75']:>7.4f} {t['mAP50_95']:>9.4f} "
          f"{t['Precision']:>7.4f} {t['Recall']:>7.4f}")

    key = f"yolov8_{r['modality']}"
    if key in PAPER:
        p = PAPER[key]
        print(f"  {'':20} {'paper':<8} "
              f"{p['mAP50']:>7.4f} {p['mAP75']:>7.4f} {p['mAP50_95']:>9.4f} "
              f"{p['Precision']:>7.4f} {p['Recall']:>7.4f}")
        print()

if not found_det:
    print("  (no detection results yet)")

print()
print("=" * 75)
print("RUNTIME SUMMARY")
print("=" * 75)
print(f"{'Model':<22} {'Device':<8} {'Train(min)':>11} {'Infer(ms)':>10} {'VRAM(MB)':>9} {'Size(MB)':>9}")
print("-" * 75)
for run_dir in sorted(RESULTS.glob("*")):
    rfile = run_dir / "results.json"
    if not rfile.exists():
        continue
    r = load(rfile)
    model_name = r.get("model", run_dir.name)
    mod = r.get("modality", "?")
    label = f"{model_name} ({mod})"
    print(f"  {label:<22} {r.get('device','?'):<8} "
          f"{r.get('total_train_time_min', 0):>11.1f} "
          f"{r.get('avg_inference_ms_per_image', 0):>10.2f} "
          f"{r.get('peak_vram_mb', 0):>9.0f} "
          f"{r.get('model_size_mb', 0):>9.1f}")
