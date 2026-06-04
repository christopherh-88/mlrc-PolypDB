"""
Reproducibility checklist: prints all information needed to exactly
reproduce the Phase 2 training runs.

Usage: conda run -n polypdb python scripts/reproducibility_checklist.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def file_md5(path, chunk=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk_data := f.read(chunk):
            h.update(chunk_data)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("POLYPDB REPRODUCTION — REPRODUCIBILITY CHECKLIST")
    print("=" * 70)

    # ── 1. Package versions ───────────────────────────────────────────────────
    print("\n[1] Package versions\n")
    packages = [
        "torch", "torchvision", "numpy", "pandas", "cv2",
        "PIL", "albumentations", "segmentation_models_pytorch",
        "ultralytics", "pycocotools", "torchmetrics",
    ]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "ok")
        except ImportError:
            ver = "NOT INSTALLED"
        print(f"  {pkg:<32} {ver}")

    # ── 2. Training configuration ─────────────────────────────────────────────
    print("\n[2] Training configuration\n")
    for run_dir in sorted((ROOT / "results").glob("*")):
        rfile = run_dir / "results.json"
        if not rfile.exists():
            continue
        r = json.load(open(rfile))
        print(f"  {run_dir.name}:")
        for k in ["model", "modality", "epochs", "batch_size", "lr",
                  "seed", "device", "model_size_mb",
                  "total_train_time_min", "avg_inference_ms_per_image"]:
            if k in r:
                print(f"    {k:<35} {r[k]}")
        print()

    # ── 3. Dataset file counts ────────────────────────────────────────────────
    print("[3] Dataset file counts\n")
    import json as _json
    total = 0
    for mod in ["BLI", "FICE", "LCI", "NBI", "WLI"]:
        imgs  = len(list((ROOT / "data" / "modality" / mod / "images").glob("*")))
        masks = len(list((ROOT / "data" / "modality" / mod / "masks").glob("*")))
        total += imgs
        print(f"  {mod:<6} images={imgs:<5} masks={masks}")
    print(f"  {'TOTAL':<6} {total}")

    # ── 4. Split counts from COCO JSONs ───────────────────────────────────────
    print("\n[4] Official split counts (from COCO JSONs)\n")
    print(f"  {'Modality':<8} {'Train':>6} {'Val':>5} {'Test':>5}")
    for mod in ["BLI", "FICE", "LCI", "NBI", "WLI"]:
        counts = {}
        for split in ["train", "val", "test"]:
            p = ROOT / "coco_labels" / f"{mod.lower()}_{split}.json"
            counts[split] = len(_json.load(open(p))["images"])
        print(f"  {mod:<8} {counts['train']:>6} {counts['val']:>5} {counts['test']:>5}")

    # ── 5. COCO JSON checksums ────────────────────────────────────────────────
    print("\n[5] COCO label file checksums (MD5)\n")
    for f in sorted((ROOT / "coco_labels").glob("*.json")):
        print(f"  {f.name:<30} {file_md5(f)}")

    # ── 6. Deviations from paper ──────────────────────────────────────────────
    print("\n[6] Deviations from original paper\n")
    deviations = [
        "Trained 30 epochs (paper unspecified; models converged faster than expected)",
        "Hardware: NVIDIA T4 GPU on Kaggle vs. original paper hardware (unspecified)",
        "PyTorch 2.12 (paper likely used earlier version)",
        "Stretch baselines (PraNet, CaraNet, SSFormer-L) not reproduced due to compute",
        "YOLOv8s used (paper uses YOLOv8 variant — exact size unspecified)",
        "2412 BKAI filenames with spaces sanitized for YOLO compatibility",
        "1 LCI mask missing — that sample excluded from training",
        "WLI train bbox coverage 67.5% — 933 images have masks but no bbox annotation",
    ]
    for d in deviations:
        print(f"  - {d}")

    print(f"\n{'='*70}")
    print("All information above is sufficient to reproduce Phase 2 results.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
