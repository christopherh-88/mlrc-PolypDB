"""
Train a segmentation model on PolypDB.

Supported architectures (via segmentation-models-pytorch):
  unet         - U-Net with ResNet-34 encoder
  deeplabv3plus - DeepLabV3+ with ResNet-50 encoder

Usage:
  conda run -n polypdb python train_segmentation.py --model unet --modality WLI
  conda run -n polypdb python train_segmentation.py --model deeplabv3plus --modality WLI

Results are saved to results/<model>_<modality>/
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import segmentation_models_pytorch as smp
from datasets.polyp_dataset import get_dataloaders
from utils.metrics import RunningMetrics

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",    default="unet",
                    choices=["unet", "deeplabv3plus"])
parser.add_argument("--modality", default="WLI",
                    choices=["BLI", "FICE", "LCI", "NBI", "WLI"])
parser.add_argument("--epochs",     type=int,   default=100)
parser.add_argument("--batch_size", type=int,   default=8)
parser.add_argument("--lr",         type=float, default=1e-4)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--num_workers",type=int,   default=4)
args = parser.parse_args()

torch.manual_seed(args.seed)

# ── device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ── output dir ────────────────────────────────────────────────────────────────
out_dir = Path("results") / f"{args.model}_{args.modality}"
out_dir.mkdir(parents=True, exist_ok=True)

# ── model ─────────────────────────────────────────────────────────────────────
def build_model(name: str) -> nn.Module:
    if name == "unet":
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
    elif name == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

model = build_model(args.model).to(device)
model_size_mb = sum(p.numel() * 4 for p in model.parameters()) / 1e6
print(f"Model: {args.model}  |  Size: {model_size_mb:.1f} MB  |  "
      f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ── loss ──────────────────────────────────────────────────────────────────────
bce_loss  = nn.BCEWithLogitsLoss()
dice_loss = smp.losses.DiceLoss(mode="binary")

def criterion(pred, target):
    return bce_loss(pred, target) + dice_loss(pred, target)

# ── data ──────────────────────────────────────────────────────────────────────
train_dl, val_dl, test_dl = get_dataloaders(
    args.modality, batch_size=args.batch_size, num_workers=args.num_workers
)
print(f"Train: {len(train_dl.dataset)}  Val: {len(val_dl.dataset)}  "
      f"Test: {len(test_dl.dataset)}")

# ── optimizer + scheduler ─────────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

# ── training loop ─────────────────────────────────────────────────────────────
history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
best_val_dice = 0.0
train_start = time.time()

for epoch in range(1, args.epochs + 1):
    # --- train ---
    model.train()
    train_loss = 0.0
    for imgs, masks in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs} [train]",
                            leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        pred = model(imgs)
        loss = criterion(pred, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dl)
    scheduler.step()

    # --- val ---
    model.eval()
    val_loss = 0.0
    val_metrics = RunningMetrics()
    with torch.no_grad():
        for imgs, masks in val_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            pred = model(imgs)
            val_loss += criterion(pred, masks).item()
            val_metrics.update(pred.cpu(), masks.cpu())
    val_loss /= len(val_dl)
    vm = val_metrics.mean()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_dice"].append(vm["dice"])
    history["val_iou"].append(vm["iou"])

    print(f"Epoch {epoch:3d} | loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
          f"val_DSC {vm['dice']:.4f} | val_IoU {vm['iou']:.4f}")

    if vm["dice"] > best_val_dice:
        best_val_dice = vm["dice"]
        torch.save(model.state_dict(), out_dir / "best_model.pth")

total_train_time = time.time() - train_start

# ── test evaluation ───────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
model.load_state_dict(torch.load(out_dir / "best_model.pth", map_location=device))
model.eval()

test_metrics = RunningMetrics()
inference_times = []

with torch.no_grad():
    for imgs, masks in tqdm(test_dl, desc="Test"):
        imgs = imgs.to(device)
        t0 = time.perf_counter()
        pred = model(imgs)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        inference_times.append((time.perf_counter() - t0) / imgs.shape[0])
        test_metrics.update(pred.cpu(), masks)

tm = test_metrics.mean()
avg_inference_ms = 1000 * sum(inference_times) / len(inference_times)

# ── runtime metrics ───────────────────────────────────────────────────────────
if device.type == "mps":
    peak_vram_mb = torch.mps.current_allocated_memory() / 1e6
elif device.type == "cuda":
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6
else:
    peak_vram_mb = 0.0

# ── results ───────────────────────────────────────────────────────────────────
results = {
    "model":            args.model,
    "modality":         args.modality,
    "epochs":           args.epochs,
    "batch_size":       args.batch_size,
    "lr":               args.lr,
    "seed":             args.seed,
    "device":           str(device),
    "model_size_mb":    round(model_size_mb, 2),
    "total_train_time_min": round(total_train_time / 60, 2),
    "avg_inference_ms_per_image": round(avg_inference_ms, 3),
    "peak_vram_mb":     round(peak_vram_mb, 1),
    "test": {
        "mIoU":      round(tm["iou"],       4),
        "mDSC":      round(tm["dice"],      4),
        "Recall":    round(tm["recall"],    4),
        "Precision": round(tm["precision"], 4),
        "F2":        round(tm["f2"],        4),
    },
    "history": history,
}

with open(out_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"Model:     {args.model}  |  Modality: {args.modality}")
print(f"Device:    {device}  |  Size: {model_size_mb:.1f} MB")
print(f"Train time: {total_train_time/60:.1f} min")
print(f"Inference:  {avg_inference_ms:.1f} ms/image")
print(f"Peak VRAM:  {peak_vram_mb:.0f} MB")
print(f"{'─'*60}")
print(f"mIoU:      {tm['iou']:.4f}")
print(f"mDSC:      {tm['dice']:.4f}")
print(f"Recall:    {tm['recall']:.4f}")
print(f"Precision: {tm['precision']:.4f}")
print(f"F2:        {tm['f2']:.4f}")
print(f"{'='*60}")
print(f"Results saved → {out_dir}/results.json")
