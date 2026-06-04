"""
Export the best segmentation model to ONNX and benchmark CPU inference.

Usage: conda run -n polypdb python scripts/export_onnx.py [--model unet]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import segmentation_models_pytorch as smp

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="unet", choices=["unet", "deeplabv3plus", "efficientnet_unet"])
args = parser.parse_args()

WEIGHTS = ROOT / "results" / f"{args.model}_WLI" / "best_model.pth"
OUT     = ROOT / "results" / f"{args.model}_WLI" / f"{args.model}_wli.onnx"

if not WEIGHTS.exists():
    print(f"Weights not found: {WEIGHTS}")
    sys.exit(1)

# ── load model ────────────────────────────────────────────────────────────────
if args.model == "unet":
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
elif args.model == "deeplabv3plus":
    model = smp.DeepLabV3Plus("resnet50", encoder_weights=None, in_channels=3, classes=1)
elif args.model == "efficientnet_unet":
    model = smp.Unet("efficientnet-b0", encoder_weights=None, in_channels=3, classes=1)

model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
model.eval()

model_size_mb = sum(p.numel() * 4 for p in model.parameters()) / 1e6
print(f"Model: {args.model}  |  Size: {model_size_mb:.1f} MB")

# ── export to ONNX ────────────────────────────────────────────────────────────
dummy = torch.randn(1, 3, 512, 512)
torch.onnx.export(
    model, dummy, str(OUT),
    input_names=["image"],
    output_names=["mask_logits"],
    dynamic_axes={"image": {0: "batch"}, "mask_logits": {0: "batch"}},
    opset_version=17,
)
onnx_size_mb = OUT.stat().st_size / 1e6
print(f"ONNX exported → {OUT.name}  ({onnx_size_mb:.1f} MB)")

# ── CPU inference benchmark ───────────────────────────────────────────────────
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    inp = dummy.numpy()

    # warmup
    for _ in range(3):
        sess.run(None, {"image": inp})

    # benchmark
    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        sess.run(None, {"image": inp})
    elapsed = (time.perf_counter() - t0) / N * 1000

    print(f"CPU inference (ONNX): {elapsed:.1f} ms/image")
except ImportError:
    # fallback: benchmark with PyTorch CPU
    print("onnxruntime not installed — benchmarking with PyTorch CPU")
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            model(dummy)
        elapsed = (time.perf_counter() - t0) / N * 1000
    print(f"CPU inference (PyTorch): {elapsed:.1f} ms/image")

print(f"\nSummary:")
print(f"  PyTorch model size: {model_size_mb:.1f} MB")
print(f"  ONNX model size:    {onnx_size_mb:.1f} MB")
print(f"  CPU inference:      {elapsed:.1f} ms/image")
print(f"  CPU deployable:     {'Yes' if elapsed < 5000 else 'Marginal (>5s/image)'}")
