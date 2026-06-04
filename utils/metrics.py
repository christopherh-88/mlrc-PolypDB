"""
Segmentation metrics for PolypDB reproduction.
All metrics computed per-image then averaged (mean over test set), matching
the evaluation protocol described in the PolypDB paper.

Metrics:
  mIoU  - mean Intersection over Union
  mDSC  - mean Dice Similarity Coefficient
  Recall    - sensitivity / true positive rate
  Precision - positive predictive value
  F2        - F-beta score with beta=2 (weights recall higher than precision)
"""
import torch
import numpy as np


def _to_binary(pred: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(pred) > threshold).float()


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    pred   = _to_binary(pred).view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    pred   = _to_binary(pred).view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return ((2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


def recall_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    pred   = _to_binary(pred).view(-1)
    target = target.view(-1)
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return ((tp + smooth) / (tp + fn + smooth)).item()


def precision_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    pred   = _to_binary(pred).view(-1)
    target = target.view(-1)
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return ((tp + smooth) / (tp + fp + smooth)).item()


def f2_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    p = precision_score(pred, target, smooth)
    r = recall_score(pred, target, smooth)
    return (5 * p * r) / (4 * p + r + smooth)


def compute_all(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Return dict of all five segmentation metrics for one batch."""
    return {
        "iou":       iou_score(pred, target),
        "dice":      dice_score(pred, target),
        "recall":    recall_score(pred, target),
        "precision": precision_score(pred, target),
        "f2":        f2_score(pred, target),
    }


class RunningMetrics:
    """Accumulates per-batch metric values and computes means."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._sums  = {"iou": 0., "dice": 0., "recall": 0., "precision": 0., "f2": 0.}
        self._count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        # evaluate each image in the batch independently
        for i in range(pred.shape[0]):
            m = compute_all(pred[i], target[i])
            for k, v in m.items():
                self._sums[k] += v
            self._count += 1

    def mean(self) -> dict:
        if self._count == 0:
            return {k: 0. for k in self._sums}
        return {k: v / self._count for k, v in self._sums.items()}

    def summary(self) -> str:
        m = self.mean()
        return (f"mIoU={m['iou']:.4f}  mDSC={m['dice']:.4f}  "
                f"Recall={m['recall']:.4f}  Prec={m['precision']:.4f}  F2={m['f2']:.4f}")
