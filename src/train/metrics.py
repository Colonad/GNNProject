# src/train/metrics.py
"""
Metrics utilities for training & evaluation.

Included:
- Regression: MAE, RMSE (+ optional MSE, R² helpers)
- Classification calibration: ECE (Expected Calibration Error) with reliability-diagram stats
- Probabilistic regression calibration: coverage curve (nominal vs empirical) and regression-ECE

Design goals:
- Accept both torch.Tensor and numpy.ndarray
- Work on CPU/GPU tensors (no implicit host sync except when returning Python scalars)
- Return plot-ready aggregates (bin edges, accuracies, confidences / coverages)

Typical usage (classification):
    probs = torch.softmax(logits, dim=-1)
    labels = y.long()
    ece, bins = classification_ece(probs, labels, n_bins=15, strategy="uniform")

Typical usage (probabilistic regression):
    # predictive mean mu, stdev sigma from your model (e.g., MC Dropout)
    reg = regression_coverage_calibration(y_true, mu, sigma, alphas=None)  # dict with 'nominal', 'empirical', 'ece'

Notes:
- Classification ECE follows the standard definition using max-confidence and correctness.
- Regression calibration follows Kuleshov et al. style coverage calibration for Gaussian predictive distributions.

DoD: paired with tests/test_metrics.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import math
import numpy as np

try:
    import torch
    from torch import Tensor
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "metrics.py requires PyTorch. Install torch before importing this module."
    ) from e

ArrayLike = Union[np.ndarray, "Tensor"]

__all__ = [
    # regression (primary)
    "mae",
    "rmse",
    # regression (helpers)
    "mse",
    "r2",
    # classification calibration
    "ReliabilityBins",
    "classification_reliability_bins",
    "classification_ece",
    # probabilistic regression calibration
    "RegressionCalibrationResult",
    "regression_coverage_calibration",
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _to_tensor(x: ArrayLike, device: Optional[Union[str, torch.device]] = None) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device) if device is not None else x
    t = torch.as_tensor(x)
    return t.to(device) if device is not None else t


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_same_shape(a: Tensor, b: Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")


def _validate_labels(labels: Tensor) -> None:
    """
    Accept integer class indices for multi-class.
    Binary-specific validation is not required since we use argmax for predictions.
    """
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
        # also allow float that are integers, but be strict to avoid silent bugs
        raise ValueError("labels must contain integer class indices")


# ---------------------------------------------------------------------
# Basic regression metrics
# ---------------------------------------------------------------------


@torch.no_grad()
def mae(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Union[float, Tensor]:
    """
    Mean Absolute Error.

    Args:
        y_true: shape (...), target values
        y_pred: shape (...), predictions (same shape as y_true)
        reduction: "mean" | "sum" | "none"

    Returns:
        float (if mean/sum) or tensor of absolute errors (if "none")
    """
    yt = _to_tensor(y_true)
    yp = _to_tensor(y_pred).to(dtype=yt.dtype, device=yt.device)
    _ensure_same_shape(yt, yp)
    err = torch.abs(yp - yt)
    if reduction == "mean":
        return float(err.mean().item())
    if reduction == "sum":
        return float(err.sum().item())
    if reduction == "none":
        return err
    raise ValueError(f"Unknown reduction: {reduction}")


@torch.no_grad()
def mse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Union[float, Tensor]:
    """
    Mean Squared Error (helper).
    """
    yt = _to_tensor(y_true)
    yp = _to_tensor(y_pred).to(dtype=yt.dtype, device=yt.device)
    _ensure_same_shape(yt, yp)
    sq = (yp - yt) ** 2
    if reduction == "mean":
        return float(sq.mean().item())
    if reduction == "sum":
        return float(sq.sum().item())
    if reduction == "none":
        return sq
    raise ValueError(f"Unknown reduction: {reduction}")


@torch.no_grad()
def rmse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Union[float, Tensor]:
    """
    Root Mean Squared Error.

    Args/Returns: see `mae`.
    """
    yt = _to_tensor(y_true)
    yp = _to_tensor(y_pred).to(dtype=yt.dtype, device=yt.device)
    _ensure_same_shape(yt, yp)
    sq = (yp - yt) ** 2
    if reduction == "mean":
        return float(torch.sqrt(sq.mean()).item())
    if reduction == "sum":
        # sum of per-element RMSEs is unconventional; provided for API symmetry
        return float(torch.sqrt(sq).sum().item())
    if reduction == "none":
        return torch.sqrt(sq)
    raise ValueError(f"Unknown reduction: {reduction}")


@torch.no_grad()
def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Coefficient of determination R² = 1 - RSS/TSS.
    """
    yt = _to_tensor(y_true).to(torch.float64)
    yp = _to_tensor(y_pred).to(dtype=yt.dtype, device=yt.device)
    _ensure_same_shape(yt, yp)
    err = yt - yp
    rss = float(torch.sum(err * err).item())
    mean_y = float(torch.mean(yt).item())
    tss = float(torch.sum((yt - mean_y) ** 2).item())
    if tss <= 0:
        # Degenerate case: constant targets → R² undefined; return 0 by convention
        return 0.0
    return 1.0 - (rss / tss)


# ---------------------------------------------------------------------
# Classification calibration (ECE + reliability bins)
# ---------------------------------------------------------------------


@dataclass
class ReliabilityBins:
    """
    Container for classification reliability diagram data.
    - bin_edges: shape (n_bins + 1,)
    - bin_centers: shape (n_bins,)
    - counts: samples per bin, shape (n_bins,)
    - mean_confidence: average predicted confidence in bin, shape (n_bins,)
    - accuracy: empirical accuracy in bin, shape (n_bins,)
    """
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    counts: np.ndarray
    mean_confidence: np.ndarray
    accuracy: np.ndarray


@torch.no_grad()
def classification_reliability_bins(
    probs: ArrayLike,
    labels: ArrayLike,
    n_bins: int = 15,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> ReliabilityBins:
    """
    Compute reliability diagram bin stats for multi-class classification.

    We use sample confidence = max(prob) and correctness = (argmax == label).

    Args:
        probs: (N, C) probabilities (each row sums to ~1). If logits are given, pass softmax(logits) here.
        labels: (N,) integer class indices.
        n_bins: number of bins along [0,1].
        strategy:
            - "uniform": equal-width bins in [0, 1]
            - "quantile": bins defined by quantiles of confidence to balance counts

    Returns:
        ReliabilityBins with arrays ready for plotting.
    """
    p = _to_tensor(probs)
    if p.dim() != 2:
        raise ValueError(f"`probs` must be 2D (N,C); got shape {tuple(p.shape)}")
    y = _to_tensor(labels).long().view(-1)
    _validate_labels(y)

    conf, pred = torch.max(p, dim=1)  # (N,)
    correct = (pred == y).to(torch.float32)  # (N,)

    conf_np = _to_numpy(conf)
    corr_np = _to_numpy(correct)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, num=n_bins + 1, dtype=np.float64)
    elif strategy == "quantile":
        # Guard: if duplicate quantiles arise, np.unique ensures strict monotonic edges
        qs = np.linspace(0.0, 1.0, num=n_bins + 1)
        # Use 'linear' interpolation name for modern NumPy; fallback handled by try/except.
        try:
            quant = np.quantile(conf_np, qs, method="linear")  # NumPy >= 1.22
        except TypeError:
            quant = np.quantile(conf_np, qs, interpolation="linear")  # legacy
        bin_edges = np.unique(quant)
        # If too many duplicates (e.g., tiny N), fall back to uniform
        if bin_edges.size < 2:
            bin_edges = np.linspace(0.0, 1.0, num=n_bins + 1, dtype=np.float64)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Digitize to bins in [0,1]; dot handling of rightmost closed interval
    # numpy.digitize returns indices in [1, len(bin_edges)-1]
    # We use the interior edges to avoid out-of-range bin index
    if bin_edges.size == 2:
        # Single bin edge case: force one bin [0,1]
        bin_edges = np.array([0.0, 1.0], dtype=np.float64)
    bin_ids = np.digitize(conf_np, bin_edges[1:-1], right=True)

    nb = len(bin_edges) - 1
    counts = np.zeros(nb, dtype=np.int64)
    mean_conf = np.zeros(nb, dtype=np.float64)
    acc = np.zeros(nb, dtype=np.float64)

    for b in range(nb):
        m = bin_ids == b
        cnt = int(m.sum())
        counts[b] = cnt
        if cnt > 0:
            mean_conf[b] = float(conf_np[m].mean())
            acc[b] = float(corr_np[m].mean())
        else:
            mean_conf[b] = 0.0
            acc[b] = 0.0

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return ReliabilityBins(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        counts=counts,
        mean_confidence=mean_conf,
        accuracy=acc,
    )


@torch.no_grad()
def classification_ece(
    probs: ArrayLike,
    labels: ArrayLike,
    n_bins: int = 15,
    strategy: Literal["uniform", "quantile"] = "uniform",
    norm: Literal["l1", "l2"] = "l1",
) -> Tuple[float, ReliabilityBins]:
    """
    Expected Calibration Error (ECE) for multi-class classification.

    ECE (L1) = sum_b w_b * |acc_b - conf_b|
    ECE (L2) = sqrt( sum_b w_b * (acc_b - conf_b)^2 )

    where w_b = count_b / N.

    Args:
        probs: (N, C) probabilities
        labels: (N,) integer class indices
        n_bins: number of bins
        strategy: "uniform" or "quantile"
        norm: "l1" or "l2"

    Returns:
        (ece_value, bins)
    """
    bins = classification_reliability_bins(probs, labels, n_bins=n_bins, strategy=strategy)
    N = int(bins.counts.sum())
    if N == 0:
        return 0.0, bins

    gap = np.abs(bins.accuracy - bins.mean_confidence)  # shape (n_bins,)
    w = bins.counts.astype(np.float64) / float(N)

    if norm == "l1":
        ece = float(np.sum(w * gap))
    elif norm == "l2":
        ece = float(np.sqrt(np.sum(w * (gap ** 2))))
    else:
        raise ValueError(f"Unknown norm: {norm}")
    return ece, bins


# ---------------------------------------------------------------------
# Probabilistic regression calibration via coverage curves
# ---------------------------------------------------------------------


@dataclass
class RegressionCalibrationResult:
    """
    Result container for probabilistic regression calibration.

    Attributes:
        nominal:   (K,) nominal coverages in [0,1] (e.g., 0.1, 0.2, ..., 0.9)
        empirical: (K,) empirical coverages measured from intervals around mu
        ece:       scalar = mean absolute deviation |empirical - nominal|
    """
    nominal: np.ndarray
    empirical: np.ndarray
    ece: float


@torch.no_grad()
def regression_coverage_calibration(
    y_true: ArrayLike,
    mu: ArrayLike,
    sigma: ArrayLike,
    alphas: Optional[Iterable[float]] = None,
    clamp_sigma: float = 1e-12,
) -> RegressionCalibrationResult:
    """
    Calibration for probabilistic regression under (assumed) Gaussian predictive distribution.

    For each nominal central coverage alpha in (0,1), we compute the symmetric interval
    [mu - z * sigma, mu + z * sigma], where z = Phi^{-1}((1 + alpha)/2). We then measure
    empirical coverage = mean( y_true in interval ), and report an ECE-like scalar:

        reg_ECE = mean_alpha | empirical(alpha) - alpha |

    This is convenient for reliability diagrams (plot empirical vs nominal).

    Args:
        y_true: shape (N,)
        mu:     shape (N,) predictive means
        sigma:  shape (N,) predictive standard deviations (non-negative)
        alphas: iterable of nominal coverages in (0,1). If None, uses np.linspace(0.1, 0.9, 9)
        clamp_sigma: lower bound to avoid zero/NaN divisions

    Returns:
        RegressionCalibrationResult(nominal, empirical, ece)
    """
    yt = _to_tensor(y_true).view(-1).to(torch.float64)
    m = _to_tensor(mu).view(-1).to(torch.float64).to(device=yt.device)
    s = _to_tensor(sigma).view(-1).to(torch.float64).to(device=yt.device)
    s = torch.clamp(s, min=float(clamp_sigma))

    if yt.numel() == 0:
        return RegressionCalibrationResult(
            nominal=np.array([], dtype=np.float64),
            empirical=np.array([], dtype=np.float64),
            ece=0.0,
        )

    if alphas is None:
        alphas = np.linspace(0.1, 0.9, 9)  # 10%..90% central coverage

    # Use torch Normal icdf for z-quantiles
    standard_normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64, device=yt.device),
        torch.tensor(1.0, dtype=torch.float64, device=yt.device),
    )

    nominal_list: List[float] = []
    empirical_list: List[float] = []

    for alpha in alphas:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Each alpha must be in (0,1); got {alpha}")
        # central interval => quantiles at (1-alpha)/2 and (1+alpha)/2
        lo_q = (1.0 - float(alpha)) / 2.0
        hi_q = (1.0 + float(alpha)) / 2.0
        z_hi = standard_normal.icdf(
            torch.tensor(hi_q, dtype=torch.float64, device=yt.device)
        )
        # symmetric, so z_lo = -z_hi; we only need magnitude
        z = torch.abs(z_hi)

        lo = m - z * s
        hi = m + z * s
        covered = ((yt >= lo) & (yt <= hi)).to(torch.float64)
        emp = float(covered.mean().item())

        nominal_list.append(float(alpha))
        empirical_list.append(emp)

    nominal = np.asarray(nominal_list, dtype=np.float64)
    empirical = np.asarray(empirical_list, dtype=np.float64)
    ece = float(np.mean(np.abs(empirical - nominal)))
    return RegressionCalibrationResult(nominal=nominal, empirical=empirical, ece=ece)


# ---------------------------------------------------------------------
# Convenience wrappers (kept small; callers can import softmax/argmax elsewhere)
# ---------------------------------------------------------------------


def _argmax_probs(probs: Tensor) -> Tensor:
    return torch.argmax(probs, dim=-1)


def _softmax(logits: Tensor) -> Tensor:
    return torch.softmax(logits, dim=-1)
