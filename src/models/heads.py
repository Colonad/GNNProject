# src/models/heads.py
from __future__ import annotations
"""
Common prediction heads and calibration utilities.

Included
--------
Blocks / Heads
- MLP: flexible fully-connected stack (norm/act/dropout/residual/init).
- RegressionHead: homoscedastic or heteroscedastic Gaussian (NLL).
- ClassificationHead: logits head with optional, pluggable post-hoc calibrator.

Calibrators (post-hoc)
- TemperatureScaler: scalar T (logits / T) with L-BFGS/Adam fit on val set.
- VectorScaler: per-class temperature + bias (diagonal affine of logits).
- MatrixScaler: full affine logits' = logits @ W^T + b (with L2 on W).
- DirichletCalibrator: linear map on log-softmax (simple/full).
- IsotonicCalibrator: binary or one-vs-rest isotonic (sklearn-backed; identity if sklearn missing).

Loss helpers
- classification_ece: lightweight L1 ECE.
- confidence_penalty: -β * H(p) for better calibration.
- label_smoothing_ce: cross-entropy with label smoothing.

Notes
-----
- Calibrators implement `forward(logits)` and, when applicable, `.probs(logits)`.
- `ClassificationHead` exposes:
    * `set_calibrator(module, apply_on_eval=True)`
    * `clear_calibrator()`
    * `fit_posthoc(kind=..., val_logits=..., val_labels=..., **hparams)`
  and applies calibration automatically at *eval time only* if enabled.
"""

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Tuple, Union

import math
import numpy as np
import torch
from torch import Tensor, nn

try:
    from sklearn.isotonic import IsotonicRegression  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _get_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in ("leaky_relu", "lrelu", "leaky-relu"):
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "elu":
        return nn.ELU(alpha=1.0, inplace=True)
    raise ValueError(f"Unknown activation: {name}")


def _init_linear(layer: nn.Linear, init: str = "kaiming", bias: float = 0.0) -> None:
    init = (init or "kaiming").lower()
    if init == "kaiming":
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    elif init == "xavier":
        nn.init.xavier_uniform_(layer.weight)
    elif init == "none":
        pass
    else:
        raise ValueError(f"Unknown init: {init}")
    if layer.bias is not None:
        nn.init.constant_(layer.bias, float(bias))


# ---------------------------------------------------------------------------
# MLP block
# ---------------------------------------------------------------------------

@dataclass
class MLPConfig:
    in_dim: int
    out_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    activation: str = "relu"
    dropout: float = 0.0
    norm: Literal["none", "batch", "layer"] = "none"
    residual: bool = False
    init: Literal["kaiming", "xavier", "none"] = "kaiming"
    last_bias: float = 0.0


class MLP(nn.Module):
    """Flexible MLP with optional norm/dropout/residual."""
    def __init__(self, cfg: MLPConfig) -> None:
        super().__init__()
        self.cfg = cfg
        L = int(cfg.num_layers)
        if L < 1:
            raise ValueError("num_layers must be >= 1")

        dims: list[int] = [cfg.in_dim]
        if L > 1:
            dims += [cfg.hidden_dim] * (L - 1)
        dims += [cfg.out_dim]

        blocks: list[nn.Module] = []
        self._use_resid = bool(cfg.residual and L > 1 and cfg.in_dim == cfg.hidden_dim == cfg.out_dim)

        for i in range(L):
            in_d, out_d = dims[i], dims[i + 1]
            linear = nn.Linear(in_d, out_d)
            is_last = (i == L - 1)
            _init_linear(linear, cfg.init, cfg.last_bias if is_last else 0.0)

            block: list[nn.Module] = [linear]
            if not is_last:
                if cfg.norm == "batch":
                    block.append(nn.BatchNorm1d(out_d))
                elif cfg.norm == "layer":
                    block.append(nn.LayerNorm(out_d))
                block.append(_get_activation(cfg.activation))
                if cfg.dropout and cfg.dropout > 0:
                    block.append(nn.Dropout(p=float(cfg.dropout)))
            blocks.append(nn.Sequential(*block))

        self.net = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for i, block in enumerate(self.net):
            h_next = block(h)
            if self._use_resid and i < len(self.net) - 1:
                h = h + h_next
            else:
                h = h_next
        return h


# ---------------------------------------------------------------------------
# Regression head
# ---------------------------------------------------------------------------

@dataclass
class RegressionHeadConfig:
    in_dim: int
    out_dim: int = 1
    heteroscedastic: bool = False
    hidden_dim: int = 128
    num_layers: int = 2
    activation: str = "relu"
    dropout: float = 0.0
    norm: Literal["none", "batch", "layer"] = "none"
    residual: bool = False
    init: Literal["kaiming", "xavier", "none"] = "kaiming"
    logvar_min: float = -10.0
    logvar_max: float = 4.0


class RegressionHead(nn.Module):
    """
    Regression head.
      - homoscedastic: predicts mean only.
      - heteroscedastic: predicts (mu, log_var); train with Gaussian NLL.
    """
    def __init__(self, cfg: RegressionHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        out_dim = cfg.out_dim if not cfg.heteroscedastic else (cfg.out_dim * 2)
        self.mlp = MLP(MLPConfig(
            in_dim=cfg.in_dim, out_dim=out_dim, hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers, activation=cfg.activation, dropout=cfg.dropout,
            norm=cfg.norm, residual=cfg.residual, init=cfg.init, last_bias=0.0
        ))

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        y = self.mlp(x)
        if self.cfg.heteroscedastic:
            mu, log_var = torch.chunk(y, 2, dim=-1)
            log_var = torch.clamp(log_var, min=self.cfg.logvar_min, max=self.cfg.logvar_max)
            return mu, log_var
        return y

    def loss(
        self,
        pred: Union[Tensor, Tuple[Tensor, Tensor]],
        target: Tensor,
        kind: Literal["mse", "gauss_nll"] = "gauss_nll",
        reduction: Literal["mean", "sum"] = "mean",
    ) -> Tensor:
        if isinstance(pred, tuple):
            mu, log_var = pred
        else:
            mu, log_var = pred, None

        target = target.to(mu.dtype)
        if kind == "mse" or (kind == "gauss_nll" and log_var is None):
            loss = (mu - target) ** 2
        else:
            var = torch.exp(log_var)
            loss = 0.5 * (log_var + (target - mu) ** 2 / (var + 1e-12))

        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        raise ValueError(f"Unknown reduction: {reduction}")


# ---------------------------------------------------------------------------
# Classification head + pluggable calibrator
# ---------------------------------------------------------------------------

@dataclass
class ClassificationHeadConfig:
    in_dim: int
    num_classes: int
    hidden_dim: int = 128
    num_layers: int = 2
    activation: str = "relu"
    dropout: float = 0.0
    norm: Literal["none", "batch", "layer"] = "none"
    residual: bool = False
    init: Literal["kaiming", "xavier", "none"] = "kaiming"

    # legacy: keep optional temperature scaler for quick wins
    use_temperature: bool = False
    init_temperature: float = 1.0
    temperature_min: float = 1e-3
    temperature_max: float = 100.0


class TemperatureScaler(nn.Module):
    """Scalar temperature: logits_scaled = logits / T, with T>0 (log-param)."""
    def __init__(self, init_T: float = 1.0, t_min: float = 1e-3, t_max: float = 100.0):
        super().__init__()
        self.logT = nn.Parameter(torch.tensor(float(math.log(max(init_T, 1e-6))), dtype=torch.float32))
        self.t_min = float(t_min)
        self.t_max = float(t_max)

    def temperature(self) -> Tensor:
        T = torch.exp(self.logT)
        return torch.clamp(T, min=self.t_min, max=self.t_max)

    def forward(self, logits: Tensor) -> Tensor:
        T = self.temperature().to(logits.device, logits.dtype)
        return logits / T

    def fit(self, logits: Tensor, labels: Tensor, max_iters: int = 200, tol: float = 1e-6,
            optimizer: Literal["lbfgs", "adam"] = "lbfgs", lr: float = 0.1, verbose: bool = False) -> float:
        self.train()
        labels = labels.long()
        params = [self.logT]

        def loss_fn() -> Tensor:
            return nn.functional.cross_entropy(self.forward(logits), labels, reduction="mean")

        if optimizer == "lbfgs":
            opt = torch.optim.LBFGS(params, lr=lr, max_iter=max_iters, line_search_fn="strong_wolfe")
            last = None
            def closure():
                opt.zero_grad(set_to_none=True)
                L = loss_fn()
                L.backward()
                return L
            for _ in range(max_iters):
                L = opt.step(closure)
                val = float(L.detach().cpu().item())
                if verbose: print(f"[TempScale] NLL={val:.6f} | T={float(self.temperature().item()):.4f}")
                if last is not None and abs(last - val) < tol: break
                last = val
        else:
            opt = torch.optim.Adam(params, lr=lr)
            last = None
            for _ in range(max_iters):
                opt.zero_grad(set_to_none=True)
                L = loss_fn()
                L.backward()
                opt.step()
                val = float(L.detach().cpu().item())
                if verbose: print(f"[TempScale] NLL={val:.6f} | T={float(self.temperature().item()):.4f}")
                if last is not None and abs(last - val) < tol: break
                last = val
        return float(self.temperature().detach().cpu().item())


class VectorScaler(nn.Module):
    """Per-class temperature + bias: logits' = logits / T_c + b_c."""
    def __init__(self, num_classes: int, init_T: float = 1.0):
        super().__init__()
        self.logT = nn.Parameter(torch.full((num_classes,), math.log(max(init_T, 1e-6)), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(num_classes, dtype=torch.float32))

    def forward(self, logits: Tensor) -> Tensor:
        T = torch.exp(self.logT).to(logits)
        return logits / T + self.bias

    def fit(self, logits: Tensor, labels: Tensor, max_iters: int = 200, tol: float = 1e-6,
            lr: float = 0.1, optimizer: Literal["lbfgs", "adam"] = "lbfgs", verbose: bool = False):
        labels = labels.long()
        params = [self.logT, self.bias]

        def loss_fn() -> Tensor:
            return nn.functional.cross_entropy(self.forward(logits), labels, reduction="mean")

        if optimizer == "lbfgs":
            opt = torch.optim.LBFGS(params, lr=lr, max_iter=max_iters, line_search_fn="strong_wolfe")
            last = None
            def closure():
                opt.zero_grad(set_to_none=True)
                L = loss_fn()
                L.backward()
                return L
            for _ in range(max_iters):
                L = opt.step(closure)
                val = float(L.detach().cpu().item())
                if verbose: print(f"[VectorScale] NLL={val:.6f}")
                if last is not None and abs(last - val) < tol: break
                last = val
        else:
            opt = torch.optim.Adam(params, lr=lr)
            last = None
            for _ in range(max_iters):
                opt.zero_grad(set_to_none=True)
                L = loss_fn()
                L.backward()
                opt.step()
                val = float(L.detach().cpu().item())
                if verbose: print(f"[VectorScale] NLL={val:.6f}")
                if last is not None and abs(last - val) < tol: break
                last = val
        return torch.exp(self.logT).detach(), self.bias.detach()


class MatrixScaler(nn.Module):
    """Full affine: logits' = logits @ W^T + b, with L2(W) regularization."""
    def __init__(self, num_classes: int, l2: float = 1e-4):
        super().__init__()
        self.W = nn.Parameter(torch.eye(num_classes, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(num_classes, dtype=torch.float32))
        self.l2 = float(l2)

    def forward(self, logits: Tensor) -> Tensor:
        return logits @ self.W.T + self.b

    def fit(self, logits: Tensor, labels: Tensor, max_iters: int = 300, tol: float = 1e-6,
            lr: float = 0.05, optimizer: Literal["lbfgs", "adam"] = "adam", verbose: bool = False):
        labels = labels.long()
        params = [self.W, self.b]

        def loss_fn() -> Tensor:
            logits_p = self.forward(logits)
            ce = nn.functional.cross_entropy(logits_p, labels, reduction="mean")
            reg = self.l2 * (self.W ** 2).sum()
            return ce + reg

        if optimizer == "lbfgs":
            opt = torch.optim.LBFGS(params, lr=lr, max_iter=max_iters, line_search_fn="strong_wolfe")
            last = None
            def closure():
                opt.zero_grad(set_to_none=True)
                L = loss_fn()
                L.backward()
                return L
            for _ in range(max_iters):
                L = opt.step(closure)
                val = float(L.detach().cpu().item())
                if verbose: print(f"[MatrixScale] Obj={val:.6f}")
                if last is not None and abs(last - val) < tol: break
                last = val
        else:
            opt = torch.optim.Adam(params, lr=lr, weight_decay=0.0)
            last = None
            for _ in range(max_iters):
                opt.zero_grad(set_to_none=True)
                L = loss_fn()
                L.backward()
                opt.step()
                val = float(L.detach().cpu().item())
                if verbose: print(f"[MatrixScale] Obj={val:.6f}")
                if last is not None and abs(last - val) < tol: break
                last = val
        return self.W.detach(), self.b.detach()


class DirichletCalibrator(nn.Module):
    """
    Linear map on log-probabilities:
      z = log_softmax(logits); scores = z @ A^T + b  (simple: A=diag; full: A full)
    """
    def __init__(self, num_classes: int, mode: Literal["simple", "full"] = "simple", l2: float = 0.0):
        super().__init__()
        self.C = int(num_classes)
        self.mode = mode
        self.l2 = float(l2)
        if mode == "simple":
            self.a = nn.Parameter(torch.ones(self.C, dtype=torch.float32))
        elif mode == "full":
            self.A = nn.Parameter(torch.eye(self.C, dtype=torch.float32))
        else:
            raise ValueError("mode must be 'simple' or 'full'")
        self.b = nn.Parameter(torch.zeros(self.C, dtype=torch.float32))

    def forward(self, logits: Tensor) -> Tensor:
        z = nn.functional.log_softmax(logits, dim=-1)
        if self.mode == "simple":
            s = z * self.a + self.b
        else:
            s = z @ self.A.T + self.b
        return s

    def probs(self, logits: Tensor) -> Tensor:
        return nn.functional.softmax(self.forward(logits), dim=-1)

    def fit(self, logits: Tensor, labels: Tensor, max_iters: int = 300, tol: float = 1e-6,
            lr: float = 0.05, optimizer: Literal["lbfgs", "adam"] = "adam", verbose: bool = False):
        labels = labels.long()
        params = [self.b] + ([self.a] if self.mode == "simple" else [self.A])

        def loss_fn() -> Tensor:
            scores = self.forward(logits)
            ce = nn.functional.cross_entropy(scores, labels, reduction="mean")
            reg = 0.0
            if self.l2 > 0:
                if self.mode == "simple":
                    reg = self.l2 * (self.a ** 2).sum()
                else:
                    reg = self.l2 * (self.A ** 2).sum()
            return ce + reg

        if optimizer == "lbfgs":
            opt = torch.optim.LBFGS(params, lr=lr, max_iter=max_iters, line_search_fn="strong_wolfe")
            last = None
            def closure():
                opt.zero_grad(set_to_none=True)
                L = loss_fn()
                L.backward()
                return L
            for _ in range(max_iters):
                L = opt.step(closure)
                val = float(L.detach().cpu().item())
                if verbose: print(f"[Dirichlet-{self.mode}] CE={val:.6f}")
                if last is not None and abs(last - val) < tol: break
                last = val
        else:
            opt = torch.optim.Adam(params, lr=lr)
            last = None
            for _ in range(max_iters):
                opt.zero_grad(set_to_none=True)
                L = loss_fn()
                L.backward()
                opt.step()
                val = float(L.detach().cpu().item())
                if verbose: print(f"[Dirichlet-{self.mode}] CE={val:.6f}")
                if last is not None and abs(last - val) < tol: break
                last = val


class IsotonicCalibrator(nn.Module):
    """
    Isotonic regression calibrator.
    - Binary: single isotonic map on the positive-class score.
    - Multiclass: OVR isotonic on each class score, then renormalize.
    Requires scikit-learn; otherwise this becomes identity.
    """
    def __init__(self, num_classes: int, y_min: float = 0.0, y_max: float = 1.0):
        super().__init__()
        self.C = int(num_classes)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self._models: list[Optional[IsotonicRegression]] = [None] * self.C  # type: ignore[name-defined]
        self._enabled = _HAS_SKLEARN

    def forward(self, logits: Tensor) -> Tensor:
        # Return log-probs (log p) for compatibility with CE; for probs, call `.probs`
        p = self.probs(logits)
        eps = 1e-12
        return torch.log(torch.clamp(p, min=eps))

    @torch.no_grad()
    def probs(self, logits: Tensor) -> Tensor:
        if not self._enabled or self._models[0] is None:
            return nn.functional.softmax(logits, dim=-1)

        N, C = logits.shape
        out = torch.zeros_like(logits, dtype=torch.float32)
        for c in range(C):
            s = logits[:, c].detach().cpu().numpy()
            p_c = self._models[c].predict(s)  # type: ignore[index]
            out[:, c] = torch.as_tensor(p_c, dtype=torch.float32, device=logits.device)
        out = torch.clamp(out, 0.0, 1.0)
        denom = out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return out / denom

    def fit(self, logits: Tensor, labels: Tensor, verbose: bool = False):
        if not _HAS_SKLEARN:
            if verbose:
                print("[Isotonic] scikit-learn not available; isotonic calibration disabled (identity).")
            return
        labels = labels.long().detach().cpu().numpy()
        N, C = logits.shape
        scores = logits.detach().cpu().numpy()

        if C == 2:
            pos = 1
            y = (labels == pos).astype(np.float64)
            s = scores[:, pos]
            self._models = [None, IsotonicRegression(y_min=self.y_min, y_max=self.y_max, out_of_bounds="clip")]
            self._models[pos].fit(s, y)  # type: ignore[index]
            if verbose:
                print("[Isotonic] Fitted binary isotonic on class-1 scores.")
        else:
            self._models = [IsotonicRegression(y_min=self.y_min, y_max=self.y_max, out_of_bounds="clip") for _ in range(C)]  # type: ignore[name-defined]
            for c in range(C):
                y_c = (labels == c).astype(np.float64)
                s_c = scores[:, c]
                self._models[c].fit(s_c, y_c)  # type: ignore[index]
            if verbose:
                print(f"[Isotonic] Fitted {C} one-vs-rest isotonic models.")


# ---------------------------------------------------------------------------
# Classification head — pluggable calibrator wiring
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    Logit head with optional, pluggable post-hoc calibrator.

    * During training mode (self.training == True): returns **uncalibrated** logits.
    * During eval mode (self.training == False):
        - if a calibrator is attached and `apply_calibration_on_eval=True`,
          returns calibrated logits (suitable for CE/NLL/metrics).
        - otherwise, returns raw logits.
    """
    def __init__(self, cfg: ClassificationHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mlp = MLP(MLPConfig(
            in_dim=cfg.in_dim, out_dim=cfg.num_classes, hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers, activation=cfg.activation, dropout=cfg.dropout,
            norm=cfg.norm, residual=cfg.residual, init=cfg.init, last_bias=0.0
        ))

        # Backwards-compatible temperature knob
        self.temp: Optional[TemperatureScaler] = TemperatureScaler(
            init_T=cfg.init_temperature, t_min=cfg.temperature_min, t_max=cfg.temperature_max
        ) if cfg.use_temperature else None

        # Pluggable post-hoc calibrator (fitted after training)
        self._calibrator: Optional[nn.Module] = None
        self.apply_calibration_on_eval: bool = True

    # --- core ---
    def raw_logits(self, x: Tensor) -> Tensor:
        logits = self.mlp(x)
        if self.temp is not None:
            logits = self.temp(logits)
        return logits

    def forward(self, x: Tensor) -> Tensor:
        logits = self.raw_logits(x)
        if (not self.training) and self.apply_calibration_on_eval and (self._calibrator is not None):
            # Use calibrated *scores/logits*
            return self._calibrator(logits)
        return logits

    def loss(self, logits: Tensor, labels: Tensor, reduction: str = "mean") -> Tensor:
        return nn.functional.cross_entropy(logits, labels.long(), reduction=reduction)

    @torch.no_grad()
    def probs(self, logits_or_features: Tensor, *, from_logits: bool = True) -> Tensor:
        logits = logits_or_features if from_logits else self.forward(logits_or_features)
        if (not self.training) and self.apply_calibration_on_eval and (self._calibrator is not None):
            # If calibrator has probs(), prefer that; else softmax of forward()
            if hasattr(self._calibrator, "probs"):
                return self._calibrator.probs(logits)  # type: ignore[attr-defined]
            return nn.functional.softmax(self._calibrator(logits), dim=-1)
        return nn.functional.softmax(logits, dim=-1)

    # --- calibrator management ---
    def set_calibrator(self, module: Optional[nn.Module], apply_on_eval: bool = True) -> None:
        """Attach (or remove with None) a post-hoc calibrator."""
        self._calibrator = module
        self.apply_calibration_on_eval = bool(apply_on_eval)

    def clear_calibrator(self) -> None:
        self._calibrator = None

    def get_calibrator(self) -> Optional[nn.Module]:
        return self._calibrator

    # --- convenience: fit a calibrator on validation logits ---
    @torch.no_grad()
    def fit_posthoc(
        self,
        kind: Literal["temperature", "vector", "matrix", "dirichlet_simple", "dirichlet_full", "isotonic"],
        val_logits: Tensor,
        val_labels: Tensor,
        *,
        optimizer: Literal["lbfgs", "adam"] = "lbfgs",
        lr: float = 0.1,
        max_iters: int = 300,
        l2: float = 1e-4,
        verbose: bool = False,
        attach: bool = True,
    ) -> Optional[nn.Module]:
        """
        Fit a calibrator on (val_logits, val_labels) and (optionally) attach it.

        Returns:
            The fitted calibrator (or None if 'kind' is unrecognized).
        """
        C = val_logits.size(-1)
        cal: Optional[nn.Module] = None

        if kind == "temperature":
            cal = TemperatureScaler()
            cal.fit(val_logits, val_labels, optimizer=optimizer, lr=lr, max_iters=max_iters, verbose=verbose)  # type: ignore[attr-defined]
        elif kind == "vector":
            cal = VectorScaler(C)
            cal.fit(val_logits, val_labels, optimizer=optimizer, lr=lr, max_iters=max_iters, verbose=verbose)  # type: ignore[attr-defined]
        elif kind == "matrix":
            cal = MatrixScaler(C, l2=l2)
            cal.fit(val_logits, val_labels, optimizer=optimizer, lr=lr, max_iters=max_iters, verbose=verbose)  # type: ignore[attr-defined]
        elif kind == "dirichlet_simple":
            cal = DirichletCalibrator(C, mode="simple", l2=l2)
            cal.fit(val_logits, val_labels, optimizer=optimizer, lr=lr, max_iters=max_iters, verbose=verbose)  # type: ignore[attr-defined]
        elif kind == "dirichlet_full":
            cal = DirichletCalibrator(C, mode="full", l2=l2)
            cal.fit(val_logits, val_labels, optimizer=optimizer, lr=lr, max_iters=max_iters, verbose=verbose)  # type: ignore[attr-defined]
        elif kind == "isotonic":
            cal = IsotonicCalibrator(C)
            cal.fit(val_logits, val_labels, verbose=verbose)  # type: ignore[attr-defined]
        else:
            return None

        if attach:
            self.set_calibrator(cal, apply_on_eval=True)
        return cal


# ---------------------------------------------------------------------------
# Loss helpers / small metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def classification_ece(
    logits_or_probs: Tensor,
    labels: Tensor,
    n_bins: int = 15,
    from_logits: bool = True,
) -> float:
    if from_logits:
        probs = torch.softmax(logits_or_probs, dim=-1)
    else:
        probs = logits_or_probs
    conf, pred = torch.max(probs, dim=1)
    correct = (pred == labels.long()).to(torch.float32)

    conf_np = conf.detach().cpu().numpy()
    corr_np = correct.detach().cpu().numpy()
    bin_edges = np.linspace(0.0, 1.0, num=n_bins + 1)
    bin_ids = np.digitize(conf_np, bin_edges[1:-1], right=True)

    ece = 0.0
    N = len(conf_np)
    for b in range(n_bins):
        m = bin_ids == b
        cnt = int(m.sum())
        if cnt > 0:
            conf_b = float(conf_np[m].mean())
            acc_b = float(corr_np[m].mean())
            ece += (cnt / N) * abs(acc_b - conf_b)
    return float(ece)


def confidence_penalty(
    logits: Tensor,
    beta: float = 0.1,
    from_logits: bool = True,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    if from_logits:
        p = nn.functional.softmax(logits, dim=-1)
    else:
        p = logits
    eps = 1e-12
    ent = -(p * (p.clamp_min(eps).log())).sum(dim=-1)
    penalty = -beta * ent
    if reduction == "mean":
        return penalty.mean()
    if reduction == "sum":
        return penalty.sum()
    return penalty


def label_smoothing_ce(
    logits: Tensor,
    targets: Tensor,
    smoothing: float = 0.1,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    num_classes = logits.size(-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.long().unsqueeze(1), 1.0 - smoothing)
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    loss = -(true_dist * log_probs).sum(dim=-1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


__all__ = [
    # Blocks / Heads
    "MLPConfig", "MLP",
    "RegressionHeadConfig", "RegressionHead",
    "ClassificationHeadConfig", "ClassificationHead",
    # Calibrators
    "TemperatureScaler", "VectorScaler", "MatrixScaler",
    "DirichletCalibrator", "IsotonicCalibrator",
    # Helpers
    "classification_ece", "confidence_penalty", "label_smoothing_ce",
]
