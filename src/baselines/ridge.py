# src/baselines/ridge.py
"""
Ridge Regression Baseline on RDKit Descriptors (with optional Morgan FP)
=======================================================================

- Loads SMILES + targets from ESOL or QM9 (subset).
- Computes descriptor matrix via `src.baselines.descriptors`.
- Uses the SAME split logic as GNN runs: random or scaffold (Bemis–Murcko).
- Performs a small hyperparameter sweep over `alpha` and `fit_intercept`,
  picks the best on **validation MAE**, retrains on train+val, reports test.
- Saves a **CSV row** under `outputs/baselines/` with metrics and config.
- Optionally saves predictions to CSV and the fitted sklearn model (joblib).

Usage:
  python -m src.baselines.ridge --dataset ESOL --split scaffold --scale standard
  python -m src.baselines.ridge --dataset QM9 --limit-n 15000 --target U0 --split random

Definition of Done (Phase 2):
- Train/val/test with same splits; save CSV metrics to outputs/baselines/.
"""

from __future__ import annotations
import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.debug")
RDLogger.DisableLog("rdApp.info")

# Local utilities
# NOTE: We alias DescriptorMatrixConfig as DescriptorConfig so your code below stays unchanged.
from .descriptors import (
    DescriptorMatrixConfig as DescriptorConfig,
    build_descriptor_matrix,
    load_smiles_from_dataset,
    ensure_dir,
)
from src.datamodules.splits import SplitSpec, random_split_indices, scaffold_split_indices

# QM9 target index mapping (keep in sync with qm9.py)
QM9_TARGET_INDEX = {
    "mu": 0, "alpha": 1, "homo": 2, "lumo": 3, "gap": 4, "r2": 5, "zpve": 6,
    "U0": 7, "U": 8, "H": 9, "G": 10, "Cv": 11
}


@dataclass
class RidgeConfig:
    # Data
    dataset: str = "ESOL"               # {"ESOL", "QM9"}
    root: str = "data"
    split: str = "scaffold"             # {"scaffold","random"}
    seed: int = 0
    train_frac: float = 0.8
    val_frac: float = 0.1
    limit_n: Optional[int] = None       # QM9 subsample; None = full

    # QM9 target selection
    target_key: Optional[str] = "U0"
    target_index: Optional[int] = None  # overrides key if set

    # Descriptors
    use_morgan: bool = True
    morgan_radius: int = 2
    morgan_bits: int = 1024
    morgan_use_chirality: bool = True
    drop_constant: bool = True
    scale: str = "standard"             # scale features for Ridge by default

    # Hyperparameters (grid)
    alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0)
    fit_intercept_opts: Sequence[bool] = (True, False)

    # IO
    out_dir: str = "outputs/baselines"
    save_preds: bool = True
    tag: Optional[str] = None           # optional tag appended to filename

    # Verbosity
    verbose: bool = True


def _resolve_qm9_target_index(key: Optional[str], index: Optional[int]) -> int:
    if index is not None:
        return int(index)
    if key is not None and key in QM9_TARGET_INDEX:
        return QM9_TARGET_INDEX[key]
    return QM9_TARGET_INDEX["U0"]


def _load_targets(dataset: str, root: str, limit_n: Optional[int], target_idx: int) -> np.ndarray:
    """Return y vector aligned with SMILES order from `load_smiles_from_dataset`."""
    ds_name = dataset.strip().upper()
    if ds_name == "ESOL":
        from torch_geometric.datasets import MoleculeNet
        ds = MoleculeNet(root=root, name="ESOL")
        y = np.array([float(d.y.view(-1)[0].item()) for d in ds], dtype=float)
        return y
    elif ds_name == "QM9":
        from torch_geometric.datasets import QM9 as PYG_QM9
        ds = PYG_QM9(root=root)
        if limit_n is not None:
            ds = ds[: int(limit_n)]
        ys = []
        for d in ds:
            y = d.y.view(-1).cpu().numpy()
            ys.append(float(y[target_idx]))
        return np.array(ys, dtype=float)
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")


def _make_splits(smiles: List[str], cfg: RidgeConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = SplitSpec(cfg.train_frac, cfg.val_frac)
    N = len(smiles)
    if cfg.split == "random":
        tr, va, te = random_split_indices(N, spec=spec, seed=cfg.seed)
    elif cfg.split == "scaffold":
        tr, va, te = scaffold_split_indices(smiles, spec=spec, seed=cfg.seed)
    else:
        raise ValueError(f"Unknown split {cfg.split!r}")
    return np.array(tr, dtype=int), np.array(va, dtype=int), np.array(te, dtype=int)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"MAE": mae, "RMSE": rmse}


def train_and_eval(cfg: RidgeConfig) -> Dict[str, object]:
    # 1) Load SMILES + targets
    smiles = load_smiles_from_dataset(cfg.dataset, cfg.root, limit_n=cfg.limit_n)
    tgt_idx = _resolve_qm9_target_index(cfg.target_key, cfg.target_index)
    y_all = _load_targets(cfg.dataset, cfg.root, limit_n=cfg.limit_n, target_idx=tgt_idx)

    # 2) Features
    desc_cfg = DescriptorConfig(
        dataset=cfg.dataset, root=cfg.root, limit_n=cfg.limit_n,
        use_morgan=cfg.use_morgan, morgan_radius=cfg.morgan_radius, morgan_bits=cfg.morgan_bits,
        morgan_use_chirality=cfg.morgan_use_chirality, drop_constant=cfg.drop_constant, scale=cfg.scale,
        out_dir="outputs/descriptors", file_prefix=None, seed=cfg.seed, verbose=cfg.verbose
    )
    X_all, colnames, _desc_meta = build_descriptor_matrix(smiles, desc_cfg)

    # 3) Splits
    tr_idx, va_idx, te_idx = _make_splits(smiles, cfg)
    Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
    Xva, yva = X_all[va_idx], y_all[va_idx]
    Xte, yte = X_all[te_idx], y_all[te_idx]

    # 4) Small hyperparameter sweep on validation MAE
    best = None
    for alpha in cfg.alphas:
        for fit_intercept in cfg.fit_intercept_opts:
            # sklearn Ridge has no random_state param; keep it deterministic via data.
            model = Ridge(alpha=float(alpha), fit_intercept=bool(fit_intercept))
            model.fit(Xtr, ytr)
            pva = model.predict(Xva)
            m = _metrics(yva, pva)
            cand = {
                "alpha": float(alpha),
                "fit_intercept": bool(fit_intercept),
                "val_MAE": m["MAE"],
                "val_RMSE": m["RMSE"],
            }
            if best is None or cand["val_MAE"] < best["val_MAE"]:
                best = cand
    assert best is not None

    # 5) Retrain on train+val with best hyperparams, test once
    trval_idx = np.concatenate([tr_idx, va_idx])
    Xtrval, ytrval = X_all[trval_idx], y_all[trval_idx]
    model = Ridge(alpha=best["alpha"], fit_intercept=best["fit_intercept"])
    model.fit(Xtrval, ytrval)
    pte = model.predict(Xte)
    test_metrics = _metrics(yte, pte)

    # 6) Save metrics row
    ensure_dir(cfg.out_dir)
    ts = time.strftime("%Y%m%d-%H%M%S")
    tag = f"_{cfg.tag}" if cfg.tag else ""
    fname = f"{cfg.dataset.upper()}_ridge_{cfg.split}_seed{cfg.seed}{tag}_{ts}.csv"
    out_csv = os.path.join(cfg.out_dir, fname)

    header = [
        "dataset","split","seed","N","D","train_size","val_size","test_size",
        "alpha","fit_intercept",
        "val_MAE","val_RMSE","test_MAE","test_RMSE",
        "use_morgan","morgan_bits","morgan_radius","morgan_use_chirality","drop_constant","scale"
    ]
    row = [
        cfg.dataset.upper(), cfg.split, cfg.seed, X_all.shape[0], X_all.shape[1],
        len(tr_idx), len(va_idx), len(te_idx),
        best["alpha"], best["fit_intercept"],
        best["val_MAE"], best["val_RMSE"], test_metrics["MAE"], test_metrics["RMSE"],
        int(cfg.use_morgan), cfg.morgan_bits, cfg.morgan_radius, int(cfg.morgan_use_chirality), int(cfg.drop_constant), cfg.scale
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        writer = _csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)

    # 7) Optionally save predictions (train+val+test, aligned by original index)
    preds_csv = None
    if cfg.save_preds:
        preds_csv = out_csv.replace(".csv", "_preds.csv")
        all_indices = np.concatenate([tr_idx, va_idx, te_idx]).tolist()
        all_preds = model.predict(X_all[all_indices]).tolist()
        all_true = y_all[all_indices].tolist()
        with open(preds_csv, "w", newline="", encoding="utf-8") as f:
            import csv as _csv
            writer = _csv.writer(f)
            writer.writerow(["index","y_true","y_pred"])
            for idx, yt, yp in zip(all_indices, all_true, all_preds):
                writer.writerow([int(idx), float(yt), float(yp)])

    if cfg.verbose:
        print(f"[ridge] Saved metrics: {out_csv}")
        if preds_csv:
            print(f"[ridge] Saved predictions: {preds_csv}")
        print(f"[ridge] Best alpha={best['alpha']}, fit_intercept={best['fit_intercept']}; "
              f"Val MAE={best['val_MAE']:.4f}, Test MAE={test_metrics['MAE']:.4f}")

    return {
        "metrics_csv": out_csv,
        "preds_csv": preds_csv,
        "best": best,
        "test_metrics": test_metrics,
        "feature_shape": X_all.shape,
        "columns": colnames,
    }


# ---------------------------
# CLI glue
# ---------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ridge baseline on RDKit descriptors (ESOL/QM9).")
    p.add_argument("--dataset", type=str, default="ESOL", choices=["ESOL","QM9"])
    p.add_argument("--root", type=str, default="data")
    p.add_argument("--split", type=str, default="scaffold", choices=["scaffold","random"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--limit-n", type=int, default=None, help="QM9 subset size; None for full")
    p.add_argument("--target", type=str, default="U0", help="QM9 target key (ignored for ESOL)")
    p.add_argument("--target-index", type=int, default=None, help="QM9 target index (overrides key)")
    p.add_argument("--no-fp", action="store_true", help="Disable Morgan fingerprints")
    p.add_argument("--fp-bits", type=int, default=1024)
    p.add_argument("--fp-radius", type=int, default=2)
    p.add_argument("--fp-no-chiral", action="store_true", help="Disable chirality in Morgan FP")
    p.add_argument("--no-drop-constant", action="store_true")
    p.add_argument("--scale", type=str, default="standard", choices=["none","standard","minmax"])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.01,0.1,1.0,10.0,100.0])
    p.add_argument("--fit-intercept", type=int, nargs="+", default=[1,0], help="1/0 values tested in grid")
    p.add_argument("--out-dir", type=str, default="outputs/baselines")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--no-save-preds", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def _cfg_from_args(args: argparse.Namespace) -> RidgeConfig:
    return RidgeConfig(
        dataset=args.dataset, root=args.root, split=args.split, seed=int(args.seed),
        train_frac=float(args.train_frac), val_frac=float(args.val_frac),
        limit_n=(None if args.limit_n in (None, -1) else int(args.limit_n)),
        target_key=args.target, target_index=args.target_index,
        use_morgan=(not args.no_fp),
        morgan_radius=int(args.fp_radius), morgan_bits=int(args.fp_bits),
        morgan_use_chirality=(not args.fp_no_chiral),
        drop_constant=(not args.no_drop_constant),
        scale=args.scale,
        alphas=[float(a) for a in args.alphas],
        fit_intercept_opts=[bool(int(v)) for v in args.fit_intercept],
        out_dir=args.out_dir, tag=args.tag, save_preds=(not args.no_save_preds),
        verbose=(not args.quiet),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = _cfg_from_args(_build_arg_parser().parse_args(argv))
    train_and_eval(cfg)
    return 0


# -----------------------------------------------------------------------------
# Backwards-compat wrappers for src/cli/baseline.py
# -----------------------------------------------------------------------------

def train_ridge(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
    fit_intercept: bool = True,
    normalize: bool = False,  # ignored; kept for API compat
    seed: Optional[int] = None,  # unused; kept for API compat
) -> Ridge:
    """Lightweight wrapper to fit a Ridge model (kept for CLI baseline imports)."""
    X = np.asarray(X_train)
    y = np.asarray(y_train)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1:
        raise ValueError(f"y must be 1D or (N,1), got {y.shape}")
    model = Ridge(alpha=float(alpha), fit_intercept=bool(fit_intercept))
    model.fit(X, y)
    return model


def predict_ridge(model: Ridge, X: np.ndarray) -> np.ndarray:
    """Wrapper for predictions (kept for CLI baseline imports)."""
    Xp = np.asarray(X)
    if Xp.ndim != 2:
        raise ValueError(f"X must be 2D, got {Xp.shape}")
    pred = model.predict(Xp)
    return np.asarray(pred, dtype=np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
