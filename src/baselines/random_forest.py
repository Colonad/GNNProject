# src/baselines/random_forest.py
"""
Random Forest Baseline on RDKit Descriptors (with optional Morgan FP)
====================================================================

- Loads SMILES + targets from ESOL or QM9 (subset).
- Computes descriptor matrix via `src.baselines.descriptors`.
- Uses SAME splits as GNN runs: random or scaffold (Bemis–Murcko).
- Small hyperparameter sweep over n_estimators / max_depth / min_samples_leaf.
- Picks best on **validation MAE**, retrains on train+val, evaluates on test.
- Writes CSV row under `outputs/baselines/` (+ optional predictions CSV).

Usage:
  python -m src.baselines.random_forest --dataset ESOL --split scaffold
  python -m src.baselines.random_forest --dataset QM9 --limit-n 15000 --split random
"""

from __future__ import annotations
import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Silence RDKit info/debug chatter (no wildcard channel support)
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.info")
RDLogger.DisableLog("rdApp.debug")

# Local utilities
from .descriptors import DescriptorConfig, build_descriptor_matrix, load_smiles_from_dataset, ensure_dir
from src.datamodules.splits import SplitSpec, random_split_indices, scaffold_split_indices

# QM9 target index mapping (keep in sync with qm9.py)
QM9_TARGET_INDEX = {
    "mu": 0, "alpha": 1, "homo": 2, "lumo": 3, "gap": 4, "r2": 5, "zpve": 6,
    "U0": 7, "U": 8, "H": 9, "G": 10, "Cv": 11
}


@dataclass
class RFConfig:
    # Data
    dataset: str = "ESOL"
    root: str = "data"
    split: str = "scaffold"            # {"scaffold","random"}
    seed: int = 0
    train_frac: float = 0.8
    val_frac: float = 0.1
    limit_n: Optional[int] = None      # QM9 subset

    # QM9 target selection
    target_key: Optional[str] = "U0"
    target_index: Optional[int] = None

    # Descriptors
    use_morgan: bool = True
    morgan_radius: int = 2
    morgan_bits: int = 1024
    morgan_use_chirality: bool = True
    drop_constant: bool = True
    scale: str = "none"                # RF is tree-based → scaling not needed

    # Hyperparameters (grid)
    n_estimators: Sequence[int] = (200, 500, 800)
    max_depth: Sequence[Optional[int]] = (None, 12, 20)
    min_samples_leaf: Sequence[int] = (1, 2, 4)

    # IO
    out_dir: str = "outputs/baselines"
    save_preds: bool = True
    tag: Optional[str] = None

    # Verbosity
    verbose: bool = True


def _resolve_qm9_target_index(key: Optional[str], index: Optional[int]) -> int:
    if index is not None:
        return int(index)
    if key is not None and key in QM9_TARGET_INDEX:
        return QM9_TARGET_INDEX[key]
    return QM9_TARGET_INDEX["U0"]


def _load_targets(dataset: str, root: str, limit_n: Optional[int], target_idx: int) -> np.ndarray:
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


def _make_splits(smiles: List[str], cfg: RFConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def train_and_eval(cfg: RFConfig) -> Dict[str, object]:
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

    # 4) Hyperparameter sweep on validation MAE
    best: Optional[Dict[str, Any]] = None
    for n in cfg.n_estimators:
        for md in cfg.max_depth:
            for msl in cfg.min_samples_leaf:
                model = RandomForestRegressor(
                    n_estimators=int(n),
                    max_depth=(None if md is None else int(md)),
                    min_samples_leaf=int(msl),
                    random_state=cfg.seed,
                    n_jobs=-1,
                )
                model.fit(Xtr, ytr)
                pva = model.predict(Xva)
                m = _metrics(yva, pva)
                cand = {
                    "n_estimators": int(n),
                    "max_depth": (None if md is None else int(md)),
                    "min_samples_leaf": int(msl),
                    "val_MAE": m["MAE"],
                    "val_RMSE": m["RMSE"],
                }
                if best is None or cand["val_MAE"] < best["val_MAE"]:
                    best = cand
    assert best is not None

    # 5) Retrain on train+val
    trval_idx = np.concatenate([tr_idx, va_idx])
    Xtrval, ytrval = X_all[trval_idx], y_all[trval_idx]
    model = RandomForestRegressor(
        n_estimators=best["n_estimators"],
        max_depth=best["max_depth"],
        min_samples_leaf=best["min_samples_leaf"],
        random_state=cfg.seed,
        n_jobs=-1,
    )
    model.fit(Xtrval, ytrval)
    pte = model.predict(Xte)
    test_metrics = _metrics(yte, pte)

    # 6) Save metrics CSV
    ensure_dir(cfg.out_dir)
    ts = time.strftime("%Y%m%d-%H%M%S")
    tag = f"_{cfg.tag}" if cfg.tag else ""
    fname = f"{cfg.dataset.upper()}_rf_{cfg.split}_seed{cfg.seed}{tag}_{ts}.csv"
    out_csv = os.path.join(cfg.out_dir, fname)

    header = [
        "dataset","split","seed","N","D","train_size","val_size","test_size",
        "n_estimators","max_depth","min_samples_leaf",
        "val_MAE","val_RMSE","test_MAE","test_RMSE",
        "use_morgan","morgan_bits","morgan_radius","morgan_use_chirality","drop_constant","scale"
    ]
    row = [
        cfg.dataset.upper(), cfg.split, cfg.seed, X_all.shape[0], X_all.shape[1],
        len(tr_idx), len(va_idx), len(te_idx),
        best["n_estimators"], ("" if best["max_depth"] is None else best["max_depth"]), best["min_samples_leaf"],
        best["val_MAE"], best["val_RMSE"], test_metrics["MAE"], test_metrics["RMSE"],
        int(cfg.use_morgan), cfg.morgan_bits, cfg.morgan_radius, int(cfg.morgan_use_chirality), int(cfg.drop_constant), cfg.scale
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        writer = _csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)

    # 7) Optional predictions
    preds_csv = None
    if cfg.save_preds:
        preds_csv = out_csv.replace(".csv", "_preds.csv")
        with open(preds_csv, "w", newline="", encoding="utf-8") as f:
            import csv as _csv
            writer = _csv.writer(f)
            writer.writerow(["index","y_true","y_pred"])
            for idx, yt, yp in zip(te_idx.tolist(), yte.tolist(), pte.tolist()):
                writer.writerow([int(idx), float(yt), float(yp)])

    if cfg.verbose:
        print(f"[rf] Saved metrics: {out_csv}")
        if preds_csv:
            print(f"[rf] Saved predictions: {preds_csv}")
        print(
            f"[rf] Best n_estimators={best['n_estimators']}, max_depth={best['max_depth']}, "
            f"min_samples_leaf={best['min_samples_leaf']}; "
            f"Val MAE={best['val_MAE']:.4f}, Test MAE={test_metrics['MAE']:.4f}"
        )

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
    p = argparse.ArgumentParser(description="RandomForest baseline on RDKit descriptors (ESOL/QM9).")
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
    p.add_argument("--scale", type=str, default="none", choices=["none","standard","minmax"])
    p.add_argument("--n-estimators", type=int, nargs="+", default=[200,500,800])
    p.add_argument("--max-depth", type=int, nargs="+", default=[-1,12,20], help="-1 means None")
    p.add_argument("--min-samples-leaf", type=int, nargs="+", default=[1,2,4])
    p.add_argument("--out-dir", type=str, default="outputs/baselines")  # <-- fixed here
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--no-save-preds", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def _cfg_from_args(args: argparse.Namespace) -> RFConfig:
    max_depths = [None if d == -1 else int(d) for d in args.max_depth]
    return RFConfig(
        dataset=args.dataset, root=args.root, split=args.split, seed=int(args.seed),
        train_frac=float(args.train_frac), val_frac=float(args.val_frac),
        limit_n=(None if args.limit_n in (None, -1) else int(args.limit_n)),
        target_key=args.target, target_index=args.target_index,
        use_morgan=(not args.no_fp),
        morgan_radius=int(args.fp_radius), morgan_bits=int(args.fp_bits),
        morgan_use_chirality=(not args.fp_no_chiral),
        drop_constant=(not args.no_drop_constant),
        scale=args.scale,
        n_estimators=[int(n) for n in args.n_estimators],
        max_depth=max_depths,
        min_samples_leaf=[int(m) for m in args.min_samples_leaf],
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

__all__ = ["train_random_forest", "predict_random_forest"]

def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {X.shape}")
    return X

def _as_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1:
        raise ValueError(f"y must be 1D (or 2D with a single column), got shape {y.shape}")
    return y

def train_random_forest(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Union[str, float, int, None] = "auto",
    n_jobs: int = -1,
    seed: int = 0,
) -> RandomForestRegressor:
    """
    Thin wrapper used by the CLI to fit a RandomForestRegressor.
    Deterministic for a fixed seed and input.
    """
    X = _as_2d(np.asarray(X_train, dtype=np.float64))
    y = _as_1d(y_train).astype(np.float64, copy=False)
    if isinstance(max_depth, str) and max_depth.lower() == "none":
        max_depth = None
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=None if max_depth is None else int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        n_jobs=int(n_jobs),
        random_state=int(seed),
        bootstrap=True,
    )
    model.fit(X, y)
    return model

def predict_random_forest(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Thin wrapper used by the CLI for predictions.
    Returns float32 predictions shaped (N,).
    """
    if not hasattr(model, "predict"):
        raise TypeError("`model` must expose a scikit-learn-like `predict` method.")
    Xp = _as_2d(np.asarray(X, dtype=np.float64))
    pred = model.predict(Xp)
    return np.asarray(pred, dtype=np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
