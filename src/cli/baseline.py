# src/cli/baseline.py
from __future__ import annotations
from copy import deepcopy
"""
Hydra-powered CLI for classical baselines (Ridge / RandomForest) on ESOL / QM9.

Key features
------------
- Uses the same datamodule logic and splits as GNN training (so results are comparable).
- RDKit-based featurization via src.baselines.descriptors (Morgan, physchem, combo).
- Optional target standardization (train mean/std) with de-standardized metrics.
- Feature scaling: standard / minmax / none.
- Optional hyper-parameter sweep; selects best on validation MAE.
- Deterministic with seed for scikit-learn where applicable.
- Artifacts:
    out_dir/
      ├─ model.pkl
      ├─ metrics.csv
      ├─ summary.json
      ├─ preds_val.csv   (if dump_preds=true)
      ├─ preds_test.csv  (if dump_preds=true)
      └─ features_cache/ (feature .npz cache keyed by split & params)

Examples
--------
# Simple ESOL → Ridge on Morgan fingerprints
python -m src.cli.baseline data.name=ESOL model.name=ridge feat.kind=morgan runtime.out_dir=runs/baseline_esol_ridge

# RandomForest with physchem+Morgan, target standardization and sweep
python -m src.cli.baseline \
  data.name=ESOL model.name=random_forest \
  feat.kind=morgan_physchem feat.morgan_bits=4096 feat.morgan_radius=3 \
  train.standardize_targets=true train.sweep=true \
  model.rf_n_estimators='[200,400]' model.rf_max_depth='[None,20,40]' \
  runtime.out_dir=runs/baseline_esol_rf
"""

import csv
import hashlib
import json
import os
import sys
import time
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch

# Reuse canonical split creation via the training loop + datamodules
from src.train.loop import (
    build_argparser,
    build_dataloaders,
)

# Baselines & featurizers
from src.baselines.descriptors import (
    build_featurizer,               # (cfg) -> callable(smiles_list) -> np.ndarray
    DescriptorConfig,               # kept for compatibility; not directly used here
)
from src.baselines.ridge import train_ridge, predict_ridge  # wrappers
from src.baselines.random_forest import train_random_forest, predict_random_forest

try:
    import joblib  # for model persistence
except Exception as _e:
    joblib = None


# --------------------------------------------------------------------------
# Hydra config schema
# --------------------------------------------------------------------------

@dataclass
class DataCfg:
    name: str = "ESOL"              # ESOL | QM9
    root: str = "data"
    limit_n: Optional[int] = None   # QM9 subset
    target: str = "U0"              # QM9 target key (ignored for ESOL)
    target_index: Optional[int] = None
    split: str = "scaffold"         # scaffold | random
    train_frac: float = 0.8
    val_frac: float = 0.1


@dataclass
class FeatCfg:
    # Featurization kind (as implemented in src/baselines/descriptors.py)
    kind: str = "morgan"            # morgan | physchem | morgan_physchem
    # Morgan fingerprint params
    morgan_bits: int = 2048
    morgan_radius: int = 2
    morgan_use_chirality: bool = True
    morgan_use_features: bool = False
    # Physchem params (feature set key handled inside descriptors.py)
    physchem_set: str = "basic"     # "basic" or any set you defined
    # General
    n_jobs: int = 0                 # parallelism for featurization (0/1 = inline)
    cache: bool = True              # cache features to disk
    cache_dir: Optional[str] = None # if None -> <out_dir>/features_cache


@dataclass
class ModelCfg:
    # Which baseline to use
    name: str = "ridge"             # ridge | random_forest

    # Ridge params
    ridge_alpha: float = 1.0
    ridge_fit_intercept: bool = True
    ridge_normalize: bool = False   # historical; ridge.py may ignore (we still pass through)

    # RandomForest params
    rf_n_estimators: int = 300
    rf_max_depth: Optional[Any] = None
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    rf_max_features: str = "auto"   # or "sqrt", "log2", or float (fraction)
    rf_n_jobs: int = -1

    # Feature scaling for linear models; also allowed with RF (then it's applied but less impactful)
    scaler: str = "standard"        # standard | minmax | none

    # Optional sweep (simple grids). Use JSON-ish strings or lists in CLI.
    ridge_alphas: Optional[List[float]] = None
    rf_n_estimators_grid: Optional[List[int]] = None
    rf_max_depth_grid: Optional[List[Any]] = None


@dataclass
class TrainCfg:
    standardize_targets: bool = False  # z-score y on train; report metrics on original scale
    sweep: bool = False                # enable hyper-parameter sweep
    seed: int = 0


@dataclass
class RuntimeCfg:
    cpu: bool = False
    num_workers: int = 0               # unused here (we don't use torch dataloader); kept for symmetry
    out_dir: Optional[str] = None
    quiet: bool = False
    dump_preds: bool = True            # write preds_val.csv & preds_test.csv
    save_model: bool = True


@dataclass
class RootCfg:
    data: DataCfg = field(default_factory=DataCfg)
    feat: FeatCfg = field(default_factory=FeatCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)


cs = ConfigStore.instance()
cs.store(name="baseline_config", node=RootCfg)


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _resolve_out_dir(original_cwd: str, cfg: RootCfg) -> str:
    t = time.strftime("%Y%m%d-%H%M%S")
    if cfg.runtime.out_dir:
        return cfg.runtime.out_dir if os.path.isabs(cfg.runtime.out_dir) else os.path.join(original_cwd, cfg.runtime.out_dir)
    tag = f"{cfg.data.name.lower()}_{cfg.model.name}_baseline_{cfg.data.split}_seed{cfg.train.seed}_{t}"
    return os.path.join(original_cwd, "runs", tag)


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader_args_from_cfg(cfg: RootCfg, original_cwd: str) -> List[str]:
    data_root = cfg.data.root if os.path.isabs(cfg.data.root) else os.path.join(original_cwd, cfg.data.root)
    args = [
        "--dataset", cfg.data.name.upper(),
        "--root", data_root,
        "--split", cfg.data.split,
        "--target", cfg.data.target,
        "--seed", str(cfg.train.seed),
        "--train-frac", str(cfg.data.train_frac),
        "--val-frac", str(cfg.data.val_frac),
        "--epochs", "1",
        "--batch-size", "256",
        "--num-workers", "0",
        "--lr", "0.001", "--weight-decay", "0.0001",
        "--scheduler", "none",
        "--model", "gin",
        "--hidden-dim", "64",
        "--num-layers", "2",
        "--pool", "mean", "--act", "relu",
        "--dropout", "0.0",
        "--readout-layers", "1",
        "--readout-hidden-mult", "1.0",
        "--init", "kaiming",
        "--aggr", "add",
        "--out-dir", os.path.join(_resolve_out_dir(original_cwd, cfg), "_tmp"),
        "--no-shuffle-train",
        "--quiet",
    ]
    # Only include when not None
    if cfg.data.limit_n is not None:
        args += ["--limit-n", str(int(cfg.data.limit_n))]
    if cfg.data.target_index is not None:
        args += ["--target-index", str(int(cfg.data.target_index))]
    return args



def _extract_smiles_and_targets(dset) -> Tuple[List[str], np.ndarray]:
    """Extract SMILES (as strings) and targets (float32 array) from a PyG InMemoryDataset or a simple list of Data.
    We look for a `smiles` attribute on Data.
    """
    smiles: List[str] = []
    ys: List[float] = []
    for item in dset:
        s = getattr(item, "smiles", None)
        if s is None:
            raise RuntimeError("Data item has no `smiles` attribute; baseline featurizer requires SMILES. "
                               "Please ensure datamodules attach `smiles` to Data objects.")
        y = item.y
        if y is None:
            raise RuntimeError("Data item has no `y` target.")
        yi = float(y.view(-1)[0].item())
        smiles.append(str(s))
        ys.append(yi)
    return smiles, np.asarray(ys, dtype=np.float32)


def _hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()[:16]


def _feature_cache_paths(out_dir: str, feat_cfg: FeatCfg, split_name: str, smiles: List[str]) -> Tuple[str, str]:
    """Return (npz_path, meta_json_path) for caching feature matrices."""
    cache_dir = feat_cfg.cache_dir or os.path.join(out_dir, "features_cache")
    _ensure_dir(cache_dir)
    key = _hash_key(
        split_name,
        feat_cfg.kind,
        f"bits={feat_cfg.morgan_bits},radius={feat_cfg.morgan_radius},chir={feat_cfg.morgan_use_chirality},feat={feat_cfg.morgan_use_features}",
        f"phys={feat_cfg.physchem_set}",
        f"n={len(smiles)}",
        f"h_smiles={_hash_key(str(len(smiles)), str(sum(len(s) for s in smiles[:100])))}",
    )
    return os.path.join(cache_dir, f"{split_name}_{key}.npz"), os.path.join(cache_dir, f"{split_name}_{key}.json")


def _build_features(smiles: List[str], feat_cfg: FeatCfg, out_dir: str, split_name: str, use_cache: bool = True) -> np.ndarray:
    if use_cache and feat_cfg.cache:
        npz_path, meta_path = _feature_cache_paths(out_dir, feat_cfg, split_name, smiles)
        if os.path.exists(npz_path) and os.path.exists(meta_path):
            z = np.load(npz_path)
            X = z["X"]
            return X

    featurizer = build_featurizer(
        kind=feat_cfg.kind,
        morgan_bits=feat_cfg.morgan_bits,
        morgan_radius=feat_cfg.morgan_radius,
        use_chirality=feat_cfg.morgan_use_chirality,
        use_features=feat_cfg.morgan_use_features,
        physchem_set=feat_cfg.physchem_set,
        n_jobs=feat_cfg.n_jobs,
    )
    X = featurizer(smiles)  # -> np.ndarray [n_samples, n_features]

    if use_cache and feat_cfg.cache:
        npz_path, meta_path = _feature_cache_paths(out_dir, feat_cfg, split_name, smiles)
        np.savez_compressed(npz_path, X=X)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split": split_name,
                    "feat_cfg": {
                        "kind": feat_cfg.kind,
                        "morgan_bits": feat_cfg.morgan_bits,
                        "morgan_radius": feat_cfg.morgan_radius,
                        "use_chirality": feat_cfg.morgan_use_chirality,
                        "use_features": feat_cfg.morgan_use_features,
                        "physchem_set": feat_cfg.physchem_set,
                        "n_jobs": feat_cfg.n_jobs,
                    },
                    "shape": list(map(int, X.shape)),
                },
                f,
                indent=2,
            )
    return X


def _scale_features(train_X: np.ndarray, X: np.ndarray, scaler: str) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    if scaler == "none":
        return X, None
    if scaler == "standard":
        mu = train_X.mean(axis=0, dtype=np.float64)
        sd = train_X.std(axis=0, dtype=np.float64, ddof=0)
        sd[sd == 0.0] = 1.0
        Xs = (X - mu) / sd
        return Xs, {"type": "standard", "mean": mu, "std": sd}
    if scaler == "minmax":
        mn = train_X.min(axis=0)
        mx = train_X.max(axis=0)
        rng = mx - mn
        rng[rng == 0.0] = 1.0
        Xs = (X - mn) / rng
        return Xs, {"type": "minmax", "min": mn, "max": mx}
    raise ValueError(f"Unknown scaler: {scaler}")


def _apply_scaler(X: np.ndarray, scaler_meta: Optional[Dict[str, Any]]) -> np.ndarray:
    if scaler_meta is None:
        return X
    t = scaler_meta["type"]
    if t == "standard":
        return (X - scaler_meta["mean"]) / scaler_meta["std"]
    if t == "minmax":
        return (X - scaler_meta["min"]) / (scaler_meta["max"] - scaler_meta["min"])
    return X


def _standardize_targets(y: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    mu = float(np.mean(y))
    sd = float(np.std(y))
    if sd <= 0:
        sd = 1.0
    return (y - mu) / sd, {"mean": mu, "std": sd}


def _destandardize_preds(p: np.ndarray, stats: Optional[Dict[str, float]]) -> np.ndarray:
    if stats is None:
        return p
    return p * stats["std"] + stats["mean"]


def _mae_rmse_mse_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mse = float(np.mean(err ** 2))
    tss = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    rss = float(np.sum(err ** 2))
    r2 = 1.0 - (rss / tss if tss > 0 else 0.0)
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "R2": r2}


def _fit_and_eval_once(
    cfg: RootCfg,
    out_dir: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    scaler_meta: Optional[Dict[str, Any]],
    y_stats: Optional[Dict[str, float]],
) -> Tuple[Dict[str, Any], Any]:
    """
    Fit the chosen baseline, evaluate on val/test (de-standardized), and return (metrics, model).
    """
    model_obj: Any
    if cfg.model.name == "ridge":
        model_obj = train_ridge(
            X_train=_apply_scaler(train_X, scaler_meta),
            y_train=train_y,                                   # already standardized if enabled
            alpha=float(cfg.model.ridge_alpha),
            fit_intercept=bool(cfg.model.ridge_fit_intercept),
            normalize=bool(cfg.model.ridge_normalize),
            seed=int(cfg.train.seed),
        )
        val_pred_std = predict_ridge(model_obj, _apply_scaler(val_X, scaler_meta))
        test_pred_std = predict_ridge(model_obj, _apply_scaler(test_X, scaler_meta))
    elif cfg.model.name == "random_forest":
        model_obj = train_random_forest(
            X_train=_apply_scaler(train_X, scaler_meta),
            y_train=train_y,
            n_estimators=int(cfg.model.rf_n_estimators),
            max_depth=None if cfg.model.rf_max_depth in (None, "None") else int(cfg.model.rf_max_depth),
            min_samples_split=int(cfg.model.rf_min_samples_split),
            min_samples_leaf=int(cfg.model.rf_min_samples_leaf),
            max_features=_coerce_max_features(cfg.model.rf_max_features) if isinstance(cfg.model.rf_max_features, str)
                        else float(cfg.model.rf_max_features),
            n_jobs=int(cfg.model.rf_n_jobs),
            seed=int(cfg.train.seed),
        )

        val_pred_std = predict_random_forest(model_obj, _apply_scaler(val_X, scaler_meta))
        test_pred_std = predict_random_forest(model_obj, _apply_scaler(test_X, scaler_meta))
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    # de-standardize if needed
    val_pred = _destandardize_preds(val_pred_std, y_stats)
    test_pred = _destandardize_preds(test_pred_std, y_stats)
    val_metrics = _mae_rmse_mse_r2(val_y, val_pred)
    test_metrics = _mae_rmse_mse_r2(test_y, test_pred)

    metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "scaler": cfg.model.scaler,
        "model": cfg.model.name,
        "params": {
            "ridge_alpha": cfg.model.ridge_alpha,
            "rf_n_estimators": cfg.model.rf_n_estimators,
            "rf_max_depth": cfg.model.rf_max_depth,
        },
    }
    return metrics, model_obj


def _maybe_dump_preds(path: str, smiles: List[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "y_true", "y_pred"])
        for s, yt, yp in zip(smiles, y_true.tolist(), y_pred.tolist()):
            w.writerow([s, f"{float(yt):.6f}", f"{float(yp):.6f}"])


def _grid(values: Optional[List[Any]], default_single: Any) -> List[Any]:
    if values is None:
        return [default_single]
    return list(values)


def _to_opt_int_or_none(v):
    """Coerce common stringy Nones to real None; else int()."""
    if v in (None, "None", "none", "null", "~", ""):
        return None
    return int(v)

def _coerce_max_features(v):
    """Map deprecated sklearn value 'auto' to a valid one for regressors."""
    if isinstance(v, str) and v.lower() == "auto":
        return "sqrt"    # sklearn's former behavior for regressors
    return v

# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

@hydra.main(version_base=None, config_name="baseline_config")
def main(cfg: DictConfig) -> int:
    conf = OmegaConf.to_object(cfg)
    if isinstance(conf, dict):
        conf = RootCfg(**conf)  # type: ignore[call-arg]

    original_cwd = get_original_cwd()
    out_dir = _resolve_out_dir(original_cwd, conf)
    _ensure_dir(out_dir)

    # Seed
    _set_seed(conf.train.seed)

    # Build loaders via loop parser to reuse exact splitting
    parser = build_argparser()
    argv = _loader_args_from_cfg(conf, original_cwd)
    args = parser.parse_args(argv)

    # Create dataloaders; we only need access to the underlying datasets
    train_loader, val_loader, test_loader, meta = build_dataloaders(args)
    train_ds, val_ds, test_ds = train_loader.dataset, val_loader.dataset, test_loader.dataset

    # Extract smiles and y from each split
    train_smiles, train_y = _extract_smiles_and_targets(train_ds)
    val_smiles, val_y = _extract_smiles_and_targets(val_ds)
    test_smiles, test_y = _extract_smiles_and_targets(test_ds)

    # Build features (with on-disk caching)
    X_train = _build_features(train_smiles, conf.feat, out_dir, "train", use_cache=True)
    X_val   = _build_features(val_smiles, conf.feat, out_dir, "val", use_cache=True)
    X_test  = _build_features(test_smiles, conf.feat, out_dir, "test", use_cache=True)

    # Optional target standardization
    y_train_std, y_stats = (train_y, None)
    if conf.train.standardize_targets:
        y_train_std, y_stats = _standardize_targets(train_y)

    # Feature scaling fit on *train* only, then apply to all splits
    X_train_scaled, scaler_meta = _scale_features(X_train, X_train, conf.model.scaler)
    X_val_scaled = _apply_scaler(X_val, scaler_meta)
    X_test_scaled = _apply_scaler(X_test, scaler_meta)

    # If sweep -> iterate grid; else single run
    best_summary: Dict[str, Any] = {}
    best_model: Any = None
    best_val_mae: float = float("inf")

    ridge_alphas = _grid(conf.model.ridge_alphas, conf.model.ridge_alpha) if conf.model.name == "ridge" else [None]
    rf_n_estimators_grid = _grid(conf.model.rf_n_estimators_grid, conf.model.rf_n_estimators) if conf.model.name == "random_forest" else [None]
    rf_max_depth_grid = _grid(conf.model.rf_max_depth_grid, conf.model.rf_max_depth) if conf.model.name == "random_forest" else [None]


    # Normalize RF grids to safe types so Hydra string "None" etc. don't break later
    if conf.model.name == "random_forest":
        rf_n_estimators_grid = [int(v) for v in rf_n_estimators_grid if v is not None]
        rf_max_depth_grid = [_to_opt_int_or_none(v) for v in rf_max_depth_grid]



    trial_records: List[Dict[str, Any]] = []


    if not conf.train.sweep and conf.model.name == "random_forest":
        conf.model.rf_max_depth = _to_opt_int_or_none(conf.model.rf_max_depth)


    if conf.train.sweep:
        # Simple grid over model hyperparameters
        

        if conf.model.name == "ridge":
            for alpha in ridge_alphas:
                c2 = deepcopy(conf)
                c2.model.ridge_alpha = float(alpha)
                metrics, model_obj = _fit_and_eval_once(
                    c2, out_dir, X_train, y_train_std, X_val, val_y, X_test, test_y, scaler_meta, y_stats
                )
                rec = {
                    "model": "ridge",
                    "alpha": float(alpha),
                    "val_MAE": metrics["val"]["MAE"],
                    "test_MAE": metrics["test"]["MAE"],
                    "metrics": metrics,
                }
                trial_records.append(rec)
                if rec["val_MAE"] < best_val_mae:
                    best_val_mae = rec["val_MAE"]
                    best_summary = metrics
                    best_model = model_obj

        elif conf.model.name == "random_forest":
            for ne in rf_n_estimators_grid:
                for md in rf_max_depth_grid:
                    c2 = deepcopy(conf)
                    c2.model.rf_n_estimators = int(ne)
                    c2.model.rf_max_depth = _to_opt_int_or_none(md)
                    metrics, model_obj = _fit_and_eval_once(
                        c2, out_dir, X_train, y_train_std, X_val, val_y, X_test, test_y, scaler_meta, y_stats
                    )
                    rec = {
                        "model": "random_forest",
                        "n_estimators": int(ne),
                        "max_depth": c2.model.rf_max_depth,
                        "val_MAE": metrics["val"]["MAE"],
                        "test_MAE": metrics["test"]["MAE"],
                        "metrics": metrics,
                    }
                    trial_records.append(rec)
                    if rec["val_MAE"] < best_val_mae:
                        best_val_mae = rec["val_MAE"]
                        best_summary = metrics
                        best_model = model_obj

        else:
            raise ValueError(f"Unknown model {conf.model.name}")
    else:
        # Single configuration
        best_summary, best_model = _fit_and_eval_once(
            conf, out_dir, X_train, y_train_std, X_val, val_y, X_test, test_y, scaler_meta, y_stats
        )
        best_val_mae = float(best_summary["val"]["MAE"])

    # Persist model if requested
    model_path = os.path.join(out_dir, "model.pkl")
    if conf.runtime.save_model:
        if joblib is None:
            print("[baseline] WARNING: joblib not available; skipping model save.")
        else:
            joblib.dump(
                {
                    "model": best_model,
                    "model_name": conf.model.name,
                    "feature_cfg": OmegaConf.to_container(cfg.feat, resolve=True),
                    "scaler_meta": scaler_meta,
                    "target_norm": y_stats,
                    "data": {
                        "dataset": conf.data.name,
                        "split": conf.data.split,
                        "target": conf.data.target,
                        "target_index": conf.data.target_index,
                    },
                    "seed": conf.train.seed,
                },
                model_path,
            )

    # Optionally dump predictions (using best model & scaler, de-standardized)
    if conf.runtime.dump_preds:
        if conf.model.name == "ridge":
            from src.baselines.ridge import predict_ridge as _pred
        else:
            from src.baselines.random_forest import predict_random_forest as _pred

        val_pred_std = _pred(best_model, _apply_scaler(X_val, scaler_meta))
        test_pred_std = _pred(best_model, _apply_scaler(X_test, scaler_meta))
        val_pred = _destandardize_preds(val_pred_std, y_stats)
        test_pred = _destandardize_preds(test_pred_std, y_stats)

        _maybe_dump_preds(os.path.join(out_dir, "preds_val.csv"), val_smiles, val_y, val_pred)
        _maybe_dump_preds(os.path.join(out_dir, "preds_test.csv"), test_smiles, test_y, test_pred)

    # Metrics CSV (append or create)
    csv_path = os.path.join(out_dir, "metrics.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "dataset", "split", "model", "scaler",
                "val_MAE", "val_RMSE", "val_MSE", "val_R2",
                "test_MAE", "test_RMSE", "test_MSE", "test_R2",
                "feat_kind", "feat_bits", "feat_radius",
                "ridge_alpha", "rf_n_estimators", "rf_max_depth",
                "seed", "out_dir",
            ])
        w.writerow([
            conf.data.name.upper(), conf.data.split, conf.model.name, conf.model.scaler,
            best_summary["val"]["MAE"], best_summary["val"]["RMSE"], best_summary["val"]["MSE"], best_summary["val"]["R2"],
            best_summary["test"]["MAE"], best_summary["test"]["RMSE"], best_summary["test"]["MSE"], best_summary["test"]["R2"],
            conf.feat.kind, conf.feat.morgan_bits, conf.feat.morgan_radius,
            conf.model.ridge_alpha if conf.model.name == "ridge" else "",
            conf.model.rf_n_estimators if conf.model.name == "random_forest" else "",
            conf.model.rf_max_depth if conf.model.name == "random_forest" else "",
            conf.train.seed, out_dir,
        ])

    # Summary JSON
    summary = {
        "dataset": conf.data.name.upper(),
        "split": conf.data.split,
        "model": conf.model.name,
        "feature": {
            "kind": conf.feat.kind,
            "morgan_bits": conf.feat.morgan_bits,
            "morgan_radius": conf.feat.morgan_radius,
            "physchem_set": conf.feat.physchem_set,
        },
        "scaler": conf.model.scaler,
        "target_standardize": conf.train.standardize_targets,
        "val": best_summary["val"],
        "test": best_summary["test"],
        "paths": {
            "out_dir": out_dir,
            "model": model_path if conf.runtime.save_model else None,
            "metrics_csv": csv_path,
            "preds_val": os.path.join(out_dir, "preds_val.csv") if conf.runtime.dump_preds else None,
            "preds_test": os.path.join(out_dir, "preds_test.csv") if conf.runtime.dump_preds else None,
        },
        "sweep": {
            "enabled": conf.train.sweep,
            "trials": None,
        },
        "seed": conf.train.seed,
    }
    # attach sweep trials if applicable
    if conf.train.sweep:
        summary["sweep"]["trials"] = trial_records

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[baseline] Done. Split={conf.data.split} | "
          f"val MAE={best_summary['val']['MAE']:.4f} | "
          f"test MAE={best_summary['test']['MAE']:.4f} | out={out_dir}")
    if conf.train.sweep:
        print(f"[baseline] trials={len(trial_records)} | best val MAE={min(t['val_MAE'] for t in trial_records):.4f}")
    if conf.runtime.save_model:
        print(f"[baseline] Saved model -> {model_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
