# src/cli/evaluate.py
from __future__ import annotations
"""
Hydra-powered evaluation CLI for molecular GNNs.

Features
--------
- Load a single checkpoint or an ensemble (glob/list) and evaluate on val/test.
- Use EMA weights (if present in checkpoint) or raw weights.
- Reuse the training run's saved target normalization (summary.json["target_norm"])
  when available; else safely recompute from the training split.
- Export per-sample predictions to CSV (optional).
- Reuses the canonical dataloaders/model factory from src.train.loop to avoid drift.

Examples
--------
# Evaluate the best checkpoint saved by train CLI:
python -m src.cli.evaluate \
  data.name=ESOL eval=scaffold model=gin runtime.out_dir=runs/esol_gin_scaffold_seed0_20250101-120000

# Evaluate with a specific checkpoint and EMA disabled:
python -m src.cli.evaluate \
  data.name=ESOL eval=scaffold model=gin eval_cfg.which_split=test \
  eval_cfg.ckpt_path=runs/esol_gin/best.ckpt eval_cfg.use_ema=false

# Simple ensemble by glob:
python -m src.cli.evaluate \
  data.name=ESOL eval=scaffold model=gin \
  eval_cfg.ensemble_glob='runs/sweep/hd*/best.ckpt' eval_cfg.which_split=val \
  eval_cfg.dump_preds=true eval_cfg.preds_filename=val_preds.csv
"""

import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader

# Canonical training loop bits
from src.train.loop import (
    build_argparser,
    build_dataloaders,
    build_model,
    evaluate as loop_evaluate,
    _compute_target_stats,   # use the same normalization helpers as training
    EMA,
)

# --------------------------------------------------------------------------
# Hydra config schema
# --------------------------------------------------------------------------

@dataclass
class DataCfg:
    name: str = "ESOL"              # ESOL | QM9
    root: str = "data"
    limit_n: Optional[int] = None   # QM9 subset
    target: str = "U0"              # QM9 target key
    target_index: Optional[int] = None  # QM9 target index (overrides key)


@dataclass
class ModelCfg:
    name: str = "gin"               # gin | mpnn
    # shared model hyperparams: used only to reconstruct the model correctly
    hidden_dim: int = 128
    num_layers: int = 5
    pool: str = "mean"              # sum | mean | max
    act: str = "relu"               # relu | gelu | leaky_relu | elu
    dropout: float = 0.10
    init: str = "kaiming"           # kaiming | xavier | none
    no_batch_norm: bool = False
    no_residual: bool = False
    use_edge_attr: bool = False
    no_learn_eps: bool = False
    virtual_node: bool = False
    readout_layers: int = 2
    readout_hidden_mult: float = 1.0
    aggr: str = "add"               # MPNN only: add | mean | max
    no_gru: bool = False            # MPNN only


@dataclass
class EvalCfg:
    # Which split of the datamodule to evaluate on
    which_split: str = "test"       # "test" | "val"
    # Path to a single checkpoint; if None, we try <out_dir>/best.ckpt
    ckpt_path: Optional[str] = None
    # Simple ensemble support: glob or list of paths. If provided, supersedes ckpt_path.
    ensemble_glob: Optional[str] = None
    ensemble_paths: List[str] = field(default_factory=list)
    # Prefer EMA shadow if available in the checkpoint
    use_ema: bool = True
    # Target normalization handling
    standardize_targets: Optional[bool] = None  # None => auto (prefer training's summary.json)
    prefer_training_norm: bool = True           # read summary.json['target_norm'] when available
    # Export predictions to CSV
    dump_preds: bool = False
    preds_filename: str = "preds.csv"
    # AMP for forward pass (CUDA only)
    amp: bool = False


@dataclass
class RuntimeCfg:
    seed: int = 0
    cpu: bool = False
    out_dir: Optional[str] = None    # training run dir (to derive default ckpt)
    quiet: bool = False
    # split/split-ratios to generate loaders consistently with training
    eval: str = "scaffold"           # scaffold | random  (same as training's --split)
    train_frac: float = 0.8
    val_frac: float = 0.1
    num_workers: int = 4
    batch_size: int = 256
    pin_memory: bool = False
    persistent_workers: bool = True
    drop_last: bool = False
    shuffle_train: bool = True       # not used here but passed to datamodule for consistency


@dataclass
class RootCfg:
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    eval_cfg: EvalCfg = field(default_factory=EvalCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)


cs = ConfigStore.instance()
cs.store(name="evaluate_config", node=RootCfg)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _resolve_out_dir(original_cwd: str, out_dir: Optional[str]) -> Optional[str]:
    if out_dir is None:
        return None
    return out_dir if os.path.isabs(out_dir) else os.path.join(original_cwd, out_dir)


def _default_ckpt_from_out_dir(out_dir: Optional[str]) -> Optional[str]:
    if out_dir is None:
        return None
    candidate = os.path.join(out_dir, "best.ckpt")
    return candidate if os.path.exists(candidate) else None


def _compose_loop_argv_for_loaders(cfg: RootCfg, original_cwd: str) -> List[str]:
    """
    Build a minimal argv for src.train.loop's parser so we can reuse
    build_dataloaders() / build_model() without code drift.
    """
    data_root = cfg.data.root
    if not os.path.isabs(data_root):
        data_root = os.path.join(original_cwd, data_root)

    args: List[str] = [
        "--dataset", cfg.data.name.upper(),
        "--root", data_root,
        "--split", cfg.runtime.eval,
        "--epochs", "1",  # irrelevant here; required by parser
        "--batch-size", str(cfg.runtime.batch_size),
        "--num-workers", str(cfg.runtime.num_workers),
        "--lr", "0.001",
        "--weight-decay", "0.0001",
        "--scheduler", "none",
        "--seed", str(cfg.runtime.seed),
        "--out-dir", _resolve_out_dir(original_cwd, cfg.runtime.out_dir) or os.path.join(original_cwd, "runs", "eval_tmp"),
        "--train-frac", str(cfg.runtime.train_frac),
        "--val-frac", str(cfg.runtime.val_frac),
        "--model", cfg.model.name.lower(),
        "--hidden-dim", str(cfg.model.hidden_dim),
        "--num-layers", str(cfg.model.num_layers),
        "--pool", cfg.model.pool,
        "--act", cfg.model.act,
        "--dropout", str(cfg.model.dropout),
        "--readout-layers", str(cfg.model.readout_layers),
        "--readout-hidden-mult", str(cfg.model.readout_hidden_mult),
        "--init", cfg.model.init,
        "--aggr", cfg.model.aggr,
        "--target", cfg.data.target,
    ]

    # Flags mirroring training parser
    if cfg.data.limit_n is not None:
        args += ["--limit-n", str(cfg.data.limit_n)]
    if cfg.data.target_index is not None:
        args += ["--target-index", str(cfg.data.target_index)]
    if cfg.runtime.pin_memory:
        args.append("--pin-memory")
    if not cfg.runtime.persistent_workers:
        args.append("--no-persistent-workers")
    if cfg.runtime.drop_last:
        args.append("--drop-last")
    if not cfg.runtime.shuffle_train:
        args.append("--no-shuffle-train")
    if cfg.runtime.cpu:
        args.append("--cpu")
    if cfg.runtime.quiet:
        args.append("--quiet")

    # Model boolean flips
    if cfg.model.no_batch_norm:
        args.append("--no-batch-norm")
    if cfg.model.no_residual:
        args.append("--no-residual")
    if cfg.model.use_edge_attr:
        args.append("--use-edge-attr")
    if cfg.model.no_learn_eps:
        args.append("--no-learn-eps")
    if cfg.model.virtual_node:
        args.append("--virtual-node")
    if cfg.model.no_gru:
        args.append("--no-gru")

    # Target normalization setting (if explicitly requested)
    if cfg.eval_cfg.standardize_targets is True:
        args.append("--standardize-targets")

    return args


def _safe_torch_load(path: str) -> Dict[str, Any]:
    # Favor safe mode if available (PyTorch >= 2.4)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location="cpu")


def _collect_ckpt_paths(cfg: RootCfg, resolved_out_dir: Optional[str]) -> List[str]:
    # Priority: explicit list > glob > single ckpt path > default best.ckpt
    if cfg.eval_cfg.ensemble_paths:
        return [p for p in cfg.eval_cfg.ensemble_paths if os.path.exists(p)]
    if cfg.eval_cfg.ensemble_glob:
        paths = sorted(glob.glob(cfg.eval_cfg.ensemble_glob))
        return [p for p in paths if os.path.exists(p)]
    if cfg.eval_cfg.ckpt_path is not None:
        return [cfg.eval_cfg.ckpt_path] if os.path.exists(cfg.eval_cfg.ckpt_path) else []
    # infer from out_dir
    best = _default_ckpt_from_out_dir(resolved_out_dir)
    return [best] if best else []


def _load_model_with_ckpt(args, node_dim: int, edge_dim: Optional[int], ckpt_path: str, use_ema: bool, device) -> Tuple[torch.nn.Module, Optional[EMA]]:
    """
    Instantiate the model from args, load weights from ckpt, recreate EMA if present.
    """
    model = build_model(args, node_dim=node_dim, edge_dim=edge_dim).to(device)
    state = _safe_torch_load(ckpt_path)
    model.load_state_dict(state["model"])

    ema = None
    if use_ema and "ema_shadow" in state:
        ema = EMA(model, decay=state.get("args", {}).get("ema_decay", 0.999))
        for k, v in state["ema_shadow"].items():
            ema.shadow[k] = v.to(device)
    return model, ema


def _get_target_stats_for_eval(cfg: RootCfg, out_dir: Optional[str], train_loader: DataLoader, device) -> Optional[dict]:
    """
    Decide which target normalization to use during evaluation:
    1) If prefer_training_norm, try to read summary.json from the out_dir of the model.
    2) Else, fall back to recomputing from the current training loader if cfg.eval_cfg.standardize_targets is True.
    3) Otherwise, return None.
    """
    # explicit True/False overrides auto behavior
    if cfg.eval_cfg.standardize_targets is False:
        return None

    if cfg.eval_cfg.standardize_targets is True:
        # Always compute from current training loader if requested
        return _compute_target_stats(train_loader, device)

    # Auto mode: prefer training's norm
    if cfg.eval_cfg.prefer_training_norm and out_dir is not None:
        summary_path = os.path.join(out_dir, "summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                tn = summary.get("target_norm", None)
                if isinstance(tn, dict) and "std" in tn and tn["std"] > 0:
                    return tn
            except Exception:
                pass

    # Fallback: no normalization (None)
    return None


def _evaluate_single(model, ema, loader: DataLoader, device, use_amp: bool, target_stats: Optional[dict]) -> Dict[str, float]:
    return loop_evaluate(model, loader, device, amp=use_amp, ema=ema, target_stats=target_stats)


def _evaluate_ensemble(models: List[torch.nn.Module],
                       emas: List[Optional[EMA]],
                       loader: DataLoader,
                       device,
                       use_amp: bool,
                       target_stats: Optional[dict]) -> Dict[str, float]:
    """
    Evaluate an ensemble by averaging predictions across constituent models.
    We mimic loop.evaluate(), but generate averaged preds.

    NOTE: We don't call loop.evaluate() directly because EMA must be applied per-model.
    """
    loss_fn = torch.nn.MSELoss(reduction="mean")
    total_loss = 0.0
    n = 0
    all_y: List[torch.Tensor] = []
    all_p: List[torch.Tensor] = []

    # Lazily import training helpers to reuse their behavior for (de)standardization
    from src.train.loop import _standardize_targets as _std, _destandardize_preds as _destd

    # Apply EMA shadow per-model; restore after loop
    applied = []
    try:
        for m, e in zip(models, emas):
            if e is not None:
                e.apply_shadow(m)
            applied.append(e is not None)

        for batch in loader:
            batch = batch.to(device)
            y = batch.y.view(-1, 1).to(device)
            y_std = _std(y, target_stats)

            # Collect predictions from each model, then average
            preds = []
            if use_amp:
                with torch.cuda.amp.autocast():
                    for m in models:
                        p = m(batch)
                        preds.append(p)
            else:
                for m in models:
                    p = m(batch)
                    preds.append(p)

            pred_mean = torch.stack(preds, dim=0).mean(dim=0)
            loss = loss_fn(pred_mean, y_std)

            total_loss += float(loss.item()) * y.size(0)
            n += y.size(0)

            all_y.append(y.detach().cpu())
            all_p.append(pred_mean.detach().cpu())

        y_true = torch.cat(all_y, dim=0)
        y_pred_std = torch.cat(all_p, dim=0)
        y_pred = _destd(y_pred_std, target_stats)
        from src.train.loop import compute_mae_rmse
        mae, rmse = compute_mae_rmse(y_true, y_pred)
        return {"MSE": total_loss / max(n, 1), "MAE": mae, "RMSE": rmse}
    finally:
        # Restore models if EMA was applied
        for m, e, was_applied in zip(models, emas, applied):
            if e is not None and was_applied:
                e.restore(m)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

@hydra.main(version_base=None, config_name="evaluate_config")
def main(cfg: DictConfig) -> int:
    # Convert to object (Hydra -> dataclass-like)
    conf = OmegaConf.to_object(cfg)
    if isinstance(conf, dict):
        conf = RootCfg(**conf)  # type: ignore[call-arg]

    original_cwd = get_original_cwd()
    resolved_out_dir = _resolve_out_dir(original_cwd, conf.runtime.out_dir)

    # Build base args and create loaders with the same code used in training
    parser = build_argparser()
    argv = _compose_loop_argv_for_loaders(conf, original_cwd)
    args = parser.parse_args(argv)

    # Seed & device (keep behavior consistent with training loop)
    torch.manual_seed(conf.runtime.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not conf.runtime.cpu else "cpu")

    # Dataloaders & dims
    train_loader, val_loader, test_loader, meta = build_dataloaders(args)
    from src.train.loop import _guess_dims_from_loader
    node_dim, edge_dim = _guess_dims_from_loader(train_loader)

    # Select split
    if conf.eval_cfg.which_split.lower() == "val":
        eval_loader = val_loader
    elif conf.eval_cfg.which_split.lower() == "test":
        eval_loader = test_loader
    else:
        raise ValueError("eval_cfg.which_split must be one of {'val','test'}")

    # Target normalization to use
    target_stats = _get_target_stats_for_eval(conf, resolved_out_dir, train_loader, device)

    # Collect checkpoint paths
    ckpts = _collect_ckpt_paths(conf, resolved_out_dir)
    if len(ckpts) == 0:
        raise SystemExit("No checkpoint(s) found. Provide eval_cfg.ckpt_path or runtime.out_dir (with best.ckpt), "
                         "or set ensemble_glob/ensemble_paths.")

    # Build model(s) and evaluate
    use_amp = bool(conf.eval_cfg.amp and device.type == "cuda")
    if len(ckpts) == 1:
        model, ema = _load_model_with_ckpt(args, node_dim, edge_dim, ckpts[0], conf.eval_cfg.use_ema, device)
        metrics = _evaluate_single(model, ema, eval_loader, device, use_amp, target_stats)
    else:
        models: List[torch.nn.Module] = []
        emas: List[Optional[EMA]] = []
        for p in ckpts:
            m, e = _load_model_with_ckpt(args, node_dim, edge_dim, p, conf.eval_cfg.use_ema, device)
            models.append(m)
            emas.append(e)
        metrics = _evaluate_ensemble(models, emas, eval_loader, device, use_amp, target_stats)

    # Optionally dump predictions
    if conf.eval_cfg.dump_preds:
        # Re-run a single forward collect to get predictions row-wise
        # (Avoid refactoring training loop; keep logic contained here.)
        from src.train.loop import _standardize_targets as _std, _destandardize_preds as _destd
        import csv

        header = ["row_idx", "y_true", "y_pred"]
        preds_path = conf.eval_cfg.preds_filename
        if not os.path.isabs(preds_path):
            base = resolved_out_dir or os.path.join(original_cwd, "runs")
            preds_path = os.path.join(base, preds_path)

        os.makedirs(os.path.dirname(preds_path), exist_ok=True)

        # For ensembles, reuse the averaging path
        def _predict_one_loader() -> List[Tuple[int, float, float]]:
            rows: List[Tuple[int, float, float]] = []
            idx = 0
            if len(ckpts) == 1:
                model.eval()
                applied = False
                try:
                    if ema is not None:
                        ema.apply_shadow(model); applied = True
                    for batch in eval_loader:
                        batch = batch.to(device)
                        y = batch.y.view(-1, 1)
                        with torch.no_grad():
                            if use_amp:
                                with torch.cuda.amp.autocast():
                                    p = model(batch).cpu()
                            else:
                                p = model(batch).cpu()
                        y = y.cpu()
                        p = _destd(p, target_stats)
                        for yi, pi in zip(y.view(-1).tolist(), p.view(-1).tolist()):
                            rows.append((idx, float(yi), float(pi)))
                            idx += 1
                finally:
                    if ema is not None and applied:
                        ema.restore(model)
            else:
                # ensemble averaging per batch
                applied = []
                try:
                    for mm, ee in zip(models, emas):
                        if ee is not None:
                            ee.apply_shadow(mm)
                        applied.append(ee is not None)

                    for batch in eval_loader:
                        batch = batch.to(device)
                        y = batch.y.view(-1, 1).cpu()
                        preds = []
                        with torch.no_grad():
                            if use_amp:
                                with torch.cuda.amp.autocast():
                                    for mm in models:
                                        preds.append(mm(batch).cpu())
                            else:
                                for mm in models:
                                    preds.append(mm(batch).cpu())
                        p = torch.stack(preds, dim=0).mean(dim=0)
                        p = _destd(p, target_stats)
                        for yi, pi in zip(y.view(-1).tolist(), p.view(-1).tolist()):
                            rows.append((idx, float(yi), float(pi)))
                            idx += 1
                finally:
                    for mm, ee, was_applied in zip(models, emas, applied):
                        if ee is not None and was_applied:
                            ee.restore(mm)
            return rows

        rows = _predict_one_loader()
        with open(preds_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"[cli.evaluate] Wrote predictions to: {preds_path}")

    # Print summary
    print(f"[cli.evaluate] Split={conf.eval_cfg.which_split} | "
          f"MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | MSE={metrics['MSE']:.4f}")
    if len(ckpts) > 1:
        print(f"[cli.evaluate] Ensemble size: {len(ckpts)}")
    else:
        print(f"[cli.evaluate] Checkpoint: {ckpts[0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
