from __future__ import annotations

"""
src/analysis/plots.py

Phase 9 — Plots & tables pipeline

This module builds standard figures from the per-run metrics and params:
  * Depth curves: test MAE / RMSE vs num_layers for GIN & MPNN.
  * Regularization ablation bars: dropout × weight_decay for GIN & MPNN.
  * Calibration curves: reliability diagrams from a calibration CSV (optional).

Usage examples (from repo root, after Phase 6 runs exist):

  # Depth curves (ESOL, scaffold split)
  python -m src.analysis.plots depth \
      --roots runs outputs/phase6 \
      --dataset ESOL \
      --split scaffold \
      --outdir report/figures

  # Regularization ablation bars
  python -m src.analysis.plots ablation \
      --roots runs outputs/phase6 \
      --dataset ESOL \
      --split scaffold \
      --outdir report/figures

  # Calibration curves (if you have a calibration_bins CSV)
  python -m src.analysis.plots calibration \
      --calibration-csv report/tables/calibration_bins.csv \
      --outdir report/figures

  # Run all standard plots in one go
  python -m src.analysis.plots all \
      --roots runs outputs/phase6 \
      --dataset ESOL \
      --split scaffold \
      --outdir report/figures \
      --calibration-csv report/tables/calibration_bins.csv
"""

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .aggregate_csv import (
    _gather_metrics_csv_paths,
    _read_csv_rows,
    _select_latest_test_row,
    _read_json,
    _to_float_or_none,
)

# ---------------------------------------------------------------------------
# Small helpers / containers
# ---------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def _normalize_model_name(model_raw: Any) -> str:
    """
    Map various strings to canonical model names: 'gin', 'mpnn', 'ridge', 'random_forest', etc.
    """
    s = str(model_raw).strip()
    sl = s.lower()
    if "gin" in sl:
        return "gin"
    if "mpnn" in sl:
        return "mpnn"
    if "random_forest" in sl or sl == "rf":
        return "random_forest"
    if "ridge" in sl:
        return "ridge"
    return sl


def _deep_find_key(obj: Any, target: str) -> Optional[Any]:
    """
    Recursively search a nested dict/list structure for the first occurrence of `target` as a key.
    Returns the corresponding value if found, else None.
    """
    if isinstance(obj, dict):
        if target in obj:
            return obj[target]
        for v in obj.values():
            out = _deep_find_key(v, target)
            if out is not None:
                return out
    elif isinstance(obj, list):
        for item in obj:
            out = _deep_find_key(item, target)
            if out is not None:
                return out
    return None


def _coerce_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None


def _coerce_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


@dataclass
class BestRun:
    """Best test-row for a single run, plus its params.json."""

    dataset: str
    split: str  # split field from metrics.csv (may be 'test' depending on logger)
    model: str  # normalized model name (gin, mpnn, ridge, random_forest, ...)
    metrics: Dict[str, float]  # e.g. {"test_MAE": ..., "test_RMSE": ...}
    params: Dict[str, Any]
    run_dir: str


def load_best_runs(
    roots: Sequence[str],
    dataset_filter: Optional[str] = None,
) -> List[BestRun]:
    """
    Walk all `metrics.csv` under the given roots and collect the latest 'test'
    row plus its params.json. We also robustly infer the dataset split strategy
    (scaffold/random) and store it in BestRun.split.
    """
    csv_paths = _gather_metrics_csv_paths(roots)
    runs: List[BestRun] = []

    for metrics_path in csv_paths:
        rows = _read_csv_rows(metrics_path)
        if not rows:
            continue
        latest = _select_latest_test_row(rows)
        if latest is None:
            continue

        dataset = str(latest.get("dataset", "")).strip() or "ESOL"
        if dataset_filter is not None and dataset != dataset_filter:
            continue

        # Model (normalized)
        raw_model = latest.get("model", "")
        model = _normalize_model_name(raw_model)

        run_dir = os.path.dirname(metrics_path)
        params = _read_json(os.path.join(run_dir, "params.json")) or {}

        # ---------- infer split strategy (scaffold/random) ----------
        strategy: Optional[str] = None
        if isinstance(params, dict):
            ev = params.get("eval")
            if isinstance(ev, str) and ev.strip():
                strategy = ev.strip()
            if strategy is None:
                data_cfg = params.get("data")
                if isinstance(data_cfg, dict):
                    ds = data_cfg.get("split")
                    if isinstance(ds, str) and ds.strip():
                        strategy = ds.strip()

        if strategy is None:
            base = os.path.basename(run_dir).lower()
            if "_scaffold_" in base or base.endswith("_scaffold") or "scaffold" in base:
                strategy = "scaffold"
            elif "_random_" in base or base.endswith("_random") or "random" in base:
                strategy = "random"

        if strategy is None:
            # If metrics row itself already carries scaffold/random, use it; else fall back.
            s = str(latest.get("split", "")).strip().lower()
            if s in ("scaffold", "random"):
                strategy = s
            else:
                strategy = s or "scaffold"  # last-resort fallback

        # Pull numeric metrics we care about
        metrics: Dict[str, float] = {}
        for key in ("test_MAE", "test_RMSE", "val_MAE", "val_RMSE"):
            v = _to_float_or_none(latest.get(key))
            if v is not None:
                metrics[key] = v

        # Stash latest row for downstream fallbacks (e.g., num_layers search)
        if isinstance(params, dict):
            params["_latest_row"] = latest  # harmless helper pocket

        runs.append(
            BestRun(
                dataset=dataset,
                split=strategy,
                model=model,
                metrics=metrics,
                params=params,
                run_dir=run_dir,
            )
        )

    print(
        f"[plots] Collected {len(runs)} runs from roots={list(roots)}"
        + (f" (dataset={dataset_filter})" if dataset_filter else "")
    )
    return runs



def _effective_split_from_params(params: Dict[str, Any], fallback_split: str) -> str:
    """
    Infer the *dataset* split (scaffold/random) from params, falling back to the metrics 'split'
    column if needed.
    """
    if not isinstance(params, dict):
        return fallback_split
    # Try eval=scaffold style
    ev = params.get("eval")
    if isinstance(ev, str) and ev:
        return ev.strip()
    # Try nested data.split
    data_cfg = params.get("data")
    if isinstance(data_cfg, dict):
        ds = data_cfg.get("split")
        if isinstance(ds, str) and ds:
            return ds.strip()
    return fallback_split


# ---------------------------------------------------------------------------
# Depth curves: test MAE / RMSE vs num_layers for GIN & MPNN
# ---------------------------------------------------------------------------


def make_depth_curves(
    roots: Sequence[str],
    dataset: str,
    split: str,
    outdir: str,
) -> None:
    """
    Build depth curves for GIN and MPNN:

      x-axis: num_layers (L)
      y-axis: test MAE (and separately test RMSE)
      errorbars: ±1 std across seeds / runs

    We aggregate over *all* runs with the same (model, num_layers, dataset, split),
    regardless of which experiment (A/B/C/...) they came from. That is fine for
    the overall story: "how does depth affect performance, marginalizing over
    other hyperparameters?"
    """
    runs = load_best_runs(roots, dataset_filter=dataset)

    # Nested dicts: model -> L -> list of metrics
    mae_by_model_depth: Dict[str, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    rmse_by_model_depth: Dict[str, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in runs:
        # Infer dataset-split (scaffold/random) from params first; fall back to metrics split
        run_split = _effective_split_from_params(r.params, r.split)
        if split and run_split != split:
            continue
        if r.model not in {"gin", "mpnn"}:
            continue

        # Find num_layers somewhere in params (model.num_layers or nested)
        num_layers_val: Optional[Any] = None
        if isinstance(r.params, dict):
            mcfg = r.params.get("model")
            if isinstance(mcfg, dict):
                num_layers_val = mcfg.get("num_layers")
            if num_layers_val is None:
                num_layers_val = _deep_find_key(r.params, "num_layers")

        L = _coerce_int(num_layers_val)
        if L is None:
            # Skip runs where we genuinely can't infer the depth
            continue

        mae = r.metrics.get("test_MAE")
        rmse = r.metrics.get("test_RMSE")
        if mae is not None:
            mae_by_model_depth[r.model][L].append(float(mae))
        if rmse is not None:
            rmse_by_model_depth[r.model][L].append(float(rmse))

    if not mae_by_model_depth:
        print(
            f"[plots.depth] No suitable runs found for dataset={dataset}, split={split}"
        )
        return

    _ensure_dir(outdir)

    # ---- MAE curve ----
    fig, ax = plt.subplots()
    for model, depth_dict in sorted(mae_by_model_depth.items()):
        depths = sorted(depth_dict.keys())
        means = [float(np.mean(depth_dict[d])) for d in depths]
        stds = [float(np.std(depth_dict[d])) for d in depths]

        ax.errorbar(
            depths,
            means,
            yerr=stds,
            marker="o",
            linestyle="-",
            label=model.upper(),
        )

    ax.set_xlabel("Number of GNN layers L")
    ax.set_ylabel("Test MAE")
    ax.set_title(f"{dataset} ({split}) — Test MAE vs depth")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    mae_path = os.path.join(
        outdir, f"{dataset.lower()}_{split}_depth_mae.png"
    )
    fig.savefig(mae_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plots.depth] Saved MAE depth curve -> {mae_path}")

    # ---- RMSE curve ----
    fig, ax = plt.subplots()
    for model, depth_dict in sorted(rmse_by_model_depth.items()):
        if not depth_dict:
            continue
        depths = sorted(depth_dict.keys())
        means = [float(np.mean(depth_dict[d])) for d in depths]
        stds = [float(np.std(depth_dict[d])) for d in depths]

        ax.errorbar(
            depths,
            means,
            yerr=stds,
            marker="o",
            linestyle="-",
            label=model.upper(),
        )

    ax.set_xlabel("Number of GNN layers L")
    ax.set_ylabel("Test RMSE")
    ax.set_title(f"{dataset} ({split}) — Test RMSE vs depth")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    rmse_path = os.path.join(
        outdir, f"{dataset.lower()}_{split}_depth_rmse.png"
    )
    fig.savefig(rmse_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plots.depth] Saved RMSE depth curve -> {rmse_path}")


# ---------------------------------------------------------------------------
# Regularization ablation bars: dropout × weight_decay for GIN & MPNN
# ---------------------------------------------------------------------------


def make_ablation_bars(
    roots: Sequence[str],
    dataset: str,
    split: str,
    outdir: str,
) -> None:
    """
    Build ablation bar plots for dropout × weight_decay on ESOL / scaffold.

    For each model in {GIN, MPNN}:
      * x-axis: dropout values
      * grouped bars: different weight_decay values
      * y-axis: mean test MAE across seeds / runs

    We infer dropout and weight_decay from params.json (recursively if needed).
    """
    runs = load_best_runs(roots, dataset_filter=dataset)

    # model -> (dropout, wd) -> list of test_MAE
    mae_by_model_cfg: Dict[str, Dict[Tuple[float, float], List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in runs:
        run_split = _effective_split_from_params(r.params, r.split)
        if split and run_split != split:
            continue
        if r.model not in {"gin", "mpnn"}:
            continue

        params = r.params if isinstance(r.params, dict) else {}

        # Try to find dropout
        dropout_val: Any = None
        mcfg = params.get("model")
        if isinstance(mcfg, dict):
            dropout_val = mcfg.get("dropout")
        if dropout_val is None:
            dropout_val = _deep_find_key(params, "dropout")
        dropout = _coerce_float(dropout_val, default=0.0)

        # Try to find weight_decay
        wd_val: Any = None
        tcfg = params.get("train")
        if isinstance(tcfg, dict):
            wd_val = tcfg.get("weight_decay")
        if wd_val is None:
            wd_val = _deep_find_key(params, "weight_decay")
        wd = _coerce_float(wd_val, default=0.0)

        mae = r.metrics.get("test_MAE")
        if mae is None:
            continue

        mae_by_model_cfg[r.model][(dropout, wd)].append(float(mae))

    if not mae_by_model_cfg:
        print(
            f"[plots.ablation] No suitable runs found for dataset={dataset}, split={split}"
        )
        return

    _ensure_dir(outdir)

    for model, cfg_dict in sorted(mae_by_model_cfg.items()):
        if not cfg_dict:
            continue

        dropouts = sorted({cfg[0] for cfg in cfg_dict.keys()})
        wds = sorted({cfg[1] for cfg in cfg_dict.keys()})

        x = np.arange(len(dropouts), dtype=float)
        width = 0.8 / max(len(wds), 1)

        fig, ax = plt.subplots()

        for j, wd in enumerate(wds):
            means: List[float] = []
            stds: List[float] = []

            for d in dropouts:
                vals = cfg_dict.get((d, wd), [])
                if vals:
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)))
                else:
                    means.append(np.nan)
                    stds.append(0.0)

            # Horizontal shift per weight decay
            ax.bar(
                x + j * width,
                means,
                width,
                yerr=stds,
                label=f"wd={wd:g}",
                align="edge",
            )

        ax.set_xticks(x + width * (len(wds) - 1) / 2.0)
        ax.set_xticklabels([f"drop={d:g}" for d in dropouts])
        ax.set_ylabel("Test MAE")
        ax.set_xlabel("Dropout")
        ax.set_title(
            f"{dataset} ({split}) — {model.upper()} dropout × weight_decay ablation"
        )
        ax.grid(axis="y", linestyle=":", linewidth=0.5)
        ax.legend(title="weight_decay")

        fname = os.path.join(
            outdir,
            f"{dataset.lower()}_{split}_ablation_{model}.png",
        )
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[plots.ablation] Saved ablation bars for {model} -> {fname}")


# ---------------------------------------------------------------------------
# Calibration curves: reliability diagram from a CSV (optional)
# ---------------------------------------------------------------------------


def _load_calibration_curves(
    calib_csv: str,
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Load calibration bins from a CSV.

    We try to be flexible about column names:
      x-axis: one of {"bin_mid", "prob", "conf"}
      y-axis: one of {"accuracy", "acc", "empirical"}

    Curves are grouped by a "label" column (fallback: "tag", then "model",
    then a single default label).
    """
    curves: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    with open(calib_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_raw = row.get("bin_mid") or row.get("prob") or row.get("conf")
            y_raw = (
                row.get("accuracy") or row.get("acc") or row.get("empirical")
            )

            x = _to_float_or_none(x_raw)
            y = _to_float_or_none(y_raw)
            if x is None or y is None:
                continue

            label = (
                row.get("label")
                or row.get("tag")
                or row.get("model")
                or "curve"
            )
            curves[label].append((x, y))

    return curves


def make_calibration_curves(
    calib_csv: str,
    outdir: str,
    title: Optional[str] = None,
) -> None:
    """
    Build a reliability diagram from a calibration_bins-style CSV.

    This is intentionally generic and does *not* assume a particular Phase 5/8
    implementation. If `report/tables/calibration_bins.csv` does not exist yet,
    we simply skip plotting.
    """
    if not os.path.exists(calib_csv):
        print(
            f"[plots.calibration] No calibration CSV found at {calib_csv}; "
            "skipping calibration curves."
        )
        return

    curves = _load_calibration_curves(calib_csv)
    if not curves:
        print(
            f"[plots.calibration] Calibration CSV at {calib_csv} has no usable rows; skipping."
        )
        return

    _ensure_dir(outdir)

    fig, ax = plt.subplots()

    # Perfect calibration diagonal
    grid = np.linspace(0.0, 1.0, 101)
    ax.plot(grid, grid, linestyle="--", linewidth=1.0, label="Perfect")

    for label, pts in sorted(curves.items()):
        pts_sorted = sorted(pts, key=lambda t: t[0])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        ax.plot(xs, ys, marker="o", linestyle="-", label=label)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title or "Reliability diagram")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()

    fname = os.path.join(outdir, "calibration_reliability.png")
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plots.calibration] Saved calibration plot -> {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 9 plotting pipeline (depth, ablation, calibration)."
    )
    subparsers = p.add_subparsers(dest="cmd", required=True)

    # depth
    p_depth = subparsers.add_parser(
        "depth", help="Depth curves: test MAE/RMSE vs num_layers."
    )
    p_depth.add_argument(
        "--roots",
        nargs="+",
        default=["runs", "outputs/phase6"],
        help="Roots to search for metrics.csv (default: runs outputs/phase6).",
    )
    p_depth.add_argument(
        "--dataset",
        default="ESOL",
        help="Dataset name to filter on (default: ESOL).",
    )
    p_depth.add_argument(
        "--split",
        default="scaffold",
        help="Split name to filter on (default: scaffold).",
    )
    p_depth.add_argument(
        "--outdir",
        default="report/figures",
        help="Directory to write figures into.",
    )

    # ablation
    p_ablation = subparsers.add_parser(
        "ablation",
        help="Regularization ablation bars: dropout × weight_decay.",
    )
    p_ablation.add_argument(
        "--roots",
        nargs="+",
        default=["runs", "outputs/phase6"],
        help="Roots to search for metrics.csv (default: runs outputs/phase6).",
    )
    p_ablation.add_argument(
        "--dataset",
        default="ESOL",
        help="Dataset name to filter on (default: ESOL).",
    )
    p_ablation.add_argument(
        "--split",
        default="scaffold",
        help="Split name to filter on (default: scaffold).",
    )
    p_ablation.add_argument(
        "--outdir",
        default="report/figures",
        help="Directory to write figures into.",
    )

    # calibration
    p_calib = subparsers.add_parser(
        "calibration",
        help="Calibration curves (reliability diagram) from a CSV.",
    )
    p_calib.add_argument(
        "--calibration-csv",
        default="report/tables/calibration_bins.csv",
        help="Path to calibration CSV (default: report/tables/calibration_bins.csv).",
    )
    p_calib.add_argument(
        "--outdir",
        default="report/figures",
        help="Directory to write figures into.",
    )
    p_calib.add_argument(
        "--title",
        default=None,
        help="Optional title override for the calibration plot.",
    )

    # all
    p_all = subparsers.add_parser(
        "all",
        help="Run depth + ablation (+ calibration if CSV exists) in one go.",
    )
    p_all.add_argument(
        "--roots",
        nargs="+",
        default=["runs", "outputs/phase6"],
        help="Roots to search for metrics.csv (default: runs outputs/phase6).",
    )
    p_all.add_argument(
        "--dataset",
        default="ESOL",
        help="Dataset name to filter on (default: ESOL).",
    )
    p_all.add_argument(
        "--split",
        default="scaffold",
        help="Split name to filter on (default: scaffold).",
    )
    p_all.add_argument(
        "--outdir",
        default="report/figures",
        help="Directory to write figures into.",
    )
    p_all.add_argument(
        "--calibration-csv",
        default="report/tables/calibration_bins.csv",
        help="Path to calibration CSV (used if present).",
    )
    p_all.add_argument(
        "--title",
        default=None,
        help="Optional title override for the calibration plot.",
    )

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.cmd == "depth":
        make_depth_curves(args.roots, args.dataset, args.split, args.outdir)
    elif args.cmd == "ablation":
        make_ablation_bars(args.roots, args.dataset, args.split, args.outdir)
    elif args.cmd == "calibration":
        make_calibration_curves(
            calib_csv=args.calibration_csv,
            outdir=args.outdir,
            title=args.title,
        )
    elif args.cmd == "all":
        make_depth_curves(args.roots, args.dataset, args.split, args.outdir)
        make_ablation_bars(args.roots, args.dataset, args.split, args.outdir)
        if args.calibration_csv and os.path.exists(args.calibration_csv):
            make_calibration_curves(
                calib_csv=args.calibration_csv,
                outdir=args.outdir,
                title=args.title,
            )
        else:
            print(
                "[plots.all] No calibration CSV found; "
                "skipping calibration curves."
            )
    else:
        raise ValueError(f"Unknown command: {args.cmd}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
