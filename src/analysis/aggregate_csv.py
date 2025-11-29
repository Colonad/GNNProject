# src/analysis/aggregate_csv.py
from __future__ import annotations

"""
Aggregate per-run metrics.csv files into a single summary table.

Typical usage
-------------
python -m src.analysis.aggregate_csv \
  --roots runs outputs \
  --out report/tables/summary.csv

Behavior
--------
- Walks each --roots directory recursively and collects paths ending with "metrics.csv".
- For each run directory (parent of metrics.csv):
    - Parses metrics.csv and selects the most recent `split == "test"` row
      (if 'epoch' column exists, uses the max epoch; else last row with split=="test").
    - Reads params.json (optional) to extract 'seed' (helps count unique seeds).
    - Reads summary.json (optional) to capture 'best_epoch' and 'best_by'.
- Aggregates across runs using group-by keys (default):
    ["dataset", "model", "split", "scaler", "feat_kind", "feat_bits", "feat_radius"]
- For numeric metric columns (MAE, RMSE, MSE, R2, loss, etc.), computes
  mean, std, min, max, and a pretty "mean ± std" string.
- Writes CSV to --out (ensures parent dirs exist).

Notes
-----
- Non-numeric columns are excluded from aggregation.
- If you pass --metrics, only those metric columns are aggregated (if present).
- Missing metrics in some runs are ignored for that metric (NaNs dropped).
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False
    raise RuntimeError("aggregate_csv.py requires NumPy. Please `pip install numpy`.")

# -------------------------------
# Helpers
# -------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _is_number(x: Any) -> bool:
    if isinstance(x, (int, float)):
        return True
    try:
        import numpy as _np  # type: ignore
        if isinstance(x, _np.generic):
            return True
        if isinstance(x, _np.ndarray):
            return x.size == 1 and _np.issubdtype(x.dtype, _np.number)
    except Exception:
        pass
    # strings that look like numbers?
    if isinstance(x, str):
        try:
            float(x)
            return True
        except Exception:
            return False
    return False

def _to_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        f = float(x)
        if math.isnan(f):
            return None
        return f
    if isinstance(x, str):
        try:
            f = float(x)
            if math.isnan(f):
                return None
            return f
        except Exception:
            return None
    try:
        import numpy as _np  # type: ignore
        if isinstance(x, _np.generic):
            f = float(x)
            if math.isnan(f):
                return None
            return f
        if isinstance(x, _np.ndarray) and x.size == 1:
            f = float(x.reshape(-1)[0])
            if math.isnan(f):
                return None
            return f
    except Exception:
        return None
    return None

def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _gather_metrics_csv_paths(roots: Sequence[str], filename: str = "metrics.csv") -> List[str]:
    paths: List[str] = []
    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, files in os.walk(root):
            if filename in files:
                paths.append(os.path.join(dirpath, filename))
    return sorted(paths)

def _read_csv_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    except Exception:
        pass
    return rows

def _select_latest_test_row(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Choose the row corresponding to the test split.

    - If there are explicit rows with split == "test", pick the most recent one
      (max epoch, then max step), matching the old baseline logging style.
    - If there are NO such rows (e.g. our new GNN loop writes a single
      summary row that already contains val_* and test_* columns), just
      return the last row in the CSV.
    """
    if not rows:
        return None

    # Old-style logs: multiple rows with split=train/val/test
    test_rows = [r for r in rows if str(r.get("split", "")).strip().lower() == "test"]
    if test_rows:
        def _epoch_val(r: Dict[str, Any]) -> Tuple[int, int]:
            e = _to_float_or_none(r.get("epoch"))
            step = _to_float_or_none(r.get("step"))
            return (int(e) if e is not None else -1,
                    int(step) if step is not None else -1)
        test_rows.sort(key=_epoch_val)
        return test_rows[-1]

    # New-style logs: per-run summary only (no split column or no "test" rows)
    return rows[-1]

# -------------------------------
# Core aggregation
# -------------------------------

DEFAULT_GROUP_KEYS = ["dataset", "model", "split", "scaler", "feat_kind", "feat_bits", "feat_radius"]
DEFAULT_METRICS_ORDER = ["MAE", "RMSE", "MSE", "R2", "loss", "ECE"]  # we'll keep any numeric if present

@dataclass
class RunRecord:
    group: Dict[str, Any]
    metrics: Dict[str, float]   # numeric-only
    seed: Optional[int]
    run_dir: str
    best_epoch: Optional[int]
    best_by: Optional[str]

def _extract_group_keys(row: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    g: Dict[str, Any] = {}
    for k in keys:
        v = row.get(k)
        # normalize texty keys
        if v is None:
            g[k] = ""
        else:
            g[k] = v
    return g

def _discover_numeric_metrics(row: Dict[str, Any], allowed: Optional[Sequence[str]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if allowed is not None and len(allowed) > 0:
        # keep only allowed metrics that are numeric
        for k in allowed:
            if k in row:
                f = _to_float_or_none(row[k])
                if f is not None:
                    out[k] = f
        return out
    # else: keep any numeric-looking column that's not in common non-metric fields
    exclude = {"step", "epoch", "split", "lr", "wall_time",
               "dataset", "model", "scaler", "feat_kind", "feat_bits", "feat_radius"}
    for k, v in row.items():
        if k in exclude:
            continue
        f = _to_float_or_none(v)
        if f is not None:
            out[k] = f
    return out

def _load_run_record(metrics_csv_path: str,
                     group_keys: Sequence[str],
                     only_metrics: Optional[Sequence[str]] = None) -> Optional[RunRecord]:
    rows = _read_csv_rows(metrics_csv_path)
    if not rows:
        return None
    latest = _select_latest_test_row(rows)
    if latest is None:
        return None

    run_dir = os.path.dirname(metrics_csv_path)
    params = _read_json(os.path.join(run_dir, "params.json")) or {}
    summary = _read_json(os.path.join(run_dir, "summary.json")) or {}

    seed = None
    # try params["seed"], or nested fields
    for candidate in ["seed", ("runtime", "seed"), ("train", "seed")]:
        try:
            if isinstance(candidate, tuple):
                v = params
                for c in candidate:
                    v = v[c]
                seed = int(v)
                break
            else:
                if candidate in params:
                    seed = int(params[candidate])
                    break
        except Exception:
            continue

    best_epoch = None
    best_by = None
    if isinstance(summary, dict):
        be = summary.get("best_epoch", None)
        if be is not None:
            try:
                best_epoch = int(be)
            except Exception:
                best_epoch = None
        bb = summary.get("best_by", None)
        if isinstance(bb, str):
            best_by = bb

    group = _extract_group_keys(latest, group_keys)
    metrics = _discover_numeric_metrics(latest, only_metrics)

    return RunRecord(group=group, metrics=metrics, seed=seed, run_dir=run_dir,
                     best_epoch=best_epoch, best_by=best_by)

def _agg_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr, ddof=0)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "n": int(arr.size),
    }

def _format_pm(mean: float, std: float, precision: int = 4) -> str:
    if any(map(lambda x: x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))), [mean, std])):
        return "NaN ± NaN"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"

# -------------------------------
# CLI and pipeline
# -------------------------------

def aggregate(
    roots: Sequence[str],
    out_csv: str,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    metrics: Optional[Sequence[str]] = None,
    float_precision: int = 4,
) -> str:
    """Return path to the written CSV."""
    csv_paths = _gather_metrics_csv_paths(roots)
    if not csv_paths:
        _ensure_dir(os.path.dirname(out_csv) or ".")
        # Write an empty-but-header CSV to satisfy DoD
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "model", "split", "scaler", "feat_kind", "feat_bits", "feat_radius",
                        "n_runs", "n_seeds"])
        return out_csv

    records: List[RunRecord] = []
    for p in csv_paths:
        rec = _load_run_record(p, group_keys, metrics)
        if rec is not None and rec.metrics:
            records.append(rec)

    if not records:
        _ensure_dir(os.path.dirname(out_csv) or ".")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(group_keys + ["n_runs", "n_seeds"])
        return out_csv

    # Collect all metric names to aggregate
    metric_names: List[str] = []
    if metrics:
        metric_names = [m for m in metrics if any(m in r.metrics for r in records)]
    else:
        # union of all numeric metrics
        seen = set()
        for r in records:
            for k in r.metrics.keys():
                if k not in seen:
                    seen.add(k)
        # preserve intuitive order
        ordered = [m for m in DEFAULT_METRICS_ORDER if m in seen]
        # then append any others
        ordered += [m for m in sorted(seen) if m not in ordered]
        metric_names = ordered

    # Group
    groups: Dict[Tuple[Any, ...], List[RunRecord]] = defaultdict(list)
    for r in records:
        key = tuple((r.group.get(k, "") for k in group_keys))
        groups[key].append(r)

    # Aggregate
    rows_out: List[List[Any]] = []

    # Header
    header: List[str] = list(group_keys) + ["n_runs", "n_seeds"]
    for m in metric_names:
        header += [f"{m}_mean", f"{m}_std", f"{m}_min", f"{m}_max", f"{m}_pm"]  # pm = pretty "mean ± std"

    # Write
    _ensure_dir(os.path.dirname(out_csv) or ".")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for gkey, runs in sorted(groups.items()):
            # counts
            n_runs = len(runs)
            # count unique seeds (if missing, count as distinct by run_dir to avoid collapsing)
            seeds = set()
            for r in runs:
                seeds.add((r.seed if r.seed is not None else f"run@{os.path.basename(r.run_dir)}"))
            n_seeds = len(seeds)

            base = list(gkey) + [n_runs, n_seeds]

            # Collect metric arrays
            metric_cells: List[Any] = []
            for m in metric_names:
                vals = [r.metrics[m] for r in runs if m in r.metrics and _to_float_or_none(r.metrics[m]) is not None]
                stats = _agg_stats(vals)
                pm = _format_pm(stats["mean"], stats["std"], precision=int(float_precision))
                metric_cells += [
                    round(stats["mean"], int(float_precision)) if not math.isnan(stats["mean"]) else "NaN",
                    round(stats["std"], int(float_precision)) if not math.isnan(stats["std"]) else "NaN",
                    round(stats["min"], int(float_precision)) if not math.isnan(stats["min"]) else "NaN",
                    round(stats["max"], int(float_precision)) if not math.isnan(stats["max"]) else "NaN",
                    pm,
                ]

            w.writerow(base + metric_cells)

    return out_csv


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate run metrics into a summary CSV (mean ± std across seeds).")
    p.add_argument("--roots", nargs="+", default=["runs"], help="Root directories to search (recursively) for metrics.csv.")
    p.add_argument("--out", default="report/tables/summary.csv", help="Output CSV path.")
    p.add_argument("--group-by", nargs="+", default=DEFAULT_GROUP_KEYS,
                   help=f"Grouping columns (default: {DEFAULT_GROUP_KEYS}).")
    p.add_argument("--metrics", nargs="+", default=None,
                   help="Specific metric names to aggregate (e.g., MAE RMSE). If omitted, aggregate all numeric metrics.")
    p.add_argument("--precision", type=int, default=4, help="Float precision for output numbers and ± formatting.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    out_path = aggregate(
        roots=args.roots,
        out_csv=args.out,
        group_keys=args.group_by,
        metrics=args.metrics,
        float_precision=args.precision,
    )
    print(f"[aggregate_csv] Wrote summary -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# EOF