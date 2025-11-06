# src/cli/grid_run.py
"""
Grid runner for the Molecular GNN project.

Features
--------
- Accepts grid specs via `+grids=<name>` (resolved to configs/grids/<name>.yaml)
  or via `--grid <path/to/grid.yaml>`.
- Optional base overrides applied to every trial (e.g., `data.name=ESOL eval=scaffold`).
- Expands cartesian product of list-valued entries in the grid YAML.
- Runs each combination by invoking: `python -m src.cli.train <overrides...>`.
- Writes a manifest (YAML + JSON) for reproducibility.
- Saves per-trial stdout/stderr logs under the grid output directory.
- Aggregates metrics from each run into a single CSV:
  tries `metrics.json`, `metrics.csv` in `runtime.out_dir` or `runs/...` folder.
- Parallel execution via `--jobs N` (N>1 uses multiprocessing).
- Robust error handling; failures are captured in the summary.

Usage
-----
    # From the repo root:
    python -m src.cli.grid_run +grids=esol_small_grid data.name=ESOL eval=scaffold --jobs 1

    # Or with an explicit grid file:
    python -m src.cli.grid_run --grid configs/grids/esol_small_grid.yaml data.name=ESOL eval=scaffold

Notes
-----
- This script intentionally uses subprocess calls to avoid tight coupling with
  the training module internals and to respect Hydra-like overrides you pass in.
- It does NOT require Hydra to run, but happily forwards any Hydra-style
  key=value overrides you include.
"""

from __future__ import annotations
import hashlib
import argparse
import csv
import datetime as dt
import itertools
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception as e:
    print(
        "ERROR: PyYAML is required for grid_run. Please `pip install pyyaml` (or add to environment.yml).",
        file=sys.stderr,
    )
    raise

# --------- Constants ---------

DEFAULT_GRIDS_DIR = Path("configs/grids")
DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_OUTPUTS_DIR = Path("outputs/grids")


# --------- Utilities ---------





# Only include the most informative knobs in folder names (everything else is in manifest/command.txt)
SELECTED_DIR_KEYS = {
    "model.name",
    "model.num_layers",
    "model.hidden_dim",
    "model.dropout",
    "train.optimizer.weight_decay",
    "runtime.seed",
    "train.epochs",
    "train.scheduler.name",
}

# Friendly abbreviations for common keys
KEY_ABBR = {
    "model.name": "m",
    "model.num_layers": "nl",
    "model.hidden_dim": "hd",
    "model.dropout": "do",
    "train.optimizer.weight_decay": "wd",
    "train.epochs": "ep",
    "train.scheduler.name": "sch",
    "runtime.seed": "s",
}

def _abbr_key(k: str) -> str:
    return KEY_ABBR.get(k, k.replace(".", "_"))

def _abbr_val(v: Any) -> str:
    # Compact, filesystem-safe representation
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        # Short but precise; avoid scientific unless needed
        s = f"{v:.6g}"
        # strip trailing zeros/dot
        if "." in s:
            s = s.rstrip("0").rstrip(".") or "0"
        return s
    # Avoid brackets/commas/spaces
    s = str(v).replace("/", "-").replace(" ", "")
    for ch in ["[", "]", "{", "}", "(", ")", ",", ":", "'", '"']:
        s = s.replace(ch, "")
    return s

def _truncate_with_hash(name: str, max_len: int = 140) -> str:
    """
    Ensure directory name stays under max_len by appending a short hash if needed.
    """
    if len(name) <= max_len:
        return name
    h = hashlib.blake2b(name.encode("utf-8"), digest_size=8).hexdigest()
    keep = max_len - (len(h) + 2)  # 2 for '__'
    if keep < 16:  # safety guard
        keep = 16
    return f"{name[:keep]}__{h}"




def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_listlike(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dictionaries using dot-separated keys.
    {"model": {"hidden_dim": 128, "dropout": [0.0, 0.1]}} ->
    {"model.hidden_dim": 128, "model.dropout": [0.0, 0.1]}
    """
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def cartesian_product(grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Given a flattened dict of name->value (each value can be scalar or list),
    produce a list of dicts representing each combination.

    Rules:
    - If a value is listlike, we sweep over it.
    - If a value is scalar, it is fixed.
    """
    keys = list(grid.keys())
    sweep_values: List[List[Any]] = []
    for k in keys:
        v = grid[k]
        sweep_values.append(list(v) if is_listlike(v) else [v])
    combos = []
    for prod in itertools.product(*sweep_values):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos


def dict_to_overrides(d: Dict[str, Any]) -> List[str]:
    """
    Convert a flat dict into CLI overrides: key=value.
    Strings get quoted if they contain spaces or special chars.
    """
    overrides: List[str] = []
    for k, v in d.items():
        if isinstance(v, bool):
            val = "true" if v else "false"
        elif v is None:
            val = "null"
        elif isinstance(v, (int, float)):
            val = str(v)
        else:
            s = str(v)
            # Quote if spaces or characters likely to confuse shells/Hydra
            if any(ch.isspace() for ch in s) or any(ch in s for ch in ['"', "'", "=", ",", ":", "{", "}", "[", "]"]):
                s = shlex.quote(s)
            val = s
        overrides.append(f"{k}={val}")
    return overrides


def extract_grid_name_from_plus(overrides: List[str]) -> Optional[str]:
    """
    Find and remove a '+grids=NAME' style override from a list of overrides.
    Returns NAME if present; modifies the list in place to remove the token.
    """
    idx = None
    name = None
    for i, tok in enumerate(overrides):
        if tok.startswith("+grids="):
            idx = i
            name = tok.split("=", 1)[1].strip()
            break
    if idx is not None:
        overrides.pop(idx)
    return name


def locate_grid_yaml(grid_name: str) -> Path:
    """
    Map 'esol_small_grid' -> configs/grids/esol_small_grid.yaml
    """
    candidate = DEFAULT_GRIDS_DIR / f"{grid_name}.yaml"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not locate grid '{grid_name}'. Expected file: {candidate}"
        )
    return candidate


def discover_metrics(run_dir: Path) -> Dict[str, Any]:
    """
    Try to discover metrics from a run directory. Supports:
    - metrics.json
    - metrics.csv (expects a header row; takes the last row)
    Returns a dict with at least {'status': 'ok'|'fail'} and best-effort keys.
    """
    result: Dict[str, Any] = {"status": "fail"}
    # Prefer JSON if present
    for candidate in ["metrics.json", "final_metrics.json"]:
        p = run_dir / candidate
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data = {k.lower(): v for k, v in data.items()}
                    result.update(data)
                    result["status"] = "ok"
                    return result
            except Exception:
                pass

    # Fallback to CSV
    csv_path = run_dir / "metrics.csv"
    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                last_row = None
                for row in reader:
                    last_row = row
                if last_row:
                    # Normalize keys to lowercase
                    last_row = {k.lower(): v for k, v in last_row.items()}
                    result.update(last_row)
                    result["status"] = "ok"
                    return result
        except Exception:
            pass

    return result


def make_trial_out_dir(grid_out_dir: Path, trial_idx: int, combo: Dict[str, Any]) -> Path:
    """
    Build a concise, stable per-trial directory name that won't exceed filesystem limits.

    Example:
      trial0003__m=gin__nl=6__hd=128__do=0.1__wd=0.0001__sch=cosine__ep=200__s=1
    """
    parts: List[str] = [f"trial{trial_idx:04d}"]

    # Keep only selected, informative keys; sort for stability
    for key in sorted(k for k in combo.keys() if k in SELECTED_DIR_KEYS):
        parts.append(f"{_abbr_key(key)}={_abbr_val(combo[key])}")

    name = "__".join(parts)
    name = _truncate_with_hash(name, max_len=140)  # safe length for deep paths

    out_dir = grid_out_dir / name
    safe_mkdir(out_dir)
    return out_dir



# --------- Data classes ---------

@dataclass
class TrialSpec:
    idx: int
    combo: Dict[str, Any]
    base_overrides: List[str]
    out_dir: Path
    status: str = "pending"
    returncode: Optional[int] = None
    run_dir: Optional[Path] = None
    metrics: Dict[str, Any] = None  # type: ignore


# --------- Core runner ---------

def build_train_command(
    python_exec: str,
    trial: TrialSpec,
    train_module: str = "src.cli.train",
) -> List[str]:
    """
    Construct a subprocess command list for this trial.
    Always includes a `runtime.out_dir=<out_dir>` override so the train script writes there.
    """
    combo_overrides = dict_to_overrides(trial.combo)
    out_override = [f"runtime.out_dir={str(trial.out_dir)}"]
    cmd = [python_exec, "-m", train_module] + trial.base_overrides + combo_overrides + out_override
    return cmd


def run_trial(trial: TrialSpec, python_exec: str, log_root: Path) -> TrialSpec:
    """
    Execute a single trial and capture logs. Does not raise: records status in the TrialSpec.
    """
    log_dir = log_root / f"trial{trial.idx:04d}"
    safe_mkdir(log_dir)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    cmd = build_train_command(python_exec, trial)

    # Record the exact command
    (log_dir / "command.txt").write_text(" ".join(shlex.quote(c) for c in cmd), encoding="utf-8")

    try:
        with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
            proc = subprocess.run(cmd, stdout=out_f, stderr=err_f, check=False)
        trial.returncode = proc.returncode
        trial.status = "ok" if proc.returncode == 0 else "fail"
    except Exception as e:
        trial.status = "fail"
        trial.returncode = -1
        (log_dir / "exception.txt").write_text(repr(e), encoding="utf-8")

    # After run, try to find metrics in the designated out_dir
    # Some training scripts may nest outputs (e.g., save best.ckpt under out_dir)
    trial.run_dir = trial.out_dir
    trial.metrics = discover_metrics(trial.out_dir)
    if trial.status != "ok":
        try:
            err_text = stderr_path.read_text(encoding="utf-8", errors="ignore").strip()
            snippet = (err_text.splitlines() or [""])[-1][:240]
            trial.metrics = trial.metrics or {}
            trial.metrics["error"] = snippet
        except Exception:
            pass
    return trial


def aggregate_results(trials: List[TrialSpec], grid_out_dir: Path) -> Path:
    """
    Build a summary CSV across all trials.
    We include selected common metric keys if present.
    """
    # Collect all metric keys
    metric_keys = set()
    for t in trials:
        if t.metrics:
            metric_keys.update(t.metrics.keys())

    # Normalize and choose an ordering
    common_keys = ["mae", "rmse", "r2", "ece", "status"]
    other_keys = sorted(k for k in metric_keys if k not in common_keys)
    header = ["trial_idx", "status", "returncode", "run_dir"] + sorted(t.combo.keys()) + common_keys + other_keys

    summary_path = grid_out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for t in trials:
            row: Dict[str, Any] = {
                "trial_idx": t.idx,
                "status": t.status,
                "returncode": t.returncode,
                "run_dir": str(t.run_dir) if t.run_dir else "",
            }
            row.update(t.combo)
            # Fill metrics
            if t.metrics:
                for k, v in t.metrics.items():
                    row[k] = v
            writer.writerow(row)
    return summary_path




def parallel_map_star(func, items: List[tuple], jobs: int):
    """
    Multiprocessing starmap wrapper. `items` is a list of tuples,
    each tuple is expanded into func(*args).
    Falls back to serial if jobs <= 1.
    """
    if jobs <= 1:
        return [func(*it) for it in items]
    import multiprocessing as mp
    with mp.Pool(processes=jobs) as pool:
        return pool.starmap(func, items)



def parallel_map(func, items: List[Any], jobs: int) -> List[Any]:
    """
    Simple parallel map with multiprocessing. Falls back to serial if jobs==1.
    """
    if jobs <= 1:
        return [func(it) for it in items]
    import multiprocessing as mp

    with mp.Pool(processes=jobs) as pool:
        return pool.map(func, items)


# --------- CLI ---------

def parse_args(argv: Optional[List[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    """
    Parse known flags and return (args, remaining_overrides).
    Remaining overrides include Hydra-style key=value pairs the user wants to
    apply to every trial.
    """
    parser = argparse.ArgumentParser(description="Grid runner for Molecular GNN project")
    parser.add_argument("--grid", type=str, default=None,
                        help="Path to a grid YAML (e.g., configs/grids/esol_small_grid.yaml). "
                             "If not given, we look for +grids=<name> in overrides.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument("--train-module", type=str, default="src.cli.train",
                        help="Python module to launch for training (default: src.cli.train).")
    parser.add_argument("--python-exec", type=str, default=sys.executable,
                        help="Python executable to use for subprocess calls.")
    parser.add_argument("--out-root", type=str, default=str(DEFAULT_OUTPUTS_DIR),
                        help="Root directory for grid outputs (default: outputs/grids).")
    parser.add_argument("--dry-run", action="store_true", help="Expand the grid and exit without running trials.")
    parser.add_argument("--max-trials", type=int, default=None, help="Limit the number of trials to run.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip trials whose out_dir already contains metrics.json/metrics.csv.")

    # Split known vs unknown so we can forward unknown as base overrides
    args, unknown = parser.parse_known_args(argv)
    return args, unknown


def main(argv: Optional[List[str]] = None) -> None:
    args, base_overrides = parse_args(argv)

    # Extract +grids=name if provided in the overrides, and remove it from overrides
    grid_name_from_plus = extract_grid_name_from_plus(base_overrides)

    # Resolve grid YAML path
    grid_path: Optional[Path] = None
    if args.grid:
        grid_path = Path(args.grid)
        if not grid_path.exists():
            raise FileNotFoundError(f"Grid file not found: {grid_path}")
    elif grid_name_from_plus:
        grid_path = locate_grid_yaml(grid_name_from_plus)
    else:
        raise SystemExit(
            "You must specify a grid via `--grid <path>` OR `+grids=<name>` (e.g., +grids=esol_small_grid)."
        )

    grid_name = grid_path.stem
    grid_spec = load_yaml(grid_path)

    # Flatten and expand
    flat = flatten_dict(grid_spec)
    combos = cartesian_product(flat)

    # Respect --max-trials if set
    if args.max_trials is not None:
        combos = combos[: max(0, args.max_trials)]

    # Make grid output root
    grid_stamp = timestamp()
    out_root = Path(args.out_root)
    grid_out_dir = out_root / f"{grid_name}_{grid_stamp}"
    safe_mkdir(grid_out_dir)

    # Write manifest
    manifest = {
        "grid_name": grid_name,
        "grid_file": str(grid_path),
        "generated_at": dt.datetime.now().isoformat(),
        "out_dir": str(grid_out_dir),
        "n_trials": len(combos),
        "jobs": args.jobs,
        "train_module": args.train_module,
        "python_exec": args.python_exec,
        "base_overrides": base_overrides,
        "grid_spec_flattened": flat,
    }
    write_yaml(grid_out_dir / "manifest.yaml", manifest)
    write_json(grid_out_dir / "manifest.json", manifest)

    # Prepare trial specs
    trials: List[TrialSpec] = []
    for i, combo in enumerate(combos):
        out_dir = make_trial_out_dir(grid_out_dir, i, combo)
        trials.append(TrialSpec(idx=i, combo=combo, base_overrides=base_overrides, out_dir=out_dir))

    # Dry run?
    if args.dry_run:
        print(f"[DRY-RUN] Expanded {len(trials)} trials. Manifest written to: {grid_out_dir}")
        for t in trials[:10]:
            print(f"  - trial {t.idx}: overrides -> {dict_to_overrides(t.combo)} out_dir={t.out_dir}")
        if len(trials) > 10:
            print(f"  ... and {len(trials) - 10} more.")
        return

    # Optionally resume: skip trials with existing metrics
    if args.resume:
        kept = []
        skipped = 0
        for t in trials:
            found = discover_metrics(t.out_dir)
            if found.get("status") == "ok":
                t.status = "skipped"
                t.metrics = found
                kept.append(t)
                skipped += 1
            else:
                kept.append(t)
        trials = kept
        print(f"[RESUME] Skipped {skipped} trials with existing metrics.")

    # Logs root
    logs_root = grid_out_dir / "logs"
    safe_mkdir(logs_root)

    # Build argument tuples for starmap (avoid local closures for pickling)
    trial_args = [(t, args.python_exec, logs_root) for t in trials]


    if args.jobs > 1:
        results = parallel_map_star(run_trial, trial_args, jobs=args.jobs)
    else:
        results = [run_trial(t, args.python_exec, logs_root) for t in trials]
    # Aggregate
    summary_csv = aggregate_results(results, grid_out_dir)

    # Print a minimal report
    n_ok = sum(1 for t in results if t.status == "ok")
    n_fail = sum(1 for t in results if t.status == "fail")
    n_skip = sum(1 for t in results if t.status == "skipped")
    print("\n=== GRID RUN COMPLETE ===")
    print(f"Grid:          {grid_name}")
    print(f"Output dir:    {grid_out_dir}")
    print(f"Trials:        {len(results)} (ok={n_ok}, fail={n_fail}, skipped={n_skip})")
    print(f"Summary CSV:   {summary_csv}")
    print("Manifest:     ", grid_out_dir / "manifest.yaml")


if __name__ == "__main__":
    main()
