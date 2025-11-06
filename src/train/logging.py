# src/train/logging.py
from __future__ import annotations

"""
Training/Evaluation logging utilities.

Features
--------
- CSV logging per epoch (train/val/test) to `metrics.csv`
- Optional TensorBoard (tb) logging; gracefully no-ops if TB isn't available
- Run metadata capture: params.json, run_info.json, summary.json
- Robust conversion of numpy / torch scalars to Python floats
- Stable CSV header with common fields; consistent precision & flushing

Design goals
------------
- Minimal coupling: loggers accept plain dict metrics from your loop
- DoD: ensure `outputs/*/metrics.csv` exists and is non-empty by first log call
- Silence-friendly: TensorBoard is optional and auto-disabled if unavailable
"""

import csv
import json
import os
import time
import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

# Optional torch import for dtype checks (we do not *require* torch here)
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False

# Optional TB import
_TB_WRITER_CLS = None
try:
    # Prefer torch.utils.tensorboard if available
    from torch.utils.tensorboard import SummaryWriter as _TorchTBWriter
    _TB_WRITER_CLS = _TorchTBWriter  # type: ignore
except Exception:
    try:
        # Fallback to tensorboardX if installed
        from tensorboardX import SummaryWriter as _TBXWriter  # type: ignore
        _TB_WRITER_CLS = _TBXWriter
    except Exception:
        _TB_WRITER_CLS = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

Scalar = Union[int, float, bool, str]
Number = Union[int, float]


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _as_float(x: Any) -> Optional[float]:
    """Try to convert numbers (numpy / torch / python) to float; return None if not convertible."""
    # torch scalar
    if _HAS_TORCH and isinstance(x, (torch.Tensor,)):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return None
    # numpy scalar or array with single element
    try:
        import numpy as _np  # local import, optional
        if isinstance(x, _np.generic):
            return float(x)
        if isinstance(x, _np.ndarray):
            if x.size == 1:
                return float(x.reshape(-1)[0])
            return None
    except Exception:
        pass
    # python native
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _sanitize_value(v: Any, float_precision: int) -> Any:
    """
    Convert v into CSV/TB-friendly scalar.
    Numbers -> round to precision.
    Bool/str -> pass through.
    Unsupported types -> stringified JSON (best-effort).
    """
    f = _as_float(v)
    if f is not None:
        # Keep NaN/inf as strings to avoid CSV readers choking
        if f != f:  # NaN
            return "NaN"
        if f in (float("inf"), float("-inf")):
            return "inf" if f > 0 else "-inf"
        return round(f, int(float_precision))
    if isinstance(v, (bool, str)):
        return v
    # Dicts or lists: try stable JSON; fallback to str
    try:
        return json.dumps(v, sort_keys=True)
    except Exception:
        return str(v)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# --------------------------------------------------------------------------
# CSV Logger
# --------------------------------------------------------------------------

@dataclass
class CSVLogger:
    """
    Append-only CSV logger with stable header.

    The header contains:
        ["step", "epoch", "split", "lr", "wall_time", *metric_keys]

    Assumptions:
    - Your loop passes consistent metric keys for subsequent rows.
    - If new keys appear later, they are written as additional columns at the end
      (we detect and extend header in-memory and write them too).
    """
    out_dir: str
    filename: str = "metrics.csv"
    float_precision: int = 6
    auto_flush: bool = True

    # internal
    _path: str = field(init=False)
    _f: Optional[Any] = field(init=False, default=None)
    _writer: Optional[csv.DictWriter] = field(init=False, default=None)
    _header: List[str] = field(init=False, default_factory=list)
    _opened: bool = field(init=False, default=False)
    _seen_metric_keys: List[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        _ensure_dir(self.out_dir)
        self._path = os.path.join(self.out_dir, self.filename)
        # Defer file open until first write so the file is created only when used.

    # ---------------------------- public API ---------------------------- #

    def log_epoch(
        self,
        *,
        epoch: Optional[int],
        split: str,
        metrics: Mapping[str, Any],
        step: Optional[int] = None,
        lr: Optional[Number] = None,
        wall_time: Optional[float] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Log a row for a given split ("train" | "val" | "test").

        Args:
            epoch: integer epoch index (or None for final rows)
            split: "train" | "val" | "test" | etc.
            metrics: Dict of metric_name -> value (MAE, RMSE, loss, etc.)
            step: global step (optional)
            lr: learning rate (optional)
            wall_time: seconds since run start (optional)
            extra: additional columns to include
        """
        if extra is None:
            extra = {}

        # Lazily open the file on first write
        if not self._opened:
            self._open_writer(initial_metrics=metrics, extra=extra)

        # Merge all values and sanitize
        row: Dict[str, Any] = {
            "step": step if step is not None else "",
            "epoch": epoch if epoch is not None else "",
            "split": split,
            "lr": _sanitize_value(lr, self.float_precision) if lr is not None else "",
            "wall_time": _sanitize_value(wall_time, self.float_precision) if wall_time is not None else "",
        }

        # Ensure new metric keys are added if first seen
        for k in metrics.keys():
            if k not in self._seen_metric_keys:
                self._seen_metric_keys.append(k)
                if k not in self._header:
                    self._header.append(k)
                    # Recreate writer with extended header
                    self._reopen_with_header()

        # Also include extras
        for k in extra.keys():
            if k not in self._header:
                self._header.append(k)
                self._reopen_with_header()

        # Fill metrics and extras (sanitized)
        for k in self._seen_metric_keys:
            row[k] = _sanitize_value(metrics.get(k, ""), self.float_precision)
        for k, v in extra.items():
            row[k] = _sanitize_value(v, self.float_precision)

        # Write row
        assert self._writer is not None
        self._writer.writerow(row)
        if self.auto_flush and self._f is not None:
            self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            try:
                self._f.flush()
            except Exception:
                pass
            try:
                self._f.close()
            except Exception:
                pass
        self._f = None
        self._writer = None
        self._opened = False

    # ---------------------------- internals ----------------------------- #

    def _open_writer(self, initial_metrics: Mapping[str, Any], extra: Mapping[str, Any]) -> None:
        """Open CSV, write header if file is empty."""
        is_new = not os.path.exists(self._path) or os.path.getsize(self._path) == 0
        self._f = open(self._path, mode="a", newline="", encoding="utf-8")
        # baseline header
        base = ["step", "epoch", "split", "lr", "wall_time"]
        # initial metric keys
        self._seen_metric_keys = list(initial_metrics.keys())
        # compose header
        self._header = base + self._seen_metric_keys + list(extra.keys())

        self._writer = csv.DictWriter(self._f, fieldnames=self._header, extrasaction="ignore")
        if is_new:
            self._writer.writeheader()
            self._f.flush()
        self._opened = True

    def _reopen_with_header(self) -> None:
        """Recreate DictWriter with updated header (keeps the same file handle)."""
        if self._f is None:
            return
        # Recreate writer with new header; no header rewrite mid-file (CSV readers handle missing fields as empty)
        self._writer = csv.DictWriter(self._f, fieldnames=self._header, extrasaction="ignore")


# --------------------------------------------------------------------------
# TensorBoard logger (optional)
# --------------------------------------------------------------------------

@dataclass
class TensorBoardLogger:
    """
    Thin wrapper around SummaryWriter (torch TB or tensorboardX).
    No-ops if neither backend is available.
    """
    out_dir: str
    enabled: bool = True

    _writer: Optional[Any] = field(init=False, default=None)
    _step: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not self.enabled or _TB_WRITER_CLS is None:
            self._writer = None
            return
        _ensure_dir(self.out_dir)
        try:
            self._writer = _TB_WRITER_CLS(log_dir=self.out_dir)
        except Exception:
            self._writer = None

    def log_scalars(self, tag_prefix: str, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        if self._writer is None:
            return
        s = self._step if step is None else int(step)
        for k, v in metrics.items():
            f = _as_float(v)
            if f is not None:
                try:
                    self._writer.add_scalar(f"{tag_prefix}/{k}", f, global_step=s)
                except Exception:
                    pass
        if step is None:
            self._step += 1

    def close(self) -> None:
        w = self._writer
        if w is not None:
            try:
                w.flush()
            except Exception:
                pass
            try:
                w.close()
            except Exception:
                pass
        self._writer = None


# --------------------------------------------------------------------------
# Run recorder (params & summary)
# --------------------------------------------------------------------------

@dataclass
class RunRecorder:
    """
    Persists run metadata and final summary artifacts alongside metrics.csv.

    Files produced in out_dir:
        - params.json     : the (possibly nested) dict of run hyperparameters/config
        - run_info.json   : minimal metadata: start_time, git info (if you pass), hostname (optional)
        - summary.json    : final test/val/train metrics + best epoch, best-by key, and paths

    Tips:
        - Call `start()` right after creating the run folder.
        - Call `finalize()` after training to persist a compact summary.
    """
    out_dir: str
    float_precision: int = 6

    def start(self, *, params: Mapping[str, Any], extra_info: Optional[Mapping[str, Any]] = None) -> None:
        _ensure_dir(self.out_dir)

        # params.json
        try:
            with open(os.path.join(self.out_dir, "params.json"), "w", encoding="utf-8") as f:
                json.dump(_sanitize_nested(params, self.float_precision), f, indent=2)
        except Exception:
            pass

        # run_info.json
        info = {
            "start_time": _now_iso(),
            "cwd": os.getcwd(),
        }
        if extra_info:
            info.update(_sanitize_nested(extra_info, self.float_precision))
        try:
            with open(os.path.join(self.out_dir, "run_info.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)
        except Exception:
            pass

    def finalize(
        self,
        *,
        best_epoch: Optional[int],
        best_by: Optional[str],
        train: Optional[Mapping[str, Any]] = None,
        val: Optional[Mapping[str, Any]] = None,
        test: Optional[Mapping[str, Any]] = None,
        paths: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        summary = {
            "end_time": _now_iso(),
            "best_epoch": best_epoch,
            "best_by": best_by,
            "train": _sanitize_nested(train or {}, self.float_precision),
            "val": _sanitize_nested(val or {}, self.float_precision),
            "test": _sanitize_nested(test or {}, self.float_precision),
            "paths": paths or {},
        }
        if extra:
            summary["extra"] = _sanitize_nested(extra, self.float_precision)
        try:
            with open(os.path.join(self.out_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass


# --------------------------------------------------------------------------
# Public convenience: build a default logging bundle
# --------------------------------------------------------------------------

@dataclass
class LoggingBundle:
    """
    Convenience container to hold all loggers for a run.
    Typical usage in your loop:

        logs = build_logging_bundle(out_dir, tb=True, params=cfg_dict)
        logs.recorder.start(params=cfg_dict)
        for epoch in range(epochs):
            ...
            logs.csv.log_epoch(epoch=epoch, split="train", metrics=train_metrics, step=step, lr=lr, wall_time=t)
            logs.tb.log_scalars("train", train_metrics, step=step)
            ...
        logs.recorder.finalize(best_epoch=best_ep, best_by="val/MAE", val=val_best, test=test_metrics, paths=paths)
        logs.close()
    """
    csv: CSVLogger
    tb: TensorBoardLogger
    recorder: RunRecorder

    def close(self) -> None:
        try:
            self.csv.close()
        except Exception:
            pass
        try:
            self.tb.close()
        except Exception:
            pass


def build_logging_bundle(
    out_dir: str,
    *,
    enable_tb: bool = True,
    float_precision: int = 6,
    params: Optional[Mapping[str, Any]] = None,
    extra_info: Optional[Mapping[str, Any]] = None,
) -> LoggingBundle:
    _ensure_dir(out_dir)
    csv_logger = CSVLogger(out_dir=out_dir, filename="metrics.csv", float_precision=int(float_precision))
    tb_logger = TensorBoardLogger(out_dir=os.path.join(out_dir, "tb"), enabled=bool(enable_tb))
    recorder = RunRecorder(out_dir=out_dir, float_precision=int(float_precision))
    if params is not None:
        recorder.start(params=params, extra_info=extra_info or {})
    return LoggingBundle(csv=csv_logger, tb=tb_logger, recorder=recorder)


# --------------------------------------------------------------------------
# Internal: sanitize nested structures for JSON
# --------------------------------------------------------------------------

def _sanitize_nested(obj: Any, float_precision: int) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_nested(v, float_precision) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nested(v, float_precision) for v in obj]
    sv = _sanitize_value(obj, float_precision)
    return sv
