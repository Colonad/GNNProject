# tests/test_cli_baseline.py
import os
import sys
import csv
import json
import shutil
import tempfile
import subprocess
from glob import glob


def _run(cmd):
    subprocess.check_call(cmd)


def test_cli_baseline_ridge_esol_smoke():
    tmp = tempfile.mkdtemp(prefix="cli_baseline_ridge_")
    try:
        out = os.path.join(tmp, "out")
        os.makedirs(out, exist_ok=True)

        # Run a quick Ridge baseline with Morgan features and target z-scoring
        _run(
            [
                sys.executable, "-m", "src.cli.baseline",
                "data.name=ESOL",
                "data.split=scaffold",
                "model.name=ridge",
                "feat.kind=morgan",
                "feat.morgan_bits=1024",
                "feat.morgan_radius=2",
                "feat.n_jobs=0",
                "train.standardize_targets=true",
                f"runtime.out_dir={out}",
                "runtime.dump_preds=true",
                "runtime.quiet=true",
            ]
        )

        # Artifacts
        summary = os.path.join(out, "summary.json")
        metrics = os.path.join(out, "metrics.csv")
        model = os.path.join(out, "model.pkl")
        preds_val = os.path.join(out, "preds_val.csv")
        preds_test = os.path.join(out, "preds_test.csv")
        cache_dir = os.path.join(out, "features_cache")

        assert os.path.exists(summary), "summary.json not found"
        assert os.path.exists(metrics), "metrics.csv not found"
        assert os.path.exists(preds_val), "preds_val.csv not found"
        assert os.path.exists(preds_test), "preds_test.csv not found"
        # model persistence is optional if joblib missing; only assert when present
        if os.path.exists(model):
            assert os.path.getsize(model) > 0

        # summary content
        with open(summary, "r", encoding="utf-8") as f:
            s = json.load(f)
        assert "val" in s and "test" in s, "summary missing metrics"
        for k in ("MAE", "RMSE", "MSE"):
            assert k in s["val"] and k in s["test"], f"metric {k} missing"

        # metrics.csv has header and a row
        with open(metrics, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) >= 2, "metrics.csv should have header + at least one row"

        # preds CSVs have header and at least one row
        for p in (preds_val, preds_test):
            with open(p, "r", encoding="utf-8") as f:
                r = csv.reader(f)
                header = next(r)
                data_rows = list(r)
            assert header == ["smiles", "y_true", "y_pred"], f"unexpected preds header in {p}: {header}"
            assert len(data_rows) > 0, f"no prediction rows in {p}"

        # features_cache should have some .npz files for splits
        if os.path.isdir(cache_dir):
            cached = glob(os.path.join(cache_dir, "*.npz"))
            assert len(cached) >= 2, "expected cached feature matrices in features_cache/"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cli_baseline_rf_tiny_sweep():
    tmp = tempfile.mkdtemp(prefix="cli_baseline_rf_")
    try:
        out = os.path.join(tmp, "out")
        os.makedirs(out, exist_ok=True)

        # Tiny RF sweep to exercise sweep path (kept small for test runtime)
        _run(
            [
                sys.executable, "-m", "src.cli.baseline",
                "data.name=ESOL",
                "data.split=scaffold",
                "model.name=random_forest",
                "model.rf_n_estimators_grid=[10,20]",
                "model.rf_max_depth_grid=[None,10]",
                "feat.kind=morgan",
                "feat.morgan_bits=512",
                "feat.morgan_radius=2",
                "feat.n_jobs=0",
                "train.sweep=true",
                "train.standardize_targets=true",
                "train.seed=1",
                f"runtime.out_dir={out}",
                "runtime.dump_preds=false",
                "runtime.quiet=true",
            ]
        )

        # Check summary contains sweep trials and best metrics
        summary = os.path.join(out, "summary.json")
        assert os.path.exists(summary), "summary.json not found"
        with open(summary, "r", encoding="utf-8") as f:
            s = json.load(f)
        assert "sweep" in s and s["sweep"]["enabled"] is True, "sweep not recorded as enabled"
        trials = s["sweep"]["trials"]
        assert isinstance(trials, list) and len(trials) >= 1, "no sweep trials recorded"
        assert "val" in s and "test" in s, "summary missing best metrics"
        for k in ("MAE", "RMSE", "MSE"):
            assert k in s["val"] and k in s["test"], f"metric {k} missing in best summary"

        # metrics.csv exists and has at least one row
        metrics = os.path.join(out, "metrics.csv")
        assert os.path.exists(metrics), "metrics.csv not found"
        with open(metrics, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) >= 2, "metrics.csv should have header + at least one row"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
