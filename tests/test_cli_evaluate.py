# tests/test_cli_evaluate.py
import os
import sys
import csv
import json
import shutil
import tempfile
import subprocess


def _run(cmd):
    # Helpful wrapper to raise nice errors on failure
    subprocess.check_call(cmd)


def test_cli_evaluate_end_to_end_single_and_ensemble():
    tmp = tempfile.mkdtemp(prefix="cli_eval_")
    try:
        # 1) Train a tiny ESOL+GIN run to generate artifacts
        out = os.path.join(tmp, "train_out")
        os.makedirs(out, exist_ok=True)
        _run(
            [
                sys.executable, "-m", "src.train.loop",
                "--dataset", "ESOL",
                "--model", "gin",
                "--split", "scaffold",
                "--epochs", "1",
                "--out-dir", out,
                "--quiet",
            ]
        )

        # Check training artifacts
        ckpt = os.path.join(out, "best.ckpt")
        summ = os.path.join(out, "summary.json")
        assert os.path.exists(ckpt), "best.ckpt not found after training"
        assert os.path.exists(summ), "summary.json not found after training"
        with open(summ, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert "target_norm" in summary, "summary.json missing target_norm"

        # 2) Evaluate using runtime.out_dir and dump preds
        preds_csv = os.path.join(out, "preds.csv")
        _run(
            [
                sys.executable, "-m", "src.cli.evaluate",
                f"runtime.out_dir={out}",
                "eval_cfg.which_split=test",
                "eval_cfg.dump_preds=true",
                f"eval_cfg.preds_filename={preds_csv}",
                "eval_cfg.use_ema=true",
            ]
        )
        assert os.path.exists(preds_csv), "preds.csv not written by evaluate CLI"

        # Verify CSV has header and at least one row
        with open(preds_csv, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r)
            rows = list(r)
        assert header == ["row_idx", "y_true", "y_pred"], f"Unexpected header: {header}"
        assert len(rows) > 0, "No predictions written"

        # 3) Evaluate again with explicit ckpt path and EMA disabled
        _run(
            [
                sys.executable, "-m", "src.cli.evaluate",
                f"eval_cfg.ckpt_path={ckpt}",
                "eval_cfg.use_ema=false",
                "eval_cfg.which_split=test",
                "eval_cfg.dump_preds=false",
                f"runtime.out_dir={out}",
            ]
        )

        # 4) Ensemble: duplicate checkpoint and evaluate via glob
        dup1 = os.path.join(out, "copy1.ckpt")
        dup2 = os.path.join(out, "copy2.ckpt")
        shutil.copy2(ckpt, dup1)
        shutil.copy2(ckpt, dup2)

        _run(
            [
                sys.executable, "-m", "src.cli.evaluate",
                f"runtime.out_dir={out}",
                "eval_cfg.which_split=test",
                f"eval_cfg.ensemble_glob={os.path.join(out, '*.ckpt')}",
                "eval_cfg.dump_preds=false",
            ]
        )

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
