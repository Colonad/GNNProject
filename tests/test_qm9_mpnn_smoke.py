# tests/test_qm9_mpnn_smoke.py
import json, os, shutil, tempfile, subprocess, sys

def test_qm9_mpnn_smoke_subset():
    out = tempfile.mkdtemp(prefix="run_qm9_mpnn_")
    try:
        cmd = [
            sys.executable, "-m", "src.train.loop",
            "--dataset", "QM9",
            "--model", "mpnn",
            "--split", "random",
            "--limit-n", "800",
            "--target", "U0",
            "--epochs", "1",
            "--batch-size", "256",
            "--standardize-targets",
            "--out-dir", out,
            "--quiet",
        ]
        subprocess.check_call(cmd)

        # Artifacts exist
        summary_path = os.path.join(out, "summary.json")
        metrics_path = os.path.join(out, "metrics.csv")
        ckpt_path = os.path.join(out, "best.ckpt")
        assert os.path.exists(summary_path), "summary.json missing"
        assert os.path.exists(metrics_path), "metrics.csv missing"
        assert os.path.exists(ckpt_path), "best.ckpt missing"

        # Summary is well-formed
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert "test" in summary and "best_val_MAE" in summary
        tn = summary.get("target_norm", None)
        assert isinstance(tn, dict) and tn["std"] > 0
    finally:
        shutil.rmtree(out, ignore_errors=True)
