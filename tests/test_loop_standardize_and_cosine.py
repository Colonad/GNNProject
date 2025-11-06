# tests/test_loop_standardize_and_cosine.py
import json, os, shutil, tempfile, subprocess, sys

def test_loop_standardize_and_cosine_runs():
    out = tempfile.mkdtemp(prefix="run_esol_gin_")
    try:
        cmd = [
            sys.executable, "-m", "src.train.loop",
            "--dataset", "ESOL",
            "--model", "gin",
            "--split", "scaffold",
            "--epochs", "2",
            "--standardize-targets",
            "--scheduler", "cosine_warmup",
            "--warmup-epochs", "1",
            "--out-dir", out,
            "--quiet",
        ]
        subprocess.check_call(cmd)

        summary_path = os.path.join(out, "summary.json")
        assert os.path.exists(summary_path), "summary.json missing"

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        tn = summary.get("target_norm", None)
        assert isinstance(tn, dict) and "mean" in tn and "std" in tn, "target_norm absent in summary"
        assert tn["std"] > 0, "std must be > 0"

        assert os.path.exists(os.path.join(out, "best.ckpt")), "best.ckpt missing"
        assert os.path.exists(os.path.join(out, "metrics.csv")), "metrics.csv missing"
    finally:
        shutil.rmtree(out, ignore_errors=True)
