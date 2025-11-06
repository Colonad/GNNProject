# tests/test_sweep_smoke.py
import json, os, shutil, tempfile, subprocess, sys

def test_sweep_smoke():
    out = tempfile.mkdtemp(prefix="sweep_esol_gin_")
    try:
        cmd = [
            sys.executable, "-m", "src.train.sweep",
            "--dataset", "ESOL",
            "--model", "gin",
            "--split", "scaffold",
            "--epochs", "1",
            "--out-dir", out,
            "--hidden-dims", "32", "48",
            "--lrs", "0.001",
            "--quiet",
        ]
        subprocess.check_call(cmd)
        path = os.path.join(out, "sweep_summary.json")
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert "best" in data and "all" in data and len(data["all"]) >= 1
    finally:
        shutil.rmtree(out, ignore_errors=True)
