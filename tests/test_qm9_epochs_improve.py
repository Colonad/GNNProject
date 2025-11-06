# tests/test_qm9_epochs_improve.py
import json, os, shutil, tempfile, subprocess, sys

def test_qm9_more_epochs_nonworse_val_mae():
    out_base = tempfile.mkdtemp(prefix="qm9_improve_")
    try:
        out1 = os.path.join(out_base, "e1")
        out3 = os.path.join(out_base, "e3")

        # Run for 1 epoch
        cmd1 = [
            sys.executable, "-m", "src.train.loop",
            "--dataset", "QM9",
            "--model", "mpnn",
            "--split", "random",
            "--limit-n", "800",
            "--target", "U0",
            "--epochs", "1",
            "--batch-size", "256",
            "--standardize-targets",
            "--seed", "42",
            "--out-dir", out1,
            "--quiet",
        ]
        subprocess.check_call(cmd1)

        # Run for 3 epochs (same seed, should be <= val MAE)
        cmd3 = [
            sys.executable, "-m", "src.train.loop",
            "--dataset", "QM9",
            "--model", "mpnn",
            "--split", "random",
            "--limit-n", "800",
            "--target", "U0",
            "--epochs", "3",
            "--batch-size", "256",
            "--standardize-targets",
            "--seed", "42",
            "--out-dir", out3,
            "--quiet",
        ]
        subprocess.check_call(cmd3)

        with open(os.path.join(out1, "summary.json"), "r", encoding="utf-8") as f:
            s1 = json.load(f)
        with open(os.path.join(out3, "summary.json"), "r", encoding="utf-8") as f:
            s3 = json.load(f)

        mae1 = float(s1["best_val_MAE"])
        mae3 = float(s3["best_val_MAE"])
        # Allow tie, but 3 epochs should not be worse than 1 epoch
        assert mae3 <= mae1 + 1e-8, f"Expected <= best val MAE when training longer, got {mae3} vs {mae1}"
    finally:
        shutil.rmtree(out_base, ignore_errors=True)
