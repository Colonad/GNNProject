# tests/test_ckpt_determinism_ema.py
import json, os, shutil, tempfile, subprocess, sys
import torch

def test_ckpt_eval_determinism_with_ema():
    out = tempfile.mkdtemp(prefix="ckpt_det_")
    try:
        # Train a tiny ESOL+GIN with EMA to produce a checkpoint
        cmd = [
            sys.executable, "-m", "src.train.loop",
            "--dataset", "ESOL",
            "--model", "gin",
            "--split", "scaffold",
            "--epochs", "2",
            "--ema",
            "--out-dir", out,
            "--quiet",
        ]
        subprocess.check_call(cmd)

        # Load checkpoint and evaluate twice; metrics should be identical
        from src.train.loop import build_argparser, build_dataloaders, build_model, _guess_dims_from_loader, evaluate
        parser = build_argparser()
        args = parser.parse_args(
            ["--dataset","ESOL","--model","gin","--split","scaffold","--epochs","2","--out-dir",out,"--quiet"]
        )
        device = torch.device("cpu")

        # Build loaders/model
        train_loader, val_loader, test_loader, meta = build_dataloaders(args)
        node_dim, edge_dim = _guess_dims_from_loader(train_loader)
        model = build_model(args, node_dim=node_dim, edge_dim=edge_dim).to(device)

        # Load weights + EMA shadow if present
        ckpt_path = os.path.join(out, "best.ckpt")
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location="cpu")

        model.load_state_dict(state["model"])

        # Recreate EMA object if shadow saved
        ema = None
        if "ema_shadow" in state:
            from src.train.loop import EMA
            ema = EMA(model, decay=0.999)
            for k, v in state["ema_shadow"].items():
                ema.shadow[k] = v.to(device)

        m1 = evaluate(model, test_loader, device, amp=False, ema=ema)
        m2 = evaluate(model, test_loader, device, amp=False, ema=ema)

        assert m1 == m2, f"Eval with EMA should be deterministic: {m1} vs {m2}"
    finally:
        shutil.rmtree(out, ignore_errors=True)
