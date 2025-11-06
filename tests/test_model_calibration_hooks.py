# tests/test_model_calibration_hooks.py
import math
import torch
import pytest
from torch import nn
from torch_geometric.data import Data, Batch

from src.models.gin import GINConfig, GINNet
from src.models.mpnn import MPNNConfig, MPNNNet


class MultiplyCalibrator(nn.Module):
    """Simple, deterministic 'calibrator' for testing:
    multiplies logits by a fixed scalar. We use this to detect whether
    calibration was applied (eval) or not (train/regression)."""
    def __init__(self, factor: float = 2.0):
        super().__init__()
        self.factor = float(factor)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.factor


def _toy_batch(edge_attr_dim: int | None = None):
    # Two tiny graphs with or without edge attributes
    x1 = torch.randn(3, 10)
    ei1 = torch.tensor([[0, 1, 2, 1, 2, 0],
                        [1, 2, 0, 0, 1, 2]], dtype=torch.long)
    x2 = torch.randn(4, 10)
    ei2 = torch.tensor([[0, 1, 2],
                        [1, 2, 3]], dtype=torch.long)
    if edge_attr_dim is not None:
        e1 = torch.randn(ei1.size(1), edge_attr_dim)
        e2 = torch.randn(ei2.size(1), edge_attr_dim)
        d1 = Data(x=x1, edge_index=ei1, edge_attr=e1)
        d2 = Data(x=x2, edge_index=ei2, edge_attr=e2)
    else:
        d1 = Data(x=x1, edge_index=ei1)
        d2 = Data(x=x2, edge_index=ei2)
    return Batch.from_data_list([d1, d2])


@pytest.mark.parametrize("Net, Cfg, edge_dim, use_edge_attr", [
    # GIN: no edge_attr
    (GINNet, GINConfig, None, False),
    # MPNN: requires edge_attr
    (MPNNNet, MPNNConfig, 6, True),
])
def test_calibrator_noop_for_regression(Net, Cfg, edge_dim, use_edge_attr):
    # out_dim=1 → regression → calibrator must be a no-op regardless of mode
    batch = _toy_batch(edge_attr_dim=edge_dim)
    cfg_kwargs = dict(
        in_dim=10,
        out_dim=1,
        hidden_dim=32,
        num_layers=3,
        pool="mean",
        dropout=0.0,
        batch_norm=False,
    )
    if Net is MPNNNet:
        cfg_kwargs["edge_dim"] = edge_dim
        cfg_kwargs["use_gru"] = False
    else:
        cfg_kwargs["edge_dim"] = edge_dim
        cfg_kwargs["use_edge_attr"] = use_edge_attr

    model = Net(Cfg(**cfg_kwargs))
    model.set_calibrator(MultiplyCalibrator(5.0), apply_on_eval=True)  # should not be used for out_dim=1

    # Train mode
    model.train()
    y_train = model(batch)
    assert y_train.shape == (2, 1)

    # Eval mode
    model.eval()
    y_eval = model(batch)
    assert y_eval.shape == (2, 1)

    # Because calibrator applies only when logits have C>1, outputs must match numerically
    torch.testing.assert_close(y_train, y_eval, rtol=0, atol=0)


@pytest.mark.parametrize("Net, Cfg, edge_dim, use_edge_attr", [
    (GINNet, GINConfig, None, False),     # GIN: logits without edge_attr
    (GINNet, GINConfig, 6, True),         # GINE: logits with edge_attr
    (MPNNNet, MPNNConfig, 6, True),       # MPNN: logits with edge_attr
])
def test_calibrator_applies_only_in_eval(Net, Cfg, edge_dim, use_edge_attr):
    # out_dim=3 → multi-class logits
    batch = _toy_batch(edge_attr_dim=edge_dim)
    cfg_kwargs = dict(
        in_dim=10,
        out_dim=3,
        hidden_dim=32,
        num_layers=3,
        pool="mean",
        dropout=0.0,
        batch_norm=False,
    )
    
    if Net is MPNNNet:
        cfg_kwargs["edge_dim"] = edge_dim
        cfg_kwargs["use_gru"] = False
    else:
        cfg_kwargs["edge_dim"] = edge_dim
        cfg_kwargs["use_edge_attr"] = use_edge_attr

    model = Net(Cfg(**cfg_kwargs))
    model.set_calibrator(MultiplyCalibrator(2.5), apply_on_eval=True)

    # Train mode → calibrator must NOT apply
    model.train()
    logits_train = model(batch)
    assert logits_train.shape == (2, 3)

    # Eval mode → calibrator MUST apply (multiply by factor)
    model.eval()
    logits_eval = model(batch)
    assert logits_eval.shape == (2, 3)

    # Because everything is deterministic across modes except the calibrator,
    # eval logits should be scaled by ~2.5 vs train logits (up to numerical noise).
    # We avoid equality because nonlinearities may introduce tiny differences;
    # check proportionality roughly by comparing norms and per-element ratios.
    ratio = (logits_eval / (logits_train + 1e-12)).mean().item()
    assert math.isfinite(ratio)
    assert 2.2 <= ratio <= 2.8, f"Expected ~2.5x after calibration; got {ratio:.3f}"

    # probs() should softmax logits (and include calibrator in eval)
    p_eval = model.probs(batch)
    assert p_eval.shape == (2, 3)
    row_sums = p_eval.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=0, atol=1e-5)


@pytest.mark.parametrize("Net, Cfg, edge_dim, use_edge_attr", [
    (GINNet, GINConfig, None, False),
    (MPNNNet, MPNNConfig, 6, True),
])
def test_clear_calibrator(Net, Cfg, edge_dim, use_edge_attr):
    # Verify clear_calibrator removes effect in eval for classification
    batch = _toy_batch(edge_attr_dim=edge_dim)
    cfg_kwargs = dict(
        in_dim=10,
        out_dim=3,
        hidden_dim=32,
        num_layers=3,
        pool="mean",
        dropout=0.0,
        batch_norm=False,
    )    
    
    
    
    
    
    
    if Net is MPNNNet:
        cfg_kwargs["edge_dim"] = edge_dim
        cfg_kwargs["use_gru"] = False
    else:
        cfg_kwargs["edge_dim"] = edge_dim
        cfg_kwargs["use_edge_attr"] = use_edge_attr

    model = Net(Cfg(**cfg_kwargs))
    model.set_calibrator(MultiplyCalibrator(3.0), apply_on_eval=True)

    model.eval()
    logits_with = model(batch)

    model.clear_calibrator()
    logits_without = model(batch)

    # Removing calibrator should change eval outputs noticeably
    diff = (logits_with - logits_without).abs().mean().item()
    assert diff > 1e-6
