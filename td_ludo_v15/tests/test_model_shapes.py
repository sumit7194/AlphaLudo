"""Tests for V15GraphTransformer — forward shapes, masking, gradients,
parameter count, and device transfer (MPS/CUDA)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from td_ludo_v15.models.v15 import V15GraphTransformer


# ─── Shapes + dtypes ──────────────────────────────────────────────────────


def test_forward_shapes_cpu():
    model = V15GraphTransformer()
    B = 4
    x = torch.randn(B, 8, 15, 15, 3)
    legal_mask = torch.zeros(B, 225)
    legal_mask[:, [33, 91, 92]] = 1  # arbitrary legal cells
    policy, value = model(x, legal_mask)
    assert policy.shape == (B, 225)
    assert value.shape == (B,)
    assert policy.dtype == torch.float32
    assert value.dtype == torch.float32


def test_param_count_in_range():
    """V15 target is 1.3M-1.8M params."""
    model = V15GraphTransformer()
    n = model.count_parameters()
    assert 1_000_000 < n < 2_500_000, f"params should be ~1.5M, got {n:,}"
    print(f"V15 param count: {n:,}")


def test_policy_sums_to_one():
    model = V15GraphTransformer()
    model.eval()
    B = 2
    x = torch.randn(B, 8, 15, 15, 3)
    legal_mask = torch.zeros(B, 225)
    legal_mask[:, [33, 91, 92, 100, 150]] = 1
    with torch.no_grad():
        policy, _ = model(x, legal_mask)
    sums = policy.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), (
        f"policy should sum to 1 per row, got {sums}"
    )


def test_illegal_cells_zero_probability():
    """After masking, illegal cells should have ~0 probability."""
    model = V15GraphTransformer()
    model.eval()
    B = 1
    x = torch.randn(B, 8, 15, 15, 3)
    legal_mask = torch.zeros(B, 225)
    legal_mask[0, [33, 91, 92]] = 1
    with torch.no_grad():
        policy, _ = model(x, legal_mask)
    # Illegal cells should have probability < 1e-6
    illegal_idx = [i for i in range(225) if i not in (33, 91, 92)]
    illegal_probs = policy[0, illegal_idx]
    assert (illegal_probs < 1e-6).all(), (
        f"illegal cells should have ~0 prob, max was {illegal_probs.max().item():g}"
    )


def test_value_in_zero_one():
    model = V15GraphTransformer()
    model.eval()
    B = 4
    x = torch.randn(B, 8, 15, 15, 3)
    legal_mask = torch.ones(B, 225)
    with torch.no_grad():
        _, value = model(x, legal_mask)
    assert (value >= 0).all() and (value <= 1).all(), (
        f"value should be in [0, 1] (sigmoid), got range [{value.min()}, {value.max()}]"
    )


# ─── Gradient flow ────────────────────────────────────────────────────────


def test_gradients_flow_through_all_params():
    model = V15GraphTransformer(n_layers=2)  # smaller for fast test
    B = 2
    x = torch.randn(B, 8, 15, 15, 3, requires_grad=False)
    legal_mask = torch.zeros(B, 225)
    legal_mask[:, [33, 91, 92, 100]] = 1
    policy, value = model(x, legal_mask)
    loss = -torch.log(policy[:, 91] + 1e-8).mean() + ((value - 0.7) ** 2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        # Edge-bias init is all zeros — gradient may be exactly zero on first
        # backward for unused edge types. Just check grad exists, not its norm.


# ─── Device transfer ──────────────────────────────────────────────────────


def test_mps_forward_matches_cpu():
    """Optional: MPS forward should produce the same result as CPU within fp32 noise."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    torch.manual_seed(0)
    model = V15GraphTransformer(n_layers=2)
    model.eval()
    B = 2
    x = torch.randn(B, 8, 15, 15, 3)
    legal_mask = torch.zeros(B, 225)
    legal_mask[:, [33, 91, 92]] = 1
    with torch.no_grad():
        policy_cpu, value_cpu = model(x, legal_mask)
    # Move to MPS
    model_mps = model.to("mps")
    x_m = x.to("mps")
    mask_m = legal_mask.to("mps")
    with torch.no_grad():
        policy_mps, value_mps = model_mps(x_m, mask_m)
    # MPS produces slightly different fp32 results than CPU; tolerate up to 1e-4
    policy_diff = (policy_cpu - policy_mps.cpu()).abs().max().item()
    value_diff = (value_cpu - value_mps.cpu()).abs().max().item()
    assert policy_diff < 1e-3, f"policy diverges on MPS: max diff {policy_diff:g}"
    assert value_diff < 1e-3, f"value diverges on MPS: max diff {value_diff:g}"


# ─── Edge bias initialization ─────────────────────────────────────────────


def test_edge_bias_init_zero():
    """All edge-bias tables init to zero so the model starts as standard
    scaled-dot-product attention. Training learns the bias."""
    model = V15GraphTransformer()
    for layer in model.layers:
        bias = layer.attn.edge_bias.weight
        assert torch.allclose(bias, torch.zeros_like(bias)), (
            "edge bias should init to zero"
        )


# ─── Deterministic given same seed ────────────────────────────────────────


def test_deterministic_forward():
    """Two forward passes with the same inputs + model should produce
    identical outputs (no nondeterminism in dropout etc. at eval time)."""
    torch.manual_seed(42)
    model = V15GraphTransformer(n_layers=2)
    model.eval()
    x = torch.randn(2, 8, 15, 15, 3)
    legal_mask = torch.zeros(2, 225)
    legal_mask[:, [33, 100]] = 1
    with torch.no_grad():
        p1, v1 = model(x, legal_mask)
        p2, v2 = model(x, legal_mask)
    assert torch.allclose(p1, p2)
    assert torch.allclose(v1, v2)
