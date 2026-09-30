"""Unit tests for V15_4P_GraphTransformer architecture."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from td_ludo_v15.models import V15_4P_GraphTransformer


def test_model_4p_param_count():
    model = V15_4P_GraphTransformer(d_model=128, n_heads=4, n_layers=4, ffn_dim=256)
    params = model.count_parameters()
    # Exactly 587,691 parameters
    assert 550_000 < params < 650_000, f"Expected ~588K params, got {params}"


def test_model_4p_forward_shapes():
    model = V15_4P_GraphTransformer(d_model=128, n_heads=4, n_layers=4, ffn_dim=256)
    model.eval()

    B = 4
    x = torch.randn(B, 1, 15, 15, 5)
    legal_mask = torch.zeros(B, 225)
    legal_mask[:, [10, 32, 55]] = 1.0  # 3 legal moves per batch item

    with torch.no_grad():
        policy, value = model(x, legal_mask=legal_mask)

    # Policy shape: (B, 225)
    assert policy.shape == (B, 225)
    # Value shape: (B, 4)
    assert value.shape == (B, 4)

    # Probabilities must sum to 1.0
    torch.testing.assert_close(policy.sum(dim=-1), torch.ones(B), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(value.sum(dim=-1), torch.ones(B), atol=1e-5, rtol=1e-5)

    # Illegal moves must have probability 0.0
    for b in range(B):
        for c in range(225):
            if c not in (10, 32, 55):
                assert policy[b, c].item() == 0.0


def test_model_4p_backward():
    model = V15_4P_GraphTransformer(d_model=128, n_heads=4, n_layers=4, ffn_dim=256)
    model.train()

    B = 2
    x = torch.randn(B, 15, 15, 5)  # 4D tensor auto-unsqeezes
    legal_mask = torch.ones(B, 225)

    policy, value, pol_logits, val_logits = model(x, legal_mask=legal_mask, return_logits=True)

    # Target actions: index 10 and 20
    target_action = torch.tensor([10, 20], dtype=torch.long)
    target_winner = torch.tensor([0, 2], dtype=torch.long)

    loss_pol = F.cross_entropy(pol_logits, target_action)
    loss_val = F.cross_entropy(val_logits, target_winner)
    loss = loss_pol + loss_val

    loss.backward()

    # Verify every parameter with requires_grad has a valid gradient
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
