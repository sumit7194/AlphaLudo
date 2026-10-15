"""MLX port == PyTorch V17Net (float32) on real 2P and 4P encodings."""
import numpy as np
import pytest
import torch

pytest.importorskip("mlx.core")
from v17.mlx_model import MlxV17
from v17.model import V17Net
from v17.tests.test_model import batch


@pytest.mark.parametrize("four", [False, True])
def test_mlx_matches_torch(four):
    torch.manual_seed(1)
    net = V17Net()
    # Non-trivial weights: perturb the zero-initialised edge biases.
    with torch.no_grad():
        for l in net.layers:
            l.attn.edge_bias.weight.normal_(0, 0.5)
    net.train(False)
    x, m = batch(four, n=256, seed=3)
    with torch.no_grad():
        p0, v0 = net(x, m)
    p1, v1 = MlxV17(net)(x.numpy(), m.numpy())
    assert np.abs(p1 - p0.numpy()).max() < 2e-5
    assert np.abs(v1 - v0.numpy()).max() < 2e-5
    assert (p1.argmax(1) == p0.numpy().argmax(1)).all()
