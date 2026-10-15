"""V17Net shapes, masking, and active-seat handling on real encodings."""
import numpy as np
import torch
import alphaludo_rs

from v17.model import V17Net, count_params


def batch(four: bool, n=64, seed=0):
    rng = np.random.default_rng(seed)
    active = [True] * 4 if four else [True, False, True, False]
    seats = [s for s in range(4) if active[s]]
    X, M = [], []
    while len(X) < n:
        pos = np.full((4, 4), -1, dtype=np.int8)
        for s in seats:
            pos[s] = rng.integers(-1, 50, 4)
        cp = int(rng.choice(seats)); dice = 6
        x, m = alphaludo_rs.encode_unified(pos, cp, dice, active, [0, 0, 0, 0])
        if m.sum() >= 1:
            X.append(x); M.append(m)
    return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(M))


def test_shapes_and_masks():
    torch.manual_seed(0)
    net = V17Net()
    assert 550_000 < count_params(net) < 650_000
    for four in (False, True):
        x, m = batch(four)
        p, v = net(x, m)
        assert p.shape == (64, 225) and v.shape == (64, 4)
        assert torch.allclose(p.sum(-1), torch.ones(64), atol=1e-5)
        assert (p[m < 0.5] == 0).all()                    # illegal cells get no mass
        assert torch.allclose(v.sum(-1), torch.ones(64), atol=1e-5)
        if not four:                                      # 2P: seats next/prev are inactive
            assert (v[:, 1] == 0).all() and (v[:, 3] == 0).all()
        else:
            assert (v > 0).all()


def test_backward():
    net = V17Net()
    x, m = batch(False, n=16)
    p, v = net.forward_logits(x, m)
    loss = -(torch.log_softmax(p, -1) * (m / m.sum(-1, keepdim=True))).sum(-1).mean() \
        + torch.nn.functional.cross_entropy(v, torch.zeros(16, dtype=torch.long))
    loss.backward()
    assert all(q.grad is not None for q in net.parameters())
