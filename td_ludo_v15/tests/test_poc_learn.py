"""Wraps the POC learning script as a pytest test.

Verifies that V15 can be trained via gradient descent — forward + backward
+ optimizer all work end-to-end without numerical issues. This is the
final gate before declaring the V15 code base ready for real training.
"""
from __future__ import annotations

import pytest

from td_ludo_v15.scripts.poc_learn import main as poc_main


@pytest.mark.slow
def test_poc_learn_strict_cpu():
    """100 states, 100 steps, CPU. Loss must drop ≥10%."""
    poc_main(["--n-states", "100", "--steps", "100", "--device", "cpu", "--strict"])


@pytest.mark.slow
def test_poc_learn_strict_mps():
    """Same as CPU test but on MPS. Skipped if MPS unavailable."""
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    poc_main(["--n-states", "100", "--steps", "100", "--device", "mps", "--strict"])
