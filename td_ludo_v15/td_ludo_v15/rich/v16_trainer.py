"""V16RichTrainer — V15RichTrainer with a routed auxiliary objective.

Subclasses the v15 trainer and overrides exactly TWO hooks. Everything that
defines the RL update — PPO clipping, GAE, terminal-only rewards, EMA return
normalisation, entropy bonus, win-BCE value head, gradient clipping, optimizer
step cadence — is inherited, unmodified, from V15RichTrainer. That is
deliberate: the previous attempt at this idea reimplemented the RL loop and
diverged from the champion's pipeline in ~10 independent ways, which made its
212k-game result uninterpretable. A subclass with two hooks cannot drift.

THE TWO ARMS
------------
Both arms use the same architecture, same parameter count, same aux head, same
aux loss, and the same two-backward structure. ONLY the mode differs:

  routed  MAIN for the RL backward, AUX for the auxiliary backward.
          => `a` receives RL gradient only, `b` receives aux gradient only.
          This is the owner's idea.

  mixed   JOINT for both backwards. Since w = a*(1+tanh(b)) depends on both
          properties, each backward reaches BOTH — so the two signals share
          all parameters.
          This is the control: it isolates "routing signals to separate
          properties helps" from "having two properties / an aux loss helps".

Because the arms differ only in the mode string, nothing else can explain a gap.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .v15_trainer import V15RichTrainer
from ..models.v16 import set_mode, MAIN, AUX, JOINT


class V16RichTrainer(V15RichTrainer):
    """V15 PPO trainer + a second, separately-routed auxiliary objective.

    Args (beyond V15RichTrainer's):
        routed:      True  -> a<-RL, b<-aux (the idea)
                     False -> both losses reach both properties (control)
        aux_coeff:   weight on the auxiliary BCE.
        aux_pos_weight: fixed positive weighting, or None (default) to ADAPT.

    ON THE POSITIVE WEIGHTING
    -------------------------
    The tactical events are SPARSE by nature — that is the point of the idea
    ("cuts, home, or others"). Unweighted BCE is then minimised by always
    predicting "no", the gradient into `b` collapses, and the arm dies for a
    reason that has nothing to do with the hypothesis: a fake KILL caused by
    class imbalance.

    A FIXED weight is also wrong here. Measured positive rates:
        teacher-game SL cache : 3.2% capture / 13.7% danger
        random self-play      : 4.9% capture / 24.7% danger
    Strong play avoids danger, so the rate MOVES as the policy improves — and
    it will move again as the opponent mix does. Pinning a constant would
    silently re-weight the aux objective over the course of the run. So the
    rate is tracked with an EMA (same idiom as the trainer's return
    normalisation) and the weight is derived from it each step.
    """

    RATE_EMA_ALPHA = 0.01
    POS_W_CLAMP = (1.0, 100.0)

    def __init__(self, *args, routed: bool = True, aux_coeff: float = 1.0,
                 aux_pos_weight: Optional[tuple] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.routed = bool(routed)
        self.aux_coeff = float(aux_coeff)
        self._fixed_pos_w = (
            torch.tensor(aux_pos_weight, dtype=torch.float32, device=self.device)
            if aux_pos_weight is not None else None)
        self._aux_rate_ema: Optional[torch.Tensor] = None

    def _pos_weight(self, mb_aux: torch.Tensor) -> torch.Tensor:
        if self._fixed_pos_w is not None:
            return self._fixed_pos_w
        rate = mb_aux.mean(0).detach()
        if self._aux_rate_ema is None:
            self._aux_rate_ema = rate.clone()
        else:
            a = self.RATE_EMA_ALPHA
            self._aux_rate_ema = (1 - a) * self._aux_rate_ema + a * rate
        r = self._aux_rate_ema.clamp_min(1e-6)
        return ((1.0 - r) / r).clamp(*self.POS_W_CLAMP)

    # ── hook 1: which property receives the RL backward ─────────────────────
    def _mode_main(self) -> None:
        set_mode(self.model, MAIN if self.routed else JOINT)

    # ── hook 2: the auxiliary backward ──────────────────────────────────────
    def _aux_backward(self, mb_states, mb_masks, mb_aux) -> float:
        if mb_aux is None or self.aux_coeff == 0.0:
            return 0.0
        set_mode(self.model, AUX if self.routed else JOINT)
        _, _, aux_logits = self.model(mb_states, mb_masks, want_aux=True)
        loss = self.aux_coeff * F.binary_cross_entropy_with_logits(
            aux_logits, mb_aux, pos_weight=self._pos_weight(mb_aux))
        # Accumulates into the SAME .grad buffers as the RL backward, so both
        # objectives share one clip_grad_norm_ and one optimizer.step().
        loss.backward()
        return float(loss.item())

    # ── diagnostics ─────────────────────────────────────────────────────────
    def gate_stats(self) -> Optional[dict]:
        fn = getattr(self.model, "gate_stats", None)
        return fn() if callable(fn) else None
