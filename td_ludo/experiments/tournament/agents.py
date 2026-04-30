"""Agent abstractions for the tournament.

Three competitor types, all behind a common `select_move(game, legal,
consec_sixes) -> int` interface:

  - HistAgent: backed by OpponentRegistry. Used for V6_big, V6_1, V6_3,
    V10 etc. Each gets its correct encoder + architecture for free.

  - ModelAgent: a custom checkpoint + architecture. Used for V12.2,
    distilled-14ch student, or any checkpoint not in the registry.
    Caller supplies an arch preset name that maps to (architecture,
    encoder, kwargs).

  - BotAgent: hand-coded bots from heuristic_bot.py. Used for Expert /
    Heuristic / Aggressive / Defensive / Random.

All inference is greedy (argmax over legal moves). Greedy is
deterministic and reproducible; sampled play would add variance that
masks model-vs-model skill differences over the 1000-game-per-pair
sample sizes typical for this tournament.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

import td_ludo_cpp as cpp


# ---------------------------------------------------------------------------
#  Architecture presets for ModelAgent
# ---------------------------------------------------------------------------

@dataclass
class _ArchPreset:
    name: str
    arch_class: Callable
    arch_kwargs: dict
    encoder_fn: Callable
    in_channels: int
    needs_consecutive_sixes: bool = False


def _build_arch_presets() -> Dict[str, _ArchPreset]:
    """Lazy-import model classes — keep top-level imports cheap."""
    from td_ludo.models.v5 import AlphaLudoV5
    from td_ludo.models.v6_3 import AlphaLudoV63
    from td_ludo.models.v10 import AlphaLudoV10
    from td_ludo.models.v12 import AlphaLudoV12
    from experiments.distillation_14ch.model_14ch import MinimalCNN14

    return {
        # V12.2 production: 3 ResBlocks × 128, V11 33ch encoder, attn.
        "v122": _ArchPreset(
            name="v122",
            arch_class=AlphaLudoV12,
            arch_kwargs=dict(
                num_res_blocks=3, num_channels=128,
                num_attn_layers=2, num_heads=4, ffn_ratio=4,
                dropout=0.0, in_channels=33,
            ),
            encoder_fn=cpp.encode_state_v11,
            in_channels=33,
        ),
        # V12 default: 4 ResBlocks × 96, V11 33ch encoder, attn.
        "v12_default": _ArchPreset(
            name="v12_default",
            arch_class=AlphaLudoV12,
            arch_kwargs=dict(
                num_res_blocks=4, num_channels=96,
                num_attn_layers=2, num_heads=4, ffn_ratio=4,
                dropout=0.0, in_channels=33,
            ),
            encoder_fn=cpp.encode_state_v11,
            in_channels=33,
        ),
        # V10: pure CNN, 6×96, 28ch.
        "v10": _ArchPreset(
            name="v10",
            arch_class=AlphaLudoV10,
            arch_kwargs=dict(num_res_blocks=6, num_channels=96, in_channels=28),
            encoder_fn=cpp.encode_state_v10,
            in_channels=28,
        ),
        # V6.3: 27ch, 10×128, no attn.
        "v6_3": _ArchPreset(
            name="v6_3",
            arch_class=AlphaLudoV63,
            arch_kwargs=dict(num_res_blocks=10, num_channels=128, in_channels=27),
            encoder_fn=cpp.encode_state_v6_3,
            in_channels=27,
            needs_consecutive_sixes=True,
        ),
        # V6.1: 24ch, 10×128, no attn (uses AlphaLudoV5 class).
        "v6_1": _ArchPreset(
            name="v6_1",
            arch_class=AlphaLudoV5,
            arch_kwargs=dict(num_res_blocks=10, num_channels=128, in_channels=24),
            encoder_fn=cpp.encode_state_v6,
            in_channels=24,
        ),
        # V6_big: 17ch (V5-era), 10×128, no attn.
        "v6_big": _ArchPreset(
            name="v6_big",
            arch_class=AlphaLudoV5,
            arch_kwargs=dict(num_res_blocks=10, num_channels=128, in_channels=17),
            encoder_fn=cpp.encode_state,
            in_channels=17,
        ),
        # 14ch distilled student: 4 own + 4 opp + 6 dice one-hot.
        # Architecture: MinimalCNN14, 10 ResBlocks × 128, pure CNN, 3-head.
        "v14_minimal": _ArchPreset(
            name="v14_minimal",
            arch_class=MinimalCNN14,
            arch_kwargs=dict(num_res_blocks=10, num_channels=128, in_channels=14),
            encoder_fn=cpp.encode_state_v14_minimal,
            in_channels=14,
        ),
    }


# ---------------------------------------------------------------------------
#  Agent base + concrete types
# ---------------------------------------------------------------------------

class _AgentBase:
    name: str = "?"

    def select_move(self, state, legal: List[int], consec_sixes: int) -> int:
        raise NotImplementedError


class HistAgent(_AgentBase):
    """Wraps a tag from OpponentRegistry. Free — registry handles arch +
    encoder dispatch internally."""

    def __init__(self, tag: str, registry, name: Optional[str] = None):
        self.tag = tag
        self.registry = registry
        self.name = name or tag

    def select_move(self, state, legal, consec_sixes):
        return self.registry.select_action_single(self.tag, state, consec_sixes)


class ModelAgent(_AgentBase):
    """Loads a custom checkpoint with one of the architecture presets."""

    def __init__(
        self,
        name: str,
        ckpt_path: str,
        arch_preset: str,
        device: torch.device,
    ):
        self.name = name
        self.preset = _build_arch_presets()[arch_preset]
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

        model = self.preset.arch_class(**self.preset.arch_kwargs).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[tournament] {name}: "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        self.device = device

    def select_move(self, state, legal, consec_sixes):
        if not legal:
            return -1
        if len(legal) == 1:
            return legal[0]

        if self.preset.needs_consecutive_sixes:
            enc = self.preset.encoder_fn(state, consec_sixes)
        else:
            enc = self.preset.encoder_fn(state)

        x = torch.from_numpy(enc).unsqueeze(0).to(self.device, dtype=torch.float32)
        mask = np.zeros(4, dtype=np.float32)
        for a in legal:
            mask[a] = 1.0
        m = torch.from_numpy(mask).unsqueeze(0).to(self.device, dtype=torch.float32)

        with torch.no_grad():
            if hasattr(self.model, "forward_policy_only"):
                logits = self.model.forward_policy_only(x, m)
            else:
                # MinimalCNN14 (and other 3-head-only models): forward()
                # returns (policy, win_prob, moves) where policy is already
                # masked + softmaxed. argmax of softmax = argmax of logits,
                # so we use it directly for greedy action selection.
                policy, _, _ = self.model(x, m)
                logits = policy
            action = int(logits.argmax(dim=1).item())
        if action not in legal:
            action = legal[0]
        return action


class BotAgent(_AgentBase):
    """Wraps a hand-coded bot from td_ludo.game.heuristic_bot."""

    def __init__(self, bot_name: str, name: Optional[str] = None):
        from td_ludo.game.heuristic_bot import (
            HeuristicLudoBot, AggressiveBot, DefensiveBot,
            RacingBot, RandomBot, ExpertBot,
        )
        bots = {
            "Expert":     ExpertBot,
            "Heuristic":  HeuristicLudoBot,
            "Aggressive": AggressiveBot,
            "Defensive":  DefensiveBot,
            "Racing":     RacingBot,
            "Random":     RandomBot,
        }
        if bot_name not in bots:
            raise ValueError(
                f"Unknown bot '{bot_name}'. Available: {list(bots.keys())}"
            )
        self.bot = bots[bot_name]()
        self.name = name or bot_name

    def select_move(self, state, legal, consec_sixes):
        if not legal:
            return -1
        return self.bot.select_move(state, legal)


# ---------------------------------------------------------------------------
#  Convenience factory — used by the CLI
# ---------------------------------------------------------------------------

def list_arch_presets() -> List[str]:
    return list(_build_arch_presets().keys())
