"""
Historical-model opponent registry for RL training (Phase 1 stub).

Replaces the saturated bot mix (Heuristic / Aggressive / Defensive /
Expert — all hand-coded) with prior-generation V-model checkpoints as
opponents. Different generations use different architectures and
encoders; this module dispatches per opponent tag.

Design goals:
  1. **Lazy loading** — checkpoints loaded on first use, cached for the
     process lifetime (these don't change across the run).
  2. **Per-tag encoder dispatch** — V6.3 uses encode_state_v6_3 (27ch),
     V10 uses encode_state_v10 (28ch), V11/V12 use encode_state_v11
     (33ch). Calling code passes a tag; this module handles the rest.
  3. **Batched forward** — given a list of (game, tag) pairs, group by
     tag and run one forward pass per group, return actions in the same
     order as input.
  4. **Frozen weights, eval mode** — opponents never train. We freeze
     parameters and call .eval() at load time.

Tags (extend as needed):
  - "Hist_V6_3"  → AlphaLudoV63, encode_state_v6_3
  - "Hist_V10"   → AlphaLudoV10, encode_state_v10
  - "Hist_V11"   → AlphaLudoV11, encode_state_v11
  - "Hist_V12"   → AlphaLudoV12 (current class) loaded with V12-default
                   shape (4 ResBlocks × 96ch), encode_state_v11

Checkpoint paths are configured in `_DEFAULT_CKPTS`. Override with the
`HISTORICAL_OPPONENT_CKPTS` env var if needed (JSON dict, tag → path).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

import td_ludo_cpp as cpp


# Opponent registry: tag → spec.
# Each spec is loaded lazily on first call to `get_model(tag)`.

@dataclass
class _OpponentSpec:
    tag: str
    arch_class: Callable           # nn.Module subclass (uninstantiated)
    arch_kwargs: dict              # passed to arch_class(**kwargs)
    encoder_fn: Callable           # cpp.encode_state_v* callable
    in_channels: int               # for sanity-checking
    needs_consecutive_sixes: bool  # V6.3-only quirk


def _build_specs() -> Dict[str, _OpponentSpec]:
    """Lazy import the model classes — keep top-level imports cheap."""
    from td_ludo.models.v6_3 import AlphaLudoV63
    from td_ludo.models.v10 import AlphaLudoV10
    from td_ludo.models.v11 import AlphaLudoV11
    from td_ludo.models.v12 import AlphaLudoV12

    return {
        "Hist_V6_3": _OpponentSpec(
            tag="Hist_V6_3",
            arch_class=AlphaLudoV63,
            arch_kwargs=dict(num_res_blocks=10, num_channels=128, in_channels=27),
            encoder_fn=cpp.encode_state_v6_3,
            in_channels=27,
            needs_consecutive_sixes=True,
        ),
        "Hist_V10": _OpponentSpec(
            tag="Hist_V10",
            arch_class=AlphaLudoV10,
            arch_kwargs=dict(num_res_blocks=6, num_channels=96, in_channels=28),
            encoder_fn=cpp.encode_state_v10,
            in_channels=28,
            needs_consecutive_sixes=False,
        ),
        "Hist_V11": _OpponentSpec(
            tag="Hist_V11",
            arch_class=AlphaLudoV11,
            arch_kwargs=dict(
                num_res_blocks=4, num_channels=96,
                num_attn_layers=2, num_heads=4, ffn_ratio=4,
                dropout=0.0, in_channels=33,
            ),
            encoder_fn=cpp.encode_state_v11,
            in_channels=33,
            needs_consecutive_sixes=False,
        ),
        "Hist_V12": _OpponentSpec(
            tag="Hist_V12",
            arch_class=AlphaLudoV12,
            # V12 default shape — NOT V12.2's 3×128.
            arch_kwargs=dict(
                num_res_blocks=4, num_channels=96,
                num_attn_layers=2, num_heads=4, ffn_ratio=4,
                dropout=0.0, in_channels=33,
            ),
            encoder_fn=cpp.encode_state_v11,
            in_channels=33,
            needs_consecutive_sixes=False,
        ),
    }


_DEFAULT_CKPTS = {
    "Hist_V6_3": "play/model_weights/historical/v6_3.pt",
    "Hist_V10":  "play/model_weights/historical/v10.pt",
    "Hist_V11":  "play/model_weights/historical/v11.pt",
    "Hist_V12":  "play/model_weights/historical/v12.pt",
}

# Repo-root-relative paths. The runner resolves them against
# td_ludo/ root (same as how chain_v122.sh resolves).
def _resolve_ckpt(rel: str) -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, rel)


def _ckpt_path(tag: str) -> str:
    """Resolve checkpoint path for a tag, with env-var override."""
    overrides_json = os.environ.get("HISTORICAL_OPPONENT_CKPTS")
    if overrides_json:
        try:
            overrides = json.loads(overrides_json)
            if tag in overrides:
                return overrides[tag]
        except json.JSONDecodeError:
            pass
    return _resolve_ckpt(_DEFAULT_CKPTS[tag])


class OpponentRegistry:
    """Per-process cache of historical opponent models.

    Usage:
        reg = OpponentRegistry(device=torch.device("cuda"))
        action = reg.select_action_single("Hist_V11", game)
        # or batched:
        actions = reg.select_actions_batched([
            ("Hist_V11", game_a, 0),  # 0 = consecutive_sixes (only used for V6_3)
            ("Hist_V10", game_b, 0),
        ])
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._specs = _build_specs()
        self._models: Dict[str, torch.nn.Module] = {}

    def available_tags(self) -> List[str]:
        return list(self._specs.keys())

    def get_model(self, tag: str) -> torch.nn.Module:
        """Lazy-load + cache. Frozen, eval mode."""
        if tag in self._models:
            return self._models[tag]
        if tag not in self._specs:
            raise KeyError(f"Unknown opponent tag: {tag}. "
                           f"Known tags: {list(self._specs.keys())}")
        spec = self._specs[tag]
        ckpt_path = _ckpt_path(tag)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint for {tag} not found at {ckpt_path}. "
                f"Set HISTORICAL_OPPONENT_CKPTS env var to override paths."
            )

        model = spec.arch_class(**spec.arch_kwargs).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        # strict=False: tolerate minor key mismatches across architecture
        # versions (e.g., dropped aux heads). Mismatches are logged but
        # don't fail the load.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[opp_registry] {tag}: "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._models[tag] = model
        return model

    def encode(self, tag: str, game, consecutive_sixes: int = 0) -> np.ndarray:
        """Encode a single GameState with the opponent's encoder."""
        spec = self._specs[tag]
        if spec.needs_consecutive_sixes:
            return spec.encoder_fn(game, consecutive_sixes)
        return spec.encoder_fn(game)

    @staticmethod
    def _legal_mask(legal_moves: List[int]) -> np.ndarray:
        m = np.zeros(4, dtype=np.float32)
        for a in legal_moves:
            m[a] = 1.0
        return m

    def select_action_single(
        self, tag: str, game, consecutive_sixes: int = 0,
    ) -> int:
        """Greedy argmax(legal-masked policy) for a single game/state."""
        legal = list(cpp.get_legal_moves(game))
        if not legal:
            return -1
        if len(legal) == 1:
            return legal[0]

        spec = self._specs[tag]
        model = self.get_model(tag)

        state = self.encode(tag, game, consecutive_sixes)
        x = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32)
        mask = torch.from_numpy(self._legal_mask(legal)).unsqueeze(0).to(
            self.device, dtype=torch.float32,
        )
        with torch.no_grad():
            logits = model.forward_policy_only(x, mask)
            action = int(logits.argmax(dim=1).item())
        if action not in legal:
            # Defensive — should never happen with proper masking.
            action = legal[0]
        return action

    def select_actions_batched(
        self,
        items: List[tuple],  # list of (tag, game, consecutive_sixes)
    ) -> List[int]:
        """Group by tag, batch-encode + batch-forward, return actions.

        Args:
            items: list of (tag, game, consecutive_sixes) — one per
                game-row that this opponent should act on.

        Returns:
            list of int actions, in the same order as `items`. -1 for
            rows with no legal moves.
        """
        n = len(items)
        if n == 0:
            return []

        actions: List[Optional[int]] = [None] * n

        # Group by tag for batched forward.
        by_tag: Dict[str, List[int]] = {}
        for k, (tag, _g, _c) in enumerate(items):
            by_tag.setdefault(tag, []).append(k)

        for tag, idxs in by_tag.items():
            spec = self._specs[tag]
            model = self.get_model(tag)

            states_list = []
            masks_list = []
            legal_per_row = []
            singletons = []  # rows with exactly 1 legal move

            for k in idxs:
                _, game, csix = items[k]
                legal = list(cpp.get_legal_moves(game))
                legal_per_row.append(legal)
                if not legal:
                    actions[k] = -1
                    continue
                if len(legal) == 1:
                    actions[k] = legal[0]
                    singletons.append(k)
                    continue
                states_list.append(self.encode(tag, game, csix))
                masks_list.append(self._legal_mask(legal))

            if not states_list:
                continue

            states = np.stack(states_list)
            masks = np.stack(masks_list)
            x = torch.from_numpy(states).to(self.device, dtype=torch.float32)
            m = torch.from_numpy(masks).to(self.device, dtype=torch.float32)
            with torch.no_grad():
                logits = model.forward_policy_only(x, m)
                argmax = logits.argmax(dim=1).cpu().numpy()

            # Walk back through idxs and fill in non-singleton, non-empty rows.
            j = 0
            for k in idxs:
                if actions[k] is not None:
                    continue  # already assigned (singleton or empty)
                a = int(argmax[j])
                if a not in legal_per_row[idxs.index(k)]:
                    a = legal_per_row[idxs.index(k)][0]
                actions[k] = a
                j += 1

        # Sanity: every row must have an action assigned.
        return [a if a is not None else -1 for a in actions]
