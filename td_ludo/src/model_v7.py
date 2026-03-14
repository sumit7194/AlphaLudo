"""
AlphaLudo V7 — Sequence Transformer for Strategic Board Game Play

Replaces the V6 CNN (15×15×17 spatial, 3M params) with a lightweight
transformer over a context window of K past turns.

Architecture:
  Input per turn:
    - 8 token positions (int 0-58) → nn.Embedding(59, embed_dim)  → 8 × embed_dim
    - 9 continuous features (3 global + 6 dice) → Linear(9, embed_dim) → 1 × embed_dim
    - 1 historical action (int 0-4) → nn.Embedding(5, embed_dim)  → 1 × embed_dim
  Per-turn encoding:
    - Concatenate 10 token embeddings → (10, embed_dim) per turn
    - Mean-pool within turn → (1, embed_dim)
    - Add learned temporal position embedding
  Sequence:
    - K turns → (K, embed_dim) sequence
    - Standard transformer encoder layers with causal masking
  Output:
    - Policy head: Linear(embed_dim, 4) from last turn's representation
    - Value head: Linear(embed_dim, 1) with tanh from last turn's representation

Estimated params: ~400K-800K (vs V6's 3M)
Target device: 16GB M4 Mac Mini (MPS)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.state_encoder_1d import (
    NUM_POSITION_CLASSES,
    NUM_ACTION_CLASSES,
    CONTINUOUS_DIM,
    NUM_TOKEN_POSITIONS,
)


class TurnEncoder(nn.Module):
    """
    Encodes a single turn's 1D state into a fixed-size embedding.

    Takes:
      - token_positions: (B, 8) int  → 8 position embeddings
      - continuous: (B, 9) float     → projected to embed_dim
      - action: (B,) int            → 1 action embedding
    Returns:
      - (B, embed_dim) turn embedding
    """

    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim

        # Position embedding for token locations (0-58)
        self.pos_embed = nn.Embedding(NUM_POSITION_CLASSES, embed_dim)

        # Action embedding (0-4: tokens 0-3 or pass)
        self.action_embed = nn.Embedding(NUM_ACTION_CLASSES, embed_dim)

        # Project continuous features (global context + dice) to embed_dim
        self.continuous_proj = nn.Linear(CONTINUOUS_DIM, embed_dim)

        # Final projection after aggregation (10 slots → mean → project)
        # 10 slots: 8 token positions + 1 continuous + 1 action
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, token_positions, continuous, action):
        """
        Args:
            token_positions: (B, 8) int64 — token position indices
            continuous: (B, 9) float32 — global context + dice one-hot
            action: (B,) int64 — historical action index

        Returns:
            (B, embed_dim) — turn embedding
        """
        # Embed 8 token positions: (B, 8) → (B, 8, E)
        tok_emb = self.pos_embed(token_positions)

        # Project continuous: (B, 9) → (B, 1, E)
        cont_emb = self.continuous_proj(continuous).unsqueeze(1)

        # Embed action: (B,) → (B, 1, E)
        act_emb = self.action_embed(action).unsqueeze(1)

        # Concatenate all slots: (B, 10, E)
        all_slots = torch.cat([tok_emb, cont_emb, act_emb], dim=1)

        # Mean pool across slots: (B, E)
        turn_repr = all_slots.mean(dim=1)

        # Project and normalize
        turn_repr = self.layer_norm(self.output_proj(turn_repr))

        return turn_repr


class AlphaLudoV7(nn.Module):
    """
    V7 Sequence Transformer — processes K past turns to select actions.

    Forward pass:
      1. Encode each of K turns into embed_dim vectors (via TurnEncoder)
      2. Add temporal position embeddings
      3. Pass through transformer encoder (causal self-attention)
      4. Extract last turn's representation
      5. Policy head → (B, 4) logits; Value head → (B, 1) score
    """

    def __init__(
        self,
        context_length=16,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        self.context_length = context_length
        self.embed_dim = embed_dim

        # Per-turn encoder
        self.turn_encoder = TurnEncoder(embed_dim)

        # Temporal position embedding (learned, one per slot in context window)
        self.temporal_pos_embed = nn.Embedding(context_length, embed_dim)

        # Padding mask token (learnable, for turns before game start)
        self.pad_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.pad_token, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Policy Head (Actor): embed_dim → 4 token logits
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4),
        )

        # Value Head (Critic): embed_dim → 1 score
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        # Causal mask (registered as buffer — not a parameter)
        # Upper-triangular mask: position i can only attend to positions <= i
        causal_mask = torch.triu(
            torch.ones(context_length, context_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer('causal_mask', causal_mask)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _apply_legal_mask(self, policy_logits, legal_mask):
        """Apply legal move mask to policy logits (same as V5/V6)."""
        if legal_mask is not None:
            all_illegal = (legal_mask.sum(dim=1, keepdim=True) == 0)
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), float('-inf'))
            if all_illegal.any():
                policy_logits = torch.where(
                    all_illegal.expand_as(policy_logits),
                    torch.zeros_like(policy_logits),
                    policy_logits,
                )
        return policy_logits

    def _encode_sequence(self, token_positions, continuous, actions, seq_mask):
        """
        Encode a sequence of K turns into transformer input.

        Args:
            token_positions: (B, K, 8) int64 — token positions per turn
            continuous: (B, K, 9) float32 — continuous features per turn
            actions: (B, K) int64 — historical actions per turn
            seq_mask: (B, K) bool — True for padded (invalid) positions

        Returns:
            (B, embed_dim) — representation of the last valid turn
        """
        B, K = token_positions.shape[:2]
        device = token_positions.device

        # Encode each turn independently
        # Reshape to (B*K, ...) for batched turn encoding
        tok_flat = token_positions.reshape(B * K, -1)       # (B*K, 8)
        cont_flat = continuous.reshape(B * K, -1)            # (B*K, 9)
        act_flat = actions.reshape(B * K)                    # (B*K,)

        turn_embeds = self.turn_encoder(tok_flat, cont_flat, act_flat)  # (B*K, E)
        turn_embeds = turn_embeds.reshape(B, K, self.embed_dim)         # (B, K, E)

        # Replace padded turns with learned pad token
        pad_expanded = self.pad_token.unsqueeze(0).unsqueeze(0).expand(B, K, -1)
        turn_embeds = torch.where(
            seq_mask.unsqueeze(-1).expand_as(turn_embeds),
            pad_expanded,
            turn_embeds,
        )

        # Add temporal position embeddings
        positions = torch.arange(K, device=device)
        temporal_emb = self.temporal_pos_embed(positions)  # (K, E)
        turn_embeds = turn_embeds + temporal_emb.unsqueeze(0)

        # Create attention mask for transformer
        # Combine causal mask with padding mask
        # causal_mask: (K, K) bool — True means "do not attend"
        # seq_mask: (B, K) bool — True means "this position is padding"
        causal = self.causal_mask[:K, :K]  # in case K < context_length

        # Transformer forward
        # src_key_padding_mask: (B, K) — True for padded positions
        # mask: (K, K) — causal attention mask
        out = self.transformer(
            turn_embeds,
            mask=causal,
            src_key_padding_mask=seq_mask,
        )  # (B, K, E)

        # Extract the last turn's representation
        # Find the last non-padded position for each batch
        # seq_mask is True for padding, so we want the last False position
        valid_mask = ~seq_mask  # (B, K) — True for valid positions
        # Last valid index per batch
        last_valid_idx = (valid_mask.cumsum(dim=1) * valid_mask).argmax(dim=1)  # (B,)

        # Gather the last valid turn's output
        last_repr = out[torch.arange(B, device=device), last_valid_idx]  # (B, E)

        return last_repr

    def forward(self, token_positions, continuous, actions, seq_mask, legal_mask=None):
        """
        Full forward pass.

        Args:
            token_positions: (B, K, 8) int64
            continuous: (B, K, 9) float32
            actions: (B, K) int64
            seq_mask: (B, K) bool — True for padded turns
            legal_mask: (B, 4) float32 — 1.0 for legal actions, 0.0 for illegal

        Returns:
            policy: (B, 4) probability distribution
            value: (B, 1) predicted value
        """
        features = self._encode_sequence(token_positions, continuous, actions, seq_mask)

        # Policy head
        policy_logits = self.policy_head(features)
        policy_logits = self._apply_legal_mask(policy_logits, legal_mask)
        policy = F.softmax(policy_logits, dim=1)

        # Value head
        value = self.value_head(features)  # No tanh — let it be unbounded like V5

        return policy, value

    def forward_policy_only(self, token_positions, continuous, actions, seq_mask, legal_mask=None):
        """
        Fast forward pass returning only policy logits (for inference).

        Returns:
            policy_logits: (B, 4) raw logits
        """
        features = self._encode_sequence(token_positions, continuous, actions, seq_mask)
        policy_logits = self.policy_head(features)
        return self._apply_legal_mask(policy_logits, legal_mask)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    import numpy as np

    # Test the model
    model = AlphaLudoV7(context_length=16, embed_dim=128, num_heads=4, num_layers=4)
    print(f"AlphaLudo V7 Parameters: {model.count_parameters():,}")

    # Simulate a batch of 4 games, each with K=16 turns of context
    B, K = 4, 16
    token_positions = torch.randint(0, 59, (B, K, 8))
    continuous = torch.randn(B, K, 9)
    actions = torch.randint(0, 5, (B, K))
    seq_mask = torch.zeros(B, K, dtype=torch.bool)
    # First 3 turns are padding for games 0 and 1
    seq_mask[0, :3] = True
    seq_mask[1, :5] = True

    legal_mask = torch.tensor([
        [1, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 1, 1],
    ], dtype=torch.float32)

    # Test forward
    policy, value = model(token_positions, continuous, actions, seq_mask, legal_mask)
    print(f"Policy shape: {policy.shape}")  # (4, 4)
    print(f"Value shape: {value.shape}")    # (4, 1)
    print(f"Policy sums: {policy.sum(dim=1)}")  # Should be ~1.0
    print(f"Policy[0]: {policy[0]}")  # Tokens 2,3 should be 0 (masked)
    print(f"Policy[2]: {policy[2]}")  # Tokens 0,3 should be 0 (masked)

    # Test forward_policy_only
    logits = model.forward_policy_only(token_positions, continuous, actions, seq_mask, legal_mask)
    print(f"Policy logits shape: {logits.shape}")  # (4, 4)

    # Test with minimal context (K=1, rest padding)
    seq_mask_min = torch.ones(B, K, dtype=torch.bool)
    seq_mask_min[:, -1] = False  # Only last turn is valid
    policy2, value2 = model(token_positions, continuous, actions, seq_mask_min, legal_mask)
    print(f"\nMinimal context:")
    print(f"Policy shape: {policy2.shape}")
    print(f"Value shape: {value2.shape}")
