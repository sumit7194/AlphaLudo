"""
AlphaLudo V6.2 — V6.1 CNN (24ch, 128ch, 5 ResBlocks) + Temporal Transformer

Combines V6.1's proven strategic CNN backbone with V9-style temporal transformer.
The CNN processes each turn's board state independently, then the transformer
attends across K=8 past turns to learn multi-turn patterns.

Architecture:
  Input: (B, K, 24, 15, 15) — K frames of 24ch strategic encoding

  CNN Backbone (shared weights, per-turn):
    Stem: Conv2d(24→128, 3×3) + BN + ReLU
    5× ResidualBlock(128)
    GAP → (B, 128)

  Temporal Transformer:
    Action embedding: Embedding(5, 128) — encodes previous action
    Position embedding: Embedding(K, 128) — temporal position
    LayerNorm for turn normalization
    4× TransformerEncoderLayer(128-dim, 4 heads, ff=512, GELU, norm_first)
    Causal mask + padding mask
    Alpha gate: tanh(0)=0 → pure CNN at init, learns to use context

  Heads:
    Policy: Linear(128→64) → ReLU → Linear(64→4)
    Value:  Linear(128→64) → ReLU → Linear(64→1)

  Total: ~4.7M params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ACTION_CLASSES = 5  # 0-3 pieces, 4=pass/none


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AlphaLudoV62(nn.Module):
    """V6.2: V6.1 CNN backbone + Temporal Transformer."""

    def __init__(
        self,
        context_length=8,
        in_channels=24,
        num_res_blocks=5,
        num_channels=128,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        self.context_length = context_length
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        # --- CNN Backbone (same as V6.1) ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # Project CNN output to embed_dim if needed
        if num_channels != embed_dim:
            self.cnn_proj = nn.Linear(num_channels, embed_dim)
        else:
            self.cnn_proj = nn.Identity()

        # --- Temporal Transformer ---
        self.action_embed = nn.Embedding(NUM_ACTION_CLASSES, embed_dim)
        self.turn_norm = nn.LayerNorm(embed_dim)
        self.temporal_pos_embed = nn.Embedding(context_length, embed_dim)

        # Padding token for turns before game start
        self.pad_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.pad_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,  # 512
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Alpha gate: tanh(0)=0 → pure CNN at init
        self.transformer_alpha = nn.Parameter(torch.zeros(1))

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(context_length, context_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer('causal_mask', causal_mask)

        # --- Policy Head ---
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        # --- Value Head ---
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
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
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _backbone(self, x):
        """CNN backbone: stem + res blocks + GAP → (B, num_channels)."""
        out = self.stem(x)
        for block in self.res_blocks:
            out = block(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return out.flatten(start_dim=1)

    def _apply_legal_mask(self, policy_logits, legal_mask):
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

    def _transformer_forward(self, cnn_features, prev_actions, seq_mask):
        """Transformer + alpha gate on pre-computed CNN features."""
        B, K = cnn_features.shape[:2]
        device = cnn_features.device

        raw_cnn = cnn_features

        # Add action history
        act_emb = self.action_embed(prev_actions)
        turn_embeds = self.turn_norm(cnn_features + act_emb)

        # Replace padded turns with learned pad token
        pad_expanded = self.pad_token.unsqueeze(0).unsqueeze(0).expand(B, K, -1)
        turn_embeds = torch.where(
            seq_mask.unsqueeze(-1).expand_as(turn_embeds),
            pad_expanded,
            turn_embeds,
        )

        # Temporal position embeddings
        positions = torch.arange(K, device=device)
        turn_embeds = turn_embeds + self.temporal_pos_embed(positions).unsqueeze(0)

        # Transformer with causal + padding mask.
        #
        # NaN-safety: when position 0 is padded (early game with history < K),
        # causal-mask + key-padding-mask combine to leave position 0's attention
        # with ALL keys masked out. softmax over an empty set produces NaN, which
        # then propagates through residual + LayerNorm into every subsequent
        # position. The NaN survives even when the alpha gate is exactly 0
        # (since 0 * NaN = NaN), poisoning policy logits and destroying eval
        # win-rate (~78% V6.1 baseline -> ~55% with NaN). Two layers of defense:
        #
        #   1. Sanitize the key-padding mask so that at least one key is always
        #      visible to every query (we always allow the LAST valid position
        #      as a fallback key). This prevents the all-masked case at the
        #      source for the typical short-history pattern.
        #   2. nan_to_num after the transformer call as belt-and-suspenders for
        #      any other pathological mask combination.
        causal = self.causal_mask[:K, :K]

        # Defense 1: ensure no row of (causal & ~padding) is all False.
        # The standard pattern is padding-at-front + valid-at-back, so the last
        # valid index is always >= n_pad. The simplest robust fix is to unmask
        # the LAST position in the key dim for all queries, making sure every
        # query has at least one attendable key. We then re-mask the OUTPUT for
        # padded queries (we don't read them anyway, but it keeps gradients clean).
        # NOTE: with the standard "pad-at-front" layout, the last position is
        # always valid (it's the current turn), so this doesn't change semantics
        # for valid queries — they could already attend to the current turn via
        # causal mask. It only fixes the degenerate first-padded-position case.
        safe_kpm = seq_mask.clone()
        safe_kpm[:, -1] = False  # always allow last position as a key

        out = self.transformer(turn_embeds, mask=causal, src_key_padding_mask=safe_kpm)

        # Defense 2: scrub any residual NaN/Inf to zero. Padded positions
        # produce undefined output regardless; we never read them since we
        # extract last_valid_idx below.
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        # Extract last valid turn
        valid_mask = ~seq_mask
        last_valid_idx = (valid_mask.cumsum(dim=1) * valid_mask).argmax(dim=1)
        batch_idx = torch.arange(B, device=device)

        last_direct = raw_cnn[batch_idx, last_valid_idx]
        last_trans = out[batch_idx, last_valid_idx]

        gate = torch.tanh(self.transformer_alpha)
        return last_direct + gate * last_trans

    def _encode_sequence(self, grids, prev_actions, seq_mask):
        """Full encode: CNN per turn + transformer across turns."""
        B, K = grids.shape[:2]

        flat_grids = grids.reshape(B * K, *grids.shape[2:])
        total = flat_grids.shape[0]
        chunk_size = 256

        if total <= chunk_size:
            cnn_features = self._backbone(flat_grids)
        else:
            chunks = []
            for i in range(0, total, chunk_size):
                chunks.append(self._backbone(flat_grids[i:i + chunk_size]))
            cnn_features = torch.cat(chunks, dim=0)

        cnn_features = self.cnn_proj(cnn_features)
        cnn_features = cnn_features.reshape(B, K, self.embed_dim)

        return self._transformer_forward(cnn_features, prev_actions, seq_mask)

    def forward(self, grids, prev_actions, seq_mask, legal_mask=None):
        """Full forward pass."""
        features = self._encode_sequence(grids, prev_actions, seq_mask)

        policy_logits = self.policy_head(features)
        policy_logits = self._apply_legal_mask(policy_logits, legal_mask)
        policy = F.softmax(policy_logits, dim=1)

        value = self.value_head(features)
        return policy, value

    def forward_policy_only(self, grids, prev_actions, seq_mask, legal_mask=None):
        """Fast forward — policy logits only."""
        features = self._encode_sequence(grids, prev_actions, seq_mask)
        policy_logits = self.policy_head(features)
        return self._apply_legal_mask(policy_logits, legal_mask)

    def compute_single_cnn_features(self, grids):
        """Process single board grids through CNN for caching during gameplay."""
        with torch.no_grad():
            features = self._backbone(grids)
        return self.cnn_proj(features.detach())

    def forward_cached(self, cached_cnn, prev_actions, seq_mask, legal_mask=None):
        """Forward using pre-computed CNN features."""
        features = self._transformer_forward(cached_cnn, prev_actions, seq_mask)

        policy_logits = self.policy_head(features)
        policy_logits = self._apply_legal_mask(policy_logits, legal_mask)
        policy = F.softmax(policy_logits, dim=1)

        value = self.value_head(features)
        return policy, value

    def forward_policy_only_cached(self, cached_cnn, prev_actions, seq_mask, legal_mask=None):
        """Policy logits from pre-computed CNN features."""
        features = self._transformer_forward(cached_cnn, prev_actions, seq_mask)
        policy_logits = self.policy_head(features)
        return self._apply_legal_mask(policy_logits, legal_mask)

    def load_v61_weights(self, v61_state_dict):
        """
        Load CNN backbone + heads from a V6.1 checkpoint.
        Transformer layers remain randomly initialized.
        The alpha gate starts at 0 (pure CNN), so the model initially
        behaves exactly like V6.1 until the transformer learns.
        """
        own_state = self.state_dict()
        loaded = 0
        skipped = 0

        # V6.1 uses AlphaLudoV5 which has different key names
        # V6.1: conv_input, bn_input, res_blocks, policy_fc1/fc2, value_fc1/fc2
        # V6.2: stem.0 (conv), stem.1 (bn), res_blocks, policy_head.0/2, value_head.0/2

        key_map = {
            'conv_input.weight': 'stem.0.weight',
            'bn_input.weight': 'stem.1.weight',
            'bn_input.bias': 'stem.1.bias',
            'bn_input.running_mean': 'stem.1.running_mean',
            'bn_input.running_var': 'stem.1.running_var',
            'bn_input.num_batches_tracked': 'stem.1.num_batches_tracked',
            'policy_fc1.weight': 'policy_head.0.weight',
            'policy_fc1.bias': 'policy_head.0.bias',
            'policy_fc2.weight': 'policy_head.2.weight',
            'policy_fc2.bias': 'policy_head.2.bias',
            'value_fc1.weight': 'value_head.0.weight',
            'value_fc1.bias': 'value_head.0.bias',
            'value_fc2.weight': 'value_head.2.weight',
            'value_fc2.bias': 'value_head.2.bias',
        }

        for v61_key, v61_val in v61_state_dict.items():
            # Try direct key mapping
            if v61_key in key_map:
                target_key = key_map[v61_key]
            elif v61_key.startswith('res_blocks.'):
                target_key = v61_key  # res_blocks have same structure
            else:
                skipped += 1
                continue

            if target_key in own_state and own_state[target_key].shape == v61_val.shape:
                own_state[target_key] = v61_val
                loaded += 1
            else:
                skipped += 1

        self.load_state_dict(own_state)
        print(f"[V6.2] Loaded {loaded} params from V6.1, skipped {skipped} "
              f"(transformer layers randomly initialized, alpha gate = 0)")

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = AlphaLudoV62(context_length=8)
    total = model.count_parameters()
    print(f"AlphaLudo V6.2 — Total: {total:,}")

    # Count by component
    cnn_params = sum(p.numel() for n, p in model.named_parameters()
                     if any(k in n for k in ['stem', 'res_blocks', 'cnn_proj']))
    tf_params = sum(p.numel() for n, p in model.named_parameters()
                    if any(k in n for k in ['transformer', 'action_embed', 'turn_norm',
                                            'temporal_pos_embed', 'pad_token', 'transformer_alpha']))
    head_params = sum(p.numel() for n, p in model.named_parameters()
                      if any(k in n for k in ['policy_head', 'value_head']))
    print(f"  CNN: {cnn_params:,} | Transformer: {tf_params:,} | Heads: {head_params:,}")

    # Test forward
    B, K = 4, 8
    grids = torch.randn(B, K, 24, 15, 15)
    prev_actions = torch.randint(0, 5, (B, K))
    seq_mask = torch.zeros(B, K, dtype=torch.bool)
    seq_mask[0, :3] = True
    legal_mask = torch.ones(B, 4)

    policy, value = model(grids, prev_actions, seq_mask, legal_mask)
    print(f"Policy: {policy.shape}, Value: {value.shape}")
    print(f"Alpha gate: {torch.tanh(model.transformer_alpha).item():.4f}")

    # Test V6.1 weight loading
    from model import AlphaLudoV5
    v61 = AlphaLudoV5(num_res_blocks=5, num_channels=128, in_channels=24)
    model2 = AlphaLudoV62(context_length=8, num_res_blocks=5)
    model2.load_v61_weights(v61.state_dict())
