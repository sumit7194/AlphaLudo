"""
AlphaLudo V9 — Slim CNN + Temporal Transformer

Informed by mech interp on V6:
- Layer knockout: all 10 ResBlocks individually removable → over-parameterized
- CKA: Blocks 5-9 nearly identical (>0.99) → massive redundancy
- Channel ablation: Score Diff lowest impact, dice channels wasteful as 6 spatial planes

Architecture:
  Input: 14 channels (optimized encoding)
    Ch 0-3:  My Token 0-3 (position on board)
    Ch 4-7:  Opp Token 0-3 (individual opponent tokens)
    Ch 8:    Safe Zones
    Ch 9:    My Home Path
    Ch 10:   Opp Home Path
    Ch 11:   My Locked % (broadcast)
    Ch 12:   Opp Locked % (broadcast)
    Ch 13:   Dice Roll / 6.0 (broadcast)

  CNN: 5 ResBlocks, 80 channels (~750K params)
    Stem: Conv2d(14→80, 3×3) + BN + ReLU
    5× ResidualBlock(80)
    GAP → (B, 80)

  Temporal Transformer: 4 layers, 80-dim (~400K params)
    Action embedding: Embedding(5, 80)
    Turn norm: LayerNorm(80)
    Temporal pos embed: Embedding(K, 80)
    4× TransformerEncoderLayer(80, 4 heads, ff=320, gelu, norm_first)
    Causal + padding mask
    Alpha gate: starts at 0 (pure CNN), learns to use context

  Heads:
    Policy: Linear(80→64) → ReLU → Linear(64→4)
    Value:  Linear(80→64) → ReLU → Linear(64→1)

  Total: ~1.2M params, all trainable end-to-end
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


class AlphaLudoV9(nn.Module):
    """
    V9: Slim CNN (5 ResBlocks, 80ch) + Temporal Transformer (4 layers, 80-dim).
    All parameters trainable end-to-end — small enough to fit in 16GB Mac memory.
    """

    def __init__(
        self,
        context_length=16,
        in_channels=14,
        num_res_blocks=5,
        num_channels=80,
        embed_dim=80,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        self.context_length = context_length
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        # --- CNN Backbone ---
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
            dim_feedforward=embed_dim * 4,  # 320
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Alpha gate: tanh(0)=0 → pure CNN at init, learns to mix transformer
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
        """
        Transformer + alpha gate on pre-computed CNN features.

        Args:
            cnn_features: (B, K, embed_dim)
            prev_actions: (B, K) int64
            seq_mask: (B, K) bool — True for padded positions
        Returns:
            (B, embed_dim)
        """
        B, K = cnn_features.shape[:2]
        device = cnn_features.device

        raw_cnn = cnn_features  # save for gate

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

        # Transformer with causal + padding mask
        causal = self.causal_mask[:K, :K]
        out = self.transformer(turn_embeds, mask=causal, src_key_padding_mask=seq_mask)

        # Extract last valid turn
        valid_mask = ~seq_mask
        last_valid_idx = (valid_mask.cumsum(dim=1) * valid_mask).argmax(dim=1)
        batch_idx = torch.arange(B, device=device)

        last_direct = raw_cnn[batch_idx, last_valid_idx]
        last_trans = out[batch_idx, last_valid_idx]

        gate = torch.tanh(self.transformer_alpha)
        return last_direct + gate * last_trans

    def _encode_sequence(self, grids, prev_actions, seq_mask):
        """
        Full encode: CNN per turn + transformer across turns.

        Args:
            grids: (B, K, 14, 15, 15)
            prev_actions: (B, K) int64
            seq_mask: (B, K) bool
        Returns:
            (B, embed_dim)
        """
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
        """
        Full forward pass.

        Args:
            grids: (B, K, 14, 15, 15)
            prev_actions: (B, K) int64
            seq_mask: (B, K) bool — True for padded turns
            legal_mask: (B, 4) float32
        Returns:
            policy: (B, 4) probabilities
            value: (B, 1)
        """
        features = self._encode_sequence(grids, prev_actions, seq_mask)

        policy_logits = self.policy_head(features)
        policy_logits = self._apply_legal_mask(policy_logits, legal_mask)
        policy = F.softmax(policy_logits, dim=1)

        value = self.value_head(features)
        return policy, value

    def forward_policy_only(self, grids, prev_actions, seq_mask, legal_mask=None):
        """Fast forward — policy logits only (skip value head)."""
        features = self._encode_sequence(grids, prev_actions, seq_mask)
        policy_logits = self.policy_head(features)
        return self._apply_legal_mask(policy_logits, legal_mask)

    def compute_single_cnn_features(self, grids):
        """
        Process single board grids through CNN for caching during gameplay.

        Args:
            grids: (B, 14, 15, 15)
        Returns:
            (B, embed_dim)
        """
        with torch.no_grad():
            features = self._backbone(grids)
        return self.cnn_proj(features.detach())

    def compute_cnn_features(self, grids):
        """
        Batch CNN features for sequence of grids (for PPO caching).

        Args:
            grids: (N, K, 14, 15, 15)
        Returns:
            (N, K, embed_dim)
        """
        N, K = grids.shape[:2]
        flat_grids = grids.reshape(N * K, *grids.shape[2:])
        total = flat_grids.shape[0]
        chunk_size = 256

        all_features = []
        for i in range(0, total, chunk_size):
            chunk = flat_grids[i:i + chunk_size]
            with torch.no_grad():
                feats = self._backbone(chunk)
            all_features.append(feats.detach())
        cnn_features = torch.cat(all_features, dim=0)
        cnn_features = self.cnn_proj(cnn_features)
        return cnn_features.reshape(N, K, self.embed_dim)

    def forward_cached(self, cached_cnn, prev_actions, seq_mask, legal_mask=None):
        """
        Forward using pre-computed CNN features (skips CNN backbone).

        Args:
            cached_cnn: (B, K, embed_dim)
            prev_actions: (B, K) int64
            seq_mask: (B, K) bool
            legal_mask: (B, 4) float32
        """
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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = AlphaLudoV9()
    total = model.count_all_parameters()
    trainable = model.count_parameters()
    print(f"AlphaLudo V9 — Total: {total:,} | Trainable: {trainable:,}")

    # Count by component
    cnn_params = sum(p.numel() for n, p in model.named_parameters()
                     if any(k in n for k in ['stem', 'res_blocks']))
    tf_params = sum(p.numel() for n, p in model.named_parameters()
                    if any(k in n for k in ['transformer', 'action_embed', 'turn_norm',
                                            'temporal_pos_embed', 'pad_token', 'transformer_alpha']))
    head_params = sum(p.numel() for n, p in model.named_parameters()
                      if any(k in n for k in ['policy_head', 'value_head']))
    print(f"  CNN: {cnn_params:,} | Transformer: {tf_params:,} | Heads: {head_params:,}")

    # Test forward pass
    B, K = 4, 16
    grids = torch.randn(B, K, 14, 15, 15)
    prev_actions = torch.randint(0, 5, (B, K))
    seq_mask = torch.zeros(B, K, dtype=torch.bool)
    seq_mask[0, :3] = True
    seq_mask[1, :5] = True

    legal_mask = torch.tensor([
        [1, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 1, 1],
    ], dtype=torch.float32)

    policy, value = model(grids, prev_actions, seq_mask, legal_mask)
    print(f"Policy shape: {policy.shape}")  # (4, 4)
    print(f"Value shape: {value.shape}")    # (4, 1)
    print(f"Policy sums: {policy.sum(dim=1)}")
    print(f"Policy[0]: {policy[0]}")
    print(f"Alpha gate: {torch.tanh(model.transformer_alpha).item():.4f}")

    # Test cached forward
    cached = model.compute_cnn_features(grids)
    print(f"Cached CNN shape: {cached.shape}")  # (4, 16, 80)
    p2, v2 = model.forward_cached(cached, prev_actions, seq_mask, legal_mask)
    print(f"Cached policy matches: {torch.allclose(policy, p2, atol=1e-5)}")
