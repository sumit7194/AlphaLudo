"""
AlphaLudo V8 — V6 CNN + Temporal Transformer

Wraps the proven V6 CNN backbone (128ch, 10 ResBlocks, 17ch input)
with a temporal transformer that attends over K=16 past turns.

Architecture:
  Per turn:
    - Board state (17, 15, 15) → V6 CNN backbone → (128,) spatial features
    - Previous action (int 0-4) → nn.Embedding(5, 128) → (128,) action embedding
    - Sum + LayerNorm → (128,) turn embedding
  Sequence:
    - K turns → (K, 128) + temporal position embeddings
    - Transformer encoder with causal masking (4 layers, 4 heads)
  Output:
    - Policy head: Linear(128, 128) → ReLU → Linear(128, 4)
    - Value head: Linear(128, 128) → ReLU → Linear(128, 1)

The CNN backbone can be frozen (only ~400K trainable transformer params)
or unfrozen for end-to-end fine-tuning (~3.4M total params).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import AlphaLudoV5

NUM_ACTION_CLASSES = 5  # 0-3 pieces, 4=pass/none


class AlphaLudoV8(nn.Module):
    """
    V8: Proven V6 CNN spatial encoder + temporal transformer.

    The CNN processes each turn's board state into a 128-dim feature vector.
    The temporal transformer attends across K past turns to learn patterns
    like opponent velocity, threat persistence, and multi-turn strategies.
    """

    def __init__(
        self,
        context_length=16,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        cnn_res_blocks=10,
        cnn_channels=128,
        in_channels=17,
    ):
        super().__init__()

        self.context_length = context_length
        self.embed_dim = embed_dim
        self.cnn_frozen = False

        # V6 CNN backbone (spatial encoder)
        self.cnn = AlphaLudoV5(
            num_res_blocks=cnn_res_blocks,
            num_channels=cnn_channels,
            in_channels=in_channels,
        )
        # CNN backbone outputs (B, cnn_channels).
        # If cnn_channels != embed_dim, project. Otherwise identity.
        if cnn_channels != embed_dim:
            self.cnn_proj = nn.Linear(cnn_channels, embed_dim)
        else:
            self.cnn_proj = nn.Identity()

        # Action history embedding (what was done on previous turn)
        self.action_embed = nn.Embedding(NUM_ACTION_CLASSES, embed_dim)

        # Combine CNN features + action embedding
        self.turn_norm = nn.LayerNorm(embed_dim)

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
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Policy Head (Actor): embed_dim → 4 token logits
        # Shape matches V6's policy_fc1/policy_fc2 so trained weights can be loaded directly
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        # Value Head (Critic): embed_dim → 1 score
        # Shape matches V6's value_fc1/value_fc2
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Transformer gate: tanh(0)=0 so at init the model is pure V6 CNN
        # Gradually opens as PPO trains it to use temporal context
        self.transformer_alpha = nn.Parameter(torch.zeros(1))

        # Causal mask (registered as buffer — not a parameter)
        causal_mask = torch.triu(
            torch.ones(context_length, context_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer('causal_mask', causal_mask)

        # Initialize transformer + heads weights (not CNN — those come from V6)
        self._init_new_weights()

    def _init_new_weights(self):
        """Initialize transformer and head weights. CNN weights come from V6 checkpoint."""
        for name, module in self.named_modules():
            # Skip CNN backbone — its weights are loaded from V6
            if name.startswith('cnn'):
                continue
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
        """Apply legal move mask to policy logits (same as V5/V6/V7)."""
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

    def compute_cnn_features(self, grids):
        """
        Pre-compute CNN backbone features for a batch of grid sequences.
        Use this to cache features before PPO updates (CNN is frozen, so features don't change).

        Args:
            grids: (N, K, C, 15, 15) float32
        Returns:
            (N, K, embed_dim) float32 — cached CNN features
        """
        N, K = grids.shape[:2]
        flat_grids = grids.reshape(N * K, *grids.shape[2:])
        total = flat_grids.shape[0]
        chunk_size = 256

        all_features = []
        for i in range(0, total, chunk_size):
            chunk = flat_grids[i:i + chunk_size]
            with torch.no_grad():
                feats = self.cnn._backbone(chunk)
            all_features.append(feats.detach())
        cnn_features = torch.cat(all_features, dim=0)
        cnn_features = self.cnn_proj(cnn_features)
        return cnn_features.reshape(N, K, self.embed_dim)

    def _transformer_forward(self, cnn_features, prev_actions, seq_mask):
        """
        Run transformer + gate on pre-computed CNN features.
        Shared by both _encode_sequence and forward_cached.

        Args:
            cnn_features: (B, K, embed_dim) — CNN backbone output
            prev_actions: (B, K) int64
            seq_mask: (B, K) bool
        Returns:
            (B, embed_dim)
        """
        B, K = cnn_features.shape[:2]
        device = cnn_features.device

        raw_cnn_features = cnn_features  # save for gate

        # Add action history embeddings
        act_emb = self.action_embed(prev_actions)
        turn_embeds = self.turn_norm(cnn_features + act_emb)

        # Replace padded turns with learned pad token
        pad_expanded = self.pad_token.unsqueeze(0).unsqueeze(0).expand(B, K, -1)
        turn_embeds = torch.where(
            seq_mask.unsqueeze(-1).expand_as(turn_embeds),
            pad_expanded,
            turn_embeds,
        )

        # Add temporal position embeddings
        positions = torch.arange(K, device=device)
        temporal_emb = self.temporal_pos_embed(positions)
        turn_embeds = turn_embeds + temporal_emb.unsqueeze(0)

        # Transformer forward with causal + padding mask
        causal = self.causal_mask[:K, :K]
        out = self.transformer(
            turn_embeds,
            mask=causal,
            src_key_padding_mask=seq_mask,
        )

        # Extract last valid turn's representation
        valid_mask = ~seq_mask
        last_valid_idx = (valid_mask.cumsum(dim=1) * valid_mask).argmax(dim=1)
        batch_idx = torch.arange(B, device=device)

        last_direct = raw_cnn_features[batch_idx, last_valid_idx]
        last_trans = out[batch_idx, last_valid_idx]

        gate = torch.tanh(self.transformer_alpha)
        return last_direct + gate * last_trans

    def _encode_sequence(self, grids, prev_actions, seq_mask):
        """
        Encode a sequence of K board states through CNN + temporal transformer.

        Args:
            grids: (B, K, C, 15, 15) float32 — board state grids per turn
            prev_actions: (B, K) int64 — previous action per turn (0-4)
            seq_mask: (B, K) bool — True for padded (invalid) positions

        Returns:
            (B, embed_dim) — representation of the last valid turn
        """
        B, K = grids.shape[:2]

        # Run CNN backbone on all K turns (chunked to limit memory)
        flat_grids = grids.reshape(B * K, *grids.shape[2:])
        total = flat_grids.shape[0]
        chunk_size = 256

        if total <= chunk_size:
            if self.cnn_frozen:
                with torch.no_grad():
                    cnn_features = self.cnn._backbone(flat_grids)
                cnn_features = cnn_features.detach()
            else:
                cnn_features = self.cnn._backbone(flat_grids)
        else:
            chunks = []
            for i in range(0, total, chunk_size):
                chunk = flat_grids[i:i + chunk_size]
                if self.cnn_frozen:
                    with torch.no_grad():
                        c = self.cnn._backbone(chunk)
                    chunks.append(c.detach())
                else:
                    chunks.append(self.cnn._backbone(chunk))
            cnn_features = torch.cat(chunks, dim=0)

        cnn_features = self.cnn_proj(cnn_features)
        cnn_features = cnn_features.reshape(B, K, self.embed_dim)

        return self._transformer_forward(cnn_features, prev_actions, seq_mask)

    def forward(self, grids, prev_actions, seq_mask, legal_mask=None):
        """
        Full forward pass.

        Args:
            grids: (B, K, 17, 15, 15) float32 — board states
            prev_actions: (B, K) int64 — previous actions
            seq_mask: (B, K) bool — True for padded turns
            legal_mask: (B, 4) float32 — 1.0 for legal actions

        Returns:
            policy: (B, 4) probability distribution
            value: (B, 1) predicted value
        """
        features = self._encode_sequence(grids, prev_actions, seq_mask)

        policy_logits = self.policy_head(features)
        policy_logits = self._apply_legal_mask(policy_logits, legal_mask)
        policy = F.softmax(policy_logits, dim=1)

        value = self.value_head(features)

        return policy, value

    def forward_cached(self, cached_cnn, prev_actions, seq_mask, legal_mask=None):
        """
        Forward pass using pre-computed CNN features (skips CNN backbone entirely).
        Use during PPO updates when CNN is frozen — saves ~16× compute.

        Args:
            cached_cnn: (B, K, embed_dim) — from compute_cnn_features()
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

    def compute_single_cnn_features(self, grids):
        """
        Process individual board grids through CNN backbone.
        For caching during gameplay — only process new turns.

        Args:
            grids: (B, C, 15, 15) float32 — single board states (NOT sequences)
        Returns:
            (B, embed_dim) float32
        """
        with torch.no_grad():
            features = self.cnn._backbone(grids)
        return self.cnn_proj(features.detach())

    def forward_policy_only_cached(self, cached_cnn, prev_actions, seq_mask, legal_mask=None):
        """Policy logits from pre-computed CNN features (for gameplay with caching)."""
        features = self._transformer_forward(cached_cnn, prev_actions, seq_mask)
        policy_logits = self.policy_head(features)
        return self._apply_legal_mask(policy_logits, legal_mask)

    def forward_policy_only(self, grids, prev_actions, seq_mask, legal_mask=None):
        """Fast forward pass returning only policy logits (for inference)."""
        features = self._encode_sequence(grids, prev_actions, seq_mask)
        policy_logits = self.policy_head(features)
        return self._apply_legal_mask(policy_logits, legal_mask)

    def load_v6_weights(self, checkpoint_path):
        """Load V6 CNN weights from a checkpoint file."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Clean up compiled model keys
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # Load only CNN backbone weights (ignore V6's policy/value heads)
        cnn_state = {}
        for k, v in state_dict.items():
            cnn_state[k] = v

        self.cnn.load_state_dict(cnn_state, strict=False)
        print(f"[V8] Loaded V6 CNN backbone weights from {checkpoint_path}")

        # Also load V6's trained policy/value heads — same shapes (128→64→4 and 128→64→1)
        if 'policy_fc1.weight' in state_dict:
            self.policy_head[0].weight.data.copy_(state_dict['policy_fc1.weight'])
            self.policy_head[0].bias.data.copy_(state_dict['policy_fc1.bias'])
            self.policy_head[2].weight.data.copy_(state_dict['policy_fc2.weight'])
            self.policy_head[2].bias.data.copy_(state_dict['policy_fc2.bias'])
            self.value_head[0].weight.data.copy_(state_dict['value_fc1.weight'])
            self.value_head[0].bias.data.copy_(state_dict['value_fc1.bias'])
            self.value_head[2].weight.data.copy_(state_dict['value_fc2.weight'])
            self.value_head[2].bias.data.copy_(state_dict['value_fc2.bias'])
            print(f"[V8] Loaded V6 policy/value head weights (model starts as V6)")
        else:
            print(f"[V8] WARNING: V6 policy/value head weights not found in checkpoint")

    def freeze_cnn(self):
        """Freeze CNN backbone — only transformer + heads are trainable."""
        self.cnn_frozen = True
        for param in self.cnn.parameters():
            param.requires_grad_(False)
        # Also put CNN in eval mode (affects BatchNorm/Dropout)
        self.cnn.eval()

    def unfreeze_cnn(self, lr_scale=0.1):
        """Unfreeze CNN backbone for end-to-end fine-tuning."""
        self.cnn_frozen = False
        for param in self.cnn.parameters():
            param.requires_grad_(True)
        self.cnn.train()
        print(f"[V8] CNN unfrozen (recommend {lr_scale}x learning rate for CNN params)")

    def train(self, mode=True):
        """Override train() to keep CNN in eval mode when frozen."""
        super().train(mode)
        if self.cnn_frozen and mode:
            self.cnn.eval()
        return self

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = AlphaLudoV8(context_length=16, embed_dim=128, num_heads=4, num_layers=4)
    total = model.count_all_parameters()
    trainable = model.count_parameters()
    print(f"AlphaLudo V8 — Total: {total:,} | Trainable: {trainable:,}")

    # Freeze CNN and check trainable count
    model.freeze_cnn()
    trainable_frozen = model.count_parameters()
    print(f"With CNN frozen — Trainable: {trainable_frozen:,}")

    # Test forward pass
    B, K = 4, 16
    grids = torch.randn(B, K, 17, 15, 15)
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

    logits = model.forward_policy_only(grids, prev_actions, seq_mask, legal_mask)
    print(f"Policy logits shape: {logits.shape}")
