"""Bidding phase models for Skat AI.

These models evaluate hand strength before seeing the skat to guide bidding decisions.
"""

import torch
import torch.nn as nn

from ..constants import BIDDING_EVALUATOR_INPUT_DIM, NUM_BID_LEVELS
from .base import ResBlock


class BiddingEvaluator(nn.Module):
    """
    Dense model for evaluating hand strength before skat pickup.

    Predicts win probability at each bid level for both pickup and hand game modes.

    Input: 35 features (32 cards one-hot + 3 position one-hot)
    Output: Two heads - pickup_probs[63] and hand_probs[63]

    Previously named: PreSkatEvaluator, Model A
    """

    def __init__(
        self,
        input_dim: int = BIDDING_EVALUATOR_INPUT_DIM,
        num_thresholds: int = NUM_BID_LEVELS,
        hidden_dim: int = 512,
        num_blocks: int = 4,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])

        # Heads - output logits (no sigmoid) for AMP-safe training
        self.pickup_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_thresholds),
        )

        self.hand_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_thresholds),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)

        pickup = self.pickup_head(x)
        hand = self.hand_head(x)
        return pickup, hand


class BiddingTransformer(nn.Module):
    """
    Transformer-based model for evaluating hand strength before skat pickup.

    Uses attention to learn card relationships for bidding:
    - Jack combinations (matador estimation)
    - Suit length patterns
    - Ace/10 density and protection

    Input: 10 hand cards (indices) + position (0-2)
    Output: pickup_probs[63], hand_probs[63]

    Previously named: PreSkatTransformer
    """

    NUM_CARDS = 10

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Card embeddings (32 cards, no padding needed for pre-skat)
        self.card_embed = nn.Embedding(32, d_model)

        # Position embedding (Forehand=0, Middlehand=1, Rearhand=2)
        self.position_embed = nn.Embedding(3, d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output heads
        self.output_norm = nn.LayerNorm(d_model)

        # Separate heads for pickup and hand (position is now added before transformer)
        # Output logits (no sigmoid) for AMP-safe training with BCEWithLogitsLoss
        self.pickup_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_BID_LEVELS),
        )

        self.hand_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_BID_LEVELS),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.card_embed.weight)
        nn.init.xavier_uniform_(self.position_embed.weight)
        nn.init.xavier_uniform_(self.cls_token)

    def forward(
        self, hand_cards: torch.Tensor, position: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hand_cards: (batch, 10) card indices (0-31)
            position: (batch,) position indices (0-2)

        Returns:
            pickup_logits: (batch, 63) logits for win probability at each bid level for pickup
            hand_logits: (batch, 63) logits for win probability at each bid level for hand game

        Note: Apply torch.sigmoid() to convert logits to probabilities for inference.
        """
        batch_size = hand_cards.size(0)

        # Embed cards
        card_emb = self.card_embed(hand_cards)  # (batch, 10, d_model)

        # Add position embedding to all card embeddings (broadcast)
        pos_emb = self.position_embed(position).unsqueeze(1)  # (batch, 1, d_model)
        card_emb = card_emb + pos_emb  # (batch, 10, d_model)

        # Add CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls, card_emb], dim=1)  # (batch, 11, d_model)

        # Transformer forward (no masking needed - all 10 cards always present)
        x = self.transformer(x)

        # Get CLS token output
        cls_output = x[:, 0]  # (batch, d_model)
        cls_output = self.output_norm(cls_output)

        # Predict probabilities for each bid level
        pickup_probs = self.pickup_head(cls_output)  # (batch, 63)
        hand_probs = self.hand_head(cls_output)  # (batch, 63)

        return pickup_probs, hand_probs
