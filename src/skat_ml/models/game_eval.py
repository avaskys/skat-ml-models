"""Game evaluation models for Skat AI.

These models evaluate hand+skat+game-type combinations for discard and game selection.
"""

import torch
import torch.nn as nn

from ..constants import GAME_EVALUATOR_INPUT_DIM, MAX_HAND, MAX_SKAT
from .base import ResBlock


class GameEvaluator(nn.Module):
    """
    Dense model for evaluating hand after skat pickup/exchange.

    Predicts win probability for a specific game configuration (hand + skat + game type).
    Used for discard optimization: searches all 66 pairs × all game types.

    Input: 75 features (32 hand + 32 skat + 6 game + 3 pos + 1 hand_flag + 1 bid)
    Output: Win probability (scalar)

    Previously named: PostSkatEvaluator, Model B
    """

    def __init__(
        self,
        input_dim: int = GAME_EVALUATOR_INPUT_DIM,
        hidden_dim: int = 512,
        num_blocks: int = 4,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])

        # Output logits (no sigmoid) for AMP-safe training
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_head(x).squeeze(-1)


class GameEvaluatorTransformer(nn.Module):
    """
    Transformer-based post-skat hand evaluator.

    Uses attention to learn card relationships:
    - Singleton detection (a 10 with no suit support is dangerous)
    - Card protection (A-T in same suit is safe)
    - Skat value (high cards in skat = safe points)

    Input sequence:
    [CLS] [GAME] [POS] [HAND_FLAG] [BID] [hand cards x10] [skat cards x0-2]

    No positional encoding on cards = permutation invariant within hand/skat.
    Segment embeddings distinguish context / hand / skat.

    Previously named: CardSetEvaluator
    """

    # Segment type IDs
    SEG_CLS = 0
    SEG_CONTEXT = 1
    SEG_HAND = 2
    SEG_SKAT = 3

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Card embeddings (32 cards + 1 padding token)
        self.card_embed = nn.Embedding(33, d_model, padding_idx=32)

        # Context embeddings
        self.game_type_embed = nn.Embedding(6, d_model)  # 6 game types
        self.position_embed = nn.Embedding(3, d_model)  # 3 positions
        self.hand_flag_embed = nn.Embedding(2, d_model)  # pickup=0, hand=1

        # Segment embeddings (what type of token is this?)
        self.segment_embed = nn.Embedding(4, d_model)  # CLS, CTX, HAND, SKAT

        # Bid projection (normalize bid to 0-1 range, then project)
        self.bid_proj = nn.Linear(1, d_model)

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

        # Output head - logits (no sigmoid) for AMP-safe training
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.card_embed.weight, std=0.02)
        nn.init.normal_(self.game_type_embed.weight, std=0.02)
        nn.init.normal_(self.position_embed.weight, std=0.02)
        nn.init.normal_(self.hand_flag_embed.weight, std=0.02)
        nn.init.normal_(self.segment_embed.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.bid_proj.weight, std=0.02)
        nn.init.zeros_(self.bid_proj.bias)

    def forward(
        self,
        hand_cards: torch.Tensor,
        skat_cards: torch.Tensor,
        skat_len: torch.Tensor,
        game_type: torch.Tensor,
        position: torch.Tensor,
        is_hand: torch.Tensor,
        bid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hand_cards: (batch, 10) int - card indices in hand (0-31)
            skat_cards: (batch, 2) int - card indices in skat (0-31), padded with 32
            skat_len: (batch,) int - actual skat length (0, 1, or 2)
            game_type: (batch,) int - game type index (0-5)
            position: (batch,) int - player position (0-2)
            is_hand: (batch,) int - 0=pickup game, 1=hand game
            bid: (batch,) float - normalized bid value (0-1)

        Returns:
            win_logit: (batch,) float - logit for win probability

        Note: Apply torch.sigmoid() to convert to probability for inference.
        """
        batch_size = hand_cards.size(0)
        device = hand_cards.device

        embeddings = []
        segments = []

        # 1. CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        embeddings.append(cls)
        segments.append(
            torch.full((batch_size, 1), self.SEG_CLS, dtype=torch.long, device=device)
        )

        # 2. Game type token
        game_emb = self.game_type_embed(game_type).unsqueeze(1)
        embeddings.append(game_emb)
        segments.append(
            torch.full((batch_size, 1), self.SEG_CONTEXT, dtype=torch.long, device=device)
        )

        # 3. Position token
        pos_emb = self.position_embed(position).unsqueeze(1)
        embeddings.append(pos_emb)
        segments.append(
            torch.full((batch_size, 1), self.SEG_CONTEXT, dtype=torch.long, device=device)
        )

        # 4. Hand flag token
        hand_flag_emb = self.hand_flag_embed(is_hand).unsqueeze(1)
        embeddings.append(hand_flag_emb)
        segments.append(
            torch.full((batch_size, 1), self.SEG_CONTEXT, dtype=torch.long, device=device)
        )

        # 5. Bid token
        bid_emb = self.bid_proj(bid.unsqueeze(-1)).unsqueeze(1)
        embeddings.append(bid_emb)
        segments.append(
            torch.full((batch_size, 1), self.SEG_CONTEXT, dtype=torch.long, device=device)
        )

        # 6. Hand cards (always 10)
        hand_emb = self.card_embed(hand_cards)  # (batch, 10, d_model)
        embeddings.append(hand_emb)
        segments.append(
            torch.full((batch_size, MAX_HAND), self.SEG_HAND, dtype=torch.long, device=device)
        )

        # 7. Skat cards (0-2, padded)
        skat_emb = self.card_embed(skat_cards)  # (batch, 2, d_model)
        embeddings.append(skat_emb)
        segments.append(
            torch.full((batch_size, MAX_SKAT), self.SEG_SKAT, dtype=torch.long, device=device)
        )

        # Concatenate all embeddings
        x = torch.cat(embeddings, dim=1)  # (batch, seq_len, d_model)
        seg = torch.cat(segments, dim=1)  # (batch, seq_len)

        # Add segment embeddings (NOT positional - permutation invariance within hand/skat)
        x = x + self.segment_embed(seg)

        # Create attention mask for padding (skat cards beyond skat_len)
        # Sequence: [CLS, GAME, POS, HAND_FLAG, BID, hand x10, skat x2]
        # Indices:    0     1    2      3       4    5-14       15-16
        seq_len = x.size(1)
        skat_start = 5 + MAX_HAND  # = 15

        # Create mask: True = masked (ignored)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        skat_end = skat_start + skat_len.unsqueeze(1)  # (batch, 1)
        attn_mask = (positions >= skat_end) & (positions >= skat_start)

        # Transformer forward
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Get CLS token output
        cls_output = x[:, 0]  # (batch, d_model)
        cls_output = self.output_norm(cls_output)

        # Predict win probability
        win_prob = self.output_head(cls_output).squeeze(-1)

        return win_prob
