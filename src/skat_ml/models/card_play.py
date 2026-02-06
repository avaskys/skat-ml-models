"""Card play models for Skat AI.

These models select which card to play during trick-by-trick gameplay.
"""

import torch
import torch.nn as nn

from ..constants import CARD_PLAY_POLICY_INPUT_DIM
from .base import ResBlock


class CardPlayPolicy(nn.Module):
    """
    Dense policy network for trick-by-trick card selection.

    Predicts the best card to play from the 32 possible cards.
    Input: 268 dims (Hand, Skat, History Me/L/R, Trick, Context, Ouvert)
    Output: 32 dims (Logits for each card)

    Features a gated Null Ouvert encoder that only activates for defenders
    in Null Ouvert games, allowing dedicated learning of "trap" strategies.

    Previously named: GameplayPolicy, Model C
    """

    # Feature positions in the 268-dim input vector
    IDX_GAME_NULL = 229  # v_game[5] - is this a NULL game?
    IDX_I_AM_DECLARER = 230  # v_decl[0] - am I the declarer?
    IDX_IS_OUVERT = 235  # v_ouvert - is this an ouvert game?
    IDX_VISIBLE_HAND_START = 236  # v_visible_hand starts here (32 dims)

    def __init__(
        self,
        input_dim: int = CARD_PLAY_POLICY_INPUT_DIM,
        hidden_dim: int = 512,
        num_blocks: int = 4,
    ):
        super().__init__()

        # Main pathway processes first 236 features (everything except visible_hand)
        self.input_proj = nn.Sequential(
            nn.Linear(236, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Dedicated encoder for Null Ouvert defender trap logic
        # Only activated when: NULL game + Ouvert + I'm a defender
        self.null_ouvert_encoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Blocks now process hidden_dim + 64 (for ouvert features)
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim + 64) for _ in range(num_blocks)]
        )

        # Policy head outputs logits for each card (0-31)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split features
        main_features = x[:, :236]
        visible_hand = x[:, self.IDX_VISIBLE_HAND_START:]  # Last 32 dims

        # Compute gate: 1.0 only for Null Ouvert defenders
        is_null = x[:, self.IDX_GAME_NULL]
        is_ouvert = x[:, self.IDX_IS_OUVERT]
        is_defender = 1.0 - x[:, self.IDX_I_AM_DECLARER]  # NOT declarer = defender
        gate = (is_null * is_ouvert * is_defender).unsqueeze(-1)  # Shape: [batch, 1]

        # Main pathway
        h = self.input_proj(main_features)

        # Null Ouvert encoder (gated - zeros for non-Null-Ouvert-defender)
        ouvert_h = self.null_ouvert_encoder(visible_hand) * gate

        # Combine pathways
        h = torch.cat([h, ouvert_h], dim=-1)

        # Process through residual blocks
        for block in self.blocks:
            h = block(h)

        return self.output_head(h)


class CardPlayTransformer(nn.Module):
    """
    Transformer-based gameplay policy for Skat.

    Instead of fixed feature vectors, processes the game as a sequence:
    [CLS] [GAME] [DECL] [OUVERT?] [hand cards...] [ouvert cards...] [history...] [trick...]

    Attention learns to:
    - Track which cards have been played
    - Compare trump rankings
    - Detect voids (opponent didn't follow suit)
    - Understand control dynamics
    - Use visible declarer cards in ouvert games

    Input: Variable-length sequences of tokens
    Output: 32-dim logits (one per card)

    Previously named: SkatTransformer
    """

    # Segment type IDs
    SEG_CLS = 0
    SEG_CONTEXT = 1
    SEG_HAND = 2
    SEG_HISTORY = 3
    SEG_TRICK = 4
    SEG_OUVERT = 5

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Token embeddings
        self.card_embed = nn.Embedding(32, d_model)
        self.player_embed = nn.Embedding(3, d_model)
        self.game_type_embed = nn.Embedding(6, d_model)
        self.declarer_embed = nn.Embedding(3, d_model)  # me=0, left=1, right=2
        self.is_ouvert_embed = nn.Embedding(2, d_model)  # 0=not ouvert, 1=ouvert

        # Segment embeddings (what type of token is this?)
        self.segment_embed = nn.Embedding(6, d_model)  # Added SEG_OUVERT

        # Position embeddings
        self.position_embed = nn.Embedding(max_seq_len, d_model)

        # Projection layer for (player, card) pairs - concatenate then project
        # Shared for both history and trick moves (same semantics)
        self.move_proj = nn.Linear(d_model * 2, d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Layer norm before output
        self.output_norm = nn.LayerNorm(d_model)

        # Output head: predict over 32 cards
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 32),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.card_embed.weight, std=0.02)
        nn.init.normal_(self.player_embed.weight, std=0.02)
        nn.init.normal_(self.game_type_embed.weight, std=0.02)
        nn.init.normal_(self.declarer_embed.weight, std=0.02)
        nn.init.normal_(self.is_ouvert_embed.weight, std=0.02)
        nn.init.normal_(self.segment_embed.weight, std=0.02)
        nn.init.normal_(self.position_embed.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.move_proj.weight)
        nn.init.zeros_(self.move_proj.bias)

    def forward(
        self,
        game_type: torch.Tensor,
        declarer: torch.Tensor,
        is_ouvert: torch.Tensor,
        hand: torch.Tensor,
        hand_len: torch.Tensor,
        ouvert_hand: torch.Tensor,
        ouvert_hand_len: torch.Tensor,
        history: torch.Tensor,
        history_len: torch.Tensor,
        trick: torch.Tensor,
        trick_len: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            game_type: (batch,) int - game type index (0-5)
            declarer: (batch,) int - relative declarer (0=me, 1=left, 2=right)
            is_ouvert: (batch,) int - 0=not ouvert, 1=ouvert game
            hand: (batch, max_hand) int - card indices in hand (padded)
            hand_len: (batch,) int - actual hand length
            ouvert_hand: (batch, max_ouvert) int - declarer's visible cards (padded)
            ouvert_hand_len: (batch,) int - actual ouvert hand length (0 if not applicable)
            history: (batch, max_history, 2) int - (player, card) pairs (padded)
            history_len: (batch,) int - actual history length
            trick: (batch, max_trick, 2) int - current trick (player, card) pairs
            trick_len: (batch,) int - cards in current trick (0-2)
            legal_mask: (batch, 32) bool - which cards are legal to play

        Returns:
            logits: (batch, 32) - score for each card
        """
        batch_size = game_type.size(0)
        device = game_type.device

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

        # 3. Declarer token
        decl_emb = self.declarer_embed(declarer).unsqueeze(1)
        embeddings.append(decl_emb)
        segments.append(
            torch.full((batch_size, 1), self.SEG_CONTEXT, dtype=torch.long, device=device)
        )

        # 4. Is Ouvert token
        ouvert_emb = self.is_ouvert_embed(is_ouvert).unsqueeze(1)
        embeddings.append(ouvert_emb)
        segments.append(
            torch.full((batch_size, 1), self.SEG_CONTEXT, dtype=torch.long, device=device)
        )

        # 5. Hand cards
        hand_emb = self.card_embed(hand)  # (batch, max_hand, d_model)
        embeddings.append(hand_emb)
        segments.append(
            torch.full(
                (batch_size, hand.size(1)), self.SEG_HAND, dtype=torch.long, device=device
            )
        )

        # 6. Ouvert hand cards (declarer's visible cards, for defenders in ouvert games)
        ouvert_hand_emb = self.card_embed(ouvert_hand)  # (batch, max_ouvert, d_model)
        embeddings.append(ouvert_hand_emb)
        segments.append(
            torch.full(
                (batch_size, ouvert_hand.size(1)),
                self.SEG_OUVERT,
                dtype=torch.long,
                device=device,
            )
        )

        # 7. History moves (player + card combined via learned projection)
        hist_player_emb = self.player_embed(history[:, :, 0])
        hist_card_emb = self.card_embed(history[:, :, 1])
        hist_emb = self.move_proj(torch.cat([hist_player_emb, hist_card_emb], dim=-1))
        embeddings.append(hist_emb)
        segments.append(
            torch.full(
                (batch_size, history.size(1)),
                self.SEG_HISTORY,
                dtype=torch.long,
                device=device,
            )
        )

        # 8. Current trick cards (same projection as history)
        trick_player_emb = self.player_embed(trick[:, :, 0])
        trick_card_emb = self.card_embed(trick[:, :, 1])
        trick_emb = self.move_proj(torch.cat([trick_player_emb, trick_card_emb], dim=-1))
        embeddings.append(trick_emb)
        segments.append(
            torch.full(
                (batch_size, trick.size(1)), self.SEG_TRICK, dtype=torch.long, device=device
            )
        )

        # Concatenate all embeddings
        x = torch.cat(embeddings, dim=1)  # (batch, seq_len, d_model)
        seg = torch.cat(segments, dim=1)  # (batch, seq_len)
        seq_len = x.size(1)

        # Add segment embeddings
        x = x + self.segment_embed(seg)

        # Add position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.position_embed(positions)

        # Create attention mask for padding
        attn_mask = self._create_padding_mask(
            batch_size,
            seq_len,
            hand_len,
            ouvert_hand_len,
            history_len,
            trick_len,
            hand.size(1),
            ouvert_hand.size(1),
            history.size(1),
            trick.size(1),
            device,
        )

        # Transformer forward
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Get CLS token output
        cls_output = x[:, 0]  # (batch, d_model)
        cls_output = self.output_norm(cls_output)

        # Compute logits
        logits = self.output_head(cls_output)  # (batch, 32)

        # Mask illegal moves
        logits = logits.masked_fill(~legal_mask, float("-inf"))

        return logits

    def _create_padding_mask(
        self,
        batch_size: int,
        seq_len: int,
        hand_len: torch.Tensor,
        ouvert_len: torch.Tensor,
        history_len: torch.Tensor,
        trick_len: torch.Tensor,
        max_hand: int,
        max_ouvert: int,
        max_history: int,
        max_trick: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create attention mask where True = masked (ignored) position.
        Vectorized implementation for better performance.

        Sequence structure:
        [CLS] [GAME] [DECL] [OUVERT?] [hand...] [ouvert...] [history...] [trick...]
          0     1      2       3      4:4+H     4+H:4+H+O   4+H+O:...    ...
        """
        # Create position indices for each section
        hand_start = 4  # After CLS, GAME, DECL, OUVERT tokens
        ouvert_start = hand_start + max_hand
        hist_start = ouvert_start + max_ouvert
        trick_start = hist_start + max_history

        # Create range tensors for comparison
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)

        # Hand mask: position >= hand_start + hand_len AND position < ouvert_start
        hand_end = hand_start + hand_len.unsqueeze(1)  # (batch, 1)
        hand_mask = (positions >= hand_end) & (positions < ouvert_start)

        # Ouvert mask: position >= ouvert_start + ouvert_len AND position < hist_start
        ouvert_end = ouvert_start + ouvert_len.unsqueeze(1)  # (batch, 1)
        ouvert_mask = (positions >= ouvert_end) & (positions < hist_start)

        # History mask: position >= hist_start + history_len AND position < trick_start
        hist_end = hist_start + history_len.unsqueeze(1)  # (batch, 1)
        hist_mask = (positions >= hist_end) & (positions < trick_start)

        # Trick mask: position >= trick_start + trick_len AND position < seq_len
        trick_end = trick_start + trick_len.unsqueeze(1)  # (batch, 1)
        trick_mask = (positions >= trick_end) & (positions < seq_len)

        # Combine all masks
        mask = hand_mask | ouvert_mask | hist_mask | trick_mask

        return mask
