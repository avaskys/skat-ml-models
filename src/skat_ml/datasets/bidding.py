"""Streaming dataset for bidding model training."""

import random
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from torch.utils.data import IterableDataset

from ..constants import BASE_VALUES, BID_VALUES, NUM_BID_LEVELS
from .sgf_parser import PassedGameError, PenaltyGameError, parse_sgf_line


class StreamingBiddingDataset(IterableDataset):
    """
    Streaming dataset for BiddingEvaluator and BiddingTransformer training.

    Yields one sample per game containing:
    - hand_cards: 10 card indices (declarer's initial hand, before skat)
    - position: declarer position (0-2)
    - label_pickup: (63,) win probability at each bid level for pickup game
    - label_hand: (63,) win probability at each bid level for hand game
    - weight: sample weight (hand games weighted higher)

    Labels use masking: -1.0 = unknown (ignored in loss)

    Streams directly from SGF file - no preprocessing required.
    """

    NUM_CARDS = 10

    def __init__(
        self,
        sgf_path: Path,
        split: str = "train",
        shuffle_buffer: int = 50000,
    ):
        """
        Args:
            sgf_path: Path to SGF file.
            split: 'train' (90%) or 'val' (10%).
            shuffle_buffer: Number of samples to buffer for shuffling.
        """
        self.sgf_path = Path(sgf_path)
        self.split = split
        self.shuffle_buffer = shuffle_buffer
        self.file_size = self.sgf_path.stat().st_size

    def __iter__(self) -> Generator[dict, None, None]:
        """Yields training samples with shuffling."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            per_worker = self.file_size // worker_info.num_workers
            start_pos = worker_info.id * per_worker
            end_pos = start_pos + per_worker
        else:
            start_pos = 0
            end_pos = float("inf")

        # Stream with optional shuffle buffer
        if self.shuffle_buffer <= 0:
            # No shuffling - yield directly
            for sample in self._stream_samples(start_pos, end_pos):
                yield sample
        else:
            buffer = []
            for sample in self._stream_samples(start_pos, end_pos):
                if len(buffer) < self.shuffle_buffer:
                    buffer.append(sample)
                else:
                    idx = random.randrange(self.shuffle_buffer)
                    yield buffer[idx]
                    buffer[idx] = sample

            # Flush remaining buffer
            random.shuffle(buffer)
            for sample in buffer:
                yield sample

    def _stream_samples(
        self, start_pos: int, end_pos: float
    ) -> Generator[dict, None, None]:
        """Internal generator that reads from file and yields samples."""

        with open(self.sgf_path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(start_pos)
            if start_pos != 0:
                f.readline()  # Discard partial line

            while f.tell() <= end_pos:
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue

                try:
                    game = parse_sgf_line(line)
                    if not game:
                        continue

                    # Train/val split based on game ID
                    is_val = game.game_id % 10 == 0
                    if self.split == "train" and is_val:
                        continue
                    if self.split == "val" and not is_val:
                        continue

                    sample = self._process_game(game)
                    if sample is not None:
                        yield sample

                except (ValueError, KeyError, AttributeError, PassedGameError, PenaltyGameError):
                    # Skip malformed or invalid game records
                    continue

    def _process_game(self, game) -> dict | None:
        """Extract bidding features and labels from a parsed game."""

        # Get declarer's initial hand (before skat)
        declarer = game.declarer
        initial_hand = list(game.initial_hands[declarer])

        if len(initial_hand) != 10:
            return None

        # Compute labels
        base_value = BASE_VALUES.get(game.game_type, 9)
        label_pickup, label_hand, weight = self._compute_labels(
            is_hand_game=game.is_hand_game,
            won=game.won,
            game_value=game.game_value,
            bid_level=game.bid_level,
            base_value=base_value,
            matadors=game.matadors,
            schneider=game.schneider,
            schwarz=game.schwarz,
            is_ouvert=game.is_ouvert,
            game_type=game.game_type,
        )

        # Cards are already indices from sgf_parser
        hand_arr = np.array(sorted(initial_hand), dtype=np.int64)

        return {
            "hand_cards": hand_arr,
            "position": declarer,
            "label_pickup": label_pickup,
            "label_hand": label_hand,
            "weight": weight,
        }

    def _compute_labels(
        self,
        is_hand_game: bool,
        won: bool,
        game_value: int,
        bid_level: int,
        base_value: int,
        matadors: int,
        schneider: bool,
        schwarz: bool,
        is_ouvert: bool,
        game_type: str,
    ) -> tuple:
        """
        Compute bidding model labels.

        Returns:
        - pickup_labels: (63,) probability targets for pickup game
        - hand_labels: (63,) probability targets for hand game
        - weight: sample weight (float)

        Values: 1.0 = win, 0.0 = loss, -1.0 = masked (ignored)
        """
        pickup = np.full(NUM_BID_LEVELS, -1.0, dtype=np.float32)
        hand = np.full(NUM_BID_LEVELS, -1.0, dtype=np.float32)
        weight = 1.0

        if not is_hand_game:
            # Pickup game
            if won:
                # Case 1: Pickup + Won at game_value
                for i, threshold in enumerate(BID_VALUES):
                    pickup[i] = 1.0 if threshold <= game_value else 0.0
            else:
                # Case 2: Pickup + Lost at bid_level
                for i, threshold in enumerate(BID_VALUES):
                    if threshold >= bid_level:
                        pickup[i] = 0.0
                    # else: stays masked

            # Inverse imputation: if player chose Pickup, Hand was probably suboptimal
            for i, threshold in enumerate(BID_VALUES):
                if threshold >= 18:
                    hand[i] = 0.0
        else:
            # Hand game - Dynamic Weighting
            weight = 10.0  # Base boost for Hand games

            if won:
                if game_value >= 144:
                    weight = 50.0  # Monster hands
                elif game_value >= 100:
                    weight = 20.0  # Very strong hands

                # Reconstruct theoretical scores from game physics
                base_multiplier = 1 + abs(matadors)
                if schneider:
                    base_multiplier += 1
                if schwarz:
                    base_multiplier += 1

                if game_type == "NULL":
                    if is_ouvert:
                        hand_score = 59  # Null Ouvert Hand
                        pickup_score = 46  # Null Ouvert
                    else:
                        hand_score = 35  # Null Hand
                        pickup_score = 23  # Null
                else:
                    hand_multiplier = base_multiplier + 1  # +1 for Hand
                    if is_ouvert:
                        hand_multiplier += 1

                    hand_score = hand_multiplier * base_value
                    pickup_score = base_multiplier * base_value

                # Case 3: Hand + Won
                for i, threshold in enumerate(BID_VALUES):
                    hand[i] = 1.0 if threshold <= hand_score else 0.0

                # Impute pickup: winning hand implies winning pickup at theoretical score
                for i, threshold in enumerate(BID_VALUES):
                    if threshold <= pickup_score:
                        pickup[i] = 1.0
                    # else: stays masked
            else:
                # Case 4: Hand + Lost at bid_level
                for i, threshold in enumerate(BID_VALUES):
                    if threshold >= bid_level:
                        hand[i] = 0.0
                    # else: stays masked

        return pickup, hand, weight


def bidding_collate_fn(batch: list) -> dict:
    """Collate function for BiddingEvaluator/Transformer DataLoader."""

    return {
        "hand_cards": torch.from_numpy(np.stack([s["hand_cards"] for s in batch])),
        "position": torch.tensor([s["position"] for s in batch], dtype=torch.long),
        "label_pickup": torch.from_numpy(np.stack([s["label_pickup"] for s in batch])),
        "label_hand": torch.from_numpy(np.stack([s["label_hand"] for s in batch])),
        "weight": torch.tensor([s["weight"] for s in batch], dtype=torch.float32),
    }
