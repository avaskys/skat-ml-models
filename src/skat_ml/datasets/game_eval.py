"""Streaming dataset for game evaluation model training."""

import random
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from torch.utils.data import IterableDataset

from ..constants import CARD_PAD_IDX, GAME_TYPE_TO_IDX, MAX_BID, MAX_SKAT, MIN_BID
from .sgf_parser import PassedGameError, PenaltyGameError, parse_sgf_line


class StreamingGameEvalDataset(IterableDataset):
    """
    Streaming dataset for GameEvaluator and GameEvaluatorTransformer training.

    Yields one sample per game containing:
    - hand_cards: 10 card indices (declarer's final playing hand)
    - skat_cards: 0-2 card indices (what was discarded, or empty for hand games)
    - game_type, position, is_hand, bid
    - label: win (1.0) or loss (0.0)

    Streams directly from SGF file - no preprocessing required.
    """

    MAX_HAND = 10

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

        # Stream with shuffle buffer
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
        """Extract game evaluation features from a parsed game."""

        # Get declarer's initial hand
        declarer = game.declarer
        initial_hand = set(game.initial_hands[declarer])

        if len(initial_hand) != 10:
            return None

        # Determine final hand and skat cards
        if game.is_hand_game:
            # Hand game: declarer plays with initial hand, skat unknown
            final_hand = initial_hand
            skat_cards = []
        else:
            # Pickup game: declarer picked up skat and discarded
            skat = set(game.skat) if game.skat else set()
            discards = set(game.discards) if game.discards else set()

            if len(skat) != 2:
                return None

            # Final hand = initial + skat - discards
            full_hand = initial_hand | skat
            final_hand = full_hand - discards

            if len(final_hand) != 10:
                return None

            # Skat cards = what was discarded (0, 1, or 2 cards)
            skat_cards = list(discards)

        # Cards are already indices from sgf_parser
        hand_indices = list(final_hand)
        skat_indices = list(skat_cards)

        # Game type
        game_type_idx = GAME_TYPE_TO_IDX.get(game.game_type, 0)

        # Position (declarer position: 0=forehand, 1=middlehand, 2=rearhand)
        position = declarer

        # Normalize bid to 0-1 range
        bid_normalized = (game.bid_level - MIN_BID) / (MAX_BID - MIN_BID)

        # Label: 1.0 if declarer won, 0.0 if lost
        label = 1.0 if game.won else 0.0

        # Build pre-padded arrays
        hand_arr = np.array(sorted(hand_indices), dtype=np.int64)

        skat_arr = np.full(MAX_SKAT, CARD_PAD_IDX, dtype=np.int64)
        skat_len = len(skat_indices)
        if skat_len > 0:
            skat_arr[:skat_len] = sorted(skat_indices)

        return {
            "hand_cards": hand_arr,
            "skat_cards": skat_arr,
            "skat_len": skat_len,
            "game_type": game_type_idx,
            "position": position,
            "is_hand": 1 if game.is_hand_game else 0,
            "bid": bid_normalized,
            "label": label,
        }


def game_eval_collate_fn(batch: list) -> dict:
    """Collate function for GameEvaluator/Transformer DataLoader."""

    return {
        "hand_cards": torch.from_numpy(np.stack([s["hand_cards"] for s in batch])),
        "skat_cards": torch.from_numpy(np.stack([s["skat_cards"] for s in batch])),
        "skat_len": torch.tensor([s["skat_len"] for s in batch], dtype=torch.long),
        "game_type": torch.tensor([s["game_type"] for s in batch], dtype=torch.long),
        "position": torch.tensor([s["position"] for s in batch], dtype=torch.long),
        "is_hand": torch.tensor([s["is_hand"] for s in batch], dtype=torch.long),
        "bid": torch.tensor([s["bid"] for s in batch], dtype=torch.float32),
        "label": torch.tensor([s["label"] for s in batch], dtype=torch.float32),
    }
