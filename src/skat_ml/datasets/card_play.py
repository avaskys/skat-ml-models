"""Streaming dataset for card play model training."""

import random
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from torch.utils.data import IterableDataset

from ..constants import (
    CARD_TO_IDX,
    CARDS,
    GAME_TYPE_TO_IDX,
    MAX_HAND,
    MAX_HISTORY,
    MAX_OUVERT,
    MAX_TRICK,
)
from .sgf_parser import GameRecord, PassedGameError, PenaltyGameError, parse_sgf_line


class FastCardPlayDataset(IterableDataset):
    """
    Optimized streaming dataset for CardPlayPolicy and CardPlayTransformer training.

    Replays games from SGF files and yields samples for each card play decision.
    Pre-pads to fixed sizes and yields numpy arrays for faster data loading.

    Features:
    - Streams directly from SGF files
    - Winner-only filtering for imitation learning
    - Multi-worker support with file sharding
    - Shuffle buffer for randomization
    """

    def __init__(
        self,
        sgf_path: Path,
        winner_only: bool = True,
        split: str = "train",
        shuffle_buffer: int = 50000,
    ):
        """
        Args:
            sgf_path: Path to SGF file.
            winner_only: If True, only train on winning players.
            split: 'train' (90%) or 'val' (10%).
            shuffle_buffer: Number of samples to buffer for local shuffling.
        """
        self.sgf_path = Path(sgf_path)
        self.winner_only = winner_only
        self.split = split
        self.shuffle_buffer = shuffle_buffer
        self.file_size = sgf_path.stat().st_size

    def __iter__(self) -> Generator[dict, None, None]:
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            per_worker = self.file_size // worker_info.num_workers
            start_pos = worker_info.id * per_worker
            end_pos = start_pos + per_worker
        else:
            start_pos = 0
            end_pos = float("inf")

        # Shuffle buffer
        buffer = []

        for sample in self._stream_samples(start_pos, end_pos):
            if len(buffer) < self.shuffle_buffer:
                buffer.append(sample)
            else:
                idx = random.randrange(self.shuffle_buffer)
                yield buffer[idx]
                buffer[idx] = sample

        random.shuffle(buffer)
        for sample in buffer:
            yield sample

    def _stream_samples(
        self, start_pos: int, end_pos: float
    ) -> Generator[dict, None, None]:
        import re

        with open(self.sgf_path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(start_pos)
            if start_pos != 0:
                f.readline()

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

                    is_val = game.game_id % 10 == 0
                    if self.split == "train" and is_val:
                        continue
                    if self.split == "val" and not is_val:
                        continue

                    yield from self._process_game(game, line)
                except (ValueError, KeyError, AttributeError, PassedGameError, PenaltyGameError):
                    # Skip malformed or invalid game records
                    continue

    def _process_game(
        self, game: GameRecord, raw_line: str
    ) -> Generator[dict, None, None]:
        """Replay a game and yield samples for each card play decision."""
        import re

        mv_match = re.search(r"MV\[([^\]]+)\]", raw_line)
        if not mv_match:
            return

        moves = mv_match.group(1).split()

        # Find start of play
        play_start_idx = -1
        for i, token in enumerate(moves):
            if i < 2:
                continue
            if token in CARD_TO_IDX:
                play_start_idx = i
                break

        if play_start_idx == -1:
            return

        play_sequence = moves[play_start_idx:]
        if len(play_sequence) < 30:  # Incomplete game
            return

        # Initialize hands
        hands = [set(h) for h in game.initial_hands]

        if not game.is_hand_game and game.discards:
            full = set(game.initial_hands[game.declarer]) | set(game.skat)
            for d in game.discards:
                if d in full:
                    full.remove(d)
            hands[game.declarer] = full

        if any(len(h) != 10 for h in hands):
            return

        game_type_idx = GAME_TYPE_TO_IDX.get(game.game_type, 0)

        # Replay tricks
        history_list = []  # (player, card) tuples
        trick_list = []
        current_player = 0

        for card_str in play_sequence:
            if card_str not in CARD_TO_IDX:
                continue

            card_idx = CARD_TO_IDX[card_str]
            if card_idx not in hands[current_player]:
                return

            is_declarer = current_player == game.declarer
            is_winner_team = is_declarer == game.won

            if not self.winner_only or is_winner_team:
                # Compute relative positions
                if game.declarer == current_player:
                    rel_declarer = 0
                elif game.declarer == (current_player + 1) % 3:
                    rel_declarer = 1
                else:
                    rel_declarer = 2

                # Pre-padded hand array
                hand_arr = np.zeros(MAX_HAND, dtype=np.int64)
                hand_cards = sorted(hands[current_player])
                hand_arr[: len(hand_cards)] = hand_cards
                hand_len = len(hand_cards)

                # Pre-padded history array
                history_arr = np.zeros((MAX_HISTORY, 2), dtype=np.int64)
                hist_len = min(len(history_list), MAX_HISTORY)
                for i, (abs_p, c) in enumerate(history_list[-MAX_HISTORY:]):
                    rel_p = (abs_p - current_player) % 3
                    history_arr[i, 0] = rel_p
                    history_arr[i, 1] = c

                # Pre-padded trick array
                trick_arr = np.zeros((MAX_TRICK, 2), dtype=np.int64)
                trick_len = len(trick_list)
                for i, (abs_p, c) in enumerate(trick_list):
                    rel_p = (abs_p - current_player) % 3
                    trick_arr[i, 0] = rel_p
                    trick_arr[i, 1] = c

                # Legal mask
                legal_mask = np.zeros(32, dtype=np.float32)
                for c in hands[current_player]:
                    legal_mask[c] = 1.0

                # Ouvert hand: declarer's visible cards for defenders
                ouvert_arr = np.zeros(MAX_OUVERT, dtype=np.int64)
                ouvert_len = 0
                if game.is_ouvert and current_player != game.declarer:
                    declarer_cards = sorted(hands[game.declarer])
                    ouvert_len = len(declarer_cards)
                    ouvert_arr[:ouvert_len] = declarer_cards

                yield {
                    "game_type": game_type_idx,
                    "declarer": rel_declarer,
                    "is_ouvert": 1 if game.is_ouvert else 0,
                    "hand": hand_arr,
                    "hand_len": hand_len,
                    "ouvert_hand": ouvert_arr,
                    "ouvert_hand_len": ouvert_len,
                    "history": history_arr,
                    "history_len": hist_len,
                    "trick": trick_arr,
                    "trick_len": trick_len,
                    "legal_mask": legal_mask,
                    "target": card_idx,
                }

            # Update state
            hands[current_player].remove(card_idx)
            trick_list.append((current_player, card_idx))

            if len(trick_list) == 3:
                winner = self._trick_winner(trick_list, game.game_type)
                history_list.extend(trick_list)
                trick_list = []
                current_player = winner
            else:
                current_player = (current_player + 1) % 3

    def _trick_winner(self, trick: list, game_type: str) -> int:
        """Simplified trick winner calculation."""
        lead_idx = trick[0][1]
        lead_suit = CARDS[lead_idx][0]

        winner_i, winner_card = 0, lead_idx

        for i in range(1, 3):
            card_idx = trick[i][1]
            if self._beats(card_idx, winner_card, lead_suit, game_type):
                winner_i, winner_card = i, card_idx

        return trick[winner_i][0]

    def _beats(self, challenger: int, current: int, lead_suit: str, gt: str) -> bool:
        """Does challenger beat current winner?"""
        c_card, w_card = CARDS[challenger], CARDS[current]
        c_suit, c_rank = c_card[0], c_card[1]
        w_suit, w_rank = w_card[0], w_card[1]

        def is_trump(suit: str, rank: str) -> bool:
            if gt == "NULL":
                return False
            if rank == "J":
                return True
            if gt == "GRAND":
                return False
            return suit == gt[0]

        def power(suit: str, rank: str) -> int:
            if gt == "NULL":
                return "789TJQKA".index(rank)
            base = {"7": 0, "8": 1, "9": 2, "Q": 3, "K": 4, "T": 5, "A": 6, "J": 100}[
                rank
            ]
            if rank == "J":
                base += {"D": 1, "H": 2, "S": 3, "C": 4}[suit]
            return base

        c_trump = is_trump(c_suit, c_rank)
        w_trump = is_trump(w_suit, w_rank)

        if w_trump:
            return c_trump and power(c_suit, c_rank) > power(w_suit, w_rank)
        if c_trump:
            return True
        if c_suit == lead_suit:
            return power(c_suit, c_rank) > power(w_suit, w_rank)
        return False


def card_play_collate_fn(batch: list) -> dict:
    """Collate function for CardPlayPolicy/Transformer DataLoader."""

    return {
        "game_type": torch.tensor([s["game_type"] for s in batch], dtype=torch.long),
        "declarer": torch.tensor([s["declarer"] for s in batch], dtype=torch.long),
        "is_ouvert": torch.tensor([s["is_ouvert"] for s in batch], dtype=torch.long),
        "hand": torch.from_numpy(np.stack([s["hand"] for s in batch])),
        "hand_len": torch.tensor([s["hand_len"] for s in batch], dtype=torch.long),
        "ouvert_hand": torch.from_numpy(np.stack([s["ouvert_hand"] for s in batch])),
        "ouvert_hand_len": torch.tensor(
            [s["ouvert_hand_len"] for s in batch], dtype=torch.long
        ),
        "history": torch.from_numpy(np.stack([s["history"] for s in batch])),
        "history_len": torch.tensor([s["history_len"] for s in batch], dtype=torch.long),
        "trick": torch.from_numpy(np.stack([s["trick"] for s in batch])),
        "trick_len": torch.tensor([s["trick_len"] for s in batch], dtype=torch.long),
        "legal_mask": torch.from_numpy(np.stack([s["legal_mask"] for s in batch])).bool(),
        "target": torch.tensor([s["target"] for s in batch], dtype=torch.long),
    }
