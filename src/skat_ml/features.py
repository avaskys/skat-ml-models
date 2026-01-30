"""Feature extraction for Skat hands."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import CARD_POINTS, CARDS, GAME_TYPE_TO_IDX


def card_points(card_idx: int) -> int:
    """Get point value for a card index."""
    card = CARDS[card_idx]
    rank = card[1]  # Second character is rank
    return CARD_POINTS[rank]


def extract_bidding_features(cards: List[int], position: int) -> np.ndarray:
    """
    Extract features for BiddingEvaluator (dense model).

    Args:
        cards: List of card indices (0-31) in hand (10 cards)
        position: Player position (0=forehand, 1=middlehand, 2=rearhand)

    Returns:
        Feature vector of length 35:
        - 32 dims: one-hot card presence
        - 3 dims: one-hot position
    """
    # One-hot cards (32)
    card_vector = np.zeros(32, dtype=np.float32)
    for c in cards:
        card_vector[c] = 1.0

    # Player position (3 dims - One Hot)
    pos_vector = np.zeros(3, dtype=np.float32)
    pos_vector[position] = 1.0

    return np.concatenate([card_vector, pos_vector])


def extract_game_eval_features(
    cards: List[int],
    position: int,
    game_type: str,
    is_hand_game: bool,
    bid_level: int,
    skat_cards: List[int],
) -> np.ndarray:
    """
    Extract features for GameEvaluator (dense model).

    Args:
        cards: List of card indices (0-31) in hand (10 cards)
        position: Player position (0=forehand, 1=middlehand, 2=rearhand)
        game_type: Game type string (DIAMONDS, HEARTS, SPADES, CLUBS, GRAND, NULL)
        is_hand_game: Whether this is a hand game
        bid_level: The bid level the declarer must meet
        skat_cards: List of card indices in the skat (2 cards for pickup games, empty for hand)

    Returns:
        Feature vector of length 75:
        - 32 dims: Hand cards one-hot
        - 32 dims: Skat cards one-hot
        - 6 dims: Game type one-hot
        - 3 dims: Position one-hot
        - 1 dim: Is hand game
        - 1 dim: Bid level (normalized)
    """
    # 1. Hand cards (32 dims)
    hand_vector = np.zeros(32, dtype=np.float32)
    for c in cards:
        hand_vector[c] = 1.0

    # 2. Skat cards (32 dims)
    skat_vector = np.zeros(32, dtype=np.float32)
    if skat_cards:
        for c in skat_cards:
            skat_vector[c] = 1.0

    # 3. Game type one-hot (6 dims)
    game_type_onehot = np.zeros(6, dtype=np.float32)
    game_type_idx = GAME_TYPE_TO_IDX[game_type]
    game_type_onehot[game_type_idx] = 1.0

    # 4. Position one-hot (3 dims)
    pos_vector = np.zeros(3, dtype=np.float32)
    pos_vector[position] = 1.0

    # 5. Is hand game (1 dim)
    hand_flag = 1.0 if is_hand_game else 0.0

    # 6. Bid level normalized (1 dim)
    bid_normalized = float(bid_level) / 264.0

    # Combine (75 dims total)
    return np.concatenate([
        hand_vector,
        skat_vector,
        game_type_onehot,
        pos_vector,
        [hand_flag, bid_normalized],
    ])


def extract_card_play_features(
    player: int,
    hand: List[int],
    skat_cards: Optional[List[int]],
    history: List[List[int]],
    current_trick: List[Tuple[int, int]],
    game_type: str,
    declarer: int,
    bid_level: int,
    trick_num: int,
    is_ouvert: bool,
    declarer_hand: List[int],
) -> np.ndarray:
    """
    Extract features for CardPlayPolicy (dense model).

    Args:
        player: Current player index (0-2)
        hand: List of card indices in current player's hand
        skat_cards: List of card indices in skat (known only if declarer and pickup)
        history: List of 3 lists, each containing card indices played by that player
        current_trick: List of (player_idx, card_idx) for current trick
        game_type: Game type string
        declarer: Declarer index (0-2)
        bid_level: The bid level
        trick_num: Current trick number (0-9)
        is_ouvert: Whether it's an ouvert game
        declarer_hand: List of card indices in declarer's current hand

    Returns:
        Feature vector of length 268
    """
    vecs = []

    # 1. My Hand (32)
    v_hand = np.zeros(32, dtype=np.float32)
    for c in hand:
        v_hand[c] = 1.0
    vecs.append(v_hand)

    # 2. Known Skat (32)
    v_skat = np.zeros(32, dtype=np.float32)
    if skat_cards:
        for c in skat_cards:
            v_skat[c] = 1.0
    vecs.append(v_skat)

    # Relative Positions
    left = (player + 1) % 3
    right = (player + 2) % 3

    # 3. History Me (32)
    v_hist_me = np.zeros(32, dtype=np.float32)
    for c in history[player]:
        v_hist_me[c] = 1.0
    vecs.append(v_hist_me)

    # 4. History Left (32)
    v_hist_left = np.zeros(32, dtype=np.float32)
    for c in history[left]:
        v_hist_left[c] = 1.0
    vecs.append(v_hist_left)

    # 5. History Right (32)
    v_hist_right = np.zeros(32, dtype=np.float32)
    for c in history[right]:
        v_hist_right[c] = 1.0
    vecs.append(v_hist_right)

    # 6. Current Trick Card 1 (32)
    v_trick1 = np.zeros(32, dtype=np.float32)
    if len(current_trick) > 0:
        v_trick1[current_trick[0][1]] = 1.0
    vecs.append(v_trick1)

    # 7. Current Trick Card 2 (32)
    v_trick2 = np.zeros(32, dtype=np.float32)
    if len(current_trick) > 1:
        v_trick2[current_trick[1][1]] = 1.0
    vecs.append(v_trick2)

    # 8. Context
    v_game = np.zeros(6, dtype=np.float32)
    if game_type in GAME_TYPE_TO_IDX:
        v_game[GAME_TYPE_TO_IDX[game_type]] = 1.0
    vecs.append(v_game)

    v_decl = np.zeros(3, dtype=np.float32)
    if declarer == player:
        v_decl[0] = 1.0
    elif declarer == left:
        v_decl[1] = 1.0
    elif declarer == right:
        v_decl[2] = 1.0
    vecs.append(v_decl)

    vecs.append(np.array([bid_level / 264.0], dtype=np.float32))
    vecs.append(np.array([trick_num / 9.0], dtype=np.float32))

    # 9. Ouvert Features (1 + 32)
    v_ouvert = np.array([1.0 if is_ouvert else 0.0], dtype=np.float32)
    vecs.append(v_ouvert)

    v_visible_hand = np.zeros(32, dtype=np.float32)
    if is_ouvert:
        for c in declarer_hand:
            v_visible_hand[c] = 1.0
    vecs.append(v_visible_hand)

    return np.concatenate(vecs)
