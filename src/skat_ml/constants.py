"""Constants for Skat game representation.

Card encoding matches Java's Card.values() order for consistency.
"""

from typing import Dict, List

# Card representation (32 cards, matching Java Card.values() order)
CARDS: List[str] = [
    "CJ", "SJ", "HJ", "DJ",  # Jacks (indices 0-3)
    "CA", "CT", "CK", "CQ", "C9", "C8", "C7",  # Clubs (4-10)
    "SA", "ST", "SK", "SQ", "S9", "S8", "S7",  # Spades (11-17)
    "HA", "HT", "HK", "HQ", "H9", "H8", "H7",  # Hearts (18-24)
    "DA", "DT", "DK", "DQ", "D9", "D8", "D7",  # Diamonds (25-31)
]

CARD_TO_IDX: Dict[str, int] = {card: i for i, card in enumerate(CARDS)}

# Card point values
CARD_POINTS: Dict[str, int] = {
    "A": 11,
    "T": 10,  # 10
    "K": 4,
    "Q": 3,
    "J": 2,
    "9": 0,
    "8": 0,
    "7": 0,
}

# Valid bid values from SkatConstants.java
BID_VALUES: List[int] = [
    18, 20, 22, 23, 24, 27, 30, 33, 35, 36, 40, 44, 45, 46, 48,
    50, 54, 55, 59, 60, 63, 66, 70, 72, 77, 80, 81, 84, 88, 90,
    96, 99, 100, 108, 110, 117, 120, 121, 126, 130, 132, 135,
    140, 143, 144, 150, 153, 154, 156, 160, 162, 165, 168, 170,
    176, 180, 187, 192, 198, 204, 216, 240, 264,
]

BID_TO_IDX: Dict[int, int] = {bid: i for i, bid in enumerate(BID_VALUES)}
NUM_BID_LEVELS = len(BID_VALUES)  # 63

# Game types
GAME_TYPES: List[str] = ["DIAMONDS", "HEARTS", "SPADES", "CLUBS", "GRAND", "NULL"]
GAME_TYPE_TO_IDX: Dict[str, int] = {gt: i for i, gt in enumerate(GAME_TYPES)}

# Base values for game types (multiplied by matadors + 1 to get game value)
BASE_VALUES: Dict[str, int] = {
    "DIAMONDS": 9,
    "HEARTS": 10,
    "SPADES": 11,
    "CLUBS": 12,
    "GRAND": 24,
    "NULL": 23,  # Base null (varies with hand/ouvert)
}

# Feature dimensions for dense models
BIDDING_EVALUATOR_INPUT_DIM = 35  # 32 (cards) + 3 (position)
GAME_EVALUATOR_INPUT_DIM = 75  # 32 (hand) + 32 (skat) + 6 (game) + 3 (pos) + 1 (hand_flag) + 1 (bid)
CARD_PLAY_POLICY_INPUT_DIM = 268  # Full gameplay context

# Fixed sizes for transformer models
MAX_HAND = 10
MAX_SKAT = 2
MAX_HISTORY = 27  # 9 complete tricks before current
MAX_TRICK = 2  # At most 2 cards before our turn
MAX_OUVERT = 10  # Declarer's visible hand in ouvert games

# Padding token for card indices
CARD_PAD_IDX = 32  # Valid cards are 0-31

# Bid normalization
MAX_BID = max(BID_VALUES)  # 264
MIN_BID = min(BID_VALUES)  # 18

# Low card indices (7s, 8s, 9s - important for Null evaluation)
# 7s: C7=10, S7=17, H7=24, D7=31
# 8s: C8=9, S8=16, H8=23, D8=30
# 9s: C9=8, S9=15, H9=22, D9=29
LOW_CARD_INDICES = {8, 9, 10, 15, 16, 17, 22, 23, 24, 29, 30, 31}

# Training estimation constants
# Average card play decisions per game: 10 tricks × 3 players = 30 cards played,
# but only ~1/3 are from winning team (with winner_only=True), so ~10 per game.
# Empirically observed as ~9.8 samples per game.
EST_CARD_PLAY_SAMPLES_PER_GAME = 9.8
