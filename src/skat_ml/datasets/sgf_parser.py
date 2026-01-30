"""Parser for ISS SGF format."""

import re
from dataclasses import dataclass
from typing import List, Optional

from ..constants import CARD_TO_IDX


@dataclass
class GameRecord:
    """Parsed game record."""

    # Player info
    player_names: List[str]  # [P0, P1, P2]
    player_ratings: List[float]  # [R0, R1, R2]

    # Deal
    initial_hands: List[List[int]]  # [forehand_cards, middlehand_cards, rearhand_cards]
    skat: List[int]  # 2 cards in skat (card indices)

    # Bidding
    declarer: int  # 0=forehand, 1=middlehand, 2=rearhand
    bid_level: int  # Final bid value

    # Game announcement
    game_type: str  # DIAMONDS, HEARTS, SPADES, CLUBS, GRAND, NULL
    is_hand_game: bool
    is_ouvert: bool
    discards: Optional[List[int]]  # 2 cards discarded (if pickup game), None if hand game

    # Result
    won: bool  # Did declarer win?
    game_value: int  # Game value (positive if won, negative if lost)
    matadors: int  # Matadors count (positive = with, negative = without)
    card_points: int  # Card points taken by declarer
    schneider: bool  # Schneider declared/achieved
    schwarz: bool  # Schwarz declared/achieved

    # Metadata
    date: str
    game_id: int


def parse_card(card_str: str) -> int:
    """Convert card string (e.g., 'CJ') to index (0-31)."""
    if card_str in CARD_TO_IDX:
        return CARD_TO_IDX[card_str]
    raise ValueError(f"Unknown card: {card_str}")


class PassedGameError(Exception):
    """Raised when the game was passed by all players."""

    pass


class PenaltyGameError(Exception):
    """Raised when the game was aborted with a penalty."""

    pass


def parse_sgf_line(line: str) -> Optional[GameRecord]:
    """
    Parse a single SGF line into a GameRecord.

    Raises:
        PassedGameError: If the game was passed
        PenaltyGameError: If the game was a penalty abort
        ValueError/Exception: If parsing failed for other reasons
    """
    # Check for passed game first (quick check)
    if "R[passed]" in line:
        raise PassedGameError("Game passed")

    # Check for penalty games early
    if "d:-1" in line or "penalty" in line:
        raise PenaltyGameError("Penalty game")

    # Extract player names
    player_names = []
    for i in range(3):
        match = re.search(rf"P{i}\[([^\]]+)\]", line)
        if not match:
            raise ValueError("Missing player names")
        player_names.append(match.group(1))

    # Extract player ratings
    player_ratings = []
    for i in range(3):
        match = re.search(rf"R{i}\[([^\]]*)\]", line)
        if not match:
            raise ValueError("Missing player ratings")
        rating_str = match.group(1)
        if rating_str == "null" or rating_str == "":
            player_ratings.append(0.0)
        else:
            player_ratings.append(float(rating_str))

    # Extract date
    date_match = re.search(r"DT\[([^/]+)", line)
    date = date_match.group(1) if date_match else ""

    # Extract game ID
    id_match = re.search(r"ID\[(\d+)\]", line)
    game_id = int(id_match.group(1)) if id_match else 0

    # Extract moves field
    mv_match = re.search(r"MV\[([^\]]+)\]", line)
    if not mv_match:
        raise ValueError("Missing MV field")
    mv_content = mv_match.group(1)
    parts = mv_content.split()

    # Parse card deal (first part is 'w' = dealer indicator, second part is 32 cards)
    if len(parts) < 2:
        raise ValueError("Invalid MV format")
    deal_str = parts[1]  # Skip 'w', get actual card deal

    # Handle alternative format where hands are separated by '|'
    deal_str = deal_str.replace("|", ".")

    card_strs = deal_str.split(".")
    if len(card_strs) != 32:
        raise ValueError("Invalid card count")

    cards = [parse_card(c) for c in card_strs]
    initial_hands = [
        cards[0:10],  # Forehand
        cards[10:20],  # Middlehand
        cards[20:30],  # Rearhand
    ]
    skat = cards[30:32]

    # Parse bidding sequence and game announcement
    bid_parts = parts[2:]

    # Find declarer from result field (more reliable than parsing bids)
    result_match = re.search(r"R\[([^\]]+)\](?!.*R\[)", line)
    if not result_match:
        raise ValueError("Missing Result field")
    result = result_match.group(1)

    declarer_match = re.search(r"d:(\d+)", result)
    if not declarer_match:
        raise ValueError("Missing declarer in Result")
    declarer = int(declarer_match.group(1))
    if declarer < 0 or declarer > 2:
        raise ValueError("Invalid declarer index")

    # Find bid level from result field
    bid_match = re.search(r"(bidok|overbid)", result)
    if not bid_match:
        raise ValueError("No valid bid status")

    # Parse game announcement to get bid level, game type, and discards
    game_type = None
    is_hand_game = False
    is_ouvert = False
    discards = None

    # Look for pattern: declarer_number followed by announcement
    for i, part in enumerate(bid_parts):
        if part != str(declarer):
            continue

        if i + 1 >= len(bid_parts):
            continue

        announcement = bid_parts[i + 1]

        # Null games
        if announcement.startswith("N"):
            game_type = "NULL"

            # Check for pickup (discards) FIRST
            if "." in announcement:
                is_hand_game = False
                if "O" in announcement.split(".")[0]:
                    is_ouvert = True
                discard_strs = announcement.split(".")[1:3]
                if len(discard_strs) == 2:
                    discards = [parse_card(c) for c in discard_strs]
            elif "O" in announcement:
                is_ouvert = True
                is_hand_game = True
            elif "H" in announcement:
                is_hand_game = True
            else:
                is_hand_game = False
            break

        # Suit/Grand
        else:
            if announcement.startswith("G"):
                game_type = "GRAND"
                suffix = announcement[1:]
            elif announcement.startswith("C"):
                game_type = "CLUBS"
                suffix = announcement[1:]
            elif announcement.startswith("S"):
                game_type = "SPADES"
                suffix = announcement[1:]
            elif announcement.startswith("H"):
                game_type = "HEARTS"
                suffix = announcement[1:]
            elif announcement.startswith("D"):
                game_type = "DIAMONDS"
                suffix = announcement[1:]
            else:
                continue

            # Parse suffix
            if suffix.startswith("."):
                is_hand_game = False
                discard_strs = announcement.split(".")[1:3]
                if len(discard_strs) == 2:
                    discards = [parse_card(c) for c in discard_strs]
            elif "O" in suffix:
                is_ouvert = True
                is_hand_game = True
            elif "H" in suffix:
                is_hand_game = True
            elif suffix == "":
                is_hand_game = False
            else:
                is_hand_game = True

            break

    if game_type is None:
        raise ValueError("Could not determine game type")

    # Find bid level - parse bidding sequence
    bid_level = 18
    current_bid = 0
    for part in bid_parts:
        if part.isdigit():
            current_bid = int(part)
            if current_bid > bid_level:
                bid_level = current_bid

    if bid_level == 0:
        bid_level = 18

    # Parse result
    won = "win" in result

    # Parse game value
    value_match = re.search(r"v:(-?\d+)", result)
    if not value_match:
        raise ValueError("Missing game value")
    game_value = int(value_match.group(1))
    game_value = abs(game_value)

    # Parse matadors
    matadors_match = re.search(r"m:(-?\d+)", result)
    matadors = int(matadors_match.group(1)) if matadors_match else 0

    # Parse card points
    points_match = re.search(r"p:(\d+)", result)
    card_points = int(points_match.group(1)) if points_match else 0

    # Parse schneider/schwarz
    schneider_match = re.search(r"s:(\d+)", result)
    schneider = bool(int(schneider_match.group(1))) if schneider_match else False

    schwarz_match = re.search(r"z:(\d+)", result)
    schwarz = bool(int(schwarz_match.group(1))) if schwarz_match else False

    return GameRecord(
        player_names=player_names,
        player_ratings=player_ratings,
        initial_hands=initial_hands,
        skat=skat,
        declarer=declarer,
        bid_level=bid_level,
        game_type=game_type,
        is_hand_game=is_hand_game,
        is_ouvert=is_ouvert,
        discards=discards,
        won=won,
        game_value=game_value,
        matadors=matadors,
        card_points=card_points,
        schneider=schneider,
        schwarz=schwarz,
        date=date,
        game_id=game_id,
    )


def get_final_hand(game: GameRecord) -> List[int]:
    """
    Get the declarer's final 10-card hand.

    For hand games: return initial hand
    For pickup games: initial_hand + skat - discards
    """
    declarer_idx = game.declarer
    initial_hand = game.initial_hands[declarer_idx]

    if game.is_hand_game:
        return initial_hand
    else:
        if game.discards is None:
            return initial_hand

        # Combine initial hand + skat
        full_hand = set(initial_hand + game.skat)

        # Remove discards
        for discard in game.discards:
            if discard in full_hand:
                full_hand.remove(discard)

        return list(full_hand)
