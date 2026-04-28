from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional

try:
    from chess_pawn_mower.board import Board
except ModuleNotFoundError:
    from board import Board  # type: ignore

Position = tuple[int, int]


@dataclass(frozen=True)
class PawnMowerState:
    board: Board
    remaining_black_pawns: FrozenSet[Position]
    active_piece: Optional[str] = None
    active_origin_position: Optional[tuple[int, int]] = None
    active_position: Optional[tuple[int, int]] = None
    king_position: Optional[tuple[int, int]] = None
    move_count: int = 0
