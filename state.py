from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Tuple

try:
    from chess_pawn_mower.board import Board
except (ModuleNotFoundError, ImportError):
    from board import Board  # type: ignore

Position = Tuple[int, int]


@dataclass(frozen=True)
class PawnMowerState:
    board: Board
    remaining_black_pawns: FrozenSet[Position]
    active_piece: Optional[str] = None
    active_origin_position: Optional[Tuple[int, int]] = None
    active_position: Optional[Tuple[int, int]] = None
    king_position: Optional[Tuple[int, int]] = None
    move_count: int = 0
    # posicoes visitadas na jogada actual (desde a ultima activacao de peca)
    # impede ciclos no movimento do rei e da peca activa
    visited_positions: FrozenSet[Tuple[int, int]] = frozenset()
