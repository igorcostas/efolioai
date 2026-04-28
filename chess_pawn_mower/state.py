from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional

from .board import Board

Position = tuple[int, int]


@dataclass(frozen=True)
class PawnMowerState:
    """Estado do problema.

    `active_piece` identifica a peça atualmente controlada.
    Quando o agente está em modo rei, `active_piece` e `active_position`
    ficam a `None` e `king_position` aponta para a casa atual do rei.
    """

    board: Board
    remaining_black_pawns: FrozenSet[Position]
    active_piece: Optional[str] = None
    active_origin_position: Optional[tuple[int, int]] = None
    active_position: Optional[tuple[int, int]] = None
    king_position: Optional[tuple[int, int]] = None
    move_count: int = 0
