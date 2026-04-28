from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

try:
    from chess_pawn_mower.board import Board
except (ModuleNotFoundError, ImportError):
    from board import Board  # type: ignore

Position = tuple[int, int]

ROOK_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
BISHOP_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
KNIGHT_DELTAS = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
KING_DELTAS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),            (0, 1),
    (1, -1),  (1, 0),   (1, 1),
)
WHITE_PIECES = {'P', 'T', 'B', 'C', 'D', 'R'}
ACTIVATABLE_PIECES = {'P', 'T', 'B', 'C', 'D'}


def is_white_piece(symbol: str) -> bool:
    return symbol in WHITE_PIECES


def is_black_pawn(symbol: str) -> bool:
    return symbol == 'p'


def _line_capture_moves(
    board: Board,
    origin: Position,
    directions: Iterable[tuple[int, int]],
    cell_at: Callable[[int, int], str] | None = None,
) -> list[Position]:
    cell_at = cell_at or board.get
    row, col = origin
    targets: list[Position] = []
    for d_row, d_col in directions:
        current_row, current_col = row + d_row, col + d_col
        while board.in_bounds(current_row, current_col):
            cell = cell_at(current_row, current_col)
            if cell == ' ':
                current_row += d_row
                current_col += d_col
                continue
            if is_black_pawn(cell):
                targets.append((current_row, current_col))
            break
    return targets


def rook_captures(board: Board, origin: Position, cell_at: Callable[[int, int], str] | None = None) -> list[Position]:
    return _line_capture_moves(board, origin, ROOK_DIRECTIONS, cell_at)


def bishop_captures(board: Board, origin: Position, cell_at: Callable[[int, int], str] | None = None) -> list[Position]:
    return _line_capture_moves(board, origin, BISHOP_DIRECTIONS, cell_at)


def queen_captures(board: Board, origin: Position, cell_at: Callable[[int, int], str] | None = None) -> list[Position]:
    return rook_captures(board, origin, cell_at) + bishop_captures(board, origin, cell_at)


def knight_captures(board: Board, origin: Position, cell_at: Callable[[int, int], str] | None = None) -> list[Position]:
    cell_at = cell_at or board.get
    row, col = origin
    targets: list[Position] = []
    for d_row, d_col in KNIGHT_DELTAS:
        next_row, next_col = row + d_row, col + d_col
        if board.in_bounds(next_row, next_col) and is_black_pawn(cell_at(next_row, next_col)):
            targets.append((next_row, next_col))
    return targets


def king_captures(board: Board, origin: Position, cell_at: Callable[[int, int], str] | None = None) -> list[Position]:
    cell_at = cell_at or board.get
    row, col = origin
    targets: list[Position] = []
    for d_row, d_col in KING_DELTAS:
        next_row, next_col = row + d_row, col + d_col
        if board.in_bounds(next_row, next_col) and is_black_pawn(cell_at(next_row, next_col)):
            targets.append((next_row, next_col))
    return targets


def pawn_captures(board: Board, origin: Position, cell_at: Callable[[int, int], str] | None = None) -> list[Position]:
    cell_at = cell_at or board.get
    row, col = origin
    targets: list[Position] = []
    for d_col in (-1, 1):
        next_row, next_col = row + 1, col + d_col
        if board.in_bounds(next_row, next_col) and is_black_pawn(cell_at(next_row, next_col)):
            targets.append((next_row, next_col))
    return targets


def capture_targets(board: Board, origin: Position, piece: str, cell_at: Callable[[int, int], str] | None = None) -> list[Position]:
    match piece:
        case 'T':
            return rook_captures(board, origin, cell_at)
        case 'B':
            return bishop_captures(board, origin, cell_at)
        case 'C':
            return knight_captures(board, origin, cell_at)
        case 'D':
            return queen_captures(board, origin, cell_at)
        case 'R':
            return king_captures(board, origin, cell_at)
        case 'P':
            return pawn_captures(board, origin, cell_at)
        case _:
            return []


def king_step_targets(board: Board, origin: Position, cell_at: Callable[[int, int], str] | None = None) -> list[Position]:
    cell_at = cell_at or board.get
    row, col = origin
    targets: list[Position] = []
    for d_row, d_col in KING_DELTAS:
        next_row, next_col = row + d_row, col + d_col
        if not board.in_bounds(next_row, next_col):
            continue
        if cell_at(next_row, next_col) == 'p':
            continue
        targets.append((next_row, next_col))
    return targets
