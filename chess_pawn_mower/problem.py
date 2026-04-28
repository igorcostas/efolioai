from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Optional

from search.algorithms import astar
from search.node import Node

from .board import Board
from .moves import WHITE_PIECES, capture_targets, king_step_targets, is_white_piece
from .state import PawnMowerState

MAX_ACTIONS = 100


def build_initial_state(board: Board) -> PawnMowerState:
    return PawnMowerState(
        board=board,
        remaining_black_pawns=frozenset(board.find('p')),
        active_piece=None,
        active_origin_position=None,
        active_position=None,
        king_position=None,
        move_count=0,
    )


def is_goal(state: PawnMowerState) -> bool:
    return not state.remaining_black_pawns


def heuristic(state: PawnMowerState) -> float:
    pawns = state.remaining_black_pawns
    if not pawns:
        return 0.0

    n = len(pawns)

    # Sem peça ativa: precisa pelo menos de 1 ação para ativar + 1 por peão
    if state.active_position is None:
        return float(n + 1)

    # Com peça ativa: distância de Chebyshev ao peão mais próximo
    r, c = state.active_position
    min_dist = min(max(abs(r - pr), abs(c - pc)) for pr, pc in pawns)

    return float(n - 1 + min_dist)


def _cell_at(state: PawnMowerState, row: int, col: int) -> str:
    position = (row, col)

    if position == state.king_position:
        return 'R'

    if position == state.active_position and state.active_piece is not None:
        return state.active_piece

    if position in state.remaining_black_pawns:
        return 'p'

    board_cell = state.board.get(row, col)
    if board_cell in WHITE_PIECES:
        if state.active_origin_position == position and state.active_position != position:
            return ' '
        return board_cell

    return ' '


def _all_white_positions(board: Board) -> Iterable[tuple[int, int, str]]:
    for row, col, symbol in board.iter_cells():
        if symbol in WHITE_PIECES:
            yield row, col, symbol


def _activate_piece(state: PawnMowerState, position: tuple[int, int], symbol: str) -> PawnMowerState:
    return replace(
        state,
        active_piece=symbol,
        active_origin_position=position,
        active_position=position,
        king_position=None,
        move_count=state.move_count + 1,
    )


def _move_active_piece(state: PawnMowerState, destination: tuple[int, int]) -> PawnMowerState:
    return replace(
        state,
        active_position=destination,
        king_position=None,
        move_count=state.move_count + 1,
    )


def _move_king(state: PawnMowerState, destination: tuple[int, int]) -> PawnMowerState:
    return replace(
        state,
        king_position=destination,
        move_count=state.move_count + 1,
    )


def successors(state: PawnMowerState) -> list[tuple[str, PawnMowerState, float]]:
    if state.move_count >= MAX_ACTIONS or is_goal(state):
        return []

    board = state.board
    results: list[tuple[str, PawnMowerState, float]] = []

    if state.active_piece is None and state.active_position is None and state.king_position is None:
        for row, col, symbol in _all_white_positions(board):
            destination = Board.index_to_square(row, col)
            results.append((destination, _activate_piece(state, (row, col), symbol), 1.0))
        return results

    if state.active_position is None or state.active_piece is None:
        return results

    if state.king_position is not None:
        for row, col in king_step_targets(board, state.king_position, cell_at=lambda r, c: _cell_at(state, r, c)):
            cell = _cell_at(state, row, col)
            destination = Board.index_to_square(row, col)
            if cell == ' ':
                results.append((destination, _move_king(state, (row, col)), 1.0))
            elif is_white_piece(cell):
                results.append((destination, _activate_piece(state, (row, col), cell), 1.0))
        return results

    # Modo peça ativa: capturas e saída para rei.
    capture_cells = capture_targets(board, state.active_position, state.active_piece, cell_at=lambda r, c: _cell_at(state, r, c))
    for row, col in capture_cells:
        if (row, col) not in state.remaining_black_pawns:
            continue
        next_remaining = frozenset(position for position in state.remaining_black_pawns if position != (row, col))
        next_state = replace(
            state,
            active_position=(row, col),
            active_origin_position=state.active_origin_position,
            remaining_black_pawns=next_remaining,
            move_count=state.move_count + 1,
        )
        results.append((Board.index_to_square(row, col), next_state, 1.0))

    for row, col in king_step_targets(board, state.active_position, cell_at=lambda r, c: _cell_at(state, r, c)):
        cell = _cell_at(state, row, col)
        destination = Board.index_to_square(row, col)
        if cell == ' ':
            results.append((destination, _move_king(state, (row, col)), 1.0))
        elif is_white_piece(cell):
            results.append((destination, _activate_piece(state, (row, col), cell), 1.0))

    return results


def solve_board(board: Board, time_limit_ms: int) -> Optional[Node]:
    initial_state = build_initial_state(board)
    return astar(
        initial_state,
        is_goal=is_goal,
        successors=successors,
        heuristic=lambda s: heuristic(s) * 5,  # peso 5 → muito mais rápido
        time_limit_ms=time_limit_ms,
    )


def solution_string(node: Optional[Node]) -> str:
    if node is None:
        return ''
    actions = [str(current.action) for current in node.path()[1:] if current.action]
    return ' '.join(actions)

