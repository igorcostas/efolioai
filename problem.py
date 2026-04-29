from __future__ import annotations

from dataclasses import replace
from typing import FrozenSet, Iterable, List, Optional, Tuple

try:
    from search.algorithms import astar
    from search.node import Node
    from chess_pawn_mower.board import Board
    from chess_pawn_mower.moves import WHITE_PIECES, capture_targets, king_step_targets, is_white_piece
    from chess_pawn_mower.state import PawnMowerState
except (ModuleNotFoundError, ImportError):
    from algorithms import astar  # type: ignore
    from node import Node  # type: ignore
    from board import Board  # type: ignore
    from moves import WHITE_PIECES, capture_targets, king_step_targets, is_white_piece  # type: ignore
    from state import PawnMowerState  # type: ignore

MAX_ACTIONS = 100


def build_initial_state(board):
    return PawnMowerState(
        board=board,
        remaining_black_pawns=frozenset(board.find('p')),
        active_piece=None,
        active_origin_position=None,
        active_position=None,
        king_position=None,
        move_count=0,
    )


def is_goal(state):
    return not state.remaining_black_pawns


def _chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _mst_cost(points):
    if len(points) <= 1:
        return 0.0
    in_tree = {0}
    min_edge = [_chebyshev(points[0], points[i]) for i in range(len(points))]
    total = 0.0
    for _ in range(len(points) - 1):
        best = float('inf')
        best_idx = -1
        for i, cost in enumerate(min_edge):
            if i not in in_tree and cost < best:
                best = cost
                best_idx = i
        total += best
        in_tree.add(best_idx)
        for i in range(len(points)):
            if i not in in_tree:
                d = _chebyshev(points[best_idx], points[i])
                if d < min_edge[i]:
                    min_edge[i] = d
    return total


def heuristic(state):
    pawns = list(state.remaining_black_pawns)
    if not pawns:
        return 0.0
    mst = _mst_cost(pawns)
    current_pos = state.active_position or state.king_position
    if current_pos is not None:
        dist_nearest = min(_chebyshev(current_pos, p) for p in pawns)
    else:
        white_positions = [
            (r, c)
            for r, c, sym in state.board.iter_cells()
            if sym in WHITE_PIECES
        ]
        if white_positions:
            dist_nearest = min(
                _chebyshev(w, p)
                for w in white_positions
                for p in pawns
            )
        else:
            dist_nearest = 0.0
    transition = 0 if (state.active_position is not None or state.king_position is not None) else 1
    return mst + dist_nearest + transition


def _cell_at(state, row, col):
    position = (row, col)
    if position == state.king_position:
        return 'R'
    if position == state.active_position and state.active_piece is not None:
        return state.active_piece
    if position in state.remaining_black_pawns:
        return 'p'
    board_cell = state.board.get(row, col)
    if board_cell in WHITE_PIECES:
        if (
            state.active_origin_position == position
            and state.active_position != position
        ):
            return ' '
        return board_cell
    return ' '


def _all_white_positions(board):
    for row, col, symbol in board.iter_cells():
        if symbol in WHITE_PIECES:
            yield row, col, symbol


def _activate_piece(state, position, symbol):
    return replace(
        state,
        active_piece=symbol,
        active_origin_position=position,
        active_position=position,
        king_position=None,
        move_count=state.move_count + 1,
    )


def _capture_with_active(state, destination):
    next_remaining = frozenset(
        p for p in state.remaining_black_pawns if p != destination
    )
    return replace(
        state,
        active_position=destination,
        remaining_black_pawns=next_remaining,
        move_count=state.move_count + 1,
    )


def _enter_king_mode(state, king_dest):
    return replace(
        state,
        active_piece=None,
        active_position=None,
        king_position=king_dest,
        move_count=state.move_count + 1,
    )


def _move_king_step(state, destination):
    return replace(
        state,
        king_position=destination,
        move_count=state.move_count + 1,
    )


def successors(state):
    if state.move_count >= MAX_ACTIONS or is_goal(state):
        return []

    board = state.board
    results = []

    # MODO 0: activar uma peca branca
    if (
        state.active_piece is None
        and state.active_position is None
        and state.king_position is None
    ):
        for row, col, symbol in _all_white_positions(board):
            sq = Board.index_to_square(row, col)
            results.append((sq, _activate_piece(state, (row, col), symbol), 1.0))
        return results

    # MODO 2: rei em movimento
    # O rei pode: mover para casa vazia OU activar uma peca branca adjacente
    if state.king_position is not None:
        row_k, col_k = state.king_position
        from moves import KING_DELTAS
        for d_row, d_col in KING_DELTAS:
            next_row, next_col = row_k + d_row, col_k + d_col
            if not board.in_bounds(next_row, next_col):
                continue
            cell = _cell_at(state, next_row, next_col)
            sq = Board.index_to_square(next_row, next_col)
            if cell == ' ':
                # mover rei para casa vazia
                results.append((sq, _move_king_step(state, (next_row, next_col)), 1.0))
            elif cell in WHITE_PIECES and cell != 'R':
                # activar peca branca adjacente
                results.append((sq, _activate_piece(state, (next_row, next_col), cell), 1.0))
            # casas com peoes pretos ou o proprio rei sao ignoradas
        return results

    # MODO 1: peca activa
    # A peca captura peoes pretos ao seu alcance
    # A transicao para MODO 2 (rei) so e valida para casas VAZIAS
    if state.active_piece is None or state.active_position is None:
        return results

    # capturas de peoes pretos
    for row, col in capture_targets(
        board, state.active_position, state.active_piece,
        cell_at=lambda r, c: _cell_at(state, r, c),
    ):
        if (row, col) not in state.remaining_black_pawns:
            continue
        sq = Board.index_to_square(row, col)
        results.append((sq, _capture_with_active(state, (row, col)), 1.0))

    # transicao MODO 1 -> MODO 2: rei entra APENAS em casas vazias
    for row, col in king_step_targets(
        board, state.active_position,
        cell_at=lambda r, c: _cell_at(state, r, c),
    ):
        cell = _cell_at(state, row, col)
        if cell == ' ':
            sq = Board.index_to_square(row, col)
            results.append((sq, _enter_king_mode(state, (row, col)), 1.0))
        # peças brancas adjacentes NÃO geram transição aqui;
        # o rei tem de primeiro entrar numa casa vazia (MODO 2)
        # e só depois activar uma peça branca a partir do MODO 2.

    return results


def solve_board(board, time_limit_ms):
    initial_state = build_initial_state(board)
    return astar(
        initial_state,
        is_goal=is_goal,
        successors=successors,
        heuristic=heuristic,
        time_limit_ms=time_limit_ms,
    )


def solution_string(node):
    if node is None:
        return ''
    return ' '.join(
        str(n.action) for n in node.path()[1:] if n.action is not None
    )
