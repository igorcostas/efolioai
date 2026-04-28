from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Optional

# Imports planos — compatíveis com VPL (pasta plana) e com estrutura de pacotes
try:
    from search.algorithms import astar
    from search.node import Node
    from chess_pawn_mower.board import Board
    from chess_pawn_mower.moves import WHITE_PIECES, capture_targets, king_step_targets, is_white_piece
    from chess_pawn_mower.state import PawnMowerState
except ModuleNotFoundError:
    from algorithms import astar  # type: ignore
    from node import Node  # type: ignore
    from board import Board  # type: ignore
    from moves import WHITE_PIECES, capture_targets, king_step_targets, is_white_piece  # type: ignore
    from state import PawnMowerState  # type: ignore

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


# ──────────────────────────────────────────────────────────────
#  Heurística — MST + distância ao peão mais próximo
# ──────────────────────────────────────────────────────────────

def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _mst_cost(points: list[tuple[int, int]]) -> float:
    """Custo do MST sobre `points` usando distância de Chebyshev (Prim O(n²))."""
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


def heuristic(state: PawnMowerState) -> float:
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


# ──────────────────────────────────────────────────────────────
#  Visão virtual do tabuleiro num dado estado
# ──────────────────────────────────────────────────────────────

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
        if (
            state.active_origin_position == position
            and state.active_position != position
        ):
            return ' '
        return board_cell
    return ' '


def _all_white_positions(board: Board) -> Iterable[tuple[int, int, str]]:
    for row, col, symbol in board.iter_cells():
        if symbol in WHITE_PIECES:
            yield row, col, symbol


# ──────────────────────────────────────────────────────────────
#  Transições de estado — 3 modos MUTUAMENTE EXCLUSIVOS
#
#  MODO 0 — inicial: active_piece=None, active_position=None, king_position=None
#           → activar uma peça branca
#
#  MODO 1 — peça activa: active_piece=X, active_position=(r,c), king_position=None
#           → capturar peões OU passar ao rei (entra MODO 2)
#
#  MODO 2 — rei em movimento: active_piece=None, active_position=None, king_position=(r,c)
#           → mover rei 1 passo OU activar peça (entra MODO 1)
# ──────────────────────────────────────────────────────────────

def _activate_piece(
    state: PawnMowerState,
    position: tuple[int, int],
    symbol: str,
) -> PawnMowerState:
    """MODO 0/2 → MODO 1."""
    return replace(
        state,
        active_piece=symbol,
        active_origin_position=position,
        active_position=position,
        king_position=None,
        move_count=state.move_count + 1,
    )


def _capture_with_active(
    state: PawnMowerState,
    destination: tuple[int, int],
) -> PawnMowerState:
    """MODO 1 → MODO 1 (captura de peão)."""
    next_remaining = frozenset(
        p for p in state.remaining_black_pawns if p != destination
    )
    return replace(
        state,
        active_position=destination,
        remaining_black_pawns=next_remaining,
        move_count=state.move_count + 1,
    )


def _enter_king_mode(
    state: PawnMowerState,
    king_dest: tuple[int, int],
) -> PawnMowerState:
    """MODO 1 → MODO 2. Limpa peça activa, coloca rei em king_dest."""
    return replace(
        state,
        active_piece=None,
        active_position=None,
        king_position=king_dest,
        move_count=state.move_count + 1,
    )


def _move_king_step(
    state: PawnMowerState,
    destination: tuple[int, int],
) -> PawnMowerState:
    """MODO 2 → MODO 2 (rei move-se para casa vazia)."""
    return replace(
        state,
        king_position=destination,
        move_count=state.move_count + 1,
    )


# ──────────────────────────────────────────────────────────────
#  Geração de sucessores
# ──────────────────────────────────────────────────────────────

def successors(
    state: PawnMowerState,
) -> list[tuple[str, PawnMowerState, float]]:
    if state.move_count >= MAX_ACTIONS or is_goal(state):
        return []

    board = state.board
    results: list[tuple[str, PawnMowerState, float]] = []

    # ── MODO 0: activar uma peça branca ──────────────────────
    if (
        state.active_piece is None
        and state.active_position is None
        and state.king_position is None
    ):
        for row, col, symbol in _all_white_positions(board):
            sq = Board.index_to_square(row, col)
            results.append((sq, _activate_piece(state, (row, col), symbol), 1.0))
        return results

    # ── MODO 2: rei em movimento ──────────────────────────────
    if state.king_position is not None:
        for row, col in king_step_targets(
            board, state.king_position,
            cell_at=lambda r, c: _cell_at(state, r, c),
        ):
            cell = _cell_at(state, row, col)
            sq = Board.index_to_square(row, col)
            if cell == ' ':
                results.append((sq, _move_king_step(state, (row, col)), 1.0))
            elif is_white_piece(cell):
                results.append((sq, _activate_piece(state, (row, col), cell), 1.0))
        return results

    # ── MODO 1: peça activa ───────────────────────────────────
    if state.active_piece is None or state.active_position is None:
        return results

    # 1a) Capturas de peões com a peça activa
    for row, col in capture_targets(
        board, state.active_position, state.active_piece,
        cell_at=lambda r, c: _cell_at(state, r, c),
    ):
        if (row, col) not in state.remaining_black_pawns:
            continue
        sq = Board.index_to_square(row, col)
        results.append((sq, _capture_with_active(state, (row, col)), 1.0))

    # 1b) Passar controlo ao rei: 1 passo a partir da posição
    #     actual da peça activa — entra directamente em MODO 2.
    for row, col in king_step_targets(
        board, state.active_position,
        cell_at=lambda r, c: _cell_at(state, r, c),
    ):
        cell = _cell_at(state, row, col)
        sq = Board.index_to_square(row, col)
        if cell == ' ':
            # entra MODO 2 com rei já em (row, col)
            results.append((sq, _enter_king_mode(state, (row, col)), 1.0))
        elif is_white_piece(cell):
            # rei chega directamente a peça branca → activa-a
            results.append((sq, _activate_piece(state, (row, col), cell), 1.0))

    return results


# ──────────────────────────────────────────────────────────────
#  Interface pública
# ──────────────────────────────────────────────────────────────

def solve_board(board: Board, time_limit_ms: int) -> Optional[Node]:
    initial_state = build_initial_state(board)
    return astar(
        initial_state,
        is_goal=is_goal,
        successors=successors,
        heuristic=heuristic,
        time_limit_ms=time_limit_ms,
    )


def solution_string(node: Optional[Node]) -> str:
    if node is None:
        return ''
    return ' '.join(
        str(n.action) for n in node.path()[1:] if n.action is not None
    )
