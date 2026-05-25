from __future__ import annotations

from typing import List, Optional, Tuple
from time import perf_counter

from game_state import GameState, active_position, opponent
from game_rules import get_valid_moves, is_terminal, game_result, ACTIVATABLE_PIECES

DEFAULT_DEPTH = 3


# ---------------------------------------------------------------------------
# Heurística
# ---------------------------------------------------------------------------

def _mobility(state: GameState, player: str) -> int:
    """Número de movimentos válidos do jogador (indica mobilidade)."""
    moves = get_valid_moves(state, player)
    if len(moves) == 1 and moves[0][0] == 'null':
        return 0
    return len(moves)


def _pieces_available(state: GameState) -> int:
    """Número de peças activáveis ainda disponíveis no tabuleiro."""
    count = 0
    for row, col, sym in state.board.iter_cells():
        if sym in ACTIVATABLE_PIECES and (row, col) not in state.used_pieces:
            count += 1
    return count


def heuristic(state: GameState, my_player: str) -> float:
    """
    Avalia o estado do ponto de vista do jogador 'my_player'.
    Valor positivo = vantagem para my_player.
    """
    opp = opponent(my_player)

    my_captured = state.green_captured if my_player == 'A' else state.red_captured
    opp_captured = state.red_captured if my_player == 'A' else state.green_captured

    # Diferença de peões capturados (componente principal)
    score = (my_captured - opp_captured) * 10.0

    # Bónus de mobilidade (ter mais opções é melhor)
    my_mob = _mobility(state, my_player)
    opp_mob = _mobility(state, opp)
    score += (my_mob - opp_mob) * 0.5

    # Bónus por estar dentro de uma peça (maior poder de captura)
    my_piece = state.green_piece if my_player == 'A' else state.red_piece
    opp_piece = state.red_piece if my_player == 'A' else state.green_piece
    if my_piece is not None:
        score += 2.0
    if opp_piece is not None:
        score -= 2.0

    # Bónus por número de peças ainda disponíveis (mais opções futuras)
    score += _pieces_available(state) * 0.1

    return score


# ---------------------------------------------------------------------------
# Minimax com poda Alpha-Beta
# ---------------------------------------------------------------------------

def _minimax(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    my_player: str,
    deadline: Optional[float],
) -> float:
    """Minimax recursivo com poda Alpha-Beta."""

    # Verifica timeout
    if deadline is not None and perf_counter() > deadline:
        return heuristic(state, my_player)

    # Nó terminal ou profundidade zero
    if is_terminal(state) or depth == 0:
        return heuristic(state, my_player)

    current_player = state.turn
    moves = get_valid_moves(state, current_player)

    if maximizing:
        max_eval = float('-inf')
        for _, next_state in moves:
            eval_val = _minimax(
                next_state, depth - 1, alpha, beta,
                False, my_player, deadline,
            )
            max_eval = max(max_eval, eval_val)
            alpha = max(alpha, eval_val)
            if beta <= alpha:
                break  # poda beta
        return max_eval
    else:
        min_eval = float('inf')
        for _, next_state in moves:
            eval_val = _minimax(
                next_state, depth - 1, alpha, beta,
                True, my_player, deadline,
            )
            min_eval = min(min_eval, eval_val)
            beta = min(beta, eval_val)
            if beta <= alpha:
                break  # poda alpha
        return min_eval


def choose_move(
    state: GameState,
    my_player: str,
    depth: int = DEFAULT_DEPTH,
    time_limit_ms: Optional[int] = 5000,
) -> Tuple[str, GameState]:
    """
    Escolhe o melhor movimento para 'my_player' usando Minimax Alpha-Beta.
    Devolve (square_destino, novo_estado).
    """
    deadline = None if time_limit_ms is None else perf_counter() + (time_limit_ms / 1000.0)

    moves = get_valid_moves(state, my_player)

    # Movimento nulo obrigatório
    if len(moves) == 1 and moves[0][0] == 'null':
        return moves[0]

    best_move = moves[0]
    best_score = float('-inf')

    for move_sq, next_state in moves:
        # O adversário vai minimizar no próximo nível
        score = _minimax(
            next_state,
            depth - 1,
            float('-inf'),
            float('inf'),
            False,          # adversário minimiza
            my_player,
            deadline,
        )
        if score > best_score:
            best_score = score
            best_move = (move_sq, next_state)

    return best_move
