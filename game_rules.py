from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Tuple

try:
    from chess_pawn_mower.board import Board
    from chess_pawn_mower.moves import (
        KING_DELTAS, WHITE_PIECES,
        capture_targets, king_step_targets,
    )
except (ModuleNotFoundError, ImportError):
    from board import Board  # type: ignore
    from moves import (  # type: ignore
        KING_DELTAS, WHITE_PIECES,
        capture_targets, king_step_targets,
    )

from game_state import GameState, Position, active_position, opponent, opponent_position

ACTIVATABLE_PIECES = {'T', 'B', 'C', 'D'}
MAX_ACTIONS = 60


# ---------------------------------------------------------------------------
# Helpers de leitura do tabuleiro combinado com o estado dinâmico
# ---------------------------------------------------------------------------

def _cell_at(state: GameState, row: int, col: int) -> str:
    """Devolve o símbolo lógico numa casa, considerando o estado dinâmico."""
    pos = (row, col)

    # Peça activa verde
    if state.green_piece and pos == state.green_piece_pos:
        return state.green_piece
    # Peça activa vermelha
    if state.red_piece and pos == state.red_piece_pos:
        return state.red_piece
    # Reis
    if pos == state.green_king_pos:
        return 'A'
    if pos == state.red_king_pos:
        return 'V'
    # Peões pretos restantes
    if pos in state.remaining_black_pawns:
        return 'p'
    # Peças já abandonadas tornam-se peões brancos
    if pos in state.used_pieces:
        return 'P'
    # Tabuleiro estático (peças brancas originais, peões brancos)
    cell = state.board.get(row, col)
    # Peças de origem das peças activas ficam vazias enquanto estão activas
    if cell in ACTIVATABLE_PIECES:
        if state.green_piece_pos == pos and state.green_piece == cell:
            return cell
        if state.red_piece_pos == pos and state.red_piece == cell:
            return cell
        return cell
    return cell


def _is_adjacent(pos_a: Position, pos_b: Position) -> bool:
    """Verifica se duas casas são adjacentes (distância de Chebyshev = 1)."""
    return max(abs(pos_a[0] - pos_b[0]), abs(pos_a[1] - pos_b[1])) == 1


def _opponent_pos(state: GameState, player: str) -> Optional[Position]:
    """Posição efectiva do adversário (peça activa ou rei)."""
    return active_position(state, opponent(player))


def _would_be_adjacent(dest: Position, opp_pos: Optional[Position]) -> bool:
    """Verifica se o destino ficaria adjacente ao adversário."""
    if opp_pos is None:
        return False
    return _is_adjacent(dest, opp_pos)


# ---------------------------------------------------------------------------
# Geração de movimentos válidos
# ---------------------------------------------------------------------------

def get_valid_moves(state: GameState, player: str) -> List[Tuple[str, GameState]]:
    """
    Devolve lista de (square_destino, novo_estado) para todos os movimentos
    válidos do jogador 'player' ('A' ou 'V').
    Se não existir nenhum movimento válido, devolve o movimento nulo.
    """
    opp_pos = _opponent_pos(state, player)
    moves: List[Tuple[str, GameState]] = []

    if player == 'A':
        piece = state.green_piece
        piece_pos = state.green_piece_pos
        king_pos = state.green_king_pos
    else:
        piece = state.red_piece
        piece_pos = state.red_piece_pos
        king_pos = state.red_king_pos

    board = state.board

    # ── MODO REI: agente fora de peça ──────────────────────────────────────
    if piece is None:
        pos = king_pos
        if pos is None:
            return _null_move(state, player)

        row_k, col_k = pos
        for d_row, d_col in KING_DELTAS:
            nr, nc = row_k + d_row, col_k + d_col
            if not board.in_bounds(nr, nc):
                continue
            dest = (nr, nc)
            if _would_be_adjacent(dest, opp_pos):
                continue
            cell = _cell_at(state, nr, nc)
            sq = Board.index_to_square(nr, nc)

            # Move para casa vazia
            if cell == ' ':
                moves.append((sq, _apply_king_move(state, player, dest)))

            # Entra numa peça activável
            elif cell in ACTIVATABLE_PIECES and dest not in state.used_pieces:
                moves.append((sq, _apply_enter_piece(state, player, dest, cell)))

        if not moves:
            return _null_move(state, player)
        return moves

    # ── MODO PEÇA: agente dentro de uma peça ───────────────────────────────
    if piece_pos is None:
        return _null_move(state, player)

    # 1) Capturas de peões pretos com a peça activa
    for (nr, nc) in capture_targets(
        board, piece_pos, piece,
        cell_at=lambda r, c: _cell_at(state, r, c),
    ):
        dest = (nr, nc)
        if dest not in state.remaining_black_pawns:
            continue
        if _would_be_adjacent(dest, opp_pos):
            continue
        sq = Board.index_to_square(nr, nc)
        moves.append((sq, _apply_capture(state, player, dest)))

    # 2) Saída da peça: movimento de rei para casa adjacente vazia
    for d_row, d_col in KING_DELTAS:
        nr, nc = piece_pos[0] + d_row, piece_pos[1] + d_col
        if not board.in_bounds(nr, nc):
            continue
        dest = (nr, nc)
        if _would_be_adjacent(dest, opp_pos):
            continue
        cell = _cell_at(state, nr, nc)
        if cell == ' ':
            sq = Board.index_to_square(nr, nc)
            moves.append((sq, _apply_exit_piece(state, player, dest)))

    if not moves:
        return _null_move(state, player)
    return moves


# ---------------------------------------------------------------------------
# Aplicação de movimentos
# ---------------------------------------------------------------------------

def _apply_king_move(state: GameState, player: str, dest: Position) -> GameState:
    """Move o rei para uma casa vazia."""
    if player == 'A':
        return replace(state,
            green_king_pos=dest,
            turn=opponent(player),
            action_count=state.action_count + 1,
        )
    return replace(state,
        red_king_pos=dest,
        turn=opponent(player),
        action_count=state.action_count + 1,
    )


def _apply_enter_piece(state: GameState, player: str, dest: Position, piece: str) -> GameState:
    """Agente entra numa peça branca."""
    if player == 'A':
        return replace(state,
            green_king_pos=None,
            green_piece=piece,
            green_piece_pos=dest,
            turn=opponent(player),
            action_count=state.action_count + 1,
        )
    return replace(state,
        red_king_pos=None,
        red_piece=piece,
        red_piece_pos=dest,
        turn=opponent(player),
        action_count=state.action_count + 1,
    )


def _apply_capture(state: GameState, player: str, dest: Position) -> GameState:
    """Peça activa captura um peão preto."""
    new_pawns = frozenset(p for p in state.remaining_black_pawns if p != dest)
    if player == 'A':
        return replace(state,
            green_piece_pos=dest,
            remaining_black_pawns=new_pawns,
            green_captured=state.green_captured + 1,
            turn=opponent(player),
            action_count=state.action_count + 1,
        )
    return replace(state,
        red_piece_pos=dest,
        remaining_black_pawns=new_pawns,
        red_captured=state.red_captured + 1,
        turn=opponent(player),
        action_count=state.action_count + 1,
    )


def _apply_exit_piece(state: GameState, player: str, dest: Position) -> GameState:
    """Agente sai da peça: peça torna-se peão branco (pos registada em used_pieces)."""
    if player == 'A':
        abandoned = state.green_piece_pos
        new_used = state.used_pieces | frozenset([abandoned]) if abandoned else state.used_pieces
        return replace(state,
            green_king_pos=dest,
            green_piece=None,
            green_piece_pos=None,
            used_pieces=new_used,
            turn=opponent(player),
            action_count=state.action_count + 1,
        )
    abandoned = state.red_piece_pos
    new_used = state.used_pieces | frozenset([abandoned]) if abandoned else state.used_pieces
    return replace(state,
        red_king_pos=dest,
        red_piece=None,
        red_piece_pos=None,
        used_pieces=new_used,
        turn=opponent(player),
        action_count=state.action_count + 1,
    )


def _null_move(state: GameState, player: str) -> List[Tuple[str, GameState]]:
    """Movimento nulo: passa a vez sem alterar posição."""
    new_state = replace(state,
        turn=opponent(player),
        action_count=state.action_count + 1,
    )
    return [('null', new_state)]


# ---------------------------------------------------------------------------
# Condições de fim de jogo
# ---------------------------------------------------------------------------

def is_terminal(state: GameState) -> bool:
    """Verifica se o jogo terminou."""
    # Sem peões pretos
    if not state.remaining_black_pawns:
        return True
    # 60 acções realizadas
    if state.action_count >= MAX_ACTIONS:
        return True
    # Um agente tomou mais de metade
    half = state.total_black_pawns / 2
    if state.green_captured > half or state.red_captured > half:
        return True
    # Sem peças no tabuleiro (todas usadas)
    pieces_on_board = [
        sym for _, _, sym in state.board.iter_cells()
        if sym in ACTIVATABLE_PIECES
    ]
    active_pieces_remaining = [
        p for p in pieces_on_board
        if state.board.find(p) and
        not all(pos in state.used_pieces for pos in state.board.find(p))
    ]
    if not active_pieces_remaining and state.green_piece is None and state.red_piece is None:
        return True
    return False


def game_result(state: GameState) -> str:
    """Devolve 'A' (verde vence), 'V' (vermelho vence) ou 'empate'."""
    if state.green_captured > state.red_captured:
        return 'A'
    if state.red_captured > state.green_captured:
        return 'V'
    return 'empate'
