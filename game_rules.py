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
    """
    Devolve o símbolo lógico numa casa, considerando o estado dinâmico.

    Prioridade (da mais alta para a mais baixa):
    1. Peça activa verde (posicao actual)
    2. Peça activa vermelha (posicao actual)
    3. Rei verde (só se estiver fora de peça)
    4. Rei vermelho (só se estiver fora de peça)
    5. Peões pretos restantes
    6. Peças abandonadas -> peões brancos
    7. Casa de origem de peça activa -> vazia (peça saiu dali)
    8. Tabuleiro estático
    """
    pos = (row, col)

    # 1. Peça activa verde na posicao actual
    if state.green_piece and pos == state.green_piece_pos:
        return state.green_piece

    # 2. Peça activa vermelha na posicao actual
    if state.red_piece and pos == state.red_piece_pos:
        return state.red_piece

    # 3. Rei verde (apenas quando fora de peça)
    if state.green_king_pos is not None and pos == state.green_king_pos:
        return 'A'

    # 4. Rei vermelho (apenas quando fora de peça)
    if state.red_king_pos is not None and pos == state.red_king_pos:
        return 'V'

    # 5. Peões pretos restantes
    if pos in state.remaining_black_pawns:
        return 'p'

    # 6. Peças abandonadas tornam-se peões brancos
    if pos in state.used_pieces:
        return 'P'

    # 7. Casa de origem de peça activa fica vazia (o agente saiu dali)
    #    Verde: se está dentro de uma peça, a sua posicao original no board está vazia
    if state.green_piece is not None:
        orig_cell = state.board.get(row, col)
        if orig_cell == state.green_piece and pos != state.green_piece_pos:
            # Só há uma peça de cada tipo no tabuleiro original
            # Se a peça activa verde está noutro sítio, esta casa está vazia
            all_positions = state.board.find(state.green_piece)
            if len(all_positions) == 1:
                return ' '

    if state.red_piece is not None:
        orig_cell = state.board.get(row, col)
        if orig_cell == state.red_piece and pos != state.red_piece_pos:
            all_positions = state.board.find(state.red_piece)
            if len(all_positions) == 1:
                return ' '

    # 8. Tabuleiro estático
    return state.board.get(row, col)


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

    # ── MODO REI: agente fora de peça ───────────────────────────────────
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

            # Entra numa peça activável (não abandonada)
            elif cell in ACTIVATABLE_PIECES and dest not in state.used_pieces:
                moves.append((sq, _apply_enter_piece(state, player, dest, cell)))

        if not moves:
            return _null_move(state, player)
        return moves

    # ── MODO PEÇA: agente dentro de uma peça ───────────────────────────
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
    """
    Agente entra numa peça branca.
    O rei desaparece (king_pos=None) e a peça fica activa na posição dest.
    A casa de origem da peça no board estático é tratada como vazia por _cell_at.
    """
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
    # Correcto: itera posições de peças activáveis e verifica se todas estão em used_pieces
    piece_positions = [
        (row, col)
        for row, col, sym in state.board.iter_cells()
        if sym in ACTIVATABLE_PIECES
    ]
    all_pieces_used = all(pos in state.used_pieces for pos in piece_positions)
    if all_pieces_used and state.green_piece is None and state.red_piece is None:
        return True
    return False


def game_result(state: GameState) -> str:
    """Devolve 'A' (verde vence), 'V' (vermelho vence) ou 'empate'."""
    if state.green_captured > state.red_captured:
        return 'A'
    if state.red_captured > state.green_captured:
        return 'V'
    return 'empate'
