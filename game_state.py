from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional, Tuple

try:
    from chess_pawn_mower.board import Board
except (ModuleNotFoundError, ImportError):
    from board import Board  # type: ignore

Position = Tuple[int, int]

# Instância única do e-fólio B (hardcoded conforme enunciado) — exactamente 64 chars
INSTANCE_STRING = "Pp p pD ppBp p   pp pp  pCpVpp PP ppApCp  pp pp   p pBpp Dp p pP"


@dataclass(frozen=True)
class GameState:
    board: Board

    # Peões pretos restantes no tabuleiro
    remaining_black_pawns: FrozenSet[Position]

    # ── Agente Verde (rei 'A') ──────────────────────────────
    green_king_pos: Optional[Position]          # posição do rei verde (fora de peça)
    green_piece: Optional[str]                  # peça activa ('T','B','C','D') ou None
    green_piece_pos: Optional[Position]         # posição da peça activa verde
    green_captured: int                         # peões pretos capturados pelo verde

    # ── Agente Vermelho (rei 'V') ───────────────────────────
    red_king_pos: Optional[Position]            # posição do rei vermelho (fora de peça)
    red_piece: Optional[str]                    # peça activa ou None
    red_piece_pos: Optional[Position]           # posição da peça activa vermelha
    red_captured: int                           # peões pretos capturados pelo vermelho

    # ── Controlo do jogo ────────────────────────────────────
    turn: str                                   # 'A' (verde) ou 'V' (vermelho)
    action_count: int                           # total de acções realizadas (máx 60)
    used_pieces: FrozenSet[Position]            # posições de peças já abandonadas
                                                # (transformadas em peões brancos)
    total_black_pawns: int                      # total inicial de peões pretos


def build_initial_state() -> GameState:
    """Constrói o estado inicial a partir da instância hardcoded."""
    board = Board.from_line(INSTANCE_STRING)

    green_positions = board.find('A')
    red_positions = board.find('V')

    if not green_positions:
        raise ValueError("Rei verde 'A' não encontrado na instância.")
    if not red_positions:
        raise ValueError("Rei vermelho 'V' não encontrado na instância.")

    green_king_pos = green_positions[0]
    red_king_pos = red_positions[0]

    black_pawns = frozenset(board.find('p'))
    total_black_pawns = len(black_pawns)

    return GameState(
        board=board,
        remaining_black_pawns=black_pawns,
        green_king_pos=green_king_pos,
        green_piece=None,
        green_piece_pos=None,
        green_captured=0,
        red_king_pos=red_king_pos,
        red_piece=None,
        red_piece_pos=None,
        red_captured=0,
        turn='A',
        action_count=0,
        used_pieces=frozenset(),
        total_black_pawns=total_black_pawns,
    )


def active_position(state: GameState, player: str) -> Optional[Position]:
    """Devolve a posição activa do agente (peça ou rei)."""
    if player == 'A':
        return state.green_piece_pos if state.green_piece else state.green_king_pos
    return state.red_piece_pos if state.red_piece else state.red_king_pos


def opponent(player: str) -> str:
    return 'V' if player == 'A' else 'A'


def opponent_position(state: GameState, player: str) -> Optional[Position]:
    """Devolve a posição activa do adversário."""
    return active_position(state, opponent(player))
