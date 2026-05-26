from __future__ import annotations

# ---------------------------------------------------------------------------
# E-folio B  --  UAb 2025/2026
# Arquitectura VPL: o programa e chamado varias vezes.
# Em cada chamada: le resultados.csv, acrescenta UMA jogada por jogo
# em curso, grava e sai. O VPL volta a chamar ate os 10 jogos terminarem.
# ---------------------------------------------------------------------------

import os
from typing import Optional

from game_state import build_initial_state, GameState, active_position
from game_rules import is_terminal, get_valid_moves, game_result
from minimax import choose_move

CSV_FILE = "resultados.csv"
NUM_GAMES = 10
DEPTH = 3
TIME_LIMIT_MS = 900

END_TOKENS = {"Brancas", "Pretas", "Empate", "Invalido", "Erro"}


# ---------------------------------------------------------------------------
# Resultado -> token
# ---------------------------------------------------------------------------

def _result_token(state: GameState) -> str:
    result = game_result(state)  # 'A', 'V' ou 'empate'
    if result == 'A':
        return "Brancas"
    if result == 'V':
        return "Pretas"
    return "Empate"


# ---------------------------------------------------------------------------
# Leitura do CSV existente
# ---------------------------------------------------------------------------

def read_csv() -> list:
    lines = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", encoding="utf-8") as f:
            for raw in f:
                lines.append(raw.rstrip("\r\n"))
    while len(lines) < NUM_GAMES:
        lines.append("")
    return lines[:NUM_GAMES]


# ---------------------------------------------------------------------------
# Auxiliares de linha
# ---------------------------------------------------------------------------

def _split_line(line: str) -> tuple[list[str], Optional[str]]:
    tokens = line.split() if line.strip() else []
    terminal_positions = [i for i, token in enumerate(tokens) if token in END_TOKENS]
    if not terminal_positions:
        return tokens, None
    if terminal_positions[-1] != len(tokens) - 1 or len(terminal_positions) > 1:
        return [t for t in tokens if t not in END_TOKENS], "Invalido"
    return tokens[:-1], tokens[-1]


def _result_points(result: Optional[str]) -> int:
    return {
        "Brancas": 2, "Pretas": 0, "Empate": 1,
        "Invalido": -1, "Erro": -1, None: 0,
    }.get(result, 0)


# ---------------------------------------------------------------------------
# Reconstrucao do estado a partir do historico de jogadas
# ---------------------------------------------------------------------------

def state_from_moves(moves: list[str]) -> Optional[GameState]:
    state = build_initial_state()
    for sq in moves:
        player = state.turn
        valid = get_valid_moves(state, player)
        matched_state = next(
            (next_state for action, next_state in valid if action == sq), None
        )
        if matched_state is None and len(valid) == 1 and valid[0][0] == 'null':
            active_pos = active_position(state, player)
            if sq == 'null' or (
                active_pos is not None
                and sq == state.board.index_to_square(*active_pos)
            ):
                matched_state = valid[0][1]
        if matched_state is None:
            return None
        state = matched_state
    return state


# ---------------------------------------------------------------------------
# Processamento de cada linha
# ---------------------------------------------------------------------------

def process_line(line: str) -> str:
    tokens = line.split() if line.strip() else []
    terminal_positions = [i for i, token in enumerate(tokens) if token in END_TOKENS]

    if terminal_positions:
        if terminal_positions[-1] != len(tokens) - 1 or len(terminal_positions) > 1:
            moves_so_far = [t for t in tokens[:terminal_positions[0]] if t not in END_TOKENS]
            return (" ".join(moves_so_far) + " Invalido").strip()
        return line

    moves_so_far = tokens
    state = state_from_moves(moves_so_far)
    if state is None:
        return (" ".join(moves_so_far) + " Invalido").strip()

    if is_terminal(state):
        return (" ".join(moves_so_far) + " " + _result_token(state)).strip()

    player = state.turn
    valid = get_valid_moves(state, player)
    if not valid:
        return (" ".join(moves_so_far) + " " + _result_token(state)).strip()

    result = choose_move(state, my_player=player, depth=DEPTH, time_limit_ms=TIME_LIMIT_MS)
    if result is None or result[0] is None:
        return (" ".join(moves_so_far) + " Erro").strip()

    action, next_state = result
    if action == 'null':
        active_pos = active_position(state, player)
        if active_pos is None:
            return (" ".join(moves_so_far) + " Erro").strip()
        action = state.board.index_to_square(*active_pos)

    new_moves = moves_so_far + [action]
    new_line = " ".join(new_moves)

    if is_terminal(next_state):
        new_line = new_line + " " + _result_token(next_state)

    return new_line


# ---------------------------------------------------------------------------
# Ponto de entrada -- compativel VPL
# with open aberto ANTES do loop; escreve e faz flush linha a linha
# Sem print(), sem argparse, sem emojis, sem sys.argv
# ---------------------------------------------------------------------------

def main() -> int:
    lines = read_csv()

    with open(CSV_FILE, "w", encoding="utf-8") as f:
        for i in range(NUM_GAMES):
            lines[i] = process_line(lines[i])
            f.write(lines[i] + "\n")
            f.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
