from __future__ import annotations

# ---------------------------------------------------------------------------
# E-folio B  --  UAb 2025/2026
# Arquitectura VPL: o programa e chamado varias vezes.
# Em cada chamada: le resultados.csv, acrescenta UMA jogada por jogo
# em curso, grava e sai. O VPL volta a chamar ate os 10 jogos terminarem.
# ---------------------------------------------------------------------------

import os

from game_state import build_initial_state, GameState
from game_rules import is_terminal, get_valid_moves, game_result
from minimax import choose_move

CSV_FILE = "resultados.csv"
NUM_GAMES = 10
DEPTH = 3
TIME_LIMIT_MS = 900

END_TOKENS = {"Brancas", "Pretas", "Empate"}


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
# Reconstrucao do estado a partir do historico de jogadas
# ---------------------------------------------------------------------------

def state_from_moves(moves: list) -> GameState:
    state = build_initial_state()
    for sq in moves:
        player = state.turn
        valid = get_valid_moves(state, player)
        matched = None
        for action, next_state in valid:
            if action == sq:
                matched = next_state
                break
        if matched is None:
            break
        state = matched
    return state


# ---------------------------------------------------------------------------
# Processamento de cada linha
# ---------------------------------------------------------------------------

def process_line(line: str) -> str:
    tokens = line.split() if line.strip() else []

    # Jogo ja terminado
    if tokens and tokens[-1] in END_TOKENS:
        return line

    moves_so_far = [t for t in tokens if t not in END_TOKENS]
    state = state_from_moves(moves_so_far)

    if is_terminal(state):
        return (" ".join(moves_so_far) + " " + _result_token(state)).strip()

    player = state.turn
    valid = get_valid_moves(state, player)
    if not valid:
        return (" ".join(moves_so_far) + " " + _result_token(state)).strip()

    result = choose_move(state, my_player=player, depth=DEPTH, time_limit_ms=TIME_LIMIT_MS)

    if result is None or result[0] is None:
        return (" ".join(moves_so_far) + " " + _result_token(state)).strip()

    action, next_state = result

    new_moves = moves_so_far + [action]
    new_line = " ".join(new_moves)

    if is_terminal(next_state):
        new_line = new_line + " " + _result_token(next_state)

    return new_line


# ---------------------------------------------------------------------------
# Ponto de entrada -- compativel VPL
# with open aberto ANTES do loop; escreve e faz flush linha a linha
# Sem print(), sem argparse, sem emojis, sem imports inuteis
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
