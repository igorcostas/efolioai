from __future__ import annotations

# ---------------------------------------------------------------------------
# E-fólio B  —  UAb 2025/2026
# Arquitectura VPL: o programa é chamado várias vezes.
# Em cada chamada: lê resultados.csv, acrescenta UMA jogada por jogo
# em curso, grava e sai. O VPL volta a chamar até os 10 jogos terminarem.
# ---------------------------------------------------------------------------

import os
from time import perf_counter

from game_state import build_initial_state, GameState
from game_rules import is_terminal, get_valid_moves, game_result
from minimax import choose_move

CSV_FILE = "resultados.csv"
NUM_GAMES = 10
DEPTH = 3
TIME_LIMIT_MS = 900  # margem para ficar abaixo de 1s por jogada

# Apenas os tokens reconhecidos pelo enunciado
END_TOKENS = {"Brancas", "Pretas", "Empate"}


# ---------------------------------------------------------------------------
# Resultado -> token
# ---------------------------------------------------------------------------

def _result_token(state: GameState) -> str:
    """
    Usa game_result() das game_rules para garantir consistência
    com a mesma lógica usada pelo Minimax e pelo is_terminal.
    """
    result = game_result(state)   # 'A', 'V' ou 'empate'
    if result == 'A':
        return "Brancas"          # verde (A) ganhou
    if result == 'V':
        return "Pretas"           # vermelho (V) ganhou
    return "Empate"


# ---------------------------------------------------------------------------
# Leitura / escrita do CSV
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


def write_csv(lines: list) -> None:
    with open(CSV_FILE, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Reconstrução do estado a partir do histórico de jogadas
# ---------------------------------------------------------------------------

def state_from_moves(moves: list) -> GameState:
    """
    Reconstrói o GameState aplicando a sequência de movimentos já registada.
    Para quando encontra uma jogada não reconhecida (estado inválido).
    """
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
    """
    - Linha já com token final -> devolve intacta.
    - Linha em curso -> calcula 1 jogada com Minimax e acrescenta.
      Se o jogo terminar após a jogada, acrescenta também o token final.
    """
    tokens = line.split() if line.strip() else []

    # Jogo já terminado
    if tokens and tokens[-1] in END_TOKENS:
        return line

    # Filtra tokens de controlo que possam ter escapado
    moves_so_far = [t for t in tokens if t not in END_TOKENS]

    # Reconstrói estado
    state = state_from_moves(moves_so_far)

    # Verifica se já é terminal antes de jogar
    if is_terminal(state):
        token = _result_token(state)
        return (" ".join(moves_so_far) + " " + token).strip()

    # Verifica movimentos válidos antes de chamar Minimax
    player = state.turn
    valid = get_valid_moves(state, player)
    if not valid:
        # Sem movimentos: não deve acontecer (get_valid_moves devolve 'null'),
        # mas por segurança fecha o jogo
        token = _result_token(state)
        return (" ".join(moves_so_far) + " " + token).strip()

    # Minimax escolhe a jogada
    result = choose_move(state, my_player=player, depth=DEPTH, time_limit_ms=TIME_LIMIT_MS)

    # Guarda contra None inesperado
    if result is None or result[0] is None:
        token = _result_token(state)
        return (" ".join(moves_so_far) + " " + token).strip()

    action, next_state = result

    # Acrescenta a jogada à linha
    new_moves = moves_so_far + [action]
    new_line = " ".join(new_moves)

    # Usa next_state directamente (sem re-simular) para verificar terminal
    if is_terminal(next_state):
        new_line = new_line + " " + _result_token(next_state)

    return new_line


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main() -> int:
    lines = read_csv()

    for i in range(NUM_GAMES):
        lines[i] = process_line(lines[i])

    write_csv(lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
