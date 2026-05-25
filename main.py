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
from game_rules import is_terminal, get_valid_moves
from minimax import choose_move

try:
    from board import Board
except ImportError:
    from chess_pawn_mower.board import Board  # type: ignore

# Instância única — exactamente 64 caracteres conforme enunciado
INSTANCE_STRING = "Pp p pD ppBp p   pp pp  pCpVpp PP ppApCp  pp pp   p pBpp Dp p pP"

CSV_FILE = "resultados.csv"
NUM_GAMES = 10
DEPTH = 3
TIME_LIMIT_MS = 900  # margem para ficar abaixo de 1s por jogada

# Tokens que indicam jogo terminado
END_TOKENS = {"Brancas", "Pretas", "Empate", "Inválido", "Erro"}


# ---------------------------------------------------------------------------
# Leitura / escrita do CSV
# ---------------------------------------------------------------------------

def read_csv() -> list[str]:
    """
    Lê resultados.csv e devolve lista de 10 linhas (sem newline).
    Se o ficheiro não existir ou tiver menos de 10 linhas, completa com linhas vazias.
    """
    lines: list[str] = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", encoding="utf-8") as f:
            for raw in f:
                lines.append(raw.rstrip("\r\n"))
    # Garante exactamente NUM_GAMES linhas
    while len(lines) < NUM_GAMES:
        lines.append("")
    return lines[:NUM_GAMES]


def write_csv(lines: list[str]) -> None:
    with open(CSV_FILE, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Estado do jogo a partir da sequência de jogadas registada
# ---------------------------------------------------------------------------

def state_from_moves(moves: list[str]) -> GameState:
    """
    Reconstrói o GameState aplicando a sequência de movimentos já registada.
    'moves' é a lista de casas destino, ex: ['e6', 'd3', 'f7']
    """
    state = build_initial_state()
    for sq in moves:
        if sq in END_TOKENS:
            break
        player = state.turn
        valid = get_valid_moves(state, player)
        matched = None
        for action, next_state in valid:
            if action == sq:
                matched = next_state
                break
        if matched is None:
            # Jogada não reconhecida — devolve estado actual sem aplicar
            break
        state = matched
    return state


# ---------------------------------------------------------------------------
# Lógica de uma linha do CSV
# ---------------------------------------------------------------------------

def process_line(line: str) -> str:
    """
    Dada uma linha do CSV:
    - Se já terminou (token final presente) — devolve intacta.
    - Se ainda está em curso — calcula UMA jogada com Minimax e acrescenta.
    """
    tokens = line.split() if line.strip() else []

    # Verifica se já terminou
    if tokens and tokens[-1] in END_TOKENS:
        return line

    # Reconstrói estado a partir das jogadas já feitas
    moves_so_far = [t for t in tokens if t not in END_TOKENS]
    state = state_from_moves(moves_so_far)

    # Verifica condição terminal após aplicar todas as jogadas
    if is_terminal(state):
        result_token = _result_token(state)
        return (line.rstrip() + " " + result_token).strip()

    # Calcula a próxima jogada com Minimax
    player = state.turn
    action, _ = choose_move(state, my_player=player, depth=DEPTH, time_limit_ms=TIME_LIMIT_MS)

    new_line = (line.rstrip() + " " + action).strip() if line.strip() else action

    # Reconstrói estado com a nova jogada para verificar se terminou
    new_tokens = new_line.split()
    new_moves = [t for t in new_tokens if t not in END_TOKENS]
    new_state = state_from_moves(new_moves)

    if is_terminal(new_state):
        result_token = _result_token(new_state)
        new_line = new_line + " " + result_token

    return new_line


def _result_token(state: GameState) -> str:
    """Devolve o token de resultado conforme quem jogou nesta linha."""
    # Verde = agente que joga primeiro (turn='A' no início)
    # O token segue a perspectiva: quem ganhou
    if state.green_captured > state.red_captured:
        return "Brancas"   # agente verde (A) ganhou
    elif state.red_captured > state.green_captured:
        return "Pretas"    # agente vermelho (V) ganhou
    else:
        return "Empate"


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main() -> int:
    start = perf_counter()

    lines = read_csv()

    for i in range(NUM_GAMES):
        lines[i] = process_line(lines[i])

    write_csv(lines)

    elapsed = int((perf_counter() - start) * 1000)
    print("resultados.csv actualizado em {}ms".format(elapsed))
    for i, line in enumerate(lines):
        print("Jogo {:2d}: {}".format(i + 1, line[:80]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
