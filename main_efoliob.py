from __future__ import annotations

# ---------------------------------------------------------------------------
# E-fólio B  —  Introdução à Inteligência Artificial  |  UAb 2025/2026
# ---------------------------------------------------------------------------
# O CSV é aberto ANTES do loop e escrito linha a linha conforme cada
# instância é resolvida.
# Formato: Instância;I2(Tempo(ms));Solução
# ---------------------------------------------------------------------------

from time import perf_counter

try:
    from chess_pawn_mower.board import Board
except (ModuleNotFoundError, ImportError):
    from board import Board  # type: ignore

from game_state import build_initial_state
from game_rules import game_result, is_terminal, get_valid_moves
from minimax import choose_move

# Instância única definida no enunciado
INSTANCE_STRING = "Pp p pD ppBp p  pp pp  pCpVpp PP ppApCp  pp pp  p pBpp Dp p pP"

# Parâmetros do Minimax
DEPTH = 3
TIME_LIMIT_MS = 3000
NUM_GAMES = 10


# ---------------------------------------------------------------------------
# Execução de um jogo completo
# ---------------------------------------------------------------------------

def play_game() -> dict:
    """
    Executa o jogo completo entre agente verde (A) e agente vermelho (V),
    ambos controlados pelo Minimax Alpha-Beta.
    Devolve dicionário com estado final, solução e tempo.
    """
    state = build_initial_state()
    actions: list = []           # todas as acções intercaladas (A, V, A, V, ...)
    start_time = perf_counter()

    while not is_terminal(state):
        player = state.turn

        # Verifica movimentos válidos antes de chamar Minimax
        valid = get_valid_moves(state, player)
        if not valid:
            break

        result = choose_move(
            state,
            my_player=player,
            depth=DEPTH,
            time_limit_ms=TIME_LIMIT_MS,
        )

        # Guarda contra None inesperado
        if result is None or result[0] is None:
            break

        action, next_state = result
        actions.append(action)
        state = next_state

    elapsed_ms = int((perf_counter() - start_time) * 1000)

    return {
        'state': state,
        'solution': ' '.join(actions),
        'elapsed_ms': elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main() -> int:
    with open('resultados.csv', 'w', encoding='utf-8') as f:
        f.write('Instância;I2(Tempo(ms));Solução\n')

        for i in range(1, NUM_GAMES + 1):
            result = play_game()
            f.write('{};{};{}\n'.format(
                i,
                result['elapsed_ms'],
                result['solution'],
            ))
            # Garante que a linha é escrita imediatamente no disco
            f.flush()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
