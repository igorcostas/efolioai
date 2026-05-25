from __future__ import annotations

# ---------------------------------------------------------------------------
# E-fólio B  —  Introdução à Inteligência Artificial  |  UAb 2025/2026
# ---------------------------------------------------------------------------
# Formato do resultados.csv esperado pelo VPL:
#   Instância;Acções;Resultado;Peões_A;Peões_V;Solução_A;Solução_V
# ---------------------------------------------------------------------------

from time import perf_counter

try:
    from chess_pawn_mower.board import Board
except (ModuleNotFoundError, ImportError):
    from board import Board  # type: ignore

from game_state import build_initial_state
from game_rules import game_result, is_terminal
from minimax import choose_move

# Instância única definida no enunciado
INSTANCE_STRING = "Pp p pD ppBp p  pp pp  pCpVpp PP ppApCp  pp pp  p pBpp Dp p pP"

# Parâmetros do Minimax
DEPTH = 3
TIME_LIMIT_MS = 3000


# ---------------------------------------------------------------------------
# Loop principal do jogo
# ---------------------------------------------------------------------------

def play_game() -> dict:
    """
    Executa o jogo completo entre agente verde (A) e agente vermelho (V),
    ambos controlados pelo Minimax Alpha-Beta.
    Devolve dicionário com estado final e histórico de acções.
    """
    state = build_initial_state()

    green_actions: list[str] = []
    red_actions: list[str] = []

    start_time = perf_counter()

    while not is_terminal(state):
        player = state.turn
        action, next_state = choose_move(
            state,
            my_player=player,
            depth=DEPTH,
            time_limit_ms=TIME_LIMIT_MS,
        )
        if player == 'A':
            green_actions.append(action)
        else:
            red_actions.append(action)

        state = next_state

    elapsed_ms = int((perf_counter() - start_time) * 1000)

    return {
        'instance': 1,
        'state': state,
        'green_actions': green_actions,
        'red_actions': red_actions,
        'winner': game_result(state),
        'total_actions': state.action_count,
        'elapsed_ms': elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Escrita do CSV
# ---------------------------------------------------------------------------

def write_csv(result: dict) -> None:
    """
    Grava resultados.csv com separador ';'.
    Cabeçalho: Instância;Acções;Resultado;Peões_A;Peões_V;Solução_A;Solução_V
    Uma linha por instância (apenas instância 1 no e-fólio B).
    """
    state = result['state']
    sol_a = ' '.join(result['green_actions'])
    sol_v = ' '.join(result['red_actions'])

    with open('resultados.csv', 'w', encoding='utf-8') as f:
        f.write('Instância;Acções;Resultado;Peões_A;Peões_V;Solução_A;Solução_V\n')
        f.write('{};{};{};{};{};{};{}\n'.format(
            result['instance'],
            result['total_actions'],
            result['winner'],
            state.green_captured,
            state.red_captured,
            sol_a,
            sol_v,
        ))


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main() -> int:
    print('E-fólio B — a iniciar jogo...')
    print('Instância: {}'.format(INSTANCE_STRING))
    print()

    result = play_game()
    write_csv(result)

    state = result['state']
    print('─' * 50)
    print('Jogo terminado!')
    print('Acções totais : {}'.format(result['total_actions']))
    print('Tempo total   : {}ms'.format(result['elapsed_ms']))
    print('Resultado     : {}'.format(result['winner']))
    print('Peões A (verde)    : {}'.format(state.green_captured))
    print('Peões V (vermelho) : {}'.format(state.red_captured))
    print()
    print('Solução A : {}'.format(' '.join(result['green_actions'])))
    print('Solução V : {}'.format(' '.join(result['red_actions'])))
    print('─' * 50)
    print('resultados.csv gravado.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
