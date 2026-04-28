python -c "
from pathlib import Path
from chess_pawn_mower.board import Board
from chess_pawn_mower.problem import solve_board, solution_string
from time import perf_counter

for i in range(1, 11):
    txt = Path(f'instancias/instancia_{i}.txt').read_text()
    board = Board.from_line(txt)
    t = perf_counter()
    node = solve_board(board, time_limit_ms=9500)
    elapsed = int((perf_counter() - t) * 1000)
    sol = solution_string(node)
    if elapsed >= 9400:
        status = 'TIMEOUT'
    elif not sol:
        status = 'IMPOSSIVEL'
    else:
        status = 'OK'
    print(f'instancia_{i}: {elapsed}ms | {status} | {sol[:60] if sol else \"\"}')
"