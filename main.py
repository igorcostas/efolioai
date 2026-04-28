from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

# Imports planos — compatíveis com VPL (todos os ficheiros na mesma pasta)
try:
    from chess_pawn_mower.problem import solution_string, solve_board
    from config.parameters import DEFAULT_RESULTS_CSV, INSTANCE_COUNT, INSTANCE_DIRECTORY, TIME_LIMIT_MS
    from io_utils.csv_writer import write_results_csv
    from io_utils.instances import load_instances
except ModuleNotFoundError:
    # Pasta plana do VPL — sem subpastas
    from problem import solution_string, solve_board  # type: ignore
    from parameters import DEFAULT_RESULTS_CSV, INSTANCE_COUNT, INSTANCE_DIRECTORY, TIME_LIMIT_MS  # type: ignore
    from csv_writer import write_results_csv  # type: ignore
    from instances import load_instances  # type: ignore


DEFAULT_DIR = Path('.')
DEFAULT_CSV = Path('resultados.csv')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='projeto-efolio-ia',
        description='Resolve as 10 instâncias do puzzle e gera resultados.csv.',
    )
    parser.add_argument(
        '--instances-dir',
        type=Path,
        default=DEFAULT_DIR,
        help='Diretório com instancia_1.txt ... instancia_10.txt.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_CSV,
        help='Caminho do ficheiro resultados.csv.',
    )
    return parser


def run_batch(instances_dir: Path, output: Path) -> list[dict]:
    rows: list[dict] = []
    resolved = 0
    for loaded in load_instances(instances_dir, count=10):
        start = perf_counter()
        node = solve_board(loaded.board, time_limit_ms=10_000)
        elapsed_ms = int((perf_counter() - start) * 1000)
        sol = solution_string(node)
        acoes = len(sol.split()) if sol else 0
        if sol:
            resolved += 1
        status = sol if sol else 'sem solucao'
        print(
            f'Instância {loaded.index} ... {"com solucao" if sol else "sem solucao"}'
            f'   acoes={acoes}    tempo={elapsed_ms}ms'
        )
        rows.append({
            'Instância': loaded.index,
            'Tempo(ms)': elapsed_ms,
            'Solução': sol,
        })
    write_results_csv(rows, output, expected_rows=10)
    print(f'Resolvidas: {resolved}/10 -> {output} gravado.')
    return rows


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run_batch(args.instances_dir, args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
