from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from chess_pawn_mower.problem import solution_string, solve_board
from config.parameters import DEFAULT_RESULTS_CSV, INSTANCE_COUNT, INSTANCE_DIRECTORY, TIME_LIMIT_MS
from io_utils.csv_writer import write_results_csv
from io_utils.instances import load_instances


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='projeto-efolio-ia',
        description='Resolve as 10 instâncias do puzzle e gera resultados.csv.',
    )
    parser.add_argument(
        '--instances-dir',
        type=Path,
        default=INSTANCE_DIRECTORY,
        help='Diretório que contém instancia_1.txt ... instancia_10.txt.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help='Caminho do ficheiro resultados.csv.',
    )
    return parser


def run_batch(instances_dir: Path, output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for loaded in load_instances(instances_dir, count=INSTANCE_COUNT):
        start = perf_counter()
        node = solve_board(loaded.board, time_limit_ms=TIME_LIMIT_MS)
        elapsed_ms = int((perf_counter() - start) * 1000)
        rows.append(
            {
                'Instância': loaded.index,
                'Tempo(ms)': elapsed_ms,
                'Solução': solution_string(node),
            }
        )

    write_results_csv(rows, output, expected_rows=INSTANCE_COUNT)
    return rows


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    rows = run_batch(args.instances_dir, args.output)
    print(f'Processadas {len(rows)} instâncias. CSV gerado em: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
