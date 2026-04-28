from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chess_pawn_mower.board import Board
from config.parameters import DEFAULT_RESULTS_CSV, INSTANCE_COUNT
from io_utils.csv_writer import write_results_csv
from io_utils.instances import load_instances
from main import run_batch
from search.algorithms import bfs


class SmokeTests(unittest.TestCase):
    def test_bfs_trivial_goal(self) -> None:
        result = bfs(0, lambda state: state == 0, lambda state: [])
        self.assertIsNotNone(result)
        self.assertEqual(result.state, 0)

    def test_board_from_line_preserves_spaces(self) -> None:
        line = ('P' + ' ' * 7) * 8
        board = Board.from_line(line)
        self.assertEqual(board.to_line(), line)

    def test_csv_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / 'results.csv'
            write_results_csv([{'Instância': '1', 'Tempo(ms)': 12, 'Solução': 'ABC'}], output)
            content = output.read_text(encoding='utf-8')
            self.assertIn('Instância;Tempo(ms);Solução', content)
            self.assertIn('1;12;ABC', content)
            self.assertEqual(len(content.strip().splitlines()), INSTANCE_COUNT + 1)

    def test_load_instances_falls_back_to_blank_boards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            instances = load_instances(Path(tmp_dir), count=INSTANCE_COUNT)
            self.assertEqual(len(instances), INSTANCE_COUNT)
            self.assertTrue(all(instance.missing for instance in instances))

    def test_run_batch_writes_ten_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output = tmp_path / 'results.csv'
            rows = run_batch(tmp_path, output)
            self.assertEqual(len(rows), INSTANCE_COUNT)
            csv_lines = output.read_text(encoding='utf-8').strip().splitlines()
            self.assertEqual(len(csv_lines), INSTANCE_COUNT + 1)

    def test_default_results_path_is_inside_project(self) -> None:
        self.assertEqual(DEFAULT_RESULTS_CSV.name, 'resultados.csv')
