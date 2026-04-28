from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTANCE_COUNT = 10
TIME_LIMIT_MS = 10_000
TIME_LIMIT_SECONDS = TIME_LIMIT_MS / 1000
DEFAULT_TIMEOUT_SECONDS = TIME_LIMIT_SECONDS
DEFAULT_RESULTS_CSV = PROJECT_ROOT / 'resultados.csv'
INSTANCE_FILE_PATTERN = 'instancia_{index}.txt'
INSTANCE_DIRECTORY = Path("instancias")
HEURISTIC_OPTIONS = ('remaining_black_pawns',)
DEFAULT_HEURISTIC = HEURISTIC_OPTIONS[0]
