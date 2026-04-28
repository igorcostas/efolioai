from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from chess_pawn_mower.board import Board
from chess_pawn_mower.problem import build_initial_state
from chess_pawn_mower.state import PawnMowerState
from config.parameters import INSTANCE_FILE_PATTERN


@dataclass(frozen=True)
class LoadedInstance:
    index: int
    path: Path
    board: Board
    initial_state: PawnMowerState
    missing: bool = False


def iter_instance_paths(directory: Path, count: int = 10) -> Iterable[Path]:
    for index in range(1, count + 1):
        yield directory / INSTANCE_FILE_PATTERN.format(index=index)


def _blank_board() -> Board:
    return Board.from_line(' ' * 64)


def load_instance(path: Path, index: Optional[int] = None) -> LoadedInstance:
    missing = not path.exists()
    board = _blank_board() if missing else Board.from_line(path.read_text(encoding='utf-8'))
    return LoadedInstance(
        index=index if index is not None else 0,
        path=path,
        board=board,
        initial_state=build_initial_state(board),
        missing=missing,
    )


def load_instances(directory: Path, count: int = 10) -> list[LoadedInstance]:
    instances: list[LoadedInstance] = []
    for index, path in enumerate(iter_instance_paths(directory, count=count), start=1):
        instances.append(load_instance(path, index=index))
    return instances
