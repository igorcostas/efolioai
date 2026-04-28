from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

FILES = 'abcdefgh'


@dataclass(frozen=True)
class Board:
    """Representação 8x8 do tabuleiro."""

    cells: tuple[tuple[str, ...], ...]

    @classmethod
    def from_line(cls, line: str) -> 'Board':
        cleaned = line.rstrip('\r\n')
        if len(cleaned) != 64:
            raise ValueError('A linha do tabuleiro deve ter exatamente 64 caracteres.')
        rows = tuple(tuple(cleaned[row * 8:(row + 1) * 8]) for row in range(8))
        return cls(rows)

    @staticmethod
    def square_to_index(square: str) -> tuple[int, int]:
        if len(square) != 2 or square[0] not in FILES or square[1] not in '12345678':
            raise ValueError(f'Casa inválida: {square!r}')
        return int(square[1]) - 1, FILES.index(square[0])

    @staticmethod
    def index_to_square(row: int, col: int) -> str:
        if not (0 <= row < 8 and 0 <= col < 8):
            raise ValueError('Coordenadas fora do tabuleiro.')
        return f'{FILES[col]}{row + 1}'

    def to_line(self) -> str:
        return ''.join(''.join(row) for row in self.cells)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < 8 and 0 <= col < 8

    def get(self, row: int, col: int) -> str:
        if not self.in_bounds(row, col):
            raise IndexError('Coordenadas fora do tabuleiro.')
        return self.cells[row][col]

    def find(self, target: str) -> list[tuple[int, int]]:
        positions: list[tuple[int, int]] = []
        for row_idx, row in enumerate(self.cells):
            for col_idx, value in enumerate(row):
                if value == target:
                    positions.append((row_idx, col_idx))
        return positions

    def iter_cells(self) -> Iterable[tuple[int, int, str]]:
        for row_idx, row in enumerate(self.cells):
            for col_idx, value in enumerate(row):
                yield row_idx, col_idx, value
