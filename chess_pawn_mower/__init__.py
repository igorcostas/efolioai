"""Domínio do problema da peça contra peões e rei."""

from .board import Board
from .moves import capture_targets, king_step_targets
from .problem import build_initial_state, heuristic, is_goal, solution_string, solve_board, successors
from .state import PawnMowerState

__all__ = [
    'Board',
    'PawnMowerState',
    'build_initial_state',
    'successors',
    'is_goal',
    'heuristic',
    'solve_board',
    'solution_string',
    'capture_targets',
    'king_step_targets',
]
