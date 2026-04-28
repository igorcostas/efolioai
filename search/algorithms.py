from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from itertools import count
from time import perf_counter
from typing import Callable, Iterable, Optional, Tuple, TypeVar

from .node import Node

State = TypeVar('State')
Action = TypeVar('Action')


def bfs(
    initial_state: State,
    is_goal: Callable[[State], bool],
    successors: Callable[[State], Iterable[Tuple[Action, State]]],
    time_limit_ms: Optional[int] = None,
) -> Optional[Node]:
    """Pesquisa em largura sobre estados hashable."""

    deadline = None if time_limit_ms is None else perf_counter() + (time_limit_ms / 1000.0)
    root = Node(state=initial_state)
    if is_goal(initial_state):
        return root

    frontier = deque([root])
    visited = {initial_state}

    while frontier:
        if deadline is not None and perf_counter() > deadline:
            return None
        node = frontier.popleft()
        for action, next_state in successors(node.state):
            if next_state in visited:
                continue
            child = Node(state=next_state, parent=node, action=action, g=node.g + 1)
            if is_goal(next_state):
                return child
            visited.add(next_state)
            frontier.append(child)

    return None


def astar(
    initial_state: State,
    is_goal: Callable[[State], bool],
    successors: Callable[[State], Iterable[Tuple[Action, State, float]]],
    heuristic: Callable[[State], float],
    time_limit_ms: Optional[int] = None,
) -> Optional[Node]:
    """Pesquisa A* genérica."""

    deadline = None if time_limit_ms is None else perf_counter() + (time_limit_ms / 1000.0)
    root = Node(state=initial_state, h=heuristic(initial_state))
    if is_goal(initial_state):
        return root

    open_set: list[tuple[float, int, Node]] = []
    order = count()
    heappush(open_set, (root.f, next(order), root))
    best_g: dict[State, float] = {initial_state: 0.0}

    while open_set:
        if deadline is not None and perf_counter() > deadline:
            return None
        _, _, node = heappop(open_set)
        if is_goal(node.state):
            return node
        if node.g > best_g.get(node.state, float('inf')):
            continue
        for action, next_state, step_cost in successors(node.state):
            tentative_g = node.g + step_cost
            if tentative_g >= best_g.get(next_state, float('inf')):
                continue
            best_g[next_state] = tentative_g
            child = Node(
                state=next_state,
                parent=node,
                action=action,
                g=tentative_g,
                h=heuristic(next_state),
            )
            heappush(open_set, (child.f, next(order), child))

    return None
