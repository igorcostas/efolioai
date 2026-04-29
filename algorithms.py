from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from itertools import count
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    from search.node import Node
except (ModuleNotFoundError, ImportError):
    from node import Node  # type: ignore


def bfs(initial_state, is_goal, successors, time_limit_ms=None):
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


def astar(initial_state, is_goal, successors, heuristic, time_limit_ms=None, max_nodes=1_000_000):
    """A* que devolve o melhor no encontrado ate ao timeout (em vez de None).
    Assim instancias dificeis retornam uma solucao suboptima em vez de falhar.
    """
    deadline = None if time_limit_ms is None else perf_counter() + (time_limit_ms / 1000.0)
    root = Node(state=initial_state, h=heuristic(initial_state))
    if is_goal(initial_state):
        return root

    open_set = []
    order = count()
    heappush(open_set, (root.f, next(order), root))
    best_g = {initial_state: 0.0}

    # guarda o no objectivo de menor custo encontrado (para timeout)
    best_goal_node = None
    nodes_expanded = 0

    while open_set:
        if deadline is not None and perf_counter() > deadline:
            break
        if nodes_expanded >= max_nodes:
            break

        _, _, node = heappop(open_set)

        if is_goal(node.state):
            # solucao optima encontrada — devolve imediatamente
            return node

        if node.g > best_g.get(node.state, float('inf')):
            continue

        nodes_expanded += 1

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

            # se este filho ja e objectivo, regista como melhor candidato
            if is_goal(next_state):
                if best_goal_node is None or tentative_g < best_goal_node.g:
                    best_goal_node = child

    # timeout ou max_nodes atingido — devolve melhor solucao parcial encontrada
    return best_goal_node
