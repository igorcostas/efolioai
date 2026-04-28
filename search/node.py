from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
@dataclass
class Node:
    """Nó genérico de pesquisa."""
    state: Any
    parent: Optional['Node'] = None
    action: Any = None
    g: float = 0.0
    h: float = 0.0
    @property
    def f(self) -> float:
        return self.g + self.h
    def path(self) -> list['Node']:
        nodes: list[Node] = []
        current: Optional[Node] = self
        while current is not None:
            nodes.append(current)
            current = current.parent
        nodes.reverse()
        return nodes
