from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GraphDataBundle:
    """Unified graph container used across rule-based, ranking, and RL stages."""

    name: str
    num_nodes: int
    edge_index: Any
    x: Any
    y: Any
    nx_graph: Any
    importance: Any
    community: Any
    edge_bc: Dict[Tuple[int, int], float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskPair:
    source: int
    target: int
    shortest_len: int
    same_community: bool
    pair_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PathRecord:
    nodes: List[int]
    edges: List[Tuple[int, int]]
    source: int
    target: int
    success: bool
    method: str
    score: Optional[float] = None
    features: Optional[Dict[str, float]] = None
    fragility: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    task: TaskPair
    path: PathRecord
    total_reward: float
    steps: int
    reached_target: bool
