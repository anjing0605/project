
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import random
import networkx as nx


class FragilityEvaluator:
    """
    Evaluate structural damage caused by removing path edges.

    Fast version:
    - approximate global efficiency by sampled node pairs
    - approximate ASP on LCC by sampled sources
    - exact LCC ratio
    - internal cache for repeated paths
    """

    def __init__(
        self,
        lambda_E: float = 0.4,
        lambda_LCC: float = 0.4,
        lambda_ASP: float = 0.2,
        efficiency_num_pairs: int = 2000,
        asp_num_sources: int = 64,
        random_seed: int = 42,
        use_bridge_shortcut: bool = False,
    ):
        self.lambda_E = float(lambda_E)
        self.lambda_LCC = float(lambda_LCC)
        self.lambda_ASP = float(lambda_ASP)

        # 采样参数：可继续调
        self.efficiency_num_pairs = int(efficiency_num_pairs)
        self.asp_num_sources = int(asp_num_sources)
        self.random_seed = int(random_seed)
        self._rng = random.Random(self.random_seed)

        # 可选：若 path 中没有桥，则快速近似为“不太脆弱”
        self.use_bridge_shortcut = bool(use_bridge_shortcut)

        # path 级缓存
        self._frag_cache: Dict[Tuple[int, ...], Dict[str, float]] = {}

        # base 图桥边缓存（首次按图构建）
        self._bridge_edge_cache: Optional[set[Tuple[int, int]]] = None

    @staticmethod
    def _canon_edge(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u <= v else (v, u)

    @staticmethod
    def _path_edges(path: List[int]) -> List[Tuple[int, int]]:
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    @staticmethod
    def _safe_sample(rng: random.Random, items: List[int], k: int) -> List[int]:
        if k <= 0:
            return []
        if len(items) <= k:
            return list(items)
        return rng.sample(items, k)

    @staticmethod
    def global_efficiency_exact(G: nx.Graph) -> float:
        n = G.number_of_nodes()
        if n <= 1:
            return 0.0
        return float(nx.global_efficiency(G))

    @staticmethod
    def avg_shortest_path_of_lcc_exact(G: nx.Graph) -> float:
        """
        Exact average shortest path length on the largest connected component.
        """
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return 0.0
        components = list(nx.connected_components(G))
        if not components:
            return 0.0
        lcc_nodes = max(components, key=len)
        if len(lcc_nodes) <= 1:
            return 0.0
        subg = G.subgraph(lcc_nodes).copy()
        try:
            return float(nx.average_shortest_path_length(subg))
        except Exception:
            return 0.0
    def global_efficiency_approx(self, G: nx.Graph) -> float:
        """
        Approximate global efficiency by sampled unordered node pairs.
        """
        nodes = list(G.nodes())
        n = len(nodes)
        if n <= 1:
            return 0.0

        total_pairs = n * (n - 1) // 2
        m = min(self.efficiency_num_pairs, total_pairs)
        if m <= 0:
            return 0.0

        acc = 0.0
        sampled = 0
        seen = set()

        # 对中小图，2000 对已经足够快很多
        while sampled < m:
            u, v = self._rng.sample(nodes, 2)
            e = self._canon_edge(u, v)
            if e in seen:
                continue
            seen.add(e)

            try:
                d = nx.shortest_path_length(G, source=u, target=v)
                if d > 0:
                    acc += 1.0 / float(d)
            except nx.NetworkXNoPath:
                pass

            sampled += 1

        # global efficiency 定义是 ordered pairs 上平均，但 unordered / ordered 只差常数，
        # 在前后差分时影响很小，这里直接用 sampled mean 作为近似。
        return float(acc) / float(m)

    @staticmethod
    def largest_cc_nodes(G: nx.Graph) -> List[int]:
        if G.number_of_nodes() == 0:
            return []
        components = list(nx.connected_components(G))
        if not components:
            return []
        return list(max(components, key=len))

    def avg_shortest_path_of_lcc_approx(self, G: nx.Graph) -> float:
        """
        Approximate average shortest path length on LCC
        by sampling several source nodes and averaging
        their shortest path lengths to all reachable nodes in LCC.
        """
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return 0.0

        lcc_nodes = self.largest_cc_nodes(G)
        if len(lcc_nodes) <= 1:
            return 0.0

        subg = G.subgraph(lcc_nodes)

        sampled_sources = self._safe_sample(self._rng, lcc_nodes, self.asp_num_sources)
        if not sampled_sources:
            return 0.0

        total = 0.0
        count = 0

        for s in sampled_sources:
            lengths = nx.single_source_shortest_path_length(subg, s)
            # 去掉自己
            for t, d in lengths.items():
                if t == s:
                    continue
                total += float(d)
                count += 1

        if count == 0:
            return 0.0
        return float(total) / float(count)

    @staticmethod
    def lcc_ratio(G: nx.Graph, num_nodes: int) -> float:
        if num_nodes <= 0 or G.number_of_nodes() == 0:
            return 0.0

        components = list(nx.connected_components(G))
        if not components:
            return 0.0

        lcc_size = len(max(components, key=len))
        return float(lcc_size) / float(num_nodes)

    @staticmethod
    def remove_path_edges(G: nx.Graph, path: List[int]) -> nx.Graph:
        """
        Faster remove: copy once + remove_edges_from once
        """
        G_removed = G.copy()
        edges = list(zip(path[:-1], path[1:]))

        # 规范成图中真实存在的边
        rm_edges = []
        for u, v in edges:
            if G_removed.has_edge(u, v):
                rm_edges.append((u, v))
            elif G_removed.has_edge(v, u):
                rm_edges.append((v, u))

        if rm_edges:
            G_removed.remove_edges_from(rm_edges)
        return G_removed

    def _build_bridge_cache_if_needed(self, G: nx.Graph) -> None:
        if self._bridge_edge_cache is not None:
            return
        bridges = set()
        for u, v in nx.bridges(G):
            bridges.add(self._canon_edge(u, v))
        self._bridge_edge_cache = bridges

    def _path_contains_bridge(self, path: List[int]) -> bool:
        if self._bridge_edge_cache is None:
            return False
        for u, v in self._path_edges(path):
            if self._canon_edge(u, v) in self._bridge_edge_cache:
                return True
        return False

    def compute_base_metrics(G: nx.Graph) -> Dict[str, float]:
        """
        保持原接口名不变，但改为近似版本的 base metrics
        """
        raise RuntimeError(
            "Please call instance method compute_base_metrics(...) instead of "
            "FragilityEvaluator.compute_base_metrics(G)."
        )

    def compute_base_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """
        Cache base metrics of the original graph.
        """
        num_nodes = G.number_of_nodes()

        if self.use_bridge_shortcut:
            self._build_bridge_cache_if_needed(G)

        return {
            "global_efficiency": self.global_efficiency_approx(G),
            "lcc_ratio": self.lcc_ratio(G, num_nodes),
            "avg_shortest_path_lcc": self.avg_shortest_path_of_lcc_approx(G),
        }

    def compute_fragility(
        self,
        G: nx.Graph,
        path: List[int],
        base_metrics: Dict[str, float],
        num_nodes: int,
    ) -> Dict[str, float]:
        """
        Compute fragility after removing path edges.

        Returns:
            {
                'delta_E': ...,
                'delta_LCC': ...,
                'delta_ASP': ...,
                'fragility_score': ...
            }
        """
        if path is None or len(path) < 2:
            return {
                "delta_E": 0.0,
                "delta_LCC": 0.0,
                "delta_ASP": 0.0,
                "fragility_score": 0.0,
            }

        path_key = tuple(path)
        if path_key in self._frag_cache:
            return self._frag_cache[path_key]

        # 可选快速捷径：
        # 若 path 中没有桥，很多时候对 LCC 的破坏趋近很小。
        # 注意：这是启发式近似，不是严格结论。
        if self.use_bridge_shortcut:
            self._build_bridge_cache_if_needed(G)
            if not self._path_contains_bridge(path):
                approx_out = {
                    "delta_E": 0.0,
                    "delta_LCC": 0.0,
                    "delta_ASP": 0.0,
                    "fragility_score": 0.0,
                }
                self._frag_cache[path_key] = approx_out
                return approx_out

        G_removed = self.remove_path_edges(G, path)

        new_E = self.global_efficiency_approx(G_removed)
        new_LCC = self.lcc_ratio(G_removed, num_nodes)
        new_ASP = self.avg_shortest_path_of_lcc_approx(G_removed)

        base_E = float(base_metrics.get("global_efficiency", 0.0))
        base_LCC = float(base_metrics.get("lcc_ratio", 0.0))
        base_ASP = float(base_metrics.get("avg_shortest_path_lcc", 0.0))

        delta_E = base_E - new_E
        delta_LCC = base_LCC - new_LCC
        delta_ASP = new_ASP - base_ASP

        # 防止采样近似带来的轻微负噪声
        delta_E = max(0.0, float(delta_E))
        delta_LCC = max(0.0, float(delta_LCC))
        delta_ASP = max(0.0, float(delta_ASP))

        fragility_score = (
            self.lambda_E * delta_E
            + self.lambda_LCC * delta_LCC
            + self.lambda_ASP * delta_ASP
        )

        out = {
            "delta_E": delta_E,
            "delta_LCC": delta_LCC,
            "delta_ASP": delta_ASP,
            "fragility_score": float(fragility_score),
        }

        self._frag_cache[path_key] = out
        return out