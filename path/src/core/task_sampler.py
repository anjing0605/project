def build_task_pairs(
    G: nx.Graph,
    key_nodes: List[int],
    community: np.ndarray,
    importance: np.ndarray,
    min_shortest_len: int = 2
) -> List[TaskPair]:
    """
    输入:
        G, key_nodes, community, importance
    输出:
        task_pairs: 所有合法任务对
    """
def sample_task_pairs(
    task_pairs: List[TaskPair],
    num_samples: int,
    mode: str = "mixed"
) -> List[TaskPair]:
    """
    输入:
        task_pairs: 候选任务池
        num_samples: 采样数
        mode: 'mixed' | 'high_importance' | 'cross_community' | 'long_distance'
    输出:
        sampled_tasks
    """