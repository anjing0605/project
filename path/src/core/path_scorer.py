def normalize_feature_dicts(records: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    输入:
        多条路径的特征字典
    输出:
        归一化后的特征字典列表
    """
def score_path(features: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    输入:
        features: 路径特征+fragility
        weights: 如 alpha,beta,gamma,delta,eta
    输出:
        score
    """
def rank_paths(
    path_records: List[PathRecord],
    weights: Dict[str, float]
) -> List[PathRecord]:
    """
    输入:
        path_records
    输出:
        按分数降序排序后的路径记录
    """