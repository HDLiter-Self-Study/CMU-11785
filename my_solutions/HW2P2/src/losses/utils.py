from pytorch_metric_learning import distances


def _get_pml_distance(metric_name: str, squared: bool = False, normalize: bool = False) -> distances.BaseDistance:
    if metric_name == "euclidean":
        power = 2 if squared else 1
        return distances.LpDistance(p=2, power=power, normalize_embeddings=normalize)
    elif metric_name == "manhattan":
        return distances.LpDistance(p=1, power=1, normalize_embeddings=normalize)
    elif metric_name == "cosine":
        return distances.CosineSimilarity(normalize_embeddings=normalize)
    elif metric_name == "dot":
        return distances.DotProductSimilarity(normalize_embeddings=normalize)
    else:
        raise ValueError(f"Unknown distance metric: {metric_name}")
