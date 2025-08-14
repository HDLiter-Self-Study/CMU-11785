from typing import Callable
import torch
import torch.nn.functional as F
import torch.nn as nn


def _get_distance_metric(
    metric_name: str, squared: bool = False, normalize: bool = False, eps: float = 1e-8
) -> Callable:
    """
    Returns a callable distance function `distance(anchor, other) -> Tensor`.

    Args:
        metric_name (str): One of "euclidean", "manhattan", "cosine", "dot".
        squared (bool): If True and metric_name is "euclidean", returns squared L2 distance.
        normalize (bool): If True, L2-normalizes inputs before computing distance.
        eps (float): Epsilon for numerical stability in pairwise_distance.
    """

    def maybe_normalize(x):
        return F.normalize(x, p=2, dim=1) if normalize else x

    if metric_name == "euclidean":
        if squared:
            return lambda x, y: ((maybe_normalize(x) - maybe_normalize(y)) ** 2).sum(dim=1)
        else:
            return lambda x, y: F.pairwise_distance(maybe_normalize(x), maybe_normalize(y), p=2, eps=eps)
    elif metric_name == "manhattan":
        return lambda x, y: F.pairwise_distance(maybe_normalize(x), maybe_normalize(y), p=1, eps=eps)
    elif metric_name == "cosine":
        return lambda x, y: (1.0 - F.cosine_similarity(maybe_normalize(x), maybe_normalize(y), dim=1))
    elif metric_name == "dot":
        # Negative dot product as distance (larger dot => smaller distance)
        return lambda x, y: -(maybe_normalize(x) * maybe_normalize(y)).sum(dim=1)
    else:
        raise ValueError(f"Unknown distance metric: {metric_name}")


class TripletMarginWithDistanceLoss(nn.TripletMarginWithDistanceLoss):
    """
    A wrapper for PyTorch's TripletMarginWithDistanceLoss that enhances it with:
    - String-based distance metric selection.
    - Optional L2 normalization of embeddings.
    - Optional use of squared Euclidean distance.
    """

    def __init__(
        self,
        distance_metric: str = "euclidean",
        normalize_embeddings: bool = False,
        squared_distance: bool = False,
        **kwargs,
    ):
        distance_function = _get_distance_metric(
            metric_name=distance_metric,
            normalize=normalize_embeddings,
            squared=squared_distance,
        )
        super().__init__(distance_function=distance_function, **kwargs)
