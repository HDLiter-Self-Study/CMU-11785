import torch
import torch.nn as nn
from src.losses.utils import _get_pml_distance


class CenterLoss(nn.Module):
    """
    Center Loss: penalizes the distance between deep features and their corresponding class centers.

    Args:
        num_classes (int): number of classes
        feat_dim (int): feature dimension
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        distance_metric: str = "euclidean",
        normalize_embeddings: bool = False,
        squared_distance: bool = False,
        distance_scale: int = 1,
    ):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.distance = _get_pml_distance(distance_metric, squared_distance, normalize_embeddings)
        self.distance_scale = distance_scale  # Scale distance for small distance like cosine distance
        # Learnable class centers, automatically registered to parameters()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, x: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, feat_dim) feature embeddings
            labels: (batch_size,) ground truth class labels
        Returns:
            loss: scalar tensor
        """
        if x.shape[1] != self.feat_dim:
            raise ValueError(f"Expected feature dim {self.feat_dim}, got {x.shape[1]}")
        if labels.max() >= self.num_classes or labels.min() < 0:
            raise ValueError(f"Labels must be in [0, {self.num_classes-1}]")
        # Select class centers according to labels (keep gradient flow)
        batch_centers = self.centers[labels]

        # Calculate distance (use inverted distance if needed, e.g. cosine distance)
        diff = -self.distance(x, batch_centers) if self.distance.is_inverted else self.distance(x, batch_centers)
        loss = 0.5 * (diff.sum(dim=1)).mean() * self.distance_scale

        return loss
