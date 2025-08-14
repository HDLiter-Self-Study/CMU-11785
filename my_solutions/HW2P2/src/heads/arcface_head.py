import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceHead(nn.Module):
    """
    An ArcFace head that produces logits with an additive angular margin.
    This module is designed to be a replacement for a standard nn.Linear classification
    head. It modifies the logits before the final loss calculation.

    Based on the paper: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    (https://arxiv.org/abs/1801.07698)

    This implementation is memory-efficient and numerically stable, combining best
    practices from multiple reviews.
    """

    def __init__(
        self, in_features: int, out_features: int, scale: float = 64.0, margin: float = 0.50, easy_margin: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        # More common implementation for mm from official repositories
        self.mm = self.sin_m * self.margin
        self.th = math.cos(math.pi - margin)

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Input feature embeddings of shape (batch_size, in_features).
            labels (torch.Tensor): Ground truth labels of shape (batch_size,).

        Returns:
            torch.Tensor: Logits of shape (batch_size, out_features), ready for loss calculation.
        """
        # Ensure labels are of the correct type
        if labels.dtype != torch.long:
            raise TypeError(f"labels must be torch.long but got {labels.dtype}")

        # Normalize input features and weight matrix
        inputs_norm = F.normalize(inputs)
        weight_norm = F.normalize(self.weight)
        cosine = F.linear(inputs_norm, weight_norm)

        # Get cosine of the angle between the feature and the ground truth class weight
        cos_theta_yi = cosine.gather(1, labels.unsqueeze(1)).squeeze(1)

        # Numerically stable sine computation using square()
        sin_theta_yi = torch.sqrt(torch.clamp(1.0 - cos_theta_yi.square(), min=1e-8))

        # Standard cos(theta + m) formula
        phi_yi = cos_theta_yi * self.cos_m - sin_theta_yi * self.sin_m

        # Apply margin conditions from the paper
        if self.easy_margin:
            phi_yi = torch.where(cos_theta_yi > 0, phi_yi, cos_theta_yi)
        else:
            phi_yi = torch.where(cos_theta_yi > self.th, phi_yi, cos_theta_yi - self.mm)

        # Use scatter to update logits efficiently without cloning
        output = cosine.scatter(1, labels.unsqueeze(1), phi_yi.unsqueeze(1))

        # Scale the final logits
        output *= self.scale

        # Return the modified logits, not the loss
        return output
