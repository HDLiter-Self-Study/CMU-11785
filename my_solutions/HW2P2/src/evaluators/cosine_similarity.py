import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CosineSimilarity(nn.Module):
    """
    A module that computes cosine similarity between embeddings, with optional
    L2 normalization and temperature scaling, aligned with YAML configuration.
    """

    def __init__(self, use_l2_norm: bool = True, temperature: float = 1.0, dim: int = 1, eps: float = 1e-8):
        """
        Args:
            use_l2_norm (bool): If True, L2-normalizes the input embeddings
                before computing similarity. Defaults to True.
            temperature (float): The temperature for scaling the similarity scores.
                Must be a positive value. Defaults to 1.0.
            dim (int): Dimension along which to compute similarity. Defaults to 1.
            eps (float): Small value to avoid division by zero. Defaults to 1e-8.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be a positive value.")
        self.use_l2_norm = use_l2_norm
        self.temperature = temperature
        self.dim = dim
        self.eps = eps

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Computes cosine similarity between two tensors.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The scaled cosine similarity.
        """
        if self.use_l2_norm:
            x1 = F.normalize(x1, p=2, dim=self.dim, eps=self.eps)
            x2 = F.normalize(x2, p=2, dim=self.dim, eps=self.eps)

        # Standard cosine similarity calculation
        similarity = F.cosine_similarity(x1, x2, self.dim, self.eps)

        # Apply temperature scaling
        if self.temperature != 1.0:
            similarity = similarity / self.temperature

        return similarity
