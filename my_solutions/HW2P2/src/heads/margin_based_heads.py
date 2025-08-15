import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class AbstractMarginHead(nn.Module, ABC):
    """
    Abstract base class for margin-based classifier heads like ArcFace, CosFace, SphereFace,
    implemented using the Template Method design pattern.

    This base class handles all common operations, and subclasses are only required to
    implement the `_calculate_phi` method, which defines the specific margin logic.

    Common Operations Handled:
    - Storing and initializing the weight matrix.
    - Normalizing input features and the weight matrix.
    - Calculating the initial cosine similarity (logits).
    - Validating input labels.
    - Applying the margin-based modification to the correct logits.
    - Scaling the final logits.
    """

    def __init__(self, in_features: int, out_features: int, scale: float = 64.0):
        super().__init__()
        if out_features <= 0:
            raise ValueError("out_features must be a positive integer.")
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    @abstractmethod
    def _calculate_phi(self, cos_theta_yi: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Subclasses must implement this method to calculate the modified logit `phi(theta_yi)`.
        This is the core of the specific margin logic (e.g., ArcFace's cos(theta+m)).

        Args:
            cos_theta_yi (torch.Tensor): The cosine values for the ground truth classes.
            labels (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The modified logits for the ground truth classes.
        """
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """The template forward pass for all margin heads."""
        # --- 1. Input Validation ---
        if labels.dtype != torch.long:
            raise TypeError(f"labels must be torch.long but got {labels.dtype}")
        if labels.dim() != 1:
            raise ValueError(f"Expected labels to have shape (batch_size,), but got {labels.shape}")
        if labels.max() >= self.out_features or labels.min() < 0:
            raise ValueError(
                f"Label values must be in the range [0, {self.out_features-1}]. "
                f"Found min: {labels.min()}, max: {labels.max()}."
            )

        # --- 2. Normalize inputs and weights & compute cosine ---
        inputs_norm = F.normalize(inputs)
        weight_norm = F.normalize(self.weight)
        cosine = F.linear(inputs_norm, weight_norm)

        # --- 3. Gather the ground truth logits ---
        labels_view = labels.view(-1, 1)
        cos_theta_yi = cosine.gather(1, labels_view).squeeze(1)

        # --- 4. Calculate the modified logit using the subclass implementation ---
        phi_yi = self._calculate_phi(cos_theta_yi, labels)

        # --- 5. Update the final logit matrix ---
        output = cosine.scatter(1, labels_view, phi_yi.unsqueeze(1))

        # --- 6. Scale the output ---
        output *= self.scale
        return output


class ArcFaceHead(AbstractMarginHead):
    """
    ArcFace head based on the paper: "ArcFace: Additive Angular Margin Loss for
    Deep Face Recognition" (https://arxiv.org/abs/1801.07698).
    """

    def __init__(
        self, in_features: int, out_features: int, scale: float = 64.0, margin: float = 0.50, easy_margin: bool = False
    ):
        super().__init__(in_features, out_features, scale)
        self.margin = margin
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = self.sin_m * margin  # Corrected from self.margin

    def _calculate_phi(self, cos_theta_yi: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sin_theta_yi = torch.sqrt(torch.clamp(1.0 - cos_theta_yi.square(), min=1e-8))
        phi_yi = cos_theta_yi * self.cos_m - sin_theta_yi * self.sin_m

        if self.easy_margin:
            return torch.where(cos_theta_yi > 0, phi_yi, cos_theta_yi)
        else:
            return torch.where(cos_theta_yi > self.th, phi_yi, cos_theta_yi - self.mm)


class CosFaceHead(AbstractMarginHead):
    """
    CosFace head based on the paper: "CosFace: Large Margin Cosine Loss for
    Deep Face Recognition" (https://arxiv.org/abs/1801.09414).
    """

    def __init__(self, in_features: int, out_features: int, scale: float = 64.0, margin: float = 0.35):
        super().__init__(in_features, out_features, scale)
        self.margin = margin

    def _calculate_phi(self, cos_theta_yi: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return cos_theta_yi - self.margin


class SphereFaceHead(AbstractMarginHead):
    """
    SphereFace head based on the paper: "SphereFace: Deep Hypersphere Embedding
    for Face Recognition" (https://arxiv.org/abs/1704.08063).
    """

    def __init__(self, in_features: int, out_features: int, scale: float = 64.0, margin: int = 4):
        super().__init__(in_features, out_features, scale)
        if not isinstance(margin, int) or margin < 1:
            raise ValueError("SphereFace margin must be a positive integer.")
        self.margin = margin
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2 * x**2 - 1,
            lambda x: 4 * x**3 - 3 * x,
            lambda x: 8 * x**4 - 8 * x**2 + 1,
            lambda x: 16 * x**5 - 20 * x**3 + 5 * x,
        ]
        if self.margin >= len(self.mlambda):
            raise NotImplementedError(f"SphereFace with margin > {len(self.mlambda)-1} is not supported.")

    def _calculate_phi(self, cos_theta_yi: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos_theta_yi_clipped = torch.clamp(cos_theta_yi, -1.0, 1.0)

        theta_yi = torch.acos(cos_theta_yi_clipped)
        k = (self.margin * theta_yi / math.pi).floor().detach()

        phi_yi = ((-1) ** k) * self.mlambda[self.margin](cos_theta_yi) - 2 * k

        return torch.where(cos_theta_yi > phi_yi, phi_yi, cos_theta_yi)
