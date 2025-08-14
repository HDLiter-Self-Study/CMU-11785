import torch
import torch.nn as nn
from typing import Optional


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss with enhancements for robustness, efficiency, and soft labels.
    """

    def __init__(
        self,
        margin: float = 1.0,
        squared: bool = True,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
        reduction: str = "mean",
        avg_by: str = "posneg",
        soft_labels: bool = False,
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean'|'sum'|'none'")
        if avg_by not in ("global", "posneg"):
            raise ValueError("avg_by must be 'global' or 'posneg'")
        if soft_labels and avg_by == "posneg":
            raise ValueError("soft_labels=True is incompatible with avg_by='posneg'. Use avg_by='global'.")

        self.margin = float(margin)
        self.squared = bool(squared)
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        self.reduction = reduction
        self.avg_by = avg_by
        self.soft_labels = bool(soft_labels)

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if embedding1.shape != embedding2.shape:
            raise ValueError("embedding1 and embedding2 must have the same shape")

        # Directly compute squared euclidean distance to avoid sqrt
        diff = embedding1 - embedding2
        sq_dist = torch.sum(diff * diff, dim=1)

        if self.squared:
            # Hinge loss on squared distance
            pos_loss = sq_dist
            margin_sq = self.margin * self.margin
            neg_loss = torch.clamp(margin_sq - sq_dist, min=0.0)
        else:
            # For non-squared, we still use sq_dist for the positive loss part
            # as it's often defined as 0.5 * d^2
            dist = torch.sqrt(sq_dist + 1e-8)
            pos_loss = sq_dist
            neg_term = torch.clamp(self.margin - dist, min=0.0)
            neg_loss = neg_term * neg_term

        labels_float = label.view(-1).to(dtype=embedding1.dtype, device=embedding1.device)

        if not self.soft_labels:
            if not ((labels_float == 0) | (labels_float == 1)).all():
                raise ValueError("labels must be binary (0 or 1) unless soft_labels=True")

        loss_pos = labels_float * self.pos_weight * 0.5 * pos_loss
        loss_neg = (1.0 - labels_float) * self.neg_weight * 0.5 * neg_loss
        loss_all = loss_pos + loss_neg

        if self.reduction == "sum":
            return loss_all.sum()
        if self.reduction == "none":
            return loss_all

        # Mean reduction logic
        if self.avg_by == "global" or self.soft_labels:
            return loss_all.mean()
        else:  # avg_by == 'posneg' and not soft_labels
            pos_mask = labels_float == 1
            neg_mask = labels_float == 0

            # Use item() to get scalar count, prevent holding tensor in memory
            pos_count = pos_mask.sum().item()
            neg_count = neg_mask.sum().item()

            pos_sum = loss_all[pos_mask].sum()
            neg_sum = loss_all[neg_mask].sum()

            # Handle cases where one class is not present in the batch
            pos_avg = pos_sum / pos_count if pos_count > 0 else torch.tensor(0.0, device=loss_all.device)
            neg_avg = neg_sum / neg_count if neg_count > 0 else torch.tensor(0.0, device=loss_all.device)

            return 0.5 * (pos_avg + neg_avg)
