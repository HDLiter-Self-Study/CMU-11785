import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.reducers import BaseReducer
from pytorch_metric_learning.samplers import MPerClassSampler
from typing import Optional

from src.losses.utils import _get_pml_distance


class PosNegWeightedReducer(BaseReducer):
    """Custom reducer to support weighted averaging of positive and negative losses."""

    def __init__(self, pos_weight: float = 1.0, neg_weight: float = 1.0, avg_by: str = "posneg"):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.avg_by = avg_by

    def forward(self, loss: torch.Tensor, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute mean of positive and negative losses separately, then average."""
        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_loss = loss[pos_mask] * self.pos_weight
        neg_loss = loss[neg_mask] * self.neg_weight
        if self.avg_by == "posneg":
            # Compute mean of positive and negative losses separately, then average
            pos_avg = pos_loss.mean() if pos_loss.numel() > 0 else torch.tensor(0.0, device=loss.device)
            neg_avg = neg_loss.mean() if neg_loss.numel() > 0 else torch.tensor(0.0, device=loss.device)
            return (pos_avg + neg_avg) * 0.5
        elif self.avg_by == "global":
            # Compute mean of all losses together, weighted by pos_weight and neg_weight
            return (pos_loss.sum() + neg_loss.sum()) / (loss.numel() or 1) * 0.5
        else:
            raise ValueError(f"Unknown avg_by: {self.avg_by}")


class ContrastiveLoss(nn.Module):
    """
    A PML-based ContrastiveLoss with support for posneg averaging and miner encapsulation.
    No soft labels support, as requested.
    """

    def __init__(
        self,
        pos_margin: float = 0,
        neg_margin: float = 1,
        miner_type: str = "pair_margin",
        distance_metric: str = "euclidean",
        normalize_embeddings: bool = False,
        squared_distance: bool = False,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
        avg_by: str = "posneg",
        miner_margin_factor: float = 1.0,
        sampler_m: int = 4,
        eps: float = 0.1,  # For MultiSimilarityMiner
    ):
        super().__init__()
        if avg_by not in ("global", "posneg"):
            raise ValueError("avg_by must be 'global' or 'posneg'")

        self.pos_margin = float(pos_margin)
        self.neg_margin = float(neg_margin)
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        self.avg_by = avg_by
        self.sampler_m = sampler_m

        distance = _get_pml_distance(distance_metric, squared_distance, normalize_embeddings)
        reducer = PosNegWeightedReducer(pos_weight, neg_weight, avg_by)
        self.loss = losses.ContrastiveLoss(
            neg_margin=neg_margin, pos_margin=pos_margin, distance=distance, reducer=reducer
        )

        pos_miner_margin = pos_margin * miner_margin_factor
        neg_miner_margin = neg_margin * miner_margin_factor

        if miner_type == "pair_margin":
            self.miner = miners.PairMarginMiner(
                distance=distance, neg_margin=neg_miner_margin, pos_margin=pos_miner_margin
            )
        elif miner_type == "batch_hard":
            self.miner = miners.BatchHardMiner(distance=distance)  # No margin needed
        elif miner_type == "multi_similarity":
            self.miner = miners.MultiSimilarityMiner(epsilon=eps, distance=distance)
        else:
            raise ValueError(f"Unknown miner type: {miner_type}")

    def get_sampler(self, labels: torch.Tensor) -> MPerClassSampler:
        # Appoint the sampler for the loss so that
        # We can get different sampler for different losses in pipeline
        return MPerClassSampler(labels, self.sampler_m)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss with optional miner."""
        if not ((labels == 0) | (labels == 1)).all():
            raise ValueError("labels must be binary (0 or 1)")

        indices_tuple = self.miner(embeddings, labels) if self.miner else None
        loss = self.loss(embeddings, labels, indices_tuple)

        return loss  # Default reducer: mean (handled by reducer or PML)
