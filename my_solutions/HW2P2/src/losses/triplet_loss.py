import torch

from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.samplers import MPerClassSampler
from torch import nn

from src.losses.utils import _get_pml_distance


class TripletLoss(nn.Module):
    def __init__(
        self,
        margin,
        miner_type: str = "triplet",
        distance_metric: str = "cosine",
        normalize_embeddings: bool = True,
        squared_distance: bool = False,
        type_of_triplets: str = "semi-hard",
        miner_margin_factor: float = 1.0,
        sampler_m: int = 4,
    ):
        super().__init__()

        distance = _get_pml_distance(distance_metric, squared_distance, normalize_embeddings)
        self.sampler_m = sampler_m
        self.loss = losses.TripletMarginLoss(margin=margin, distance=distance)
        miner_margin = margin * miner_margin_factor
        if miner_type == "triplet":
            self.miner = miners.TripletMarginMiner(
                type_of_triplets=type_of_triplets, distance=distance, margin=miner_margin
            )
        elif miner_type == "batch_hard":
            self.miner = miners.BatchHardMiner(distance=distance, margin=miner_margin)
        else:
            raise ValueError(f"Unknown miner type: {miner_type}")

    def get_sampler(self, labels: torch.Tensor) -> MPerClassSampler:
        # Appoint the sampler for the loss so that
        # We can get different sampler for different losses in pipeline
        return MPerClassSampler(labels, self.sampler_m)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        indices_tuple = self.miner(embeddings, labels)
        return self.loss(embeddings, labels, indices_tuple)
