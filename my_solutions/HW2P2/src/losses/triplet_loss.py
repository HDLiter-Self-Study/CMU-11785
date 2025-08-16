import torch

from pytorch_metric_learning import losses, distances, miners
from torch import nn


class TripletMarginWithDistanceLoss(nn.Module):
    def __init__(
        self,
        margin,
        distance_metric: str = "cosine",
        normalize_embeddings: bool = True,
        squared_distance: bool = False,
        type_of_triplets: str = "semi-hard",
        miner_type: str = "triplet",
        miner_margin_factor: float = 1.0,
    ):
        super().__init__()
        if distance_metric == "euclidean":
            power = 2 if squared_distance else 1
            distance = distances.LpDistance(p=2, power=power, normalize_embeddings=normalize_embeddings)
        elif distance_metric == "manhattan":
            distance = distances.LpDistance(p=1, power=1, normalize_embeddings=normalize_embeddings)
        elif distance_metric == "cosine":
            distance = distances.CosineSimilarity(normalize_embeddings=normalize_embeddings)
        elif distance_metric == "dot":
            distance = distances.DotProductSimilarity(normalize_embeddings=normalize_embeddings)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

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

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        indices_tuple = self.miner(embeddings, labels)
        return self.loss(embeddings, labels, indices_tuple)
