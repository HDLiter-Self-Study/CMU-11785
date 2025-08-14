import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Sequence


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class single-label classification (softmax).
    Supports:
      - alpha: None | float | sequence[float] (per-class)
               If float is provided and num_classes==2 you may interpret it as positive-class weight manually.
      - gamma: focusing parameter
      - ignore_index: index in targets to ignore (useful for segmentation)
      - reduction: 'none' | 'mean' | 'sum'
    Notes:
      - inputs are raw logits of shape (N, C, ...) where ... can be spatial dims.
      - targets are integer class indices of shape (N, ...) matching input spatial dims.
    """

    def __init__(
        self,
        alpha: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none', 'mean' or 'sum'")
        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = ignore_index

        # store alpha raw; will convert to tensor on forward when we know device/C
        self.register_buffer("_alpha_tensor", None) if isinstance(alpha, torch.Tensor) else None
        self._alpha = alpha

    def _prepare_alpha(self, device: torch.device, num_classes: int, dtype: torch.dtype):
        """
        Return a tensor of shape (num_classes,) on the right device/dtype or None.
        """
        if self._alpha is None:
            return None
        if isinstance(self._alpha, torch.Tensor):
            a = self._alpha.to(device=device, dtype=dtype)
        elif isinstance(self._alpha, (list, tuple)):
            a = torch.tensor(self._alpha, device=device, dtype=dtype)
        else:  # scalar
            # interpret scalar as uniform multiplier for all classes (user caution)
            a = torch.full((num_classes,), float(self._alpha), device=device, dtype=dtype)
        if a.numel() != num_classes:
            # If user passed scalar but we prefer to warn:
            if a.numel() == 1:
                a = a.expand(num_classes)
            else:
                raise ValueError(f"alpha length ({a.numel()}) does not match num_classes ({num_classes})")
        return a

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: logits, shape (N, C, ...)
        targets: long, shape (N, ...) with values in [0, C-1]
        returns: scalar (if reduction != 'none') or tensor same shape as flattened targets (if 'none')
        """
        if inputs.dim() < 2:
            raise ValueError("Expected inputs with shape (N, C, ...)")
        num_classes = inputs.size(1)
        orig_shape = targets.shape

        # flatten spatial dims: move class to last and flatten batch+spatial
        # inputs: (N, C, H, W, ...) -> (N, C, -1) -> transpose -> (N * S, C)
        if inputs.dim() > 2:
            # (N, C, *spatial) -> (N, *spatial, C)
            dims = list(range(inputs.dim()))
            inputs_permuted = inputs.permute(0, *range(2, inputs.dim()), 1).contiguous()
            new_shape = (-1, num_classes)
            inputs_flat = inputs_permuted.view(new_shape)  # (N*S, C)
            targets_flat = targets.view(-1)
        else:
            inputs_flat = inputs
            targets_flat = targets

        device = inputs.device
        dtype = inputs.dtype

        # compute log_softmax and probs for numerical stability
        log_probs = F.log_softmax(inputs_flat, dim=1)  # shape (M, C)
        probs = torch.exp(log_probs)  # shape (M, C)

        # gather prob and log_prob for the true class
        targets_flat = targets_flat.long()
        if self.ignore_index is not None and self.ignore_index >= 0:
            valid_mask = targets_flat != self.ignore_index
        else:
            valid_mask = torch.ones_like(targets_flat, dtype=torch.bool)

        if valid_mask.any():
            # safe gather only for valid indices
            targets_valid = targets_flat[valid_mask]
            pt = probs[valid_mask, :].gather(1, targets_valid.unsqueeze(1)).squeeze(1)  # p_t
            log_pt = log_probs[valid_mask, :].gather(1, targets_valid.unsqueeze(1)).squeeze(1)  # log p_t
        else:
            # no valid elements
            if self.reduction == "none":
                return torch.zeros_like(targets_flat, dtype=dtype, device=device)
            return torch.tensor(0.0, dtype=dtype, device=device)

        # alpha per-class handling
        alpha_tensor = self._prepare_alpha(device=device, num_classes=num_classes, dtype=dtype)
        if alpha_tensor is not None:
            at = alpha_tensor[targets_valid]  # shape (M_valid,)
        else:
            at = torch.ones_like(pt, dtype=dtype, device=device)

        # focal loss for valid positions: - alpha * (1-pt)^gamma * log_pt
        loss = -at * ((1.0 - pt) ** self.gamma) * log_pt

        if self.reduction == "none":
            # need to map back to original flattened shape, putting zeros for ignored indices
            out = torch.zeros_like(targets_flat, dtype=dtype, device=device)
            out[valid_mask] = loss
            # reshape to original target shape
            return out.view(orig_shape)
        elif self.reduction == "sum":
            return loss.sum()
        else:  # mean
            # average only over valid entries (consistent with many PyTorch losses)
            return loss.mean()
