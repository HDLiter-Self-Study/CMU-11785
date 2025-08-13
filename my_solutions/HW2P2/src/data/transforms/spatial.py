"""
Vectorized and batch-safe spatial transforms.
"""

import torch
from torch import nn, Tensor


class GridMask(nn.Module):
    """
    Applies a grid mask to a tensor in a fully vectorized manner.

    This transform is batch-safe and applies a randomly configured grid mask
    to each sample in a batch independently.

    Args:
        d_ratio_range (tuple[float, float]): Range for the grid size ratio (d / min(h, w)).
        ratio (float): The ratio of the masked (black) width to the grid period (d).
                       Note: This leads to masked area ≈ ratio**2. To match original GridMask paper
                       (where optimal masked width/d ≈ 0.4 for ImageNet), consider ratio=0.4.
                       Current default of 0.6 creates more aggressive masking (~36% area masked).
    """

    def __init__(self, d_ratio_range: tuple[float, float] = (0.1, 0.4), ratio: float = 0.6):
        super().__init__()
        self.d_ratio_range = d_ratio_range
        self.ratio = ratio

    def forward(self, x: Tensor) -> Tensor:
        is_batch = x.ndim == 4
        if not is_batch:
            x = x.unsqueeze(0)

        b, c, h, w = x.shape

        # Generate random grid parameters for each image in the batch
        d_ratios = (
            torch.rand(b, device=x.device) * (self.d_ratio_range[1] - self.d_ratio_range[0]) + self.d_ratio_range[0]
        )
        min_side = min(h, w)
        ds = (d_ratios * min_side).clamp(min=1).long()  # Ensure d >= 1 to avoid division by zero

        # Random offsets: uniform [0, d) for each batch item
        dx = (torch.rand(b, device=x.device) * ds.float()).long()
        dy = (torch.rand(b, device=x.device) * ds.float()).long()

        # Create coordinate grids [H, W]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij"
        )

        # Expand to batch: [B, H, W]
        grid_x_b = grid_x.unsqueeze(0).expand(b, -1, -1)
        grid_y_b = grid_y.unsqueeze(0).expand(b, -1, -1)
        dx_b = dx.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        dy_b = dy.unsqueeze(-1).unsqueeze(-1)
        ds_b = ds.unsqueeze(-1).unsqueeze(-1)

        # Compute mask conditions using fully vectorized operations
        mod_x = (grid_x_b + dx_b) % ds_b
        mod_y = (grid_y_b + dy_b) % ds_b
        cond_x = mod_x < (ds_b * self.ratio).floor()  # Use floor for precise int comparison
        cond_y = mod_y < (ds_b * self.ratio).floor()
        cond = cond_x & cond_y

        # Mask: 1 to keep, 0 to mask out
        grid_mask = (~cond).float()  # [B, H, W]

        # Apply to all channels: [B, C, H, W]
        mask = grid_mask.unsqueeze(1)  # [B, 1, H, W]
        x_out = x * mask

        return x_out if is_batch else x_out.squeeze(0)
