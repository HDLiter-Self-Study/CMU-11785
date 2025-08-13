#!/usr/bin/env python3
"""
FMix implementation in pure PyTorch.

This module provides a self-contained FMix implementation without external
dependencies (e.g., numpy, scipy). It follows the core idea of sampling a
low-frequency mask in the frequency domain and mixing images and labels using
the sampled mask.
"""

import math
import random
from typing import Tuple, Optional, Union

import torch
import torch.nn.functional as F


def fftfreqnd_torch(
    h: int, w: Optional[int] = None, z: Optional[int] = None, device: torch.device = None
) -> torch.Tensor:
    """
    Compute frequency bins (nd) in PyTorch.

    Args:
        h: Height dimension.
        w: Width dimension (optional).
        z: Depth dimension (optional).
        device: Torch device.

    Returns:
        Frequency distance tensor.
    """
    if device is None:
        device = torch.device("cpu")

    # Compute frequency bins
    fy = torch.fft.fftfreq(h, device=device)

    if w is not None:
        fy = fy.unsqueeze(-1)

        if w % 2 == 1:
            fx = torch.fft.fftfreq(w, device=device)[: w // 2 + 2]
        else:
            fx = torch.fft.fftfreq(w, device=device)[: w // 2 + 1]
    else:
        fx = torch.tensor(0.0, device=device)

    if z is not None:
        fy = fy.unsqueeze(-1)
        fz = torch.fft.fftfreq(z, device=device)[:, None]
    else:
        fz = torch.tensor(0.0, device=device)

    return torch.sqrt(fx**2 + fy**2 + fz**2)


def get_spectrum_torch(
    freqs: torch.Tensor, decay_power: float, ch: int, h: int, w: int = 0, z: int = 0
) -> torch.Tensor:
    """
    Generate spectrum in PyTorch.

    Args:
        freqs: Frequency bins tensor.
        decay_power: Decay power for 1 / f**decay_power scaling.
        ch: Number of channels.
        h, w, z: Spatial dimensions.

    Returns:
        Spectrum tensor.
    """
    device = freqs.device

    # Compute decay factor
    min_freq = 1.0 / max(w, h, z) if max(w, h, z) > 0 else 1.0
    scale = 1.0 / (torch.maximum(freqs, torch.tensor(min_freq, device=device)) ** decay_power)

    # Generate random spectrum parameters
    param_size = [ch] + list(freqs.shape) + [2]
    param = torch.randn(param_size, device=device)

    # Expand scale dimensions
    scale = scale.unsqueeze(-1).unsqueeze(0)

    return scale * param


def make_low_freq_image_torch(
    decay_power: float, shape: Tuple[int, ...], ch: int = 1, device: torch.device = None
) -> torch.Tensor:
    """
    Generate a low-frequency image (mask) in PyTorch.

    Args:
        decay_power: Decay power.
        shape: Mask shape.
        ch: Number of channels.
        device: Torch device.

    Returns:
        Low-frequency mask tensor scaled to [0, 1].
    """
    if device is None:
        device = torch.device("cpu")

    # Compute frequency bins
    freqs = fftfreqnd_torch(*shape, device=device)

    # Generate spectrum
    spectrum = get_spectrum_torch(freqs, decay_power, ch, *shape)

    # Convert to complex
    spectrum_complex = spectrum[:, 0] + 1j * spectrum[:, 1]

    # Inverse FFT
    if len(shape) == 1:
        mask = torch.fft.irfft(spectrum_complex, n=shape[0], dim=-1)
        mask = mask[:1, : shape[0]]
    elif len(shape) == 2:
        mask = torch.fft.irfft2(spectrum_complex, s=shape, dim=(-2, -1))
        mask = mask[:1, : shape[0], : shape[1]]
    elif len(shape) == 3:
        mask = torch.fft.irfftn(spectrum_complex, s=shape, dim=(-3, -2, -1))
        mask = mask[:1, : shape[0], : shape[1], : shape[2]]
    else:
        raise ValueError(f"Unsupported shape dimensions: {len(shape)}")

    # Normalize to [0, 1]
    mask = mask - mask.min()
    mask = mask / mask.max()

    return mask


def sample_lam_torch(alpha: float, reformulate: bool = False, device: torch.device = None) -> float:
    """
    Sample lambda from Beta distribution in PyTorch (no scipy dependency).

    Args:
        alpha: Beta distribution parameter.
        reformulate: Whether to use the reformulated variant (Beta(alpha+1, alpha)).
        device: Torch device.

    Returns:
        Sampled lambda scalar.
    """
    if device is None:
        device = torch.device("cpu")

    if reformulate:
        # Beta(alpha+1, alpha)
        beta_dist = torch.distributions.Beta(alpha + 1, alpha)
    else:
        # Beta(alpha, alpha)
        beta_dist = torch.distributions.Beta(alpha, alpha)

    return beta_dist.sample().item()


def binarise_mask_torch(
    mask: torch.Tensor, lam: float, in_shape: Tuple[int, ...], max_soft: float = 0.0
) -> torch.Tensor:
    """
    Binarize a mask according to target lambda with optional soft edges.

    Args:
        mask: Input mask.
        lam: Target lambda value in [0, 1].
        in_shape: Input spatial shape.
        max_soft: Softening ratio controlling the transition band.

    Returns:
        Binarized (or softly binarized) mask.
    """
    device = mask.device

    # Flatten mask
    mask_flat = mask.reshape(-1)

    # Sort to get indices (descending)
    _, idx = torch.sort(mask_flat, descending=True)

    # Compute number of pixels to set to 1
    total_pixels = mask_flat.numel()
    num = math.ceil(lam * total_pixels) if random.random() > 0.5 else math.floor(lam * total_pixels)

    # Compute soft band size
    eff_soft = max_soft
    if max_soft > lam or max_soft > (1 - lam):
        eff_soft = min(lam, 1 - lam)

    soft_pixels = int(total_pixels * eff_soft)
    num_low = max(0, num - soft_pixels)
    num_high = min(total_pixels, num + soft_pixels)

    # Create new mask
    new_mask = torch.zeros_like(mask_flat)

    # Set high-value region to 1
    if num_high > 0:
        new_mask[idx[:num_high]] = 1.0

    # Set low-value region to 0 (already zero by init)
    if num_low < total_pixels:
        new_mask[idx[num_low:]] = 0.0

    # Set soft transition region
    if num_low < num_high:
        transition_indices = idx[num_low:num_high]
        if len(transition_indices) > 0:
            transition_values = torch.linspace(1.0, 0.0, len(transition_indices), device=device)
            new_mask[transition_indices] = transition_values

    # Reshape back to original shape
    new_mask = new_mask.reshape(1, *in_shape)

    return new_mask


def sample_mask_torch(
    alpha: float,
    decay_power: float,
    shape: Union[int, Tuple[int, ...]],
    max_soft: float = 0.0,
    reformulate: bool = False,
    device: torch.device = None,
) -> Tuple[float, torch.Tensor]:
    """
    Sample an FMix mask and its corresponding lambda.

    Args:
        alpha: Beta distribution parameter.
        decay_power: Decay power for low-frequency sampling.
        shape: Mask shape (int or tuple).
        max_soft: Softening ratio.
        reformulate: Whether to use the reformulated Beta.
        device: Torch device.

    Returns:
        (lambda, mask) tuple.
    """
    if device is None:
        device = torch.device("cpu")

    if isinstance(shape, int):
        shape = (shape,)

    # Sample lambda
    lam = sample_lam_torch(alpha, reformulate, device)

    # Generate low-frequency image
    mask = make_low_freq_image_torch(decay_power, shape, ch=1, device=device)

    # Binarize mask
    mask = binarise_mask_torch(mask, lam, shape, max_soft)

    return lam, mask


class FMix:
    """
    FMix augmentation implemented in pure PyTorch.

    Args:
        decay_power: Frequency decay power in the 1 / f**decay_power scaling.
        alpha: Beta distribution alpha parameter controlling expected lambda.
        size: Required mask shape (up to 3 spatial dimensions).
        max_soft: Softening ratio in [0, 0.5] to smooth hard mask edges.
        reformulate: If True, use reformulated Beta(alpha+1, alpha).
        num_classes: Number of classes. Optional if labels are already one-hot/soft.
        spatial_transform: Whether to apply simple spatial transforms on the mask.
    """

    def __init__(
        self,
        decay_power: float = 3.0,
        alpha: float = 1.0,
        size: Tuple[int, int] = (32, 32),
        max_soft: float = 0.0,
        reformulate: bool = False,
        num_classes: Optional[int] = None,
        spatial_transform: bool = True,
    ):
        self.decay_power = decay_power
        self.alpha = alpha
        self.size = size
        self.max_soft = max_soft
        self.reformulate = reformulate
        self.num_classes = num_classes
        self.spatial_transform = spatial_transform

        # Debug attributes
        self._last_original_lambda = None
        self._last_effective_lambda = None

    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply FMix augmentation.

        Args:
            images: Input image tensor [B, C, H, W].
            targets: Target labels as indices [B] or one-hot/soft labels [B, num_classes].

        Returns:
            (mixed_images, mixed_targets) tuple.
        """
        batch_size = images.size(0)
        device = images.device

        # Generate mask using PyTorch FMix algorithm
        lam, mask = sample_mask_torch(self.alpha, self.decay_power, self.size, self.max_soft, self.reformulate, device)

        # Expand mask to batch dimension
        if mask.dim() == 3:  # [1, H, W]
            mask = mask.expand(batch_size, -1, -1).unsqueeze(1)  # [B, 1, H, W]

        # Resize mask to image size
        if mask.shape[-2:] != images.shape[-2:]:
            mask = F.interpolate(mask, size=images.shape[-2:], mode="bilinear", align_corners=False)

        # Expand mask to all channels
        mask = mask.expand(-1, images.size(1), -1, -1)  # [B, C, H, W]

        # Generate random permutation indices
        index = torch.randperm(batch_size, device=device)

        # Mix images
        mixed_images = images * mask + images[index] * (1 - mask)

        # Compute effective lambda (actual mixing ratio)
        effective_lam = mask.reshape(batch_size, -1).mean(dim=1)

        # Mix labels
        mixed_targets = self._mix_labels(targets, targets[index], effective_lam, device)

        # Save lambda values for debugging
        self._last_original_lambda = torch.tensor([lam] * batch_size, device=device)
        self._last_effective_lambda = effective_lam

        return mixed_images, mixed_targets

    def _mix_labels(
        self, targets1: torch.Tensor, targets2: torch.Tensor, lam_batch: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Mix labels according to the per-sample lambda.

        Args:
            targets1: First batch of labels (indices or one-hot/soft).
            targets2: Second batch of labels (indices or one-hot/soft).
            lam_batch: Per-sample lambda tensor [B].
            device: Torch device.

        Returns:
            Mixed labels (one-hot/soft) tensor.
        """
        if targets1.dim() == 1:  # class indices
            if self.num_classes is None:
                raise ValueError(
                    "FMix requires 'num_classes' when targets are class indices. "
                    "Either provide num_classes or pass one-hot/soft labels."
                )
            targets1_onehot = F.one_hot(targets1, num_classes=self.num_classes).float()
            targets2_onehot = F.one_hot(targets2, num_classes=self.num_classes).float()
        else:  # already one-hot/soft
            targets1_onehot = targets1.float()
            targets2_onehot = targets2.float()

        # Mix labels
        lam_expanded = lam_batch.reshape(-1, 1)
        mixed_targets = lam_expanded * targets1_onehot + (1 - lam_expanded) * targets2_onehot

        return mixed_targets

    def get_last_lambdas(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get last sampled lambdas for debugging purposes.

        Returns:
            (original_lambda, effective_lambda)
        """
        return self._last_original_lambda, self._last_effective_lambda

    def apply_spatial_transforms(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply simple spatial transforms to the mask if enabled.

        Args:
            mask: Input mask.

        Returns:
            Transformed mask.
        """
        if not self.spatial_transform:
            return mask

        # Simple spatial transform examples
        if random.random() > 0.5:
            # Random horizontal flip
            mask = torch.flip(mask, dims=[-1])

        if random.random() > 0.5:
            # Random vertical flip
            mask = torch.flip(mask, dims=[-2])

        return mask
