import torch
from torch import nn, Tensor


class NoiseAugmentationBase(nn.Module):
    """
    Base class for fully vectorized noise augmentations with per-sample probability.
    Subclasses must implement `_generate_noise`.

    Args:
        clip (bool): Whether to clip output to `value_range`.
        value_range (tuple[float, float]): Min and max allowed pixel values.
    """

    def __init__(self, clip: bool = True, value_range=(0.0, 1.0)):
        super().__init__()
        self.clip = clip
        self.value_range = value_range

    def forward(self, x: Tensor) -> Tensor:
        is_batch = x.ndim == 4
        if not is_batch:
            x = x.unsqueeze(0)

        min_val, max_val = self.value_range
        noise = self._generate_noise(x)
        x_out = x + noise

        if self.clip:
            x_out = torch.clamp(x_out, min_val, max_val)

        return x_out if is_batch else x_out.squeeze(0)

    def _generate_noise(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement _generate_noise().")


class ImpulseNoise(NoiseAugmentationBase):
    """
    Impulse (salt-and-pepper) noise augmentation.

    Randomly replaces a fraction of pixels with either minimum (pepper)
    or maximum (salt) pixel values, simulating impulse noise.

    The ratio of salt vs. pepper noise can be controlled via `salt_vs_pepper`.

    Noise is applied independently per sample with probability `p`.

    Args:
        ratio (float): Proportion of pixels to corrupt with salt or pepper noise.
        salt_vs_pepper (float): Proportion of salt noise relative to total noise.
            Must be between 0 and 1.
        clip (bool): Whether to clip output values to `value_range`.
        value_range (tuple[float, float]): Tuple specifying (min, max)
            allowed pixel values after adding noise.
    """

    def __init__(self, ratio: float = 0.05, salt_vs_pepper: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.salt_vs_pepper = salt_vs_pepper

    def _generate_noise(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        min_val, max_val = self.value_range

        rand_matrix = torch.rand(b, c, h, w, device=x.device)

        salt_mask = rand_matrix < (self.ratio * self.salt_vs_pepper)
        pepper_mask = (rand_matrix >= (self.ratio * self.salt_vs_pepper)) & (rand_matrix < self.ratio)

        noise = torch.zeros_like(x)
        noise = torch.where(salt_mask, torch.full_like(x, max_val), noise)
        noise = torch.where(pepper_mask, torch.full_like(x, min_val), noise)

        # Impulse noise replaces pixels (not additive), so return noise - x
        return noise - x


class PoissonNoise(NoiseAugmentationBase):
    """
    Poisson noise augmentation (signal-dependent noise).

    Note: Poisson noise requires input scaled such that pixel values correspond
    to counts (usually integers). Typically, scale input before applying.

    Args:
        scale (float): Factor to scale input before applying Poisson noise.
                       Controls noise intensity.
        clip (bool): Whether to clip output.
        value_range (tuple): Allowed pixel range.
    """

    def __init__(self, scale: float = 30.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def _generate_noise(self, x: Tensor) -> Tensor:
        # Scale input to counts
        scaled = x * self.scale

        # Apply Poisson noise
        noisy = torch.poisson(scaled)

        # Rescale back to original scale
        return (noisy / self.scale) - x


class SpeckleNoise(NoiseAugmentationBase):
    """
    Speckle noise augmentation (multiplicative noise).

    Adds noise proportional to pixel value: x + x * noise

    Args:
        std (float): Standard deviation of Gaussian noise multiplied by x.
        clip (bool): Whether to clip output.
        value_range (tuple): Allowed pixel range.
    """

    def __init__(self, std: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.std = std

    def _generate_noise(self, x: Tensor) -> Tensor:
        noise = torch.randn_like(x) * self.std
        return x * noise  # multiplicative noise
