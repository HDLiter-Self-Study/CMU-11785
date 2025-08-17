"""
Sampling utilities for dataset-level augmentation strategies.

This module implements a simple repeated-augmentation dataset wrapper that
revisits each underlying sample multiple times per epoch. Each visit goes
through the same transform pipeline but will produce different views if the
pipeline contains randomness.
"""

from typing import Any, Tuple

from torch.utils.data import Dataset


class RepeatedAugmentationWrapper(Dataset):
    """Wrap a dataset to repeat each sample multiple times per epoch.

    The wrapper expands dataset length to ``len(base_dataset) * num_repeats`` and
    maps the i-th wrapped index to the base index ``i // num_repeats``. Randomness
    should be handled by the underlying transform pipeline to create distinct
    views for repeated visits.

    Args:
        base_dataset: The original dataset to be wrapped.
        num_repeats: Number of times each sample should be revisited per epoch.
        distinct: Whether repeated visits should be considered distinct. This
            flag is informational for future extension; randomness is expected
            to be provided by transforms.

    Raises:
        ValueError: If ``num_repeats`` is less than 1.
    """

    def __init__(self, base_dataset: Dataset, num_repeats: int = 2, distinct: bool = True) -> None:
        if num_repeats < 1:
            raise ValueError("'num_repeats' must be >= 1 for RepeatedAugmentation")
        self.base_dataset = base_dataset
        self.num_repeats = int(num_repeats)
        # Note: 'distinct' is currently informational. Transform pipelines should
        # provide randomness so repeated visits produce different views. This flag
        # is reserved for potential future behavior changes (e.g., enforcing
        # deterministic diversity constraints).
        self.distinct = bool(distinct)

    def __len__(self) -> int:  # noqa: D401
        """Total number of wrapped samples in one epoch."""
        return len(self.base_dataset) * self.num_repeats

    def __getitem__(self, index: int) -> Any:  # noqa: D401
        """Get item by mapping wrapped index to base dataset index."""
        base_index = index // self.num_repeats
        return self.base_dataset[base_index]


class RepeatedAugmentation:
    """Wrapper for repeated augmentation, use it as a transform for datasets.

    Args:
        num_repeats: Number of times each sample should be revisited per epoch.
        distinct: Whether repeated visits should be considered distinct. This
            flag is informational for future extension; randomness is expected
            to be provided by transforms.
    """

    def __init__(self, num_repeats: int = 2, distinct: bool = True) -> None:
        if num_repeats < 1:
            raise ValueError("'num_repeats' must be >= 1 for RepeatedAugmentation")
        self.num_repeats = int(num_repeats)
        self.distinct = bool(distinct)

    def __call__(self, dataset: Dataset) -> Dataset:
        return RepeatedAugmentationWrapper(dataset, self.num_repeats, self.distinct)
