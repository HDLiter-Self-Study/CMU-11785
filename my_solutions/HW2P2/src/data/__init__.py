"""Data package public API.

Only expose stable, up-to-date symbols. Legacy helpers are intentionally
not imported here to avoid pulling outdated dependencies.
"""

from src.data.datasets import (
    build_datasets,
    PairVerificationDataset,
    PairSubmissionDataset,
)

__all__ = [
    "build_datasets",
    "PairVerificationDataset",
    "PairSubmissionDataset",
]
