"""
Calculates the mean and standard deviation of a large image dataset.

This script iterates through a dataset using PyTorch's DataLoader to compute
statistics in an online manner, which is memory-efficient for large datasets.
The results are saved to a JSON file for later use in data normalization.

Usage:
    python scripts/calculate_dataset_stats.py --data-dir /path/to/your/dataset --output dataset_stats.json
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def calculate_stats(
    data_dir: str, image_size: int = 112, batch_size: int = 128, num_workers: int = 4
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Calculates mean, std, and class weights of a dataset.

    This approach uses the E[X^2] - (E[X])^2 formula for variance for higher
    accuracy and efficiency compared to a two-pass algorithm. It also performs
    a single pass to count class occurrences for weight calculation.

    Args:
        data_dir: Path to the root directory of the image dataset (e.g., 'data/train').
        image_size: The size to which images will be resized.
        batch_size: The batch size to use for calculation.
        num_workers: The number of worker processes for the DataLoader.

    Returns:
        A tuple containing the mean, std, and class_weights tensors.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # Scales images to [0.0, 1.0]
        ]
    )

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    if len(dataset) == 0:
        raise ValueError(f"No images found in directory: {data_dir}")

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Variables for online calculation
    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    num_pixels = 0

    # For class weights
    # Note: Accessing dataset.targets can be slow for large datasets if they are
    # not pre-loaded. ImageFolder pre-scans and caches them, so it's acceptable here.
    labels = dataset.targets
    class_counts = Counter(labels)
    num_classes = len(dataset.classes)
    total_samples = len(labels)

    weights = torch.zeros(num_classes, dtype=torch.float32)
    for i in range(num_classes):
        class_name = dataset.classes[i]
        class_idx = dataset.class_to_idx[class_name]
        count = class_counts.get(class_idx, 0)
        if count > 0:
            weights[class_idx] = total_samples / (num_classes * count)
        else:
            # Assign weight of 0 for classes with no samples, though this is unlikely
            # for classes discovered by ImageFolder.
            weights[class_idx] = 0

    pbar = tqdm(loader, desc="Calculating dataset stats", unit="batch")
    for images, _ in pbar:
        # Get batch size and image dimensions
        b, c, h, w = images.shape

        # Reshape to (B*C, H*W) and sum over all pixels
        # Sum over all pixels for each channel
        channels_sum += torch.sum(images, dim=[0, 2, 3])
        channels_squared_sum += torch.sum(images**2, dim=[0, 2, 3])
        num_pixels += b * h * w

    # Final calculation
    mean = channels_sum / num_pixels
    # Var(X) = E[X^2] - (E[X])^2
    std = torch.sqrt((channels_squared_sum / num_pixels) - mean**2)

    return mean, std, weights


def main():
    """Main function to parse arguments and run the calculation."""
    parser = argparse.ArgumentParser(description="Calculate dataset mean and std.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the training dataset directory (e.g., data/cls_data/train).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=112,
        help="The height/width to which images will be resized for stats calculation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset_stats.json",
        help="Path to save the output JSON file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for the DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for the DataLoader.",
    )
    args = parser.parse_args()

    print(f"Processing dataset at: {args.data_dir} with image size {args.image_size}x{args.image_size}")

    try:
        mean, std, class_weights = calculate_stats(args.data_dir, args.image_size, args.batch_size, args.num_workers)

        stats = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "class_weights": class_weights.tolist(),
        }

        output_path = Path(args.output)
        output_path.write_text(json.dumps(stats, indent=4), encoding="utf-8")

        print(f"\nStats calculated successfully!")
        print(f"  Mean: {stats['mean']}")
        print(f"  Std:  {stats['std']}")
        print(f"  Class Weights: {stats['class_weights']}")
        print(f"Saved to: {output_path.resolve()}")

    except (ValueError, FileNotFoundError) as e:
        print(f"\nError: {e}")
        print("Please ensure the data directory is correct and contains image subfolders.")


if __name__ == "__main__":
    main()
