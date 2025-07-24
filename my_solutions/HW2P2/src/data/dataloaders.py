"""
Data loading utilities for HW2P2
"""

import torch
import torchvision
import os

# Handle both relative and absolute imports
try:
    from .datasets import ImagePairDataset, TestImagePairDataset
except ImportError:
    from data.datasets import ImagePairDataset, TestImagePairDataset


def get_transforms():
    """Get train and validation transforms"""
    # train transforms
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(112),  # Why are we resizing the Image?
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # val transforms
    val_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(112),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    return train_transforms, val_transforms


def get_classification_dataloaders(config):
    """Create dataloaders for classification task"""
    data_dir = config["data_dir"]
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "dev")

    train_transforms, val_transforms = get_transforms()

    # get datasets
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True, num_workers=4, sampler=None
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    return train_loader, val_loader, train_dataset, val_dataset


def get_verification_dataloaders(config):
    """Create dataloaders for verification task"""
    data_dir = config["data_ver_dir"]
    _, val_transforms = get_transforms()

    # TODO: Add your validation pair txt file
    val_pairs_path = "./data/val_pairs.txt"
    pair_dataset = ImagePairDataset(data_dir, csv_file=val_pairs_path, transform=val_transforms)
    pair_dataloader = torch.utils.data.DataLoader(
        pair_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True, num_workers=4
    )

    # TODO: Add your validation pair txt file
    test_pairs_path = "./data/test_pairs.txt"
    test_pair_dataset = TestImagePairDataset(data_dir, csv_file=test_pairs_path, transform=val_transforms)
    test_pair_dataloader = torch.utils.data.DataLoader(
        test_pair_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True, num_workers=0
    )

    return pair_dataloader, test_pair_dataloader
