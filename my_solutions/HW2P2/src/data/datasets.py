"""
Dataset classes for Image Recognition and Verification
"""

import torch
import torch.utils.data
from PIL import Image
import os
import csv


class ImagePairDataset(torch.utils.data.Dataset):
    """Dataset class for image pair verification with labels"""

    def __init__(self, data_dir, csv_file, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.pairs = []
        if csv_file.endswith(".csv"):
            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    else:
                        self.pairs.append(row)
        else:
            with open(csv_file, "r") as f:
                for line in f.readlines():
                    self.pairs.append(line.strip().split(" "))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path1, img_path2, match = self.pairs[idx]
        img1 = Image.open(os.path.join(self.data_dir, img_path1))
        img2 = Image.open(os.path.join(self.data_dir, img_path2))
        return self.transform(img1), self.transform(img2), int(match)


class TestImagePairDataset(torch.utils.data.Dataset):
    """Dataset class for test image pairs without labels"""

    def __init__(self, data_dir, csv_file, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.pairs = []
        if csv_file.endswith(".csv"):
            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    else:
                        self.pairs.append(row)
        else:
            with open(csv_file, "r") as f:
                for line in f.readlines():
                    self.pairs.append(line.strip().split(" "))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path1, img_path2 = self.pairs[idx]
        img1 = Image.open(os.path.join(self.data_dir, img_path1))
        img2 = Image.open(os.path.join(self.data_dir, img_path2))
        return self.transform(img1), self.transform(img2)
