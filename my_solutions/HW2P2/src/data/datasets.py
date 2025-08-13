"""
Dataset builders for verification/classification tasks.

This module provides a simple helper to build datasets from a `paths` dict
without involving the pipelines layer. It supports:
  1) Verification pair datasets via (root_dir, pairs_txt) tuples
  2) Generic classification datasets via ImageFolder roots

The builder returns a dict containing available splits with keys among:
"train", "eval", "test", and "train_pair", "eval_pair", "test_pair".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PairRecord:
    """Record for a single pair.

    Attributes:
        path_a: Absolute path to the first image.
        path_b: Absolute path to the second image.
        label: Integer label (typically 0 or 1) indicating match/non-match.
    """

    path_a: Path
    path_b: Path
    label: int


class PairVerificationDataset:
    """Verification dataset built from a pairs file.

    The pairs text file must contain lines of the form:
        <filename_a> <filename_b> <label>

    Filenames are resolved relative to the provided `root_dir`.

    Args:
        root_dir: Directory containing all images referenced by the pairs file.
        pairs_txt: Path to the pairs text file.
        transform: Optional transform applied independently to each image.

    Returns samples as a tuple: (image_a, image_b, label).
    Images are loaded as RGB PIL images; the provided `transform` can convert
    them to tensors if needed.
    """

    def __init__(self, root_dir: Path | str, pairs_txt: Path | str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.pairs_txt = Path(pairs_txt)
        self.transform = transform

        if not self.root_dir.is_dir():
            raise ValueError(f"root_dir does not exist or is not a directory: {self.root_dir}")
        if not self.pairs_txt.is_file():
            raise ValueError(f"pairs_txt does not exist or is not a file: {self.pairs_txt}")

        self._records: List[PairRecord] = self._load_pairs(self.root_dir, self.pairs_txt)

    @staticmethod
    def _load_pairs(root_dir: Path, pairs_txt: Path) -> List[PairRecord]:
        records: List[PairRecord] = []
        with pairs_txt.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"Invalid line {line_no} in {pairs_txt}: expected 3 fields, got {len(parts)}")
                name_a, name_b, label_str = parts
                try:
                    label = int(label_str)
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(f"Invalid label at line {line_no} in {pairs_txt}: {label_str}") from exc

                path_a = root_dir / name_a
                path_b = root_dir / name_b
                if not path_a.is_file():
                    raise ValueError(f"Missing image at line {line_no}: {path_a}")
                if not path_b.is_file():
                    raise ValueError(f"Missing image at line {line_no}: {path_b}")
                records.append(PairRecord(path_a=path_a, path_b=path_b, label=label))
        if not records:
            raise ValueError(f"Empty pairs file: {pairs_txt}")
        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int):
        rec = self._records[index]
        img_a = Image.open(rec.path_a).convert("RGB")
        img_b = Image.open(rec.path_b).convert("RGB")
        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        return img_a, img_b, rec.label


class PairSubmissionDataset(Dataset):
    """Pair dataset for test-time submission without labels.

    The pairs text file must contain lines of the form:
        <filename_a> <filename_b>

    Returns samples as a tuple: (image_a, image_b).
    """

    def __init__(self, root_dir: Path | str, pairs_txt: Path | str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.pairs_txt = Path(pairs_txt)
        self.transform = transform

        if not self.root_dir.is_dir():
            raise ValueError(f"root_dir does not exist or is not a directory: {self.root_dir}")
        if not self.pairs_txt.is_file():
            raise ValueError(f"pairs_txt does not exist or is not a file: {self.pairs_txt}")

        self._records: List[tuple[Path, Path]] = self._load_pairs(self.root_dir, self.pairs_txt)

    @staticmethod
    def _load_pairs(root_dir: Path, pairs_txt: Path) -> List[tuple[Path, Path]]:
        records: List[tuple[Path, Path]] = []
        with pairs_txt.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid line {line_no} in {pairs_txt}: expected 2 fields (no label) for test split, got {len(parts)}"
                    )
                name_a, name_b = parts
                path_a = root_dir / name_a
                path_b = root_dir / name_b
                if not path_a.is_file():
                    raise ValueError(f"Missing image at line {line_no}: {path_a}")
                if not path_b.is_file():
                    raise ValueError(f"Missing image at line {line_no}: {path_b}")
                records.append((path_a, path_b))
        if not records:
            raise ValueError(f"Empty pairs file: {pairs_txt}")
        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int):
        path_a, path_b = self._records[index]
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        return img_a, img_b


class ImageOnlyWrapper(Dataset):
    """Dataset wrapper that drops labels, returning only the image.

    This is useful for classification test splits where labels are unavailable.
    The wrapper assumes the underlying dataset returns a tuple (image, label)
    and will return only the image component.
    """

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:  # noqa: D401 - short and clear
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        sample = self.base_dataset[index]
        if isinstance(sample, tuple) and len(sample) >= 1:
            return sample[0]
        return sample


def build_datasets(paths: Dict[str, str]) -> Dict[str, object]:
    """Build datasets dict from a `paths` description (no transforms attached).

    Recognized keys (all optional):
      - Classification datasets (ImageFolder-like):
          train_dir, eval_dir, test_dir
      - Verification pair datasets (require dir+txt pairs):
          train_pairs_dir + train_pairs_txt
          val_pairs_dir + val_pairs_txt  (mapped to eval_pair)
          eval_pairs_dir + eval_pairs_txt (alternative naming for eval)
          test_pairs_dir + test_pairs_txt

    Returns:
        Dict containing available splits with keys among:
        {"train", "eval", "test", "train_pair", "eval_pair", "test_pair"}.

    Raises:
        ValueError: If a pair split specifies only one of dir/txt.
        ImportError: If torchvision is missing while building ImageFolder datasets.
    """

    # Optional import for ImageFolder-based datasets
    try:
        from torchvision.datasets import ImageFolder  # type: ignore
    except Exception:  # noqa: BLE001
        ImageFolder = None  # type: ignore

    def _make_folder_split(dir_key: str, drop_label: bool = False) -> Optional[object]:
        if dir_key in paths:
            if ImageFolder is None:
                raise ImportError("torchvision is required to build ImageFolder datasets")
            root_dir = Path(paths[dir_key])
            if not root_dir.is_dir():
                raise ValueError(f"{dir_key} does not exist or is not a directory: {root_dir}")
            # No transform here; builder/pipeline will attach transforms later
            ds = ImageFolder(root=str(root_dir), transform=None)
            return ImageOnlyWrapper(ds) if drop_label else ds
        return None

    def _make_pair_split(dir_key: str, txt_key: str, is_test: bool = False) -> Optional[Dataset]:
        if dir_key in paths and txt_key in paths:
            root_dir = Path(paths[dir_key])
            pairs_txt = Path(paths[txt_key])
            if is_test:
                return PairSubmissionDataset(root_dir=root_dir, pairs_txt=pairs_txt, transform=None)
            return PairVerificationDataset(root_dir=root_dir, pairs_txt=pairs_txt, transform=None)
        elif dir_key in paths or txt_key in paths:
            raise ValueError(f"Both '{dir_key}' and '{txt_key}' are required to build the pair dataset.")
        return None

    datasets: Dict[str, object] = {}

    # Classification-like splits
    train_cls = _make_folder_split("train_dir", drop_label=False)
    if train_cls is not None:
        datasets["train"] = train_cls
    eval_cls = _make_folder_split("eval_dir", drop_label=False)
    if eval_cls is not None:
        datasets["eval"] = eval_cls
    # Test classification split has no labels
    test_cls = _make_folder_split("test_dir", drop_label=True)
    if test_cls is not None:
        datasets["test"] = test_cls

    # Pair verification splits
    train_pair = _make_pair_split("train_pairs_dir", "train_pairs_txt", is_test=False)
    if train_pair is not None:
        datasets["train_pair"] = train_pair
    # Support both val_* and eval_* naming; map to eval_pair
    eval_pair = _make_pair_split("val_pairs_dir", "val_pairs_txt", is_test=False)
    if eval_pair is None:
        eval_pair = _make_pair_split("eval_pairs_dir", "eval_pairs_txt", is_test=False)
    if eval_pair is not None:
        datasets["eval_pair"] = eval_pair
    # Test pair split (no labels in file, returns only image pairs)
    test_pair = _make_pair_split("test_pairs_dir", "test_pairs_txt", is_test=True)
    if test_pair is not None:
        datasets["test_pair"] = test_pair

    if not datasets:
        raise ValueError("No datasets could be built from provided paths.")

    return datasets
