from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest
from PIL import Image

from src.data.datasets import build_datasets


def _save_rgb(path: Path, size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (size, size), color=(127, 127, 127))
    img.save(path)


def test_build_classification_splits_labels_and_test_no_label(tmp_path: Path) -> None:
    # Prepare minimal ImageFolder structures
    train_dir = tmp_path / "train_dir"
    val_dir = tmp_path / "val_dir"
    test_dir = tmp_path / "test_dir"

    _save_rgb(train_dir / "class0" / "t0.jpg")
    _save_rgb(val_dir / "class1" / "e0.jpg")
    _save_rgb(test_dir / "class0" / "x0.jpg")

    paths: Dict[str, str] = {
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "test_dir": str(test_dir),
    }

    datasets = build_datasets(paths)

    assert "train" in datasets and "val" in datasets and "test" in datasets

    train_ds = datasets["train"]
    val_ds = datasets["val"]
    test_ds = datasets["test"]

    # Train/val return (image, label)
    t_img, t_lbl = train_ds[0]
    e_img, e_lbl = val_ds[0]
    assert isinstance(t_lbl, int)
    assert isinstance(e_lbl, int)
    assert isinstance(t_img, Image.Image)
    assert isinstance(e_img, Image.Image)

    # Test returns only image (no label)
    test_item = test_ds[0]
    assert isinstance(test_item, Image.Image)


def test_build_pair_splits_val_and_test(tmp_path: Path) -> None:
    # Prepare verification root with images
    ver_dir = tmp_path / "ver_data"
    _save_rgb(ver_dir / "a.jpg")
    _save_rgb(ver_dir / "b.jpg")

    # val txt with label (three fields)
    val_txt = tmp_path / "val_pairs.txt"
    val_txt.write_text("a.jpg b.jpg 1\n", encoding="utf-8")

    # test txt without label (two fields)
    test_txt = tmp_path / "test_pairs.txt"
    test_txt.write_text("a.jpg b.jpg\n", encoding="utf-8")

    paths: Dict[str, str] = {
        "val_pairs_dir": str(ver_dir),
        "val_pairs_txt": str(val_txt),
        "test_pairs_dir": str(ver_dir),
        "test_pairs_txt": str(test_txt),
    }

    datasets = build_datasets(paths)

    assert "val_pair" in datasets and "test_pair" in datasets

    val_pair = datasets["val_pair"]
    test_pair = datasets["test_pair"]

    a_img, b_img, lbl = val_pair[0]
    assert isinstance(a_img, Image.Image) and isinstance(b_img, Image.Image)
    assert isinstance(lbl, int) and lbl == 1

    a_img_t, b_img_t = test_pair[0]
    assert isinstance(a_img_t, Image.Image) and isinstance(b_img_t, Image.Image)


def test_alt_val_pairs_naming_supported(tmp_path: Path) -> None:
    ver_dir = tmp_path / "ver_data"
    _save_rgb(ver_dir / "a.jpg")
    _save_rgb(ver_dir / "b.jpg")

    val_txt = tmp_path / "val_pairs.txt"
    val_txt.write_text("a.jpg b.jpg 0\n", encoding="utf-8")

    paths = {
        "val_pairs_dir": str(ver_dir),
        "val_pairs_txt": str(val_txt),
    }

    datasets = build_datasets(paths)
    assert "val_pair" in datasets
    img_a, img_b, lbl = datasets["val_pair"][0]
    assert isinstance(img_a, Image.Image) and isinstance(img_b, Image.Image) and isinstance(lbl, int)


def test_incomplete_pair_spec_raises(tmp_path: Path) -> None:
    ver_dir = tmp_path / "ver_data"
    _save_rgb(ver_dir / "a.jpg")
    # Only dir, missing txt
    paths = {"val_pairs_dir": str(ver_dir)}
    with pytest.raises(ValueError):
        build_datasets(paths)


def test_no_datasets_raises() -> None:
    with pytest.raises(ValueError):
        build_datasets({})
