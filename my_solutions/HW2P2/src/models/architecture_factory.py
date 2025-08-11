"""
Architecture Factory for dynamic model creation from Effective JSON specs.

This module ties together:
- StagePlanner (derives stage widths/depths/downsamplings and expanded fields)
- ArchitectureBuilder (merges per-stage params/extras)
- Concrete model classes (ResNet, ConvNeXt) based on BaseArchitecture
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from .architectures import ResNet, ConvNeXt
from .architecture_planner import StagePlanner
from .architecture_builder import build_spec_from_planned


def _get_stem_channels(arch_type: str, stem: Dict[str, Any]) -> int:
    """Require explicit stem.out_channels; disallow silent defaults.

    Args:
        arch_type: Architecture type name.
        stem: Stem configuration dictionary.

    Returns:
        The integer out_channels for the stem.

    Raises:
        ValueError: If stem.out_channels is missing or not an int.
    """
    if isinstance(stem, dict) and isinstance(stem.get("out_channels"), int):
        return stem["out_channels"]
    raise ValueError(f"stem.out_channels must be provided explicitly for architecture '{arch_type}'")


def create_model_from_architecture(arch_cfg: Dict[str, Any], in_channels: int = 3, num_classes: int = 2) -> nn.Module:
    planned = StagePlanner.plan(arch_cfg)
    spec = build_spec_from_planned(planned)

    arch_type: str = spec["type"]
    stages: List[int] = spec["stages"]
    out_channels: List[int] = spec["out_channels"]
    downsamplings: List[int] = spec["downsamplings"]
    block_types: List[str] = spec["block_types"]
    per_stage_params: List[Dict[str, Any]] = spec["per_stage_params"]
    stem: Dict[str, Any] = spec.get("stem", {}) or {}

    block_params = per_stage_params
    stem_channels = _get_stem_channels(arch_type, stem)
    stem_params = dict(stem)
    # Map unified key to constructor key
    if "normalization" in stem_params and "norm" not in stem_params:
        stem_params["norm"] = stem_params.pop("normalization")
    stem_params.pop("out_channels", None)

    if arch_type == "resnet":
        return ResNet(
            in_channels=in_channels,
            stages=stages,
            out_channels=out_channels,
            downsamplings=downsamplings,
            block_types=block_types,
            block_params=block_params,
            head_type="classification",
            head_params={"num_classes": num_classes},
            stem_channels=stem_channels,
            stem_params=stem_params,
        )
    elif arch_type == "convnext":
        return ConvNeXt(
            in_channels=in_channels,
            stages=stages,
            out_channels=out_channels,
            downsamplings=downsamplings,
            block_types=block_types,
            block_params=block_params,
            head_type="classification",
            head_params={"num_classes": num_classes},
            stem_channels=stem_channels,
            stem_params=stem_params,
        )
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")


class ArchitectureFactory:
    def create_model(self, arch_cfg: Dict[str, Any], in_channels: int = 3, num_classes: int = 2) -> nn.Module:
        return create_model_from_architecture(arch_cfg, in_channels=in_channels, num_classes=num_classes)
