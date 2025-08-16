from __future__ import annotations
from typing import Any, Dict
import torch.nn as nn

from src.models.architecture_planner import StagePlanner
from src.models.architecture_builder import build_spec_from_planned
from src.models.architectures import ResNet, ConvNeXt


class ModelFactory:

    ARCHITECTURE_MAP = {
        "resnet": ResNet,
        "convnext": ConvNeXt,
    }

    def create(self, arch_config: Dict[str, Any], data_config: Dict[str, Any]) -> nn.Module:
        """
        Creates a model backbone instance from configuration dictionaries.

        This method orchestrates the full pipeline from raw architecture config
        to a final nn.Module instance.

        Args:
            arch_config: The architecture configuration dictionary.
            data_config: The data configuration dictionary, used to extract
                         parameters like `in_channels`.

        Returns:
            An instantiated model backbone (nn.Module).
        """
        # 1. Plan the architecture
        planned_stages = StagePlanner.plan(arch_config)

        # 2. Build the final constructor spec from the plan
        build_spec = build_spec_from_planned(planned_stages.__dict__)

        # 3. Inject data-dependent parameters and instantiate the model
        build_spec["in_channels"] = data_config["in_channels"]

        arch_type = build_spec.pop("type", None)
        if arch_type not in self.ARCHITECTURE_MAP:
            raise ValueError(
                f"Unknown architecture type: '{arch_type}'. " f"Available types: {list(self.ARCHITECTURE_MAP.keys())}"
            )

        ArchitectureClass = self.ARCHITECTURE_MAP[arch_type]

        return ArchitectureClass(**build_spec)
