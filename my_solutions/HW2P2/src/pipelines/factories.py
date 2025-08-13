"""
This module contains concrete factory implementations for building various
pipeline components, such as data augmentations, optimizers, and model heads.

Each factory inherits from `BasePipelineFactory` and specifies its own
`SEARCH_MODULES` to locate components from standard libraries (like torchvision
or torch) or custom project modules.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import torch
from torch import nn, optim
from torchvision.transforms import v2

from .base_factory import BasePipelineFactory
from ..data.label_mixing import FMix


class AugmentationFactory(BasePipelineFactory):
    """
    Factory to build a complete data augmentation pipeline.

    This factory assembles a sequence of transformations from a list of
    configurations and automatically appends necessary final steps like
    converting to a tensor and normalizing.
    """

    SEARCH_MODULES = [
        "torchvision.transforms.v2",
        "src.data.transforms",
    ]
    STATS_FILE = Path("configs/data/dataset_stats.json")

    def create(self, config: Dict[str, Any], **injected_params: Any) -> Any:
        """
        Creates an augmentation component with intelligent probability handling.

        This method extracts 'p' or 'prob' parameters and wraps the component
        with v2.RandomApply if a valid probability is found.

        Args:
            config: The component configuration dictionary.
            **injected_params: Additional parameters to inject.

        Returns:
            Either the raw component or a v2.RandomApply wrapper.
        """
        # Use the parent's validation first
        if not isinstance(config, dict) or len(config) != 1:
            return super().create(config, **injected_params)

        # Make a copy to avoid modifying the original
        config_copy = {name: params.copy() for name, params in config.items()}

        # Extract probability parameters
        params = next(iter(config_copy.values()))
        prob = params.pop("p", params.pop("prob", -1.0))

        # Create the base transform using the parent's create method
        transform = super().create(config_copy, **injected_params)

        # Wrap with RandomApply if probability is valid
        if 0.0 <= prob <= 1.0:
            return v2.RandomApply([transform], p=prob)
        else:
            return transform

    def build(self, configs: List[Dict[str, Any]]) -> v2.Compose:
        """
        Builds the augmentation pipeline from a list of configurations.

        This method loads dataset statistics and uses the base class for mode
        processing, while adding mandatory normalization transforms.

        Args:
            configs: A list of configuration dictionaries with mode and instances.

        Returns:
            A `v2.Compose` object representing the full augmentation pipeline.

        Raises:
            FileNotFoundError: If the `dataset_stats.json` file is not found.
        """
        # Load dataset statistics for normalization
        if not self.STATS_FILE.is_file():
            raise FileNotFoundError(
                f"Dataset statistics file not found at: {self.STATS_FILE.resolve()}\n"
                "Please run 'python scripts/calculate_dataset_stats.py' first."
            )
        stats = json.loads(self.STATS_FILE.read_text())
        mean, std = stats["mean"], stats["std"]

        # Use the base class implementation for mode processing
        # This will call our overridden create method which handles probability wrapping
        augmentations_component = super().build(configs)

        # Handle the return value from base build
        if augmentations_component is None:
            augmentations = []
        elif isinstance(augmentations_component, (list, tuple)):
            augmentations = list(augmentations_component)
        else:
            augmentations = [augmentations_component]

        # Append mandatory final transformations
        final_transforms = [
            v2.ToDtype(torch.float32, scale=True),  # Replaces ToTensor
            v2.Normalize(mean=mean, std=std),
        ]

        full_pipeline = augmentations + final_transforms
        return v2.Compose(full_pipeline)


class OptimizerFactory(BasePipelineFactory):
    """Factory for creating optimizers."""

    SEARCH_MODULES = [
        "torch.optim",
    ]

    @staticmethod
    def _merge_betas_params(raw_params: Dict[str, Any], optimizer_label: str) -> Dict[str, Any]:
        """Return a copy of params with beta1/beta2 merged into betas.

        Fast-fail when both betas and any of beta1/beta2 are provided.
        """
        params = dict(raw_params or {})
        has_betas = "betas" in params
        has_beta1 = "beta1" in params
        has_beta2 = "beta2" in params
        if has_betas and (has_beta1 or has_beta2):
            raise ValueError(f"Provide either 'betas' or ('beta1' and 'beta2'), not both, for {optimizer_label}")
        if (not has_betas) and has_beta1 and has_beta2:
            beta1 = float(params.pop("beta1"))
            beta2 = float(params.pop("beta2"))
            params["betas"] = (beta1, beta2)
        return params

    @staticmethod
    def create_adam(**kwargs) -> optim.Optimizer:
        """Create Adam with optional beta1/beta2 to betas merging.

        Accepts either 'betas' or ('beta1' and 'beta2'). If both betas and
        beta1/beta2 are provided, fast-fail.
        """
        from torch.optim import Adam

        merged = OptimizerFactory._merge_betas_params(kwargs, "Adam")
        return Adam(**merged)

    @staticmethod
    def create_adamw(**kwargs) -> optim.Optimizer:
        """Create AdamW with optional beta1/beta2 to betas merging.

        Accepts either 'betas' or ('beta1' and 'beta2'). If both are provided,
        fast-fail.
        """
        from torch.optim import AdamW

        merged = OptimizerFactory._merge_betas_params(kwargs, "AdamW")
        return AdamW(**merged)

    CUSTOM_REGISTRY: Dict[str, Callable] = {
        "adam": create_adam.__func__,
        "Adam": create_adam.__func__,
        "adamw": create_adamw.__func__,
        "AdamW": create_adamw.__func__,
    }


class SchedulerFactory(BasePipelineFactory):
    """
    Factory for creating learning rate schedulers.

    Note: The creation of schedulers often requires an optimizer instance,
    which is not handled by the base `create` method. The `PipelineBuilder`
    will need to handle this special dependency injection.
    """

    SEARCH_MODULES = [
        "torch.optim.lr_scheduler",
    ]

    @staticmethod
    def create_cosine_annealing_lr(
        optimizer: "optim.Optimizer", **params: Any
    ) -> "optim.lr_scheduler.CosineAnnealingLR":
        """
        Static method to instantiate CosineAnnealingLR with optional eta_min_ratio support.

        Args:
            optimizer: The optimizer instance injected by the builder.
            **params: Parameters for CosineAnnealingLR. Supports either
                - eta_min (float), or
                - eta_min_ratio (float), which will be converted using
                  eta_min = min(param_group.lr) * eta_min_ratio.

        Returns:
            A torch.optim.lr_scheduler.CosineAnnealingLR instance.
        """
        from torch.optim.lr_scheduler import CosineAnnealingLR

        if optimizer is None:
            raise ValueError("optimizer is required for CosineAnnealingLR")

        params = dict(params or {})
        # Fast-fail when both provided
        if "eta_min" in params and "eta_min_ratio" in params:
            raise ValueError("Provide either 'eta_min' or 'eta_min_ratio', not both, for CosineAnnealingLR")
        if "eta_min" not in params and "eta_min_ratio" in params:
            ratio = float(params.pop("eta_min_ratio"))
            try:
                base_lr = min(float(pg.get("lr", 0.0)) for pg in optimizer.param_groups)
            except Exception as exc:
                raise RuntimeError("Failed to infer base lr from optimizer.param_groups") from exc
            params["eta_min"] = base_lr * ratio

        # Do not filter parameters; let constructor fast-fail on invalid kwargs
        return CosineAnnealingLR(optimizer=optimizer, **params)

    CUSTOM_REGISTRY: Dict[str, Callable] = {
        "cosine_annealing_lr": create_cosine_annealing_lr.__func__,
        "CosineAnnealingLR": create_cosine_annealing_lr.__func__,
    }


class LoaderFactory(BasePipelineFactory):
    """
    Factory for creating training and evaluation DataLoaders.

    Unified with other factories:
      - instances key: "data_loader"
      - create returns parameter dict for the loader
      - build uses super().build to obtain params, then constructs
        (train_loader, eval_loader) with split-specific defaults.
    """

    SEARCH_MODULES: List[str] = [
        "torch.utils.data",
    ]

    def build(
        self,
        configs: List[Dict[str, Any]],
        *,
        dataset_train: Any,
        dataset_eval: Any,
        collate_train: Any = None,
        collate_eval: Any = None,
    ) -> Any:
        if not configs:
            raise ValueError("LoaderFactory.build requires configuration for data_loader")

        # Expect exactly one group with one instance named 'data_loader'
        group = configs[0]
        instances = group["instances"]  # let KeyError bubble if missing
        loader_params = dict(instances["data_loader"])  # let KeyError bubble if missing

        # Train parameters: default shuffle/drop_last if absent; pass everything through
        train_params = dict(loader_params)
        train_params.setdefault("shuffle", True)
        train_params.setdefault("drop_last", True)
        train_loader = super().create({"data_loader": train_params}, dataset=dataset_train, collate_fn=collate_train)

        # Eval parameters: copy and remove shuffle/drop_last/sampler keys
        eval_params = dict(loader_params)
        for k in ["shuffle", "drop_last", "sampler", "batch_sampler"]:
            if k in eval_params:
                eval_params.pop(k)
        eval_loader = super().create({"data_loader": eval_params}, dataset=dataset_eval, collate_fn=collate_eval)

        return train_loader, eval_loader


class HeadFactory(BasePipelineFactory):
    """
    Factory for creating model heads.

    Model heads are custom `nn.Module` subclasses that perform the final
    classification or regression task.
    """

    SEARCH_MODULES = [
        "src.models.heads",  # Assuming custom heads are located here
        "src.models.common_blocks",
    ]


class LabelMixingFactory(BasePipelineFactory):
    """
    Factory for creating label mixing strategies (MixUp, CutMix, FMix, etc.).

    This factory handles the creation of label mixing transforms that modify both
    input images and their corresponding labels during training. It requires
    `num_classes` only when labels are provided as class indices. If labels are
    already one-hot/soft, `num_classes` is not required. Mode handling ("single"
    and "random_choice") is delegated to BasePipelineFactory.
    """

    SEARCH_MODULES = [
        "torchvision.transforms.v2",
    ]

    CUSTOM_REGISTRY = {
        "fmix": FMix,
    }

    # Note:
    # - Build is inherited from BasePipelineFactory. Provide `num_classes` via
    #   injected kwargs when available. Underlying components will fast-fail at
    #   runtime if class-index labels are used without `num_classes`.


class DataSamplingFactory(BasePipelineFactory):
    """
    Factory for creating data sampling wrappers, such as repeated augmentation.

    This factory typically wraps an existing dataset returned by the datasets
    factory/builder and augments its iteration behavior without changing the
    transform pipeline.
    """

    SEARCH_MODULES = ["src.data.sampling"]
