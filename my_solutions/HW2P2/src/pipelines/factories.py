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
from collections import Counter

import torch
from torch import optim
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torch.nn import CrossEntropyLoss

from src.pipelines.base_factory import BasePipelineFactory
from src.data.label_mixing import FMix

# =============================================================================
# Component Factories
# =============================================================================


class HeadsFactory(BasePipelineFactory):
    """
    Factory for creating model heads. A model can only have a single head.
    """

    SEARCH_MODULES = ["src.heads"]

    def build(self, configs: List[Dict[str, Any]], **kwargs) -> Optional[torch.nn.Module]:
        if not configs:
            return None

        # A model should only have one head. This is the primary validation.
        if len(configs) > 1:
            raise ValueError(f"A model can only have one head, but {len(configs)} configurations were provided.")

        # Since there's only one config, let the base factory build it.
        # The base `build` method will return a single instance, not a list.
        return super().build(configs, **kwargs)


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

    def build(self, configs: List[Dict[str, Any]], return_base_transform: bool = True) -> Any:
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
        if not return_base_transform:
            return v2.Compose(full_pipeline)
        else:
            # Also return the base transform pipeline for validation and testing
            return v2.Compose(full_pipeline), v2.Compose(final_transforms)


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

        params = SchedulerFactory.compute_eta_min(optimizer, **params)

        # Do not filter parameters; let constructor fast-fail on invalid kwargs
        return CosineAnnealingLR(optimizer=optimizer, **params)

    @staticmethod
    def create_cosine_annealing_warm_restarts(
        optimizer: optim.Optimizer, **params: Any
    ) -> "optim.lr_scheduler.CosineAnnealingWarmRestarts":
        """
        Static method to instantiate CosineAnnealingWarmRestarts.
        """
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

        if optimizer is None:
            raise ValueError("optimizer is required for CosineAnnealingWarmRestarts")

        params = SchedulerFactory.compute_eta_min(optimizer, **params)

        # Do not filter parameters; let constructor fast-fail on invalid kwargs
        return CosineAnnealingWarmRestarts(optimizer=optimizer, **params)

    @staticmethod
    def compute_eta_min(optimizer: optim.Optimizer, **params: Any) -> Dict[str, Any]:
        """
        Compute eta_min for CosineAnnealingWarmRestarts.
        """
        if optimizer is None:
            raise ValueError("optimizer is required for CosineAnnealingWarmRestarts")

        params = dict(params or {})
        if "eta_min" in params and "eta_min_ratio" in params:
            raise ValueError("Provide either 'eta_min' or 'eta_min_ratio', not both, for CosineAnnealingWarmRestarts")
        if "eta_min" not in params and "eta_min_ratio" in params:
            ratio = float(params.pop("eta_min_ratio"))
            try:
                base_lr = min(float(pg.get("lr", 0.0)) for pg in optimizer.param_groups)
            except Exception as exc:
                raise RuntimeError("Failed to infer base lr from optimizer.param_groups") from exc
            params["eta_min"] = base_lr * ratio
        return params

    @staticmethod
    def create_warmup(
        optimizer: optim.Optimizer, steps_per_epoch: int = 0, **params: Any
    ) -> Optional[optim.lr_scheduler.LinearLR]:
        """
        Static method to instantiate LinearLR for warmup. Fails fast if required params are missing.

        Args:
            optimizer: The optimizer instance.
            steps_per_epoch: Batches per epoch for step calculation.
            **params: Warmup parameters including 'warmup_epochs' and 'warmup_start_factor'.

        Returns:
            A torch.optim.lr_scheduler.LinearLR instance for warmup, or None if warmup_epochs <= 0.
        """
        from torch.optim.lr_scheduler import LinearLR

        # Fail fast if warmup_epochs is missing from the config.
        warmup_epochs = params["warmup_epochs"]
        if warmup_epochs <= 0:
            return None  # This is a valid configuration for 'no warmup'.

        # Prepare parameters for LinearLR, failing fast if keys are missing.
        # Let LinearLR use its default for end_factor.
        warmup_params = {
            "start_factor": params["warmup_start_factor"],
            "total_iters": warmup_epochs * steps_per_epoch,
        }

        return LinearLR(optimizer, **warmup_params)

    @staticmethod
    def create_multi_step_lr(optimizer: optim.Optimizer, **params: Any) -> optim.lr_scheduler.MultiStepLR:
        """
        Static method to instantiate MultiStepLR.
        It expects 'warmup_epochs' and 'total_epochs' to be injected into params by the build method.
        It parses 'milestones_ratio' and shifts milestones if warmup is active. Fails fast.

        Args:
            optimizer: The optimizer instance.
            **params: Parameters including 'milestones_ratio', 'gamma', and injected context
                      like 'warmup_epochs' and 'total_epochs'.

        Returns:
            A torch.optim.lr_scheduler.MultiStepLR instance.
        """
        import json
        from torch.optim.lr_scheduler import MultiStepLR

        # Pop injected context params; fail fast if missing.
        warmup_epochs = params.pop("warmup_epochs")
        total_epochs = params.pop("total_epochs")
        ratio_str = params.pop("milestones_ratio")

        # Parse and calculate milestones
        try:
            ratios = json.loads(ratio_str)
            if not isinstance(ratios, list) or not all(isinstance(r, (int, float)) for r in ratios):
                raise ValueError("milestones_ratio must be a list of numbers")
            milestones = sorted([int(total_epochs * r) for r in ratios if 0 < r < 1])
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Invalid milestones_ratio format: {ratio_str} - {e}")

        # Shift if warmup present
        if warmup_epochs > 0:
            milestones = [m + warmup_epochs for m in milestones if m + warmup_epochs < total_epochs]

        # Prepare final params for MultiStepLR. It will fail fast if 'gamma' is missing.
        final_params = {"milestones": milestones, **params}
        return MultiStepLR(optimizer=optimizer, **final_params)

    CUSTOM_REGISTRY: Dict[str, Callable] = {
        "cosine_annealing_lr": create_cosine_annealing_lr.__func__,
        "CosineAnnealingLR": create_cosine_annealing_lr.__func__,
        "multi_step_lr": create_multi_step_lr.__func__,
        "MultiStepLR": create_multi_step_lr.__func__,
        "cosine_annealing_warm_restarts": create_cosine_annealing_warm_restarts.__func__,
        "CosineAnnealingWarmRestarts": create_cosine_annealing_warm_restarts.__func__,
        "warmup": create_warmup.__func__,  # Register warmup creator
    }

    def build(
        self,
        configs: List[Dict[str, Any]],
        optimizer: optim.Optimizer,
        total_epochs: int,  # Injected from cfg["run"]["epochs"]
        steps_per_epoch: int,  # Injected from len(train_loader)
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Builds the scheduler from a list of configurations, handling chaining of warmup and main scheduler.

        This method uses a pre-processing step to inject context, then calls super().build to instantiate,
        and finally post-processes the components to create a chained scheduler if needed.

        Args:
            configs: List of scheduler group configs from effective JSON.
            optimizer: The optimizer instance.
            total_epochs: Total training epochs from config.
            steps_per_epoch: Number of batches per epoch (len(train_loader)).

        Returns:
            A chained scheduler or main scheduler, or None if no configs.
        """
        if not configs:
            return None

        # Step 1: Pre-process configs - find warmup_epochs and inject context into other configs.
        warmup_epochs = 0
        for config in configs:
            # Use direct access to fail fast if 'instances' is missing.
            instances = config["instances"]
            if "warmup" in instances:
                # Fail fast if 'warmup_epochs' is missing from the warmup instance.
                warmup_epochs = instances["warmup"]["warmup_epochs"]
                # Inject steps_per_epoch only into the warmup config.
                if warmup_epochs > 0:
                    instances["warmup"]["steps_per_epoch"] = steps_per_epoch
                break  # Assume only one warmup config

        # Inject context into multi_step_lr before instantiation.
        for config in configs:
            instances = config["instances"]
            if "multi_step_lr" in instances:
                multi_step_params = instances["multi_step_lr"]
                multi_step_params["warmup_epochs"] = warmup_epochs
                multi_step_params["total_epochs"] = total_epochs

        # Step 2: Call super().build to process modes and instantiate components.
        # steps_per_epoch is now passed via the config dict, not as a kwarg.
        components = super().build(configs, optimizer=optimizer)

        # Step 3: Post-process components to identify warmup and main, then chain if needed.
        if components is None:
            return None

        # super().build returns a single item or a list. Normalize to list.
        if not isinstance(components, list):
            components = [components]

        # Filter out None values, e.g., from a warmup config with warmup_epochs=0.
        components = [c for c in components if c is not None]

        if not components:
            return None
        if len(components) > 2:
            raise ValueError("Scheduler components must be at most 2: one main and one optional warmup.")

        warmup_scheduler = None
        main_scheduler = None

        for component in components:
            if isinstance(component, torch.optim.lr_scheduler.LinearLR):
                warmup_scheduler = component
            else:
                main_scheduler = component

        if main_scheduler is None:
            return warmup_scheduler  # Only warmup was configured.

        if warmup_scheduler:
            # Chain the two schedulers together.
            warmup_steps = warmup_epochs * steps_per_epoch
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps]
            )

        return main_scheduler  # No warmup configured, return only the main scheduler.


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


class LossesFactory(BasePipelineFactory):
    """
    Factory for creating loss functions.
    It provides special handling for CrossEntropyLoss to support class weighting.
    """

    SEARCH_MODULES = ["src.losses", "torch.nn"]
    STATS_FILE = Path("configs/data/dataset_stats.json")

    @staticmethod
    def create_cross_entropy_loss(**kwargs: Any) -> CrossEntropyLoss:
        """
        Custom creator for CrossEntropyLoss.
        - If 'class_weights' is true, loads weights from the stats file.
        - Fails fast if the stats file or 'class_weights' key is missing.
        """
        params = dict(kwargs or {})
        use_class_weights = params.pop("class_weights", False)

        if use_class_weights:
            stats_file = LossesFactory.STATS_FILE
            if not stats_file.is_file():
                raise FileNotFoundError(
                    f"Dataset statistics file not found at: {stats_file.resolve()}\n"
                    f"Run 'scripts/calculate_dataset_stats.py' to generate it before using class_weights."
                )

            stats = json.loads(stats_file.read_text())
            if "class_weights" not in stats:
                raise KeyError(
                    f"'class_weights' not found in {stats_file.resolve()}. "
                    f"Please ensure 'calculate_dataset_stats.py' was run correctly."
                )

            # Convert to tensor and inject into params
            weight_tensor = torch.tensor(stats["class_weights"], dtype=torch.float32)
            params["weight"] = weight_tensor

        return CrossEntropyLoss(**params)

    CUSTOM_REGISTRY: Dict[str, Callable] = {
        "cross_entropy_loss": create_cross_entropy_loss.__func__,
        "CrossEntropyLoss": create_cross_entropy_loss.__func__,
    }


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


class EvaluatorsFactory(BasePipelineFactory):
    """
    Factory for creating evaluators.
    It provides a special 'argmax' evaluator for classification tasks.
    """

    SEARCH_MODULES = ["src.evaluators"]

    @staticmethod
    def create_argmax(**kwargs: Any) -> Callable[[torch.Tensor], torch.Tensor]:
        """Returns a lambda function that performs argmax on dimension 1."""
        return lambda x: torch.argmax(x, **kwargs)

    CUSTOM_REGISTRY: Dict[str, Callable] = {
        "argmax": create_argmax.__func__,
    }


class DatasetFactory(BasePipelineFactory):
    """
    Factory for creating data sampling wrappers, such as repeated augmentation.

    This factory typically wraps an existing dataset returned by the datasets
    factory/builder and augments its iteration behavior without changing the
    transform pipeline.
    """

    SEARCH_MODULES = ["src.data.dataset"]

    def build(
        self,
        configs: List[Dict[str, Any]],
        datasets: Dict[str, Dataset],
        target_keys: List[str] = ["train"],  # only build for these keys
    ) -> Dict[str, Dataset]:
        dataset_wrappers = super().build(configs)
        if dataset_wrappers is None:
            # no dataset wrappers, return original datasets
            return datasets
        if not isinstance(dataset_wrappers, list):
            dataset_wrappers = [dataset_wrappers]
        for key in target_keys:
            # For each target key, apply all dataset wrappers
            if key not in datasets:
                raise ValueError(f"Dataset key '{key}' not found in datasets")
            for wrapper in dataset_wrappers:
                datasets[key] = wrapper(datasets[key])
        return datasets


class EmaFactory(BasePipelineFactory):
    """
    Factory for creating Exponential Moving Average (EMA) callbacks.
    """

    SEARCH_MODULES = ["src.utils.ema"]


class GradClipFactory(BasePipelineFactory):
    """
    Factory for creating gradient clipping functions.
    """

    SEARCH_MODULES = ["src.utils.grad_clip"]
