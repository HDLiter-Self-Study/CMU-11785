"""
This module defines the main PipelineBuilder, which orchestrates the creation
of the entire training pipeline from a configuration dictionary.
"""

from typing import Any, Dict, List, Optional, Callable

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.transforms import v2
from torch.utils.data import Dataset

from .factories import (
    AugmentationFactory,
    DataSamplingFactory,
    LabelMixingFactory,
    OptimizerFactory,
    SchedulerFactory,
    LossesFactory,
    HeadFactory,
    EvaluatorFactory,
)


class PipelineBuilder:
    """
    Constructs the complete training pipeline from a configuration dictionary.

    This class reads the `pipelines` section of the effective configuration,
    uses the appropriate factories to create all necessary components, and handles
    dependencies between them (e.g., schedulers needing optimizers).

    The built components are stored as public attributes.
    """

    _FACTORY_MAPPING = {
        "augmentations": AugmentationFactory,
        "label_mixing": LabelMixingFactory,
        "optimizer": OptimizerFactory,
        "scheduler": SchedulerFactory,
        "heads": HeadFactory,
        "evaluators": EvaluatorFactory,
    }

    def __init__(self, config: Dict[str, Any], model: Optional[nn.Module] = None):
        """
        Initializes the builder with the application's configuration.

        Args:
            config: The effective configuration dictionary, typically loaded
                    from a JSON or YAML file.
            model: An optional `nn.Module` whose parameters will be passed to
                   the optimizer.
        """
        self.config = config
        self.model = model
        self.pipeline_config = config.get("pipelines", {})

        # Public attributes to store the built components
        self.augmentations: Optional[nn.Module] = None
        self.label_mixing: Optional[nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None
        self.heads: Optional[nn.ModuleDict] = None
        self.losses: Any = None
        self.evaluators: Optional[Dict[str, Callable]] = None

    def build(self) -> "PipelineBuilder":
        """
        Builds all pipeline components based on the configuration.

        This method processes each category in the `pipelines` config section,
        instantiates the corresponding factory, and creates the components.
        It manages the creation order to resolve dependencies.

        Returns:
            The builder instance itself, with all components populated.
        """
        # Optimizer must be built first as the scheduler depends on it.
        if "optimizer" in self.pipeline_config:
            self.optimizer = self._build_optimizer()

        # Scheduler depends on the optimizer.
        if "scheduler" in self.pipeline_config and self.optimizer:
            self.scheduler = self._build_scheduler(self.optimizer)

        # Build other independent components.
        if "augmentations" in self.pipeline_config:
            self.augmentations = self._build_augmentations()

        if "label_mixing" in self.pipeline_config:
            self.label_mixing = self._build_label_mixing()

        if "heads" in self.pipeline_config:
            self.heads = self._build_heads()

        if "losses" in self.pipeline_config:
            self.losses = self._build_losses()

        if "evaluators" in self.pipeline_config:
            self.evaluators = self._build_evaluators()

        return self

    def _build_optimizer(self) -> Optimizer:
        """Builds the optimizer component."""
        factory = self._FACTORY_MAPPING["optimizer"]()
        # The optimizer config in the JSON is a list containing one item.
        optimizer_config = self.pipeline_config["optimizer"][0]["instances"]

        model_params = self.model.parameters() if self.model else []
        return factory.create(optimizer_config, params=model_params)

    def _build_scheduler(self, optimizer: Optimizer) -> _LRScheduler:
        """Builds the learning rate scheduler, injecting the optimizer."""
        factory = self._FACTORY_MAPPING["scheduler"]()
        scheduler_config = self.pipeline_config["scheduler"][0]["instances"]
        return factory.create(scheduler_config, optimizer=optimizer)

    def _build_augmentations(self) -> v2.Compose:
        """Builds a sequence of data augmentations."""
        factory = self._FACTORY_MAPPING["augmentations"]()
        # The factory's build method handles the entire pipeline creation.
        aug_configs = [item["instances"] for item in self.pipeline_config.get("augmentation", [])]
        return factory.build(aug_configs)

    def _build_label_mixing(self) -> Optional[nn.Module]:
        """Builds label mixing strategies. Optionally injects num_classes if available."""
        factory = self._FACTORY_MAPPING["label_mixing"]()
        label_mixing_configs = self.pipeline_config.get("label_mixing", [])

        # Try to extract num_classes from config (optional injection)
        num_classes: Optional[int] = None
        if "model" in self.config:
            model_config = self.config["model"]
            num_classes = model_config.get("num_classes")
            if num_classes is None and "architectures" in model_config:
                arch_config = model_config["architectures"]
                num_classes = arch_config.get("num_classes")

        if num_classes is None and "data" in self.config:
            num_classes = self.config["data"].get("num_classes")

        if num_classes is not None:
            return factory.build(label_mixing_configs, num_classes=num_classes)
        # If num_classes is not found, build without it. Components will fast-fail
        # at runtime if indices are used and num_classes is required.
        return factory.build(label_mixing_configs)

    def _build_heads(self) -> nn.ModuleDict:
        """Builds a dictionary of model heads."""
        factory = self._FACTORY_MAPPING["heads"]()
        head_configs = [item["instances"] for item in self.pipeline_config["heads"]]
        heads = {
            # Extract name from single-key dict: `{'MyHead': {...}}` -> `MyHead`
            list(cfg.keys())[0]: factory.create(cfg)
            for cfg in head_configs
        }
        return nn.ModuleDict(heads)

    def _build_losses(self) -> Any:
        """Builds the loss function(s) from the configuration."""
        factory = LossesFactory()
        losses_config = self.pipeline_config.get("losses", [])
        return factory.build(losses_config)

    def _build_evaluators(self) -> Dict[str, Callable]:
        """Builds a dictionary of evaluators."""
        factory = self._FACTORY_MAPPING["evaluators"]()
        evaluator_configs = self.pipeline_config.get("evaluators", [])
        # The build method can directly return a dictionary of named instances.
        return factory.build(evaluator_configs)
