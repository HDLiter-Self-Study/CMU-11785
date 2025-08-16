"""
This module defines the main PipelineBuilder, which orchestrates the creation
of the entire training pipeline from a configuration dictionary.
"""

from typing import Any, Dict, List, Optional, Callable, Set
import importlib
from collections import defaultdict, deque

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.transforms import v2
from torch.utils.data import Dataset

from src.pipelines import factories


class PipelineBuilder:
    """
    Constructs the complete training pipeline from a configuration dictionary.

    This class reads the `pipelines` section of the effective configuration,
    uses the appropriate factories to create all necessary components, and handles
    dependencies between them (e.g., schedulers needing optimizers).

    The built components are stored as public attributes.
    """

    # Define dependencies between components (dependent -> dependencies)
    _DEPENDENCIES = {
        "scheduler": ["optimizer"],  # scheduler depends on optimizer
        # All other components are independent
    }

    @classmethod
    def _get_factory_class(cls, component_name: str):
        """
        Dynamically get factory class from component name.

        Converts component_name (e.g., "grad_clip") to factory class name
        (e.g., "GradClipFactory") and imports it from factories module.

        Args:
            component_name: The pipeline component name

        Returns:
            The factory class

        Raises:
            AttributeError: If factory class not found
        """
        # Convert "grad_clip" -> "GradClipFactory"
        # 1. Remove underscores and capitalize each word
        # 2. Add "Factory" suffix
        class_name = "".join(word.capitalize() for word in component_name.split("_")) + "Factory"

        try:
            return getattr(factories, class_name)
        except AttributeError:
            raise AttributeError(f"Factory class '{class_name}' not found in factories module")

    @classmethod
    def _topological_sort(cls, components: Set[str]) -> List[str]:
        """
        Perform topological sort on components based on dependencies.

        Args:
            components: Set of component names to sort

        Returns:
            List of components in dependency order

        Raises:
            ValueError: If circular dependency detected
        """
        # Build graph
        in_degree = defaultdict(int)
        graph = defaultdict(list)

        # Initialize in_degree for all components
        for component in components:
            in_degree[component] = 0

        # Build dependency graph
        for dependent, dependencies in cls._DEPENDENCIES.items():
            if dependent in components:
                for dependency in dependencies:
                    if dependency in components:
                        graph[dependency].append(dependent)
                        in_degree[dependent] += 1

        # Kahn's algorithm for topological sorting
        queue = deque([component for component in components if in_degree[component] == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for circular dependencies
        if len(result) != len(components):
            raise ValueError("Circular dependency detected in pipeline components")

        return result

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
        self._built_components = {}  # Store built components by name

    def build(self) -> "PipelineBuilder":
        """
        Builds all pipeline components based on the configuration.

        This method dynamically processes each category in the `pipelines` config section,
        instantiates the corresponding factory, and creates the components.
        It uses topological sorting to automatically manage the creation order and resolve dependencies.

        Returns:
            The builder instance itself, with all components populated.
        """
        # Get components that are present in config
        available_components = set(self.pipeline_config.keys())

        # Sort components based on dependencies using topological sort
        build_order = self._topological_sort(available_components)

        # Build components in dependency order
        for component_name in build_order:
            component = self._build_component(component_name)
            self._built_components[component_name] = component
            setattr(self, component_name, component)  # Optional: for legacy direct access

        return self

    def _build_component(self, component_name: str) -> Any:
        """
        Builds a specific component dynamically based on its name.

        Args:
            component_name: The name of the component to build

        Returns:
            The built component instance
        """
        # Dynamically get factory class
        factory_class = self._get_factory_class(component_name)
        factory = factory_class()
        config = self.pipeline_config[component_name]

        # Dynamically find special implementation methods
        impl_method = getattr(self, f"_build_{component_name}_impl", None)
        if impl_method is not None:
            return impl_method(factory, config)
        else:
            return self._default_build_impl(factory, config)

    def _default_build_impl(self, factory: Any, config: Any) -> Any:
        """
        Default build implementation for most components.
        """
        return factory.build(config)

    def _build_optimizer_impl(self, factory: Any, config: List[Dict[str, Any]]) -> Optimizer:
        """Builds the optimizer component."""
        # The optimizer config in the JSON is a list containing one item.
        optimizer_config = config[0]["instances"]
        model_params = self.model.parameters() if self.model else []
        return factory.create(optimizer_config, params=model_params)

    def _build_scheduler_impl(self, factory: Any, config: List[Dict[str, Any]]) -> _LRScheduler:
        """Builds the learning rate scheduler, injecting the optimizer."""
        if not self.optimizer:
            raise ValueError("Scheduler requires an optimizer to be built first")
        scheduler_config = config[0]["instances"]
        return factory.create(scheduler_config, optimizer=self.optimizer)

    def _build_label_mixing_impl(self, factory: Any, config: List[Dict[str, Any]]) -> Optional[nn.Module]:
        """Builds label mixing strategies. Optionally injects num_classes if available."""
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
            return factory.build(config, num_classes=num_classes)
        # If num_classes is not found, build without it. Components will fast-fail
        # at runtime if indices are used and num_classes is required.
        return factory.build(config)

    def __getattr__(self, name: str) -> Any:
        """
        Dynamically access already built pipeline components.
        Args:
            name: Component name
        Returns:
            The built component object
        Raises:
            AttributeError: If the component does not exist
        """
        if name in self._built_components:
            return self._built_components[name]
        raise AttributeError(f"'PipelineBuilder' object has no attribute '{name}'")
