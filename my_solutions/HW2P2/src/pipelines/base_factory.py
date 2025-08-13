"""
Base factory for creating pipeline components (e.g., heads, augmentations, optimizers)
in a fast-fail, modular, and registry-driven manner.
"""

import importlib
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union


class BasePipelineFactory(ABC):
    """
    An abstract factory for creating pipeline components.

    This factory serves as a base for creating specific types of components
    like data augmentations, model optimizers, or learning rate schedulers.
    It standardizes the creation process by allowing components to be
    instantiated from a configuration dictionary.

    Subclasses should override the `CUSTOM_REGISTRY` and `SEARCH_MODULES`
    class attributes to define where to find components.

    Attributes:
        CUSTOM_REGISTRY: A class-level dictionary mapping custom names to
            callable constructors. This is for special cases that cannot be
            found automatically.
        SEARCH_MODULES: A class-level list of Python modules (or their string
            paths) to search for components.
    """

    CUSTOM_REGISTRY: Dict[str, Callable] = {}
    SEARCH_MODULES: List[Union[str, Any]] = []

    def __init__(self):
        """Initializes the factory, resolving search modules."""
        self._custom_registry = self.__class__.CUSTOM_REGISTRY
        self._search_modules = self._resolve_modules(self.__class__.SEARCH_MODULES)

    def create(self, config: Dict[str, Any], **injected_params: Any) -> Any:
        """
        Creates a component instance based on a configuration dictionary.

        The configuration must contain a single key representing the component's
        name, with its value being a dictionary of parameters.

        Example:
            `{"RandomHorizontalFlip": {"p": 0.5}}`
            `{"random_horizontal_flip": {"p": 0.5}}`

        Args:
            config: The component configuration dictionary.
            injected_params: Keyword arguments to be injected into the
                component's constructor. These will override any parameters
                with the same name in the configuration.

        Returns:
            An instance of the created component.

        Raises:
            ValueError: If the configuration is invalid or the component
                        cannot be found.
        """
        if not isinstance(config, dict) or len(config) != 1:
            raise ValueError(
                "Component configuration must be a dictionary with a single " f"key-value pair. Got: {config}"
            )

        name, params = list(config.items())[0]
        params = params or {}

        # Combine configured params with injected params, with injected ones taking precedence.
        final_params = {**params, **injected_params}

        # 1. Attempt to create from the custom registry first.
        if name in self._custom_registry:
            constructor = self._custom_registry[name]
            return constructor(**final_params)

        # 2. If not in registry, search in the provided modules.
        constructor = self._find_constructor_in_modules(name)
        if constructor:
            try:
                return constructor(**final_params)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to instantiate component '{name}' with params {final_params}. " f"Error: {e}"
                ) from e

        # 3. If not found anywhere, raise an error.
        available = list(self._custom_registry.keys())
        raise ValueError(f"Component '{name}' could not be found. " f"Available in custom registry: {available}")

    def _resolve_modules(self, modules: List[Union[str, Any]]) -> List[Any]:
        """Resolves module paths into actual module objects."""
        resolved = []
        for module in modules:
            if isinstance(module, str):
                try:
                    resolved.append(importlib.import_module(module))
                except ImportError:
                    # Fail silently if a search module is not available.
                    # This allows for optional dependencies.
                    pass
            else:
                resolved.append(module)
        return resolved

    def _snake_to_camel(self, snake_str: str) -> str:
        """
        Converts a snake_case string to CamelCase.

        Examples:
            "random_erasing" -> "RandomErasing"
            "gaussian_blur" -> "GaussianBlur"
        """
        return "".join(word.capitalize() for word in snake_str.split("_"))

    def _find_constructor_in_modules(self, name: str) -> Optional[Callable]:
        """
        Searches for a component constructor by name across all registered modules.

        This method employs several strategies to find a match:
        1.  Converts snake_case name to CamelCase and searches.
        2.  Searches for the original name as-is.
        3.  Performs a case-insensitive search.
        4.  Performs a case-insensitive search after removing underscores.

        Args:
            name: The name of the component to find.

        Returns:
            A callable constructor if found, otherwise None.
        """
        search_strategies = [
            self._snake_to_camel(name),
            name,
        ]

        # First, try direct and transformed names
        for strategy_name in search_strategies:
            for module in self._search_modules:
                constructor = getattr(module, strategy_name, None)
                if constructor and callable(constructor):
                    return constructor

        # If not found, try case-insensitive searches
        name_lower = name.lower()
        name_lower_no_underscore = name_lower.replace("_", "")

        for module in self._search_modules:
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue

                attr_lower = attr_name.lower()

                # Case-insensitive match
                if attr_lower == name_lower:
                    constructor = getattr(module, attr_name)
                    if callable(constructor):
                        return constructor

                # Case-insensitive match without underscores
                if attr_lower == name_lower_no_underscore:
                    constructor = getattr(module, attr_name)
                    if callable(constructor):
                        return constructor
        return None

    def build(self, configs: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Generic build method that handles mode processing for pipeline components.

        This method processes configuration lists containing mode and instances,
        supporting common patterns like "single" and "random_choice" modes.

        Args:
            configs: List of configuration dictionaries with mode and instances.
            **kwargs: Additional parameters to be injected into component creation.

        Returns:
            A component, RandomChoice of components, or None if no valid configs.

        Raises:
            ValueError: If mode is unsupported or configuration is invalid.
        """
        if not configs:
            return None

        # Process each configuration group
        components = []
        for config in configs:
            mode = config.get("mode", "single")
            instances = config.get("instances", {})

            if mode == "single":
                # Create single component from the first (and should be only) instance
                if len(instances) != 1:
                    raise ValueError(f"Single mode requires exactly one instance, got {len(instances)}")

                component_name, component_params = next(iter(instances.items()))
                component = self.create({component_name: component_params}, **kwargs)
                components.append(component)

            elif mode == "random_choice":
                # Create multiple components for RandomChoice
                choice_components = []
                for component_name, component_params in instances.items():
                    component = self.create({component_name: component_params}, **kwargs)
                    choice_components.append(component)

                if choice_components:
                    # Import here to avoid circular imports
                    from torchvision.transforms import v2

                    random_choice = v2.RandomChoice(choice_components)
                    components.append(random_choice)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        # Return appropriate result based on number of components
        if len(components) == 0:
            return None
        elif len(components) == 1:
            return components[0]
        else:
            # Multiple components - return as list for factory-specific handling
            return components
