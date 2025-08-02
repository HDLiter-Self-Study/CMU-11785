"""
Augmentation configuration parser and Optuna sampler
"""

import optuna
import re
from typing import Dict, Any, List, Optional, Union
from omegaconf import DictConfig
from config_manager import get_config


class AugmentationSampler:
    """
    Parser and sampler for augmentation configuration using Optuna
    """

    def __init__(self, config_name: str = "main", overrides: Optional[List[str]] = None):
        """
        Initialize the augmentation sampler

        Args:
            config_name: Name of the main config file
            overrides: List of parameter overrides
        """
        self.config = get_config(config_name, overrides)
        self.search_spaces = self.config.search_spaces

        # Get all available search space categories dynamically
        self.search_space_categories = []
        for key in self.search_spaces.keys():
            if hasattr(self.search_spaces, key):
                self.search_space_categories.append(key)

        # For backward compatibility, keep these properties
        self.augmentation_config = getattr(self.search_spaces, "augmentation", None)
        self.label_mixing_config = getattr(self.search_spaces, "label_mixing", None)
        self.data_sampling_config = getattr(self.search_spaces, "data_sampling", None)

    def _evaluate_condition(self, condition: str, sampled_params: Dict[str, Any]) -> bool:
        """
        Evaluate a condition string using sampled parameters

        Args:
            condition: Condition string (e.g., "$spatial_augmentation_strategy in ['random_erasing', 'random_choice']")
            sampled_params: Dictionary of sampled parameters

        Returns:
            Boolean result of condition evaluation
        """
        if not condition.startswith("$"):
            return True

        # Remove $ prefix
        condition_eval = condition[1:]

        # Create a safe evaluation environment with parameter values
        eval_env = {}
        for param_name, param_value in sampled_params.items():
            eval_env[param_name] = param_value

        try:
            # Evaluate the condition in the safe environment
            result = eval(condition_eval, {"__builtins__": {}}, eval_env)
            return bool(result)
        except Exception as e:
            print(f"Warning: Could not evaluate condition '{condition}': {e}")
            return False

    def _get_choices_for_strategy(self, choices_dict: Dict[str, List[str]], strategy_value: str) -> List[str]:
        """
        Get choices based on strategy value

        Args:
            choices_dict: Dictionary mapping strategy to choices
            strategy_value: Current strategy value

        Returns:
            List of available choices
        """
        return choices_dict.get(strategy_value, [])

    def _sample_parameter(
        self, trial: optuna.Trial, param_config: DictConfig, param_name: str, sampled_params: Dict[str, Any]
    ) -> Any:
        """
        Sample a single parameter using Optuna trial

        Args:
            trial: Optuna trial object
            param_config: Parameter configuration
            param_name: Name of the parameter
            sampled_params: Already sampled parameters (for dependency resolution)

        Returns:
            Sampled parameter value
        """
        param_type = param_config.type

        if param_type == "categorical":
            if hasattr(param_config, "choices") and isinstance(param_config.choices, dict):
                # Dictionary-based choices (depends on another parameter)
                depends_param = param_config.depends_on[0] if hasattr(param_config, "depends_on") else None
                if depends_param and depends_param in sampled_params:
                    choices = self._get_choices_for_strategy(param_config.choices, sampled_params[depends_param])
                else:
                    # Fallback to first available choices
                    choices = list(param_config.choices.values())[0]
            else:
                choices = param_config.choices
            return trial.suggest_categorical(param_name, choices)

        elif param_type == "float":
            min_val = param_config.min
            max_val = param_config.max

            # Handle dynamic min/max values
            if isinstance(min_val, str) and min_val.startswith("$"):
                min_expr = min_val[1:]  # Remove $ prefix
                for dep_name, dep_value in sampled_params.items():
                    min_expr = min_expr.replace(dep_name, str(dep_value))
                try:
                    min_val = eval(min_expr)
                except:
                    min_val = param_config.get("default", 0.0)

            if isinstance(max_val, str) and max_val.startswith("$"):
                max_expr = max_val[1:]  # Remove $ prefix
                for dep_name, dep_value in sampled_params.items():
                    max_expr = max_expr.replace(dep_name, str(dep_value))
                try:
                    max_val = eval(max_expr)
                except:
                    max_val = min_val + 1.0

            return trial.suggest_float(param_name, min_val, max_val)

        elif param_type == "int":
            min_val = param_config.min
            max_val = param_config.max
            return trial.suggest_int(param_name, min_val, max_val)

        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")

    def _sample_instance(
        self, trial: optuna.Trial, instance_config: DictConfig, instance_name: str, sampled_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sample parameters for an instance

        Args:
            trial: Optuna trial object
            instance_config: Instance configuration
            instance_name: Name of the instance
            sampled_params: Already sampled parameters

        Returns:
            Dictionary of sampled instance parameters
        """
        instance_params = {}

        # Check if instance should be active
        if hasattr(instance_config, "condition"):
            if not self._evaluate_condition(instance_config.condition, sampled_params):
                return instance_params

        # Sample each parameter in the instance
        for param_name, param_config in instance_config.items():
            # Skip non-config values (like comments or strings)
            if not isinstance(param_config, DictConfig):
                continue

            if param_config.get("class") == "param":
                full_param_name = param_config.param_name
                value = self._sample_parameter(trial, param_config, full_param_name, sampled_params)
                instance_params[full_param_name] = value
                sampled_params[full_param_name] = value

        return instance_params

    def _sample_strategy(
        self, trial: optuna.Trial, strategy_config: DictConfig, strategy_name: str, sampled_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sample parameters for a strategy

        Args:
            trial: Optuna trial object
            strategy_config: Strategy configuration
            strategy_name: Name of the strategy
            sampled_params: Already sampled parameters

        Returns:
            Dictionary of sampled strategy parameters
        """
        strategy_params = {}

        for key, config in strategy_config.items():
            # Skip non-config values (like comments or strings)
            if not isinstance(config, DictConfig):
                continue

            if config.get("class") == "param":
                # Sample strategy parameter
                param_name = config.param_name
                value = self._sample_parameter(trial, config, param_name, sampled_params)
                strategy_params[param_name] = value
                sampled_params[param_name] = value

            elif config.get("class") == "instance":
                # Sample instance parameters
                instance_params = self._sample_instance(trial, config, key, sampled_params)
                strategy_params.update(instance_params)

            elif config.get("class") == "technique":
                # Sample technique parameters (new naming)
                technique_params = self._sample_strategy(trial, config, key, sampled_params)
                strategy_params.update(technique_params)

            elif config.get("class") == "strategy":
                # Nested strategy (backward compatibility)
                nested_params = self._sample_strategy(trial, config, key, sampled_params)
                strategy_params.update(nested_params)

        return strategy_params

    def sample_category_params(self, trial: optuna.Trial, category_name: str) -> Dict[str, Any]:
        """
        Sample parameters for a specific search space category

        Args:
            trial: Optuna trial object
            category_name: Name of the search space category (e.g., 'augmentation', 'label_mixing')

        Returns:
            Dictionary of sampled parameters for this category
        """
        category_config = getattr(self.search_spaces, category_name, None)
        if not category_config:
            return {}

        # Look for the main strategy config (usually named after the category)
        main_strategy = getattr(category_config, category_name, None)
        if not main_strategy:
            return {}

        sampled_params = {}
        all_params = {}

        # Sample parameters for this category
        category_params = self._sample_strategy(trial, main_strategy, category_name, sampled_params)
        all_params.update(category_params)
        all_params.update(sampled_params)

        return all_params

    def sample_augmentation_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample augmentation parameters for a trial

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of sampled augmentation parameters
        """
        return self.sample_category_params(trial, "augmentation")

    def sample_label_mixing_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample label mixing parameters for a trial

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of sampled label mixing parameters
        """
        return self.sample_category_params(trial, "label_mixing")

    def sample_data_sampling_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample data sampling parameters for a trial

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of sampled data sampling parameters
        """
        return self.sample_category_params(trial, "data_sampling")

    def sample_all_params(
        self, trial: optuna.Trial, categories: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Sample parameters for all or specified search space categories

        Args:
            trial: Optuna trial object
            categories: List of category names to sample (if None, samples all available categories)

        Returns:
            Dictionary containing sampled parameters organized by category
        """
        if categories is None:
            categories = self.search_space_categories

        results = {}
        for category in categories:
            if category in self.search_space_categories:
                results[category] = self.sample_category_params(trial, category)

        return results

    def _build_technique_registry(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Build a registry of techniques and their parameter prefixes from config.
        Dynamically discovers all search space categories and their techniques.

        Returns:
            Dictionary mapping config categories to technique info
        """
        registry = {}

        # Iterate through all search space categories dynamically
        for category_name in self.search_space_categories:
            category_config = getattr(self.search_spaces, category_name)
            if not category_config:
                continue

            registry[category_name] = {}

            # Look for the main strategy config (usually named after the category)
            main_strategy = getattr(category_config, category_name, None)
            if not main_strategy:
                continue

            # Parse all sub-strategies and instances within this category
            for strategy_key, strategy_config in main_strategy.items():
                if not isinstance(strategy_config, DictConfig):
                    continue

                if strategy_config.get("class") == "technique":
                    # Extract techniques from this technique category
                    techniques = []
                    for key, config in strategy_config.items():
                        if isinstance(config, DictConfig) and config.get("class") == "instance":
                            techniques.append(key)

                    if techniques:
                        registry[category_name][strategy_key] = techniques
                elif strategy_config.get("class") == "strategy":
                    # Handle nested strategies (backward compatibility)
                    techniques = []
                    for key, config in strategy_config.items():
                        if isinstance(config, DictConfig) and config.get("class") == "instance":
                            techniques.append(key)

                    if techniques:
                        registry[category_name][strategy_key] = techniques

            # Also check for direct instances at the category level
            direct_techniques = []
            for key, config in main_strategy.items():
                if isinstance(config, DictConfig) and config.get("class") == "instance":
                    direct_techniques.append(key)
            if direct_techniques:
                registry[category_name]["direct"] = direct_techniques

        return registry

    def _get_technique_prefixes(self, technique_name: str) -> List[str]:
        """
        Get parameter prefixes for a technique based on naming conventions.

        Args:
            technique_name: Name of the technique

        Returns:
            List of parameter prefixes for this technique
        """
        # Standard naming convention: technique_name + "_"
        return [f"{technique_name}_"]

    def _detect_active_techniques_from_params(self, technique_list: List[str], params: Dict[str, Any]) -> List[str]:
        """
        Detect which techniques are active based on the presence of their parameters.

        Args:
            technique_list: List of possible techniques
            params: Sampled parameters

        Returns:
            List of active techniques
        """
        active = []
        param_keys = set(params.keys())

        for technique in technique_list:
            prefixes = self._get_technique_prefixes(technique)
            if any(any(key.startswith(prefix) for key in param_keys) for prefix in prefixes):
                active.append(technique)

        return active

    def get_active_techniques(self, params: Dict[str, Any], category: str = "augmentation") -> Dict[str, List[str]]:
        """
        Get list of active techniques based on sampled parameters for a specific category.
        Dynamically determines techniques from config structure.

        Args:
            params: Sampled parameters
            category: Category name (e.g., 'augmentation', 'label_mixing', 'data_sampling')

        Returns:
            Dictionary of active techniques by sub-category
        """
        # Build technique registry from config
        registry = self._build_technique_registry()

        if category not in registry:
            return {}

        active_techniques = {}
        category_registry = registry[category]

        # Initialize all sub-categories with empty lists
        for sub_category in category_registry.keys():
            active_techniques[sub_category] = []

        # Process each sub-category
        for sub_category, techniques in category_registry.items():
            # Check if there's a strategy parameter for this sub-category
            strategy_param_name = f"{sub_category}_strategy"
            if sub_category.endswith("_augmentation"):
                # Handle naming patterns like "spatial_augmentation" -> "spatial_augmentation_strategy"
                strategy_param_name = f"{sub_category}_strategy"
            elif sub_category == "mixing_techniques":
                # Handle label mixing naming
                strategy_param_name = "label_mixing_technique"
            elif sub_category == "techniques":
                # Generic technique selection
                strategy_param_name = f"{category}_technique"
            elif sub_category == "direct":
                # Direct techniques (always active if present)
                active_techniques[sub_category] = techniques
                continue

            strategy_value = params.get(strategy_param_name, "none")

            if strategy_value != "none":
                if strategy_value == "random_choice":
                    # Detect active techniques from parameters
                    active_techniques[sub_category] = self._detect_active_techniques_from_params(techniques, params)
                elif strategy_value in techniques:
                    active_techniques[sub_category] = [strategy_value]

        return active_techniques

    def get_all_active_techniques(self, all_params: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
        """
        Get active techniques for all categories.

        Args:
            all_params: Dictionary of all sampled parameters organized by category

        Returns:
            Dictionary of active techniques organized by category and sub-category
        """
        all_active = {}

        for category, params in all_params.items():
            if category in self.search_space_categories:
                all_active[category] = self.get_active_techniques(params, category)

        return all_active


# Example usage and testing
if __name__ == "__main__":
    try:
        # Example of how to use the sampler
        sampler = AugmentationSampler()
        print("Sampler initialized successfully")

        print(
            "Augmentation config keys:", list(sampler.augmentation_config.keys()) if sampler.augmentation_config else []
        )
        print(
            "Label mixing config keys:", list(sampler.label_mixing_config.keys()) if sampler.label_mixing_config else []
        )
        print(
            "Data sampling config keys:",
            list(sampler.data_sampling_config.keys()) if sampler.data_sampling_config else [],
        )
        print("Available search space categories:", sampler.search_space_categories)

        def objective(trial):
            try:
                # Sample all parameters
                params = sampler.sample_all_params(trial)

                # Get active techniques for all categories
                all_active_techniques = sampler.get_all_active_techniques(params)

                print(f"Trial {trial.number}:")
                for category, category_params in params.items():
                    print(f"  {category.capitalize()} params: {category_params}")
                print(f"  All active techniques: {all_active_techniques}")

                # Return a dummy objective value (replace with actual training result)
                return trial.suggest_float("dummy_objective", 0.0, 1.0)
            except Exception as e:
                print(f"Error in trial {trial.number}: {e}")
                import traceback

                traceback.print_exc()
                raise

        # Create study and run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=3)

        print(f"Best trial: {study.best_trial.number}")
        print(f"Best params: {study.best_params}")

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback

        traceback.print_exc()
