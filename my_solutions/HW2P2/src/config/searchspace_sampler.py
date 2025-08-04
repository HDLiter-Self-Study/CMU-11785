"""
Augmentation configuration parser and Optuna sampler
"""

import optuna
from typing import Dict, Any, List, Optional, Union
from omegaconf import DictConfig
from config_manager import get_config


class SearchSpaceSampler:
    """
    General search space parser and Optuna sampler for all categories.
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

        # Dynamically discover all search space categories
        self.search_space_categories = [k for k in self.search_spaces.keys() if hasattr(self.search_spaces, k)]

    def _evaluate_condition(self, condition: str, sampled_params: Dict[str, Any]) -> bool:
        """
        Evaluate a condition string using sampled parameters

        Args:
            condition: Condition string (e.g., "$spatial_augmentation_strategy in ['random_erasing', 'random_choice']")
            sampled_params: Dictionary of sampled parameters

        Returns:
            Boolean result of condition evaluation
        """
        if not condition or not condition.startswith("$"):
            return True

        # Remove $ prefix
        condition_eval = condition[1:]

        # Create a safe evaluation environment with parameter values
        eval_env = dict(sampled_params)
        try:
            # Evaluate the condition in the safe environment
            return bool(eval(condition_eval, {"__builtins__": {}}, eval_env))
        except Exception as e:
            print(f"Warning: Could not evaluate condition '{condition}': {e}")
            return False

    def _get_choices_for_strategy(self, choices_dict: Dict[str, List[Any]], strategy_value: str) -> List[Any]:
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
            min_val, max_val = param_config.min, param_config.max

            # 动态表达式支持
            if isinstance(min_val, str) and min_val.startswith("$"):
                min_val = eval(min_val[1:], {"__builtins__": {}}, sampled_params)
            if isinstance(max_val, str) and max_val.startswith("$"):
                max_val = eval(max_val[1:], {"__builtins__": {}}, sampled_params)
            return trial.suggest_float(param_name, min_val, max_val)

        elif param_type == "int":
            return trial.suggest_int(param_name, param_config.min, param_config.max)

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
        all_params = self._sample_strategy(trial, main_strategy, category_name, sampled_params)
        all_params.update(sampled_params)

        return all_params

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


# Example usage and testing
if __name__ == "__main__":
    try:
        # Example of how to use the sampler
        sampler = SearchSpaceSampler()
        print("Sampler initialized successfully")

        print("Available search space categories:", sampler.search_space_categories)

        def objective(trial):
            try:
                # Sample all parameters
                params = sampler.sample_all_params(trial)

                print(f"Trial {trial.number}:")
                for category, category_params in params.items():
                    print(f"  {category.capitalize()} params: {category_params}")

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
