"""
Classification Optimizer
Backward-compatible wrapper for classification tasks
"""

from typing import Optional
from .base_optimizer import BaseOptimizer, run_optimization_with_adapter
from adapters import ClassificationAdapter
from config import get_config


class ClassificationOptimizer(BaseOptimizer):
    """
    Classification-specific optimizer
    Provides backward compatibility with the original OptunaWandBOptimizer
    """

    def __init__(
        self,
        config_name: str = "main",
        config_overrides: Optional[list] = None,
        param_limit: int = None,
        wandb_project: str = None,
        study_name: str = None,
        n_trials: int = None,
    ):
        """
        Initialize classification optimizer

        Args:
            config_name: Name of the main config file to load
            config_overrides: List of configuration overrides
            param_limit: Maximum number of parameters allowed (overrides config)
            wandb_project: WandB project name (overrides config)
            study_name: Optuna study name (overrides config)
            n_trials: Number of trials to run (overrides config)
        """
        # Load config to pass to adapter
        config = get_config(config_name, config_overrides or [])

        # Create classification adapter
        classification_adapter = ClassificationAdapter(config)

        # Initialize base optimizer with classification adapter
        super().__init__(
            task_adapter=classification_adapter,
            config_name=config_name,
            config_overrides=config_overrides,
            param_limit=param_limit,
            wandb_project=wandb_project,
            study_name=study_name,
            n_trials=n_trials,
        )


# Backward compatibility functions
def run_classification_optimization(config_name: str = "main", config_overrides: Optional[list] = None):
    """
    Run classification optimization (backward compatible)

    Args:
        config_name: Name of the configuration to use
        config_overrides: List of configuration overrides

    Returns:
        Optuna study object
    """
    optimizer = ClassificationOptimizer(config_name=config_name, config_overrides=config_overrides)
    study = optimizer.optimize()

    print("Classification optimization completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")

    return study


def run_quick_classification_optimization():
    """Quick classification optimization using the quick configuration"""
    return run_classification_optimization("main", ["optuna=quick"])


def run_production_classification_optimization():
    """Production classification optimization with full configuration"""
    return run_classification_optimization("main", ["training=production"])


# Alternative approach using the general function
def run_classification_with_adapter(config_name: str = "main", config_overrides: Optional[list] = None):
    """
    Run classification optimization using the general adapter approach

    Args:
        config_name: Configuration name
        config_overrides: Configuration overrides

    Returns:
        Optuna study object
    """
    config = get_config(config_name, config_overrides or [])
    classification_adapter = ClassificationAdapter(config)
    return run_optimization_with_adapter(classification_adapter, config_name, config_overrides)


if __name__ == "__main__":
    # Default: run with standard configuration
    study = run_classification_optimization()

    # Example of running with different configurations:
    # study = run_quick_classification_optimization()  # For quick testing
    # study = run_production_classification_optimization()  # For production runs
