"""
Verification Optimizer
Specialized optimizer for face verification tasks
"""

from typing import Optional
from .base_optimizer import BaseOptimizer, run_optimization_with_adapter
from adapters import VerificationAdapter
from config import get_config


class VerificationOptimizer(BaseOptimizer):
    """
    Verification-specific optimizer
    Handles face verification task optimization with optional pretrained models
    """

    def __init__(
        self,
        config_name: str = "verification",
        config_overrides: Optional[list] = None,
        pretrained_model_path: str = None,
        param_limit: int = None,
        wandb_project: str = None,
        study_name: str = None,
        n_trials: int = None,
    ):
        """
        Initialize verification optimizer

        Args:
            config_name: Name of the main config file to load
            config_overrides: List of configuration overrides
            pretrained_model_path: Path to pretrained classification model
            param_limit: Maximum number of parameters allowed (overrides config)
            wandb_project: WandB project name (overrides config)
            study_name: Optuna study name (overrides config)
            n_trials: Number of trials to run (overrides config)
        """
        # Load config to pass to adapter
        config = get_config(config_name, config_overrides or [])

        # Create verification adapter with optional pretrained model
        verification_adapter = VerificationAdapter(config, pretrained_model_path)

        # Initialize base optimizer with verification adapter
        super().__init__(
            task_adapter=verification_adapter,
            config_name=config_name,
            config_overrides=config_overrides,
            param_limit=param_limit,
            wandb_project=wandb_project,
            study_name=study_name,
            n_trials=n_trials,
        )

    @classmethod
    def from_classification_results(
        cls,
        classification_study_path: str,
        top_k: int = 3,
        config_name: str = "verification",
        config_overrides: Optional[list] = None,
        **kwargs,
    ):
        """
        Create verification optimizer from classification study results

        Args:
            classification_study_path: Path to classification study results JSON
            top_k: Number of top classification models to use for verification
            config_name: Verification config name
            config_overrides: Configuration overrides
            **kwargs: Additional arguments for VerificationOptimizer

        Returns:
            List of VerificationOptimizer instances for each top model
        """
        import json
        from pathlib import Path

        # Load classification results
        study_path = Path(classification_study_path)
        if not study_path.exists():
            raise FileNotFoundError(f"Classification study results not found: {classification_study_path}")

        with open(study_path, "r") as f:
            study_data = json.load(f)

        # Extract top performing models
        successful_configs = study_data.get("successful_configs", [])[:top_k]

        optimizers = []
        for i, config_data in enumerate(successful_configs):
            model_path = f"./checkpoints/classification_best_{i}.pth"  # Assume models are saved

            optimizer = cls(
                config_name=config_name,
                config_overrides=config_overrides,
                pretrained_model_path=model_path if Path(model_path).exists() else None,
                study_name=f"verification_from_cls_{i}",
                **kwargs,
            )
            optimizers.append(optimizer)

        return optimizers


# Convenience functions for verification optimization
def run_verification_optimization(
    config_name: str = "verification", config_overrides: Optional[list] = None, pretrained_model_path: str = None
):
    """
    Run verification optimization

    Args:
        config_name: Name of the configuration to use
        config_overrides: List of configuration overrides
        pretrained_model_path: Path to pretrained classification model

    Returns:
        Optuna study object
    """
    optimizer = VerificationOptimizer(
        config_name=config_name, config_overrides=config_overrides, pretrained_model_path=pretrained_model_path
    )

    study = optimizer.optimize()

    print("Verification optimization completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")

    return study


def run_verification_from_classification(classification_study_path: str, top_k: int = 3):
    """
    Run verification optimization using pretrained classification models

    Args:
        classification_study_path: Path to classification study results JSON
        top_k: Number of top models to use

    Returns:
        List of study results for each pretrained model
    """
    optimizers = VerificationOptimizer.from_classification_results(classification_study_path, top_k=top_k)

    studies = []
    for i, optimizer in enumerate(optimizers):
        print(f"\n🚀 Starting verification optimization {i+1}/{len(optimizers)}...")
        study = optimizer.optimize()
        studies.append(study)

    return studies


def run_quick_verification_optimization(pretrained_model_path: str = None):
    """Quick verification optimization using the quick configuration"""
    return run_verification_optimization("verification", ["optuna=quick"], pretrained_model_path)


def run_production_verification_optimization(pretrained_model_path: str = None):
    """Production verification optimization with full configuration"""
    return run_verification_optimization("verification", ["training=production"], pretrained_model_path)


# Alternative approach using the general function
def run_verification_with_adapter(
    config_name: str = "verification", config_overrides: Optional[list] = None, pretrained_model_path: str = None
):
    """
    Run verification optimization using the general adapter approach

    Args:
        config_name: Configuration name
        config_overrides: Configuration overrides
        pretrained_model_path: Path to pretrained model

    Returns:
        Optuna study object
    """
    config = get_config(config_name, config_overrides or [])
    verification_adapter = VerificationAdapter(config, pretrained_model_path)
    return run_optimization_with_adapter(verification_adapter, config_name, config_overrides)


if __name__ == "__main__":
    # Example usage
    print("🔍 Verification Optimizer Examples:")
    print()

    # 1. Run verification from scratch
    print("1. Run verification optimization from scratch:")
    print("   study = run_verification_optimization()")
    print()

    # 2. Run verification with pretrained model
    print("2. Run verification with pretrained classification model:")
    print("   study = run_verification_optimization(pretrained_model_path='./checkpoints/best_cls.pth')")
    print()

    # 3. Run verification from classification results
    print("3. Run verification from classification study results:")
    print("   studies = run_verification_from_classification('./optuna_results/classification_study_summary.json')")
    print()

    # Default: run with standard configuration
    # study = run_verification_optimization()
