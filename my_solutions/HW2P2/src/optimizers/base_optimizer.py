"""
Base Optimizer for Task-Agnostic Hyperparameter Optimization
Unified optimizer that works with different task adapters
"""

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path
import json

from config import get_config
from adapters import TaskAdapter
from models.parameter_calculator import ParameterCalculator
from training_functions import train_model, evaluate_model


class BaseOptimizer:
    """
    Unified optimizer with task adapter pattern

    This optimizer works with any task by using a TaskAdapter to handle
    task-specific logic while keeping the optimization flow generic.
    """

    def __init__(
        self,
        task_adapter: TaskAdapter,
        config_name: str = "main",
        config_overrides: Optional[list] = None,
        param_limit: int = None,
        wandb_project: str = None,
        study_name: str = None,
        n_trials: int = None,
    ):
        """
        Initialize the task-agnostic optimizer

        Args:
            task_adapter: Task-specific adapter implementing TaskAdapter interface
            config_name: Name of the main config file to load
            config_overrides: List of configuration overrides
            param_limit: Maximum number of parameters allowed (overrides config)
            wandb_project: WandB project name (overrides config)
            study_name: Optuna study name (overrides config)
            n_trials: Number of trials to run (overrides config)
        """
        # Store task adapter
        self.task_adapter = task_adapter

        # Load configuration
        self.config = get_config(config_name, config_overrides or [])

        # Set parameters from config with optional overrides
        self.param_limit = param_limit or self.config.optuna.param_limit
        self.wandb_project = wandb_project or self.config.wandb.wandb_project
        self.study_name = study_name or self.config.optuna.study_name
        self.n_trials = n_trials or self.config.optuna.n_trials

        # Add task name to study name for clarity
        self.study_name = f"{self.task_adapter.task_name}_{self.study_name}"

        # Initialize parameter calculator
        self.param_calc = ParameterCalculator()

        # Track successful configurations
        self.successful_configs = []

        # Setup WandB callback if needed
        self.wandb_callback = self.create_wandb_callback() if self.config.wandb.use_wandb else None

        # Setup objective function based on WandB configuration
        self.objective_func = self._setup_objective()

    def create_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Create unified search space combining base and task-specific parameters
        """
        # Level 1: Architecture type (from config)
        arch_choices = self.config.optuna.hyperparameters.architecture.choices
        arch_type = trial.suggest_categorical("architecture", arch_choices)

        # Level 2: Global structure parameters (from config)
        config_dict = {
            "architecture": arch_type,
            "num_classes": self.config.optuna.model.num_classes,  # From config
        }

        # Level 3: Architecture-specific parameters (config-driven)
        arch_params = self._suggest_architecture_params(trial, arch_type)
        config_dict.update(arch_params)

        # Level 4: Training hyperparameters (using config-defined search space)
        config_dict.update(self._suggest_training_hyperparameters(trial))

        # Level 5: Task-specific parameters (delegated to task adapter)
        task_params = self.task_adapter.create_task_search_space(trial)
        config_dict.update(task_params)

        return config_dict

    def _suggest_training_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest training hyperparameters based on configuration
        (Reused from original OptunaWandBOptimizer)
        """
        hyperparams = {}

        # Get hyperparameter definitions from config
        if not hasattr(self.config.optuna, "hyperparameters"):
            raise ValueError("Missing 'hyperparameters' configuration in optuna config")

        hp_config = self.config.optuna.hyperparameters

        for param_name, param_config in hp_config.items():
            # Skip architecture parameter as it's handled separately
            if param_name == "architecture":
                continue

            try:
                # For training hyperparameters, use param_name as both key and optuna name
                config_with_param_name = dict(param_config)
                config_with_param_name["param_name"] = param_name

                # Convert to object-like access for compatibility
                class ConfigObj:
                    def __init__(self, config_dict):
                        for k, v in config_dict.items():
                            setattr(self, k, v)

                    def get(self, key, default=None):
                        return getattr(self, key, default)

                config_obj = ConfigObj(config_with_param_name)
                hyperparams[param_name] = self._suggest_param_from_config(trial, param_name, config_obj)

            except ValueError as e:
                raise ValueError(f"Invalid training hyperparameter config for '{param_name}': {e}")

        return hyperparams

    def _suggest_param_from_config(self, trial: optuna.Trial, param_name: str, param_config: Any) -> Any:
        """
        Suggest a single parameter based on its configuration
        (Reused from original OptunaWandBOptimizer)
        """
        if not hasattr(param_config, "type"):
            raise ValueError(f"Parameter '{param_name}' missing required 'type' field")

        if not hasattr(param_config, "param_name"):
            raise ValueError(f"Parameter '{param_name}' missing required 'param_name' field")

        optuna_param_name = param_config.param_name
        param_type = param_config.type

        if param_type == "int":
            if not hasattr(param_config, "min") or not hasattr(param_config, "max"):
                raise ValueError(f"Integer parameter '{param_name}' missing required 'min' or 'max' field")
            return trial.suggest_int(
                optuna_param_name, param_config.min, param_config.max, log=param_config.get("log", False)
            )
        elif param_type == "float":
            if not hasattr(param_config, "min") or not hasattr(param_config, "max"):
                raise ValueError(f"Float parameter '{param_name}' missing required 'min' or 'max' field")
            return trial.suggest_float(
                optuna_param_name, param_config.min, param_config.max, log=param_config.get("log", False)
            )
        elif param_type == "categorical":
            if not hasattr(param_config, "choices"):
                raise ValueError(f"Categorical parameter '{param_name}' missing required 'choices' field")
            return trial.suggest_categorical(optuna_param_name, param_config.choices)
        else:
            raise ValueError(f"Unsupported parameter type '{param_type}' for parameter '{param_name}'")

    def _suggest_architecture_params(self, trial: optuna.Trial, arch_type: str) -> Dict[str, Any]:
        """
        Suggest architecture-specific parameters based on configuration
        (Reused from original OptunaWandBOptimizer)
        """
        # Validate configuration exists
        if not hasattr(self.config.optuna, "architectures"):
            raise ValueError("Missing 'architectures' configuration in optuna config")

        if arch_type not in self.config.optuna.architectures:
            available_archs = list(self.config.optuna.architectures.keys())
            raise ValueError(
                f"Architecture '{arch_type}' not found in config. " f"Available architectures: {available_archs}"
            )

        arch_config = self.config.optuna.architectures[arch_type]
        params = {}

        # Process structure parameters
        if hasattr(arch_config, "structure_params"):
            for param_name, param_config in arch_config.structure_params.items():
                try:
                    params[param_name] = self._suggest_param_from_config(trial, param_name, param_config)
                except ValueError as e:
                    raise ValueError(f"Invalid structure parameter config for '{arch_type}.{param_name}': {e}")

        # Process block parameters
        if hasattr(arch_config, "block_params"):
            for param_name, param_config in arch_config.block_params.items():
                try:
                    params[param_name] = self._suggest_param_from_config(trial, param_name, param_config)
                except ValueError as e:
                    raise ValueError(f"Invalid block parameter config for '{arch_type}.{param_name}': {e}")

        print(f"✅ Generated {len(params)} parameters for {arch_type} from config: {list(params.keys())}")
        return params

    def _setup_objective(self):
        """
        Setup objective function based on WandB configuration
        """
        if self.config.wandb.use_wandb and self.wandb_callback:
            return self.wandb_callback.track_in_wandb()(self._core_objective)
        else:
            return self._core_objective

    def _core_objective(self, trial: optuna.Trial) -> float:
        """
        Core objective function using task adapter
        """
        try:
            # Create configuration from search space
            model_config = self.create_search_space(trial)

            # Phase 1: Quick parameter estimation
            estimated_params = self.param_calc.estimate_params(model_config)

            # Early pruning if clearly over limit
            if estimated_params > self.param_limit * 1.2:
                trial.report(0.0, step=0)
                raise optuna.TrialPruned()

            # Phase 2: Build model using task adapter
            model = self.task_adapter.create_model(model_config)
            actual_params = sum(p.numel() for p in model.parameters())

            # Set trial attributes for Optuna tracking
            trial.set_user_attr("parameter_count", actual_params)
            trial.set_user_attr("estimated_params", estimated_params)
            trial.set_user_attr("param_estimation_error", abs(actual_params - estimated_params))
            trial.set_user_attr("architecture_type", model_config["architecture"])
            trial.set_user_attr("task", self.task_adapter.task_name)

            # Update WandB run configuration with detailed info
            if wandb.run and self.config.wandb.use_wandb:
                self._update_wandb_config(model_config, actual_params, trial)

            # Handle parameter constraint
            if actual_params > self.param_limit:
                return self._handle_over_limit_case(model, model_config, trial, actual_params)
            else:
                return self._handle_within_limit_case(model, model_config, trial, actual_params)

        except Exception as e:
            # Log error to both Optuna and WandB
            trial.set_user_attr("error", str(e))
            if wandb.run and self.config.wandb.use_wandb:
                wandb.log({"error": str(e), "failed": True})
            raise optuna.TrialPruned()

    def _update_wandb_config(self, model_config: Dict[str, Any], actual_params: int, trial: optuna.Trial):
        """Update WandB configuration with model and task info"""
        arch = model_config["architecture"]
        lr = model_config.get("learning_rate", "default")
        batch_size = model_config.get("batch_size", "default")

        wandb.run.name = f"{self.task_adapter.task_name}-{arch}-lr{lr:.1e}-bs{batch_size}-params{actual_params//1000}k"

        # Base config update
        base_config = {
            "task": self.task_adapter.task_name,
            "architecture": arch,
            "num_classes": model_config["num_classes"],
            "parameter_count": actual_params,
            "param_limit": self.param_limit,
            "model_config": model_config,
        }

        # Get task-specific config updates
        task_config = {}
        if hasattr(self.task_adapter, "get_wandb_config_update"):
            task_config = self.task_adapter.get_wandb_config_update(model_config, {})

        wandb.run.config.update({**base_config, **task_config})

    def _handle_over_limit_case(
        self, model: nn.Module, model_config: Dict[str, Any], trial: optuna.Trial, actual_params: int
    ) -> float:
        """Handle case where model exceeds parameter limit"""
        penalty_factor = min((actual_params - self.param_limit) / self.param_limit, 0.5)
        trial.set_user_attr("penalty_applied", penalty_factor)

        if wandb.run and self.config.wandb.use_wandb:
            wandb.run.config.update({"over_param_limit": True, "penalty_factor": penalty_factor})

        # Train and evaluate using task adapter
        base_score = self._train_and_evaluate_with_adapter(model, model_config, trial)
        final_score = base_score * (1 - penalty_factor)

        trial.set_user_attr("base_score", base_score)
        trial.set_user_attr("final_score", final_score)

        if wandb.run and self.config.wandb.use_wandb:
            wandb.log({"base_score": base_score, "penalty_factor": penalty_factor, "final_score": final_score})

        return final_score

    def _handle_within_limit_case(
        self, model: nn.Module, model_config: Dict[str, Any], trial: optuna.Trial, actual_params: int
    ) -> float:
        """Handle case where model is within parameter limit"""
        trial.set_user_attr("penalty_applied", 0.0)

        if wandb.run and self.config.wandb.use_wandb:
            wandb.run.config.update({"over_param_limit": False})

        # Train and evaluate using task adapter
        score = self._train_and_evaluate_with_adapter(model, model_config, trial)

        trial.set_user_attr("base_score", score)
        trial.set_user_attr("final_score", score)

        # Track successful configuration
        self.successful_configs.append(
            {"config": model_config, "params": actual_params, "score": score, "task": self.task_adapter.task_name}
        )

        return score

    def _train_and_evaluate_with_adapter(self, model: nn.Module, config: Dict[str, Any], trial: optuna.Trial) -> float:
        """
        Train and evaluate model using task adapter
        This replaces the original _train_and_evaluate methods
        """
        # Create task-specific dataloaders
        train_dataloader, val_dataloader = self.task_adapter.create_dataloaders(config)

        # Setup training configuration
        train_config = {
            **config,
            "epochs": 10,  # Reduced for hyperparameter search
            "early_stopping_patience": 3,
        }

        best_score = 0.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))

        # Training loop with task adapter
        for epoch in range(train_config["epochs"]):
            # Training phase
            model.train()
            train_loss = 0.0
            train_metrics = {"accuracy": 0.0, "total": 0}

            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                # Use task adapter for loss computation
                loss = self.task_adapter.compute_loss(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Use task adapter for metrics computation
                batch_metrics = self.task_adapter.compute_metrics(outputs, targets)
                train_metrics["accuracy"] += batch_metrics["accuracy"] * batch_metrics["total"]
                train_metrics["total"] += batch_metrics["total"]

            # Average training metrics
            train_loss /= len(train_dataloader)
            train_accuracy = train_metrics["accuracy"] / train_metrics["total"] if train_metrics["total"] > 0 else 0.0

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_metrics = {"accuracy": 0.0, "total": 0}

            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)

                    # Use task adapter for loss and metrics
                    loss = self.task_adapter.compute_loss(outputs, targets)
                    val_loss += loss.item()

                    batch_metrics = self.task_adapter.compute_metrics(outputs, targets)
                    val_metrics["accuracy"] += batch_metrics["accuracy"] * batch_metrics["total"]
                    val_metrics["total"] += batch_metrics["total"]

            # Average validation metrics
            val_loss /= len(val_dataloader)
            val_accuracy = val_metrics["accuracy"] / val_metrics["total"] if val_metrics["total"] > 0 else 0.0

            # Report intermediate values for pruning
            trial.report(val_accuracy, epoch)

            # Update best score
            if val_accuracy > best_score:
                best_score = val_accuracy

            # WandB logging
            if wandb.run and self.config.wandb.use_wandb:
                epoch_metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "best_val_accuracy": best_score,
                }
                wandb.log(epoch_metrics)

            # Check if trial should be pruned
            if trial.should_prune():
                if wandb.run and self.config.wandb.use_wandb:
                    wandb.log({"pruned": True, "pruned_at_epoch": epoch})
                raise optuna.TrialPruned()

            # Log intermediate metrics as user attributes
            trial.set_user_attr(f"train_loss_epoch_{epoch}", train_loss)
            trial.set_user_attr(f"train_acc_epoch_{epoch}", train_accuracy)
            trial.set_user_attr(f"val_loss_epoch_{epoch}", val_loss)
            trial.set_user_attr(f"val_acc_epoch_{epoch}", val_accuracy)

        # Log final summary to WandB
        if wandb.run and self.config.wandb.use_wandb:
            wandb.log(
                {
                    "final_val_accuracy": best_score,
                    "completed_epochs": train_config["epochs"],
                    "training_completed": True,
                }
            )

        return best_score

    def create_wandb_callback(self) -> WeightsAndBiasesCallback:
        """Create WandB callback for Optuna integration"""
        wandb_mode = "online" if self.config.wandb.wandb_online else "offline"

        # Get task-specific tags
        task_tags = []
        if hasattr(self.task_adapter, "get_wandb_tags"):
            task_tags = self.task_adapter.get_wandb_tags()

        base_tags = ["optuna", "nas", "hyperparameter_search"]
        all_tags = base_tags + task_tags

        return WeightsAndBiasesCallback(
            metric_name=f"validation_{self.task_adapter.primary_metric}",
            wandb_kwargs={
                "project": self.wandb_project,
                "group": self.study_name,
                "tags": all_tags,
                "mode": wandb_mode,
                "config": {
                    "task": self.task_adapter.task_name,
                    "param_limit": self.param_limit,
                    "n_trials": self.n_trials,
                    "study_name": self.study_name,
                },
            },
            as_multirun=True,
        )

    def optimize(self, storage_url: Optional[str] = None) -> optuna.Study:
        """Run the optimization process"""
        # Create study
        if storage_url:
            study = optuna.create_study(
                study_name=self.study_name,
                direction="maximize",
                storage=storage_url,
                load_if_exists=True,
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1),
                sampler=optuna.samplers.TPESampler(seed=42),
            )
        else:
            study = optuna.create_study(
                study_name=self.study_name,
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1),
                sampler=optuna.samplers.TPESampler(seed=42),
            )

        # Add custom callback
        def custom_callback(study, trial):
            if trial.state.name == "COMPLETE":
                self._log_trial_summary(trial)
            elif trial.state.name == "PRUNED":
                print(f"❌ Trial {trial.number}: pruned")

        print(f"🚀 Starting {self.task_adapter.task_name} optimization with {self.n_trials} trials...")
        print(f"   Study name: {self.study_name}")
        print(f"   Parameter limit: {self.param_limit:,}")
        print(f"   WandB project: {self.wandb_project}")
        print(f"   WandB enabled: {self.config.wandb.use_wandb}")

        # Prepare callbacks
        callbacks = [custom_callback]
        if self.wandb_callback:
            callbacks.append(self.wandb_callback)

        # Run optimization
        study.optimize(
            self.objective_func,
            n_trials=self.n_trials,
            callbacks=callbacks,
            show_progress_bar=True,
        )

        # Save results
        self._save_optimization_results(study)

        print(f"🎉 {self.task_adapter.task_name} optimization completed!")
        return study

    def _log_trial_summary(self, trial: optuna.Trial):
        """Log enhanced summary of completed trial"""
        arch = trial.params.get("architecture", "unknown")
        param_count = trial.user_attrs.get("parameter_count", 0)
        penalty = trial.user_attrs.get("penalty_applied", 0.0)
        task = trial.user_attrs.get("task", "unknown")

        # Create status emoji
        status = "⚠️" if penalty > 0 else "✅"

        print(f"{status} [{task}] Trial {trial.number}: {trial.value:.4f}")
        print(f"   Architecture: {arch}")
        print(f"   Parameters: {param_count:,}")

        if penalty > 0:
            base_score = trial.user_attrs.get("base_score", 0.0)
            print(f"   Base score: {base_score:.4f} (before penalty)")
            print(f"   Penalty: {penalty:.3f}")

        print()  # Empty line for readability

    def _save_optimization_results(self, study: optuna.Study):
        """Save optimization results to file"""
        results_dir = Path("./optuna_results")
        results_dir.mkdir(exist_ok=True)

        # Save study summary
        study_summary = {
            "task": self.task_adapter.task_name,
            "best_trial": {
                "number": study.best_trial.number,
                "value": study.best_trial.value,
                "params": study.best_trial.params,
                "user_attrs": study.best_trial.user_attrs,
            },
            "n_trials": len(study.trials),
            "successful_configs": self.successful_configs[:10],
        }

        with open(results_dir / f"{self.study_name}_summary.json", "w") as f:
            json.dump(study_summary, f, indent=2)

        # Save detailed results
        trials_df = study.trials_dataframe()
        trials_df.to_csv(results_dir / f"{self.study_name}_trials.csv", index=False)

        print(f"Results saved to {results_dir}")


# Convenience functions for different tasks
def run_optimization_with_adapter(
    task_adapter: TaskAdapter, config_name: str = "main", config_overrides: Optional[list] = None
):
    """
    Run optimization with a specific task adapter

    Args:
        task_adapter: Task adapter instance
        config_name: Configuration name
        config_overrides: Configuration overrides

    Returns:
        Optuna study object
    """
    optimizer = BaseOptimizer(task_adapter=task_adapter, config_name=config_name, config_overrides=config_overrides)
    return optimizer.optimize()
