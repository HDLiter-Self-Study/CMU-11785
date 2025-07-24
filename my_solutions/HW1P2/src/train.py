import os
import json
import shutil
import pprint

import wandb
import torch
import torch.nn as nn
import tqdm
from torchsummaryX import summary
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from data.datasets import AudioDataset, AudioTestDataset
from models.sweep_model import SweepModel

PROJECT_NAME = "CMU_11785_HW1P2"
# Set to True to run the model on the AutoDL server (cannot use wandb, ubuntu, stronger GPU, etc.)
AUTO_DL = os.path.exists("/root/autodl-tmp")
# The total run count for the sweep
TOTAL_RUNS = 2
# When using the sweep, there is a chance that the same configuration is encountered multiple times, even when method is set to "grid".
# We don't want it to happen, so we keep track of the encountered runs.
# TODO: Figure out a way to avoid this in wandb, is it a bug or am I missing something?
ENCOUNTERED_RUNS = set()


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance.
    Paper: "Focal Loss for Dense Object Detection" by Lin et al.
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        """
        :param alpha: Weighting factor for rare class (default: 1.0)
        :param gamma: Focusing parameter to down-weight easy examples (default: 2.0)
        :param reduction: Specifies the reduction to apply to the output
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: Logits from the model [N, C]
        :param targets: Ground truth labels [N]
        """
        # Compute cross entropy loss
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")

        # Compute probabilities
        pt = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ConfusionAwareLossCriterion:
    """
    Loss criterion that handles both main output and confusion-aware heads.
    """

    def __init__(
        self,
        use_focal=False,
        focal_alpha=1.0,
        focal_gamma=2.0,
        label_smoothing=0.0,
        confusion_weight=0.3,
        phonemes_to_idx=None,
    ):
        self.use_focal = use_focal
        self.label_smoothing = label_smoothing
        self.confusion_weight = confusion_weight
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

        if use_focal:
            self.hard_label_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.hard_label_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Setup confusion pair mappings
        self.phonemes_to_idx = {}
        self.confusion_mappings = {}
        if phonemes_to_idx is not None:
            self.phonemes_to_idx = phonemes_to_idx
            self._setup_confusion_mappings()

    def _setup_confusion_mappings(self):
        """Setup mappings for confusion pairs."""
        from models.sweep_model import CONFUSION_PAIRS

        for confusion_pair, head_name in CONFUSION_PAIRS:
            confusion_indices = []
            for phoneme in confusion_pair:
                if phoneme in self.phonemes_to_idx:
                    confusion_indices.append(self.phonemes_to_idx[phoneme])

            if len(confusion_indices) > 1:
                self.confusion_mappings[head_name] = confusion_indices

    def __call__(self, outputs, original_labels):
        """
        Compute loss for main output and confusion heads.

        Args:
            outputs: Can be:
                - main_logits (tensor) for eval mode
                - (main_logits, mixed_labels) for mixup training
                - (main_logits, confusion_outputs, mixed_labels) for confusion heads training
            original_labels: Original target labels (before any mixup)
        """
        # Handle different output formats
        if isinstance(outputs, tuple):
            if len(outputs) == 2:
                # (main_logits, mixed_labels) - mixup case
                main_logits, labels = outputs
                confusion_outputs = None
            elif len(outputs) == 3:
                # (main_logits, confusion_outputs, mixed_labels) - confusion heads case
                main_logits, confusion_outputs, labels = outputs
            else:
                raise ValueError(f"Unexpected output format: {len(outputs)} elements")
        else:
            # Simple case - just main logits (eval mode)
            main_logits = outputs
            confusion_outputs = None
            labels = original_labels

        # Compute main loss
        if labels.dim() > 1 and labels.size(1) > 1:  # Soft labels (MixUp)
            log_probs = torch.nn.functional.log_softmax(main_logits, dim=1)
            main_loss = self.kl_div_loss(log_probs, labels)
        else:  # Hard labels
            main_loss = self.hard_label_loss(main_logits, labels)

        total_loss = main_loss

        # Compute confusion head losses if available
        # Use original_labels for confusion heads (not mixed labels)
        if confusion_outputs is not None and original_labels.dim() == 1:  # Only for hard labels
            confusion_loss = 0.0
            num_heads = 0

            for head_name, confusion_logits in confusion_outputs.items():
                if head_name in self.confusion_mappings:
                    confusion_indices = self.confusion_mappings[head_name]

                    # Create mask for samples that have labels in this confusion pair
                    mask = torch.zeros(original_labels.size(0), dtype=torch.bool, device=original_labels.device)
                    for idx in confusion_indices:
                        mask |= original_labels == idx

                    if mask.sum() > 0:
                        # Get the subset of data that matches this confusion pair
                        masked_logits = confusion_logits[mask]
                        masked_labels = original_labels[mask]

                        # Remap labels to confusion head indices (0, 1, 2, ...)
                        remapped_labels = torch.zeros_like(masked_labels)
                        for i, global_idx in enumerate(confusion_indices):
                            remapped_labels[masked_labels == global_idx] = i

                        # Compute loss for this confusion head
                        head_loss = self.hard_label_loss(masked_logits, remapped_labels)
                        confusion_loss += head_loss
                        num_heads += 1

            if num_heads > 0:
                confusion_loss /= num_heads
                total_loss = (1 - self.confusion_weight) * main_loss + self.confusion_weight * confusion_loss

        return total_loss


class FlexibleLossCriterion:
    """
    Custom criterion that supports both MixUp and Focal Loss.
    """

    def __init__(self, use_focal=False, focal_alpha=1.0, focal_gamma=2.0, label_smoothing=0.0):
        self.use_focal = use_focal
        self.label_smoothing = label_smoothing
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

        if use_focal:
            self.hard_label_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.hard_label_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def __call__(self, outputs, original_labels):
        """
        Handle different output formats from the model.

        Args:
            outputs: Can be:
                - main_logits (tensor) for eval mode
                - (main_logits, mixed_labels) for mixup training
                - (main_logits, confusion_outputs, mixed_labels) for confusion heads training
            original_labels: Original target labels (before any mixup)
        """
        # Handle different output formats
        if isinstance(outputs, tuple):
            if len(outputs) == 2:
                # (main_logits, mixed_labels) - mixup case
                logits, labels = outputs
            elif len(outputs) == 3:
                # (main_logits, confusion_outputs, mixed_labels) - shouldn't happen with FlexibleLossCriterion
                # but handle it gracefully
                logits, _, labels = outputs
            else:
                raise ValueError(f"Unexpected output format: {len(outputs)} elements")
        else:
            # Simple case - just main logits (eval mode)
            logits = outputs
            labels = original_labels

        if labels.dim() > 1 and labels.size(1) > 1:  # Soft labels (MixUp)
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            return self.kl_div_loss(log_probs, labels)
        else:  # Hard labels
            return self.hard_label_loss(logits, labels)


def get_optimizer(model, learning_rate=0.001, weight_decay=1e-5, name="adamw"):
    """
    Returns an optimizer for the model.
    :param model: The model to optimize.
    :param learning_rate: Learning rate for the optimizer.
    :param weight_decay: Weight decay for the optimizer.
    :param name: Name of the optimizer to use.
    :return: An optimizer instance.
    """
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif name == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(
            f"Unsupported optimizer name: {name}. Supported optimizers are: adamw, adam, sgd, rmsprop, momentum."
        )


def get_scheduler(optimizer, name="plateau", **kwargs):
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.25, patience=2, threshold=0.025
        )
    elif name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif name == "cosine_annealing":
        learning_rate = kwargs["learning_rate"]
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=learning_rate * 0.1)
    else:
        raise ValueError(
            f"Unsupported scheduler name: {name}. Supported schedulers are: plateau, exponential, cosine_annealing."
        )


def train_model(model, train_loader, criterion, optimizer, device, scaler):

    model.train()
    t_loss, t_acc = 0, 0
    pbar = tqdm.tqdm(train_loader, desc="Training", unit="batch")
    for i, batch in enumerate(pbar):
        # Unpack the batch
        frames, phonemes = batch
        frames, phonemes = frames.to(device), phonemes.to(device)
        original_phonemes = phonemes.clone()  # Keep original labels for loss computation
        # Forward pass
        optimizer.zero_grad(set_to_none=True)  # Zero the gradients
        # Runs the forward pass under autocast for mixed precision training
        # This will use float16 precision for the forward pass
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(frames, phonemes)  # Returns different formats based on model configuration
            loss = criterion(outputs, original_phonemes)  # Pass original labels to criterion
        # Backward pass
        # Scales the loss for mixed precision training
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Calculate accuracy and loss
        t_loss += loss.item()

        # Extract main logits for accuracy calculation
        if isinstance(outputs, tuple):
            if len(outputs) == 2:
                # (main_logits, labels) - mixup case without confusion heads
                main_logits, phonemes = outputs
            elif len(outputs) == 3:
                # (main_logits, confusion_outputs, labels) - with confusion heads
                main_logits, confusion_outputs, phonemes = outputs
            else:
                raise ValueError(f"Unexpected output format: {len(outputs)} elements")
        else:
            # Simple case - just main logits (shouldn't happen in training)
            main_logits = outputs

        predictions = torch.argmax(main_logits, dim=1)

        # Handle mixup labels for accuracy calculation
        if phonemes.dim() > 1 and phonemes.size(1) > 1:  # Mixup labels (soft labels)
            # For mixup, get the hard labels by taking argmax of the mixed labels
            hard_labels = torch.argmax(phonemes, dim=1)
            correct = (predictions == hard_labels).sum().item()
        else:  # Regular hard labels
            correct = (predictions == phonemes).sum().item()

        batch_acc = correct / phonemes.shape[0]
        t_acc += batch_acc

        # Update the progress bar
        pbar.set_postfix(loss="{:.4f}".format(t_loss / (i + 1)), accuracy="{:.4f}%".format(t_acc * 100 / (i + 1)))
        pbar.update()

        # release memory
        del frames, phonemes, original_phonemes, outputs, loss, predictions
        if "main_logits" in locals():
            del main_logits
        if "confusion_outputs" in locals():
            del confusion_outputs
        torch.cuda.empty_cache()

    pbar.close()
    t_loss /= len(train_loader)
    t_acc /= len(train_loader)

    return t_loss, t_acc


def evaluate_model(model, eval_loader, criterion, device, phonemes=None, error_analysis=False):
    """Evaluate model and optionally perform error analysis."""
    model.eval()
    v_loss, v_acc = 0, 0
    all_predictions = []
    all_targets = []

    pbar = tqdm.tqdm(eval_loader, desc="Evaluating", unit="batch")
    # Disable gradient calculation for evaluation
    with torch.inference_mode():
        for i, batch in enumerate(pbar):
            frames, phonemes_batch = batch
            frames, phonemes_batch = frames.to(device), phonemes_batch.to(device)

            logits = model(frames)
            # Handle different output formats from model
            if isinstance(logits, tuple):
                # This shouldn't happen in eval mode with current implementation,
                # but handle it just in case
                logits = logits[0]  # Take the main logits

            loss = criterion(logits, phonemes_batch)

            v_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == phonemes_batch).sum().item()
            batch_acc = correct / phonemes_batch.shape[0]
            v_acc += batch_acc

            # Store predictions and targets for error analysis
            if error_analysis:
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(phonemes_batch.cpu().numpy())

            pbar.set_postfix(loss="{:.4f}".format(v_loss / (i + 1)), accuracy="{:.4f}%".format(v_acc * 100 / (i + 1)))
            pbar.update()

            # release memory
            del frames, phonemes_batch, logits, loss, predictions
            torch.cuda.empty_cache()

    pbar.close()
    v_loss /= len(eval_loader)
    v_acc /= len(eval_loader)

    # Perform error analysis if requested
    if error_analysis and phonemes is not None:
        perform_error_analysis(all_targets, all_predictions, phonemes)

    return v_loss, v_acc


def print_classification_report(all_targets, all_predictions, phonemes):
    """Print detailed classification report."""
    print("\nClassification Report:")
    print("-" * 30)
    report = classification_report(all_targets, all_predictions, target_names=phonemes, zero_division=0)
    print(report)


def print_top_misclassifications(all_targets, all_predictions, phonemes, top_k=10):
    """Print top misclassified phoneme pairs."""
    cm = confusion_matrix(all_targets, all_predictions)

    print(f"\nTop {top_k} Misclassified Phoneme Pairs:")
    print("-" * 40)
    misclassified = []
    for i in range(len(phonemes)):
        for j in range(len(phonemes)):
            if i != j and cm[i, j] > 0:
                misclassified.append((phonemes[i], phonemes[j], cm[i, j]))

    # Sort by count and print top k
    misclassified.sort(key=lambda x: x[2], reverse=True)
    for true_phoneme, pred_phoneme, count in misclassified[:top_k]:
        print(f"{true_phoneme} -> {pred_phoneme}: {count} times")


def print_per_class_accuracy(all_targets, all_predictions, phonemes, worst_k=10):
    """Print per-class accuracy and worst performing phonemes."""
    cm = confusion_matrix(all_targets, all_predictions)

    print("\nPer-Class Accuracy:")
    print("-" * 20)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    worst_classes = []
    for i, acc in enumerate(class_acc):
        if not np.isnan(acc):
            worst_classes.append((phonemes[i], acc))

    # Sort by accuracy and print worst k
    worst_classes.sort(key=lambda x: x[1])
    print(f"Worst {worst_k} performing phonemes:")
    for phoneme, acc in worst_classes[:worst_k]:
        print(f"{phoneme}: {acc:.3f}")


def perform_error_analysis(all_targets, all_predictions, phonemes):
    """Perform comprehensive error analysis."""
    print("\n" + "=" * 50)
    print("ERROR ANALYSIS")
    print("=" * 50)

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Run all analysis functions
    print_classification_report(all_targets, all_predictions, phonemes)
    print_top_misclassifications(all_targets, all_predictions, phonemes)
    print_per_class_accuracy(all_targets, all_predictions, phonemes)

    print("=" * 50)


def _get_phonemes_to_indices(phonemes):
    """
    Create a mapping from phonemes to their indices.
    :param phonemes: List of phonemes.
    :return: Dictionary mapping phonemes to indices.
    """
    return {phoneme: idx for idx, phoneme in enumerate(phonemes)}


def train(run_config=None):

    # Load phonemes from the config file
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    with open(os.path.join(config_dir, "phonemes.json"), "r") as f:
        phonemes_config = json.load(f)
    phonemes = phonemes_config["PHONEMES"]
    phonemes_to_idx = _get_phonemes_to_indices(phonemes)
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with wandb.init(project=PROJECT_NAME, mode="offline") as run:
        # If run_config is provided, update the wandb run config
        if run_config is not None:
            run.config.update(run_config)
            print("Using provided run config:")
            pprint.pprint(run.config, indent=4)
        run_id = run.id
        print(f"Run ID: {run_id}")
        # Check if the run is already encountered
        if run_id in ENCOUNTERED_RUNS:
            print(f"Run {run_id} already encountered, skipping.")
            run.finish()
            return
        # Add the run ID to the encountered runs
        ENCOUNTERED_RUNS.add(run_id)
        # Load hyperparameters from the wandb run config
        batch_size = run.config["batch_size"]
        num_epochs = run.config["num_epochs"]
        learning_rate = run.config["lr"]
        context = run.config["context"]  # Context frames on each side
        hidden_shape = run.config["hidden_shape"]  # Shape of the hidden layers
        hidden_dropout = run.config["hidden_dropout"]  # Dropout rate for hidden layers
        num_hidden_layers = run.config["num_hidden_layers"]  # Number of hidden layers
        optimizer_name = run.config["optimizer_name"]  # Optimizer name
        max_total_parameters = run.config["max_total_parameters"]  # Maximum total parameters for the model
        batch_norm = run.config["batch_norm"]  # Whether to use batch normalization
        activation = run.config["activation"]  # Activation function to use
        scheduler_name = run.config["scheduler_name"]  # Scheduler name
        layer_wise_dropout = run.config.get("layer_wise_dropout", False)  # Whether to use layer-wise dropout
        label_smoothing = run.config.get("label_smoothing", 0.0)  # Label smoothing factor
        frequency_masking = run.config.get("frequency_masking", 0.0)  # Frequency masking fraction
        time_masking = run.config.get("time_masking", 0.0)  # Time masking fraction
        mixup_alpha = run.config.get("mixup_alpha", 0.4)  # Mixup alpha for data augmentation
        mixup_prob = run.config.get("mixup_prob", 0.0)  # Mixup probability for data augmentation
        use_focal_loss = run.config.get("use_focal_loss", False)  # Whether to use focal loss
        focal_alpha = run.config.get("focal_alpha", 1.0)  # Focal loss alpha parameter
        focal_gamma = run.config.get("focal_gamma", 2.0)  # Focal loss gamma parameter
        use_confusion_heads = run.config.get("use_confusion_heads", False)  # Whether to use confusion heads
        confusion_weight = run.config.get("confusion_weight", 0.3)  # Weight for confusion heads loss
        confusion_dropout = run.config.get("confusion_dropout", 0.2)  # Dropout for confusion heads
        # Enable cepstral normalization
        cepstral_normalization = True

        # Weight decay for the optimizer
        weight_decay = 1e-5

        # Initialize model, criterion, optimizer
        input_size = 28 * (2 * context + 1)  # 28 features per frame, context frames on each side
        output_size = len(phonemes)  # Number of phonemes

        # Name for the run
        run_name = f"lr={learning_rate}_opt={optimizer_name}_sch={scheduler_name}_bs={batch_size}_shp={hidden_shape}_lys={num_hidden_layers}"
        run_name += f"_drp={hidden_dropout}_bn={batch_norm}_act={activation}_ctx={context}_freq_mask={frequency_masking}_time_mask={time_masking}"
        run_name += (
            f"_mixup_alpha={mixup_alpha}_mixup_prob={mixup_prob}_focal={use_focal_loss}_confheads={use_confusion_heads}"
        )
        run.name = run_name
        model_config = {
            "input_size": input_size,
            "output_size": output_size,
            "batch_norm": batch_norm,
            "activation": activation,
            "hidden_shape": hidden_shape,
            "input_dropout": 0.1,  # Default input dropout
            "hidden_dropout": hidden_dropout,
            "num_hidden_layers": num_hidden_layers,
            "layer_wise_dropout": layer_wise_dropout,
            "max_total_parameters": max_total_parameters,
            "frequency_masking": frequency_masking,
            "time_masking": time_masking,
            "mixup_alpha": mixup_alpha,
            "mixup_prob": mixup_prob,
            "use_confusion_heads": use_confusion_heads,
            "phonemes_to_idx": phonemes_to_idx,  # Pass phonemes to indices mapping
            "confusion_weight": confusion_weight,  # Weight for confusion heads loss
            "confusion_dropout": confusion_dropout,  # Dropout for confusion heads
        }
        model = SweepModel(**model_config).to(device)

        if use_confusion_heads:
            criterion = ConfusionAwareLossCriterion(
                use_focal=use_focal_loss,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                label_smoothing=label_smoothing,
                confusion_weight=confusion_weight,
                phonemes_to_idx=phonemes_to_idx,
            )
        else:
            criterion = FlexibleLossCriterion(
                use_focal=use_focal_loss,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
        optimizer = get_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay, name=optimizer_name)
        scheduler = get_scheduler(optimizer, name=scheduler_name, learning_rate=learning_rate)
        scaler = torch.amp.GradScaler("cuda")  # For mixed precision training
        # Create datasets and data loaders
        pin_memory = True
        num_workers = 2 if not AUTO_DL else 32  # Use more workers for faster data loading if on AutoDL server
        train_dataset = AudioDataset(
            root="data",
            phonemes=phonemes,
            context=context,
            partition="train-clean-100",
            cepstral_normalization=cepstral_normalization,
            # partial_load=0.001,  # Uncomment to load only a portion of the dataset for faster training
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

        eval_dataset = AudioDataset(
            root="data",
            phonemes=phonemes,
            context=context,
            partition="dev-clean",
            cepstral_normalization=cepstral_normalization,
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        # Summary of the model
        model.eval()  # Set the model to evaluation mode for summary
        frames = torch.randn(batch_size, 2 * context + 1, 28, device=device)
        summary(model, frames.to(device))
        # Create artifact and save the model with best val accuracy
        artifact = wandb.Artifact(f"{run_id}_best_model".replace("=", "-"), type="model")
        best_val_acc = 0.855  # Hardcoded best validation accuracy to compare against
        model_path = os.path.join(wandb.run.dir, f"{run_id}_best_model.pth")
        # Training loop
        for epoch in range(num_epochs):

            print(f"Epoch {epoch + 1}/{num_epochs}")

            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device, scaler)
            print(
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}"
            )

            # Only perform error analysis on the last epoch
            error_analysis = (epoch + 1) % 4 == 0 or epoch == num_epochs - 1
            eval_loss, eval_acc = evaluate_model(
                model, eval_loader, criterion, device, phonemes=phonemes, error_analysis=error_analysis
            )
            print(f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_acc:.4f}")
            # Log metrics to wandb
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": eval_loss,
                    "val_acc": eval_acc,
                    "lr": scheduler.get_last_lr()[0],
                }
            )
            # Step the scheduler
            if scheduler_name == "plateau":
                scheduler.step(eval_loss)
            else:
                scheduler.step()
            # Save the model if it has the best validation accuracy so far
            if eval_acc > best_val_acc:
                # Save the model state dict
                best_val_acc = eval_acc
                model_state_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "model_config": model_config,  # Include the model configuration for model reconstruction
                    "cepstral_normalization": cepstral_normalization,  # Include cepstral normalization flag
                    "context": context,  # Include context frames
                    "batch_size": batch_size,  # Include batch size
                }
                torch.save(model_state_dict, model_path)
                print(f"Saved new best model with accuracy: {best_val_acc:.4f}")
        # Finish the wandb run
        if os.path.exists(model_path):
            artifact.add_file(model_path)
            run.log_artifact(artifact)


def main():
    # Load wandb sweep config
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    with open(os.path.join(config_dir, "wandb_sweep_config.json"), "r") as f:
        wandb_sweep_config = json.load(f)
    # Check if wandb is in offline mode
    if AUTO_DL:
        # If running on AutoDL server, we cannot use wandb, so we run the sweep offline
        print("Running on AutoDL server. Wandb is not available, running the sweep offline.")
        from offline_wandb.offline_sweep import OfflineSweep

        # Create an offline sweep instance
        method = wandb_sweep_config["method"]
        offline_sweep = OfflineSweep(wandb_sweep_config, method=method, seed=42, unique=True)
        # Generate samples for the sweep
        samples = offline_sweep.samples(num_samples=TOTAL_RUNS)
        print(f"Generated {len(samples)} samples for the sweep.")
        # Iterate over the samples and run the train function
        for sample in samples:
            # Set the wandb run config
            train(run_config=sample)
    else:
        # Log in to wandb and create a sweep
        wandb.login()
        sweep_id = wandb.sweep(
            sweep=wandb_sweep_config,
            project=PROJECT_NAME,
        )
        # Start the sweep agent
        wandb.agent(sweep_id, function=train, count=TOTAL_RUNS)


if __name__ == "__main__":
    main()
