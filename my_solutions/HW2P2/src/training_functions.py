"""
Training functions for HW2P2
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Handle both relative and absolute imports
try:
    from .utils import AverageMeter, accuracy, get_ver_metrics
except ImportError:
    from utils import AverageMeter, accuracy, get_ver_metrics


def train_epoch(model, dataloader, optimizer, lr_scheduler, scaler, device, config):
    """Train for one epoch"""
    model.train()

    # metric meters
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    # Loss function
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="Train", ncols=5)

    for i, (images, labels) in enumerate(dataloader):

        optimizer.zero_grad()  # Zero gradients

        # send to cuda
        images = images.to(device, non_blocking=True)
        if isinstance(labels, (tuple, list)):
            targets1, targets2, lam = labels
            labels = (targets1.to(device), targets2.to(device), lam)
        else:
            labels = labels.to(device, non_blocking=True)

        # forward
        with torch.amp.autocast(device_type=device):  # This implements mixed precision. Thats it!
            outputs = model(images)

            # Use the type of output depending on the loss function you want to use
            loss = criterion(outputs["out"], labels)

        scaler.scale(loss).backward()  # This is a replacement for loss.backward()
        scaler.step(optimizer)  # This is a replacement for optimizer.step()
        scaler.update()

        # metrics
        loss_m.update(loss.item())
        if "feats" in outputs:
            acc = accuracy(outputs["out"], labels)[0].item()
        else:
            acc = 0.0
        acc_m.update(acc)

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg),
            loss="{:.04f} ({:.04f})".format(loss.item(), loss_m.avg),
            lr="{:.04f}".format(float(optimizer.param_groups[0]["lr"])),
        )

        batch_bar.update()  # Update tqdm bar

    # You may want to call some schedulers inside the train function. What are these?
    if lr_scheduler is not None:
        lr_scheduler.step()

    batch_bar.close()

    return acc_m.avg, loss_m.avg


@torch.inference_mode()
def valid_epoch_cls(model, dataloader, device, config):
    """Validation for classification task"""
    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc="Val Cls.", ncols=5)

    # metric meters
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    # Loss function
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    for i, (images, labels) in enumerate(dataloader):

        # Move images to device
        images, labels = images.to(device), labels.to(device)

        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs["out"], labels)

        # metrics
        acc = accuracy(outputs["out"], labels)[0].item()
        loss_m.update(loss.item())
        acc_m.update(acc)

        batch_bar.set_postfix(
            acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg), loss="{:.04f} ({:.04f})".format(loss.item(), loss_m.avg)
        )

        batch_bar.update()

    batch_bar.close()
    return acc_m.avg, loss_m.avg


def valid_epoch_ver(model, pair_data_loader, device, config):
    """Validation for verification task"""
    model.eval()
    scores = []
    match_labels = []
    batch_bar = tqdm(total=len(pair_data_loader), dynamic_ncols=True, position=0, leave=False, desc="Val Veri.")

    for i, (images1, images2, labels) in enumerate(pair_data_loader):

        images = torch.cat([images1, images2], dim=0).to(device)
        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)

        feats = F.normalize(outputs["feats"], dim=1)
        feats1, feats2 = feats.chunk(2)
        similarity = F.cosine_similarity(feats1, feats2)
        scores.append(similarity.cpu().numpy())
        match_labels.append(labels.cpu().numpy())
        batch_bar.update()

    scores = np.concatenate(scores)
    match_labels = np.concatenate(match_labels)

    FPRs = ["1e-4", "5e-4", "1e-3", "5e-3", "5e-2"]
    metric_dict = get_ver_metrics(match_labels.tolist(), scores.tolist(), FPRs)
    print(metric_dict)

    return metric_dict["ACC"]


def test_epoch_ver(model, pair_data_loader, device, config):
    """Test for verification task (no labels)"""
    model.eval()
    scores = []
    batch_bar = tqdm(total=len(pair_data_loader), dynamic_ncols=True, position=0, leave=False, desc="Test Veri.")

    for i, (images1, images2) in enumerate(pair_data_loader):

        images = torch.cat([images1, images2], dim=0).to(device)
        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)

        feats = F.normalize(outputs["feats"], dim=1)
        feats1, feats2 = feats.chunk(2)
        similarity = F.cosine_similarity(feats1, feats2)
        scores.extend(similarity.cpu().numpy().tolist())
        batch_bar.update()

    return scores


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, config, num_epochs):
    """
    Complete training function for both classification and verification tasks
    """
    best_metric = 0.0
    best_model_state = None

    # Initialize mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training phase
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, config)

        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device, config)

        # Get primary metric for model selection
        primary_metric = config.get("metrics", {}).get("primary", "accuracy")
        current_metric = val_metrics.get(primary_metric, 0.0)

        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val {primary_metric}: {current_metric:.4f}")

        # Save best model
        if current_metric > best_metric:
            best_metric = current_metric
            best_model_state = model.state_dict().copy()
            print(f"New best {primary_metric}: {best_metric:.4f}")

        # Learning rate scheduling
        if scheduler:
            scheduler.step(current_metric)

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {"best_metric": best_metric, "final_model_state": model.state_dict()}


def evaluate_model(model, dataloader, device, config):
    """
    Evaluate model on given dataloader
    Returns metrics dictionary
    """
    model.eval()

    # Determine task type from config
    task = config.get("task", "classification")

    if task == "verification":
        return evaluate_verification_model(model, dataloader, device, config)
    else:
        return evaluate_classification_model(model, dataloader, device, config)


def evaluate_classification_model(model, dataloader, device, config):
    """Evaluate classification model"""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs["out"], labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs["out"], 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(dataloader)

    return {"accuracy": accuracy, "loss": avg_loss, "samples": total_samples}


def evaluate_verification_model(model, dataloader, device, config):
    """Evaluate verification model"""
    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:  # (images, labels, pairs)
                images, labels, _ = batch
            else:  # (images, labels)
                images, labels = batch

            images = images.to(device)

            # Get similarity scores
            outputs = model(images)
            feats = F.normalize(outputs["feats"], dim=1)
            feats1, feats2 = feats.chunk(2)
            similarity = F.cosine_similarity(feats1, feats2)

            all_scores.extend(similarity.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate verification metrics
    scores = np.array(all_scores)
    labels = np.array(all_labels)

    try:
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(labels, scores)
    except ImportError:
        auc = 0.0  # Fallback if sklearn not available

    # Simple accuracy with threshold 0.5
    predictions = (scores > 0.5).astype(int)
    accuracy = np.mean(predictions == labels)

    return {
        "auc": auc,
        "accuracy": accuracy,
        "scores": scores.tolist(),
        "labels": labels.tolist(),
        "samples": len(labels),
    }


class EnhancedTrainingManager:
    """
    Enhanced training manager with comprehensive checkpoint management
    Supports both classification and verification tasks with unified interface
    """

    def __init__(self, config, task_type="classification"):
        self.config = config
        self.task_type = task_type
        self.config_version = "1.0"
        self.export_readable_config = True

    def save_complete_checkpoint(
        self,
        model,
        optimizer,
        epoch,
        metrics,
        hyperparams,
        architecture_config,
        training_config,
        checkpoint_dir="./checkpoints",
    ):
        """
        Save complete checkpoint with all configuration information

        Args:
            model: PyTorch model
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Training/validation metrics
            hyperparams: Hyperparameters used
            architecture_config: Model architecture configuration
            training_config: Training configuration
            checkpoint_dir: Directory to save checkpoints
        """
        import torch
        import json
        from datetime import datetime
        from pathlib import Path
        import hashlib

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create complete checkpoint data
        complete_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            # Configuration information
            "hyperparameters": hyperparams,
            "architecture_config": architecture_config,
            "training_config": training_config,
            "task_type": self.task_type,
            # Metadata
            "config_version": self.config_version,
            "timestamp": datetime.now().isoformat(),
            "config_hash": self._compute_config_hash(hyperparams, architecture_config),
        }

        # Save main checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(complete_checkpoint, checkpoint_path)

        # Save best model if applicable
        primary_metric = self.config.get("metrics", {}).get("primary", "accuracy")
        current_value = metrics.get(f"val_{primary_metric}", metrics.get(primary_metric, 0))

        best_path = checkpoint_dir / "best_model.pth"
        if not best_path.exists() or self._is_better_metric(current_value, best_path, primary_metric):
            torch.save(complete_checkpoint, best_path)
            print(f"💾 Saved new best model with {primary_metric}: {current_value:.4f}")

        # Export human-readable config if requested
        if self.export_readable_config:
            config_path = checkpoint_dir / f"config_epoch_{epoch}.json"
            self._export_readable_config(complete_checkpoint, config_path)

        return checkpoint_path

    def load_checkpoint_for_continuation(self, checkpoint_path, target_task=None, model_class=None):
        """
        Load checkpoint for continuation training or task transfer

        Args:
            checkpoint_path: Path to checkpoint file
            target_task: Target task type (None for same task continuation)
            model_class: Model class for reconstruction (if different from original)

        Returns:
            (model, optimizer_state, config_info, metadata)
        """
        import torch
        from pathlib import Path

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract information
        original_task = checkpoint.get("task_type", "classification")
        hyperparams = checkpoint["hyperparameters"]
        arch_config = checkpoint["architecture_config"]
        train_config = checkpoint["training_config"]

        print(f"📋 Loading checkpoint from {checkpoint['timestamp']}")
        print(f"📋 Original task: {original_task}, Target task: {target_task or 'same'}")

        if target_task and target_task != original_task:
            # Task transfer mode
            model, adapted_config = self._adapt_model_for_task_transfer(checkpoint, target_task, model_class)
            return model, None, adapted_config, checkpoint
        else:
            # Same task continuation mode
            if model_class:
                model = self._reconstruct_model(arch_config, model_class)
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Assume model will be created externally
                model = None

            return (
                model,
                checkpoint["optimizer_state_dict"],
                {"hyperparams": hyperparams, "architecture": arch_config, "training": train_config},
                checkpoint,
            )

    def load_for_verification_finetuning(self, classification_checkpoint_path):
        """
        Specialized method for loading classification model for verification fine-tuning

        Args:
            classification_checkpoint_path: Path to classification checkpoint

        Returns:
            (backbone_model, verification_config)
        """
        checkpoint = torch.load(classification_checkpoint_path, map_location="cpu")

        # Extract classification configuration
        cls_arch_config = checkpoint["architecture_config"]
        cls_hyperparams = checkpoint["hyperparameters"]

        print(f"🔄 Adapting classification model for verification task")
        print(f"   Architecture: {cls_hyperparams.get('architecture', 'unknown')}")
        print(f"   Original classes: {cls_hyperparams.get('num_classes', 'unknown')}")

        # Create verification-adapted configuration
        verification_config = self._adapt_config_for_verification(cls_arch_config, cls_hyperparams)

        # Load and adapt model (remove classification head, keep backbone)
        full_model_state = checkpoint["model_state_dict"]
        backbone_state = self._extract_backbone_state(full_model_state)

        return backbone_state, verification_config

    def _adapt_config_for_verification(self, cls_arch_config, cls_hyperparams):
        """Convert classification configuration to verification configuration"""
        verification_config = {
            # Preserve backbone configuration
            "backbone": {
                "architecture": cls_hyperparams.get("architecture", "resnet50"),
                "pretrained": False,  # Already trained
                "freeze_layers": cls_hyperparams.get("freeze_layers", 0),
                # Architecture-specific parameters
                "depth": cls_hyperparams.get("depth", 50),
                "width_multiplier": cls_hyperparams.get("width_multiplier", 1.0),
                "dropout_rate": cls_hyperparams.get("dropout_rate", 0.0),
            },
            # Add verification-specific configuration
            "verification_head": {
                "type": "cosine_similarity",  # Default, can be overridden
                "embedding_dim": cls_hyperparams.get("embedding_dim", 512),
                "normalize_features": True,
                "temperature": 1.0,
            },
            # Training adaptations for fine-tuning
            "training": {
                "learning_rate": cls_hyperparams.get("learning_rate", 0.001) * 0.1,  # Lower LR
                "backbone_lr_multiplier": 0.01,  # Much lower for pretrained backbone
                "batch_size": cls_hyperparams.get("batch_size", 64),
                "epochs": 30,  # Fewer epochs for fine-tuning
            },
            # Loss configuration
            "loss": {"type": "contrastive", "margin": 1.0},  # Good default for verification
        }

        return verification_config

    def _extract_backbone_state(self, full_model_state):
        """Extract backbone weights from full model state dict"""
        backbone_state = {}

        # Common patterns for backbone parameters
        backbone_patterns = ["backbone.", "encoder.", "features.", "conv", "bn", "layer"]

        for key, value in full_model_state.items():
            # Skip classification head parameters
            if any(pattern in key for pattern in ["classifier", "fc", "head"]):
                continue

            # Include backbone parameters
            if any(pattern in key for pattern in backbone_patterns) or "." not in key:
                backbone_state[key] = value

        print(f"   Extracted {len(backbone_state)} backbone parameters")
        return backbone_state

    def _compute_config_hash(self, hyperparams, architecture_config):
        """Compute hash for configuration tracking"""
        import hashlib
        import json

        config_str = json.dumps({"hyperparams": hyperparams, "architecture": architecture_config}, sort_keys=True)

        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _export_readable_config(self, checkpoint, config_path):
        """Export human-readable configuration file"""
        import json

        readable_config = {
            "experiment_info": {
                "task_type": checkpoint["task_type"],
                "timestamp": checkpoint["timestamp"],
                "epoch": checkpoint["epoch"],
                "config_version": checkpoint["config_version"],
                "config_hash": checkpoint["config_hash"],
            },
            "metrics": checkpoint["metrics"],
            "hyperparameters": checkpoint["hyperparameters"],
            "architecture_config": checkpoint["architecture_config"],
            "training_config": checkpoint["training_config"],
        }

        with open(config_path, "w") as f:
            json.dump(readable_config, f, indent=2)

    def _is_better_metric(self, current_value, best_path, metric_name):
        """Check if current metric is better than best saved"""
        try:
            import torch

            best_checkpoint = torch.load(best_path, map_location="cpu")
            best_value = best_checkpoint["metrics"].get(
                f"val_{metric_name}", best_checkpoint["metrics"].get(metric_name, 0)
            )

            # Assume higher is better for most metrics (accuracy, auc, etc.)
            # For loss metrics, this logic would need to be inverted
            return current_value > best_value
        except:
            return True  # If can't load best, current is better by default
