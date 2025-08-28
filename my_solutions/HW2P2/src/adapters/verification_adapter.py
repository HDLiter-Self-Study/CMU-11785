"""
Verification Task Adapter
Encapsulates verification-specific logic for face verification tasks
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import optuna

from .base_adapter import TaskAdapter

from models.architecture_factory import ArchitectureFactory
from data import get_verification_dataloaders


class VerificationAdapter(TaskAdapter):
    """
    Task adapter for face verification

    Handles verification-specific:
    - Model creation with verification heads
    - Verification data loading (image pairs)
    - Contrastive/cosine similarity loss computation
    - AUC/EER metric evaluation
    - Verification-specific search space
    """

    def __init__(self, config: Any = None, pretrained_model_path: str = None):
        """
        Initialize verification adapter

        Args:
            config: Verification configuration object (optional for testing)
            pretrained_model_path: Path to pretrained classification model (optional)
        """
        super().__init__(config or {})
        # Create arch_factory only if needed
        try:
            self.arch_factory = ArchitectureFactory()
        except Exception:
            self.arch_factory = None  # For testing without dependencies
        self.pretrained_model_path = pretrained_model_path

    @property
    def task_name(self) -> str:
        """Return task name"""
        return "verification"

    @property
    def primary_metric(self) -> str:
        """Return primary optimization metric"""
        return "auc"  # Area Under Curve for verification

    def create_model(self, model_config: Dict[str, Any]) -> nn.Module:
        """
        Create verification model, optionally loading from pretrained classification model

        Args:
            model_config: Model configuration dictionary

        Returns:
            Verification model (typically Siamese network)
        """
        # Ensure model has verification configuration
        verification_config = self.prepare_model_config(
            model_config,
            {
                "task": "verification",
                "verification_head_type": model_config.get("verification_head_type", "cosine_similarity"),
            },
        )

        if self.pretrained_model_path:
            # Load pretrained classification model and adapt for verification
            classification_model = self.arch_factory.create_model(verification_config)

            # Load pretrained weights
            try:
                checkpoint = torch.load(self.pretrained_model_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    classification_model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    classification_model.load_state_dict(checkpoint)
                print(f"✅ Loaded pretrained weights from {self.pretrained_model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load pretrained weights: {e}, using random initialization")

            # Create verification model wrapper
            verification_model = VerificationModelWrapper(
                backbone=classification_model,
                head_type=verification_config.get("verification_head_type", "cosine_similarity"),
                freeze_backbone=verification_config.get("freeze_backbone", True),
            )
        else:
            # Create verification model from scratch
            verification_model = self.arch_factory.create_model(verification_config)

        return verification_model

    def create_dataloaders(self, config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
        """
        Create verification data loaders for image pairs

        Args:
            config: Data configuration dictionary

        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        # Use existing verification data loading function
        train_dataloader, val_dataloader = get_verification_dataloaders(self.config)
        return train_dataloader, val_dataloader

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute verification loss (contrastive or cosine embedding loss)

        Args:
            outputs: Model similarity scores [batch_size]
            targets: Binary similarity labels [batch_size] (1=same, 0=different)

        Returns:
            Verification loss
        """
        # Use binary cross entropy for similarity prediction
        return F.binary_cross_entropy_with_logits(outputs, targets.float())

    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute verification metrics (AUC, accuracy, EER)

        Args:
            outputs: Model similarity scores [batch_size]
            targets: Binary similarity labels [batch_size]

        Returns:
            Dictionary with verification metrics
        """
        with torch.no_grad():
            # Convert outputs to probabilities
            probs = torch.sigmoid(outputs)

            # Simple threshold-based accuracy (threshold = 0.5)
            predictions = (probs > 0.5).float()
            correct = (predictions == targets.float()).sum().item()
            total = targets.size(0)
            accuracy = correct / total

            # For AUC calculation, we need sklearn (simplified version here)
            try:
                from sklearn.metrics import roc_auc_score

                auc = roc_auc_score(targets.cpu().numpy(), probs.cpu().numpy())
            except ImportError:
                # Fallback: approximate AUC using accuracy
                auc = accuracy

            return {
                "accuracy": accuracy,
                "auc": auc,
                "correct": correct,
                "total": total,
                "mean_similarity": probs.mean().item(),
                "pos_samples": targets.sum().item(),
                "neg_samples": (targets == 0).sum().item(),
            }

    def create_task_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Create verification-specific search space

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of verification-specific parameters
        """
        # Verification-specific hyperparameters
        verification_params = {
            # Verification head type
            "verification_head_type": trial.suggest_categorical(
                "verification_head_type", ["cosine_similarity", "l2_distance", "learned_similarity"]
            ),
            # Whether to freeze backbone during training
            "freeze_backbone": trial.suggest_categorical("freeze_backbone", [True, False]),
            # Learning rate multiplier for backbone (if not frozen)
            "backbone_lr_multiplier": trial.suggest_float("backbone_lr_multiplier", 0.01, 1.0, log=True),
            # Verification loss margin (for contrastive loss)
            "verification_margin": trial.suggest_float("verification_margin", 0.5, 2.0),
            # Embedding dimension for verification head
            "embedding_dim": trial.suggest_categorical("embedding_dim", [128, 256, 512, 1024]),
        }

        return verification_params

    def get_wandb_tags(self) -> list:
        """
        Get WandB tags for verification experiments

        Returns:
            List of tags for WandB logging
        """
        return ["verification", "face_verification", "siamese", "optuna", "nas"]

    def get_wandb_config_update(self, model_config: Dict[str, Any], metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Get WandB configuration updates specific to verification

        Args:
            model_config: Model configuration
            metrics: Latest metrics

        Returns:
            Dictionary of WandB config updates
        """
        return {
            "task": self.task_name,
            "verification_head_type": model_config.get("verification_head_type", "unknown"),
            "freeze_backbone": model_config.get("freeze_backbone", True),
            "current_auc": metrics.get("auc", 0.0),
            "current_accuracy": metrics.get("accuracy", 0.0),
            "pretrained_from": self.pretrained_model_path if self.pretrained_model_path else "scratch",
        }


class VerificationModelWrapper(nn.Module):
    """
    Wrapper to convert classification model to verification model
    """

    def __init__(self, backbone: nn.Module, head_type: str = "cosine_similarity", freeze_backbone: bool = True):
        """
        Initialize verification model wrapper

        Args:
            backbone: Pretrained classification model
            head_type: Type of verification head
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()

        # Extract feature extractor (remove classification head)
        self.backbone = self._extract_feature_extractor(backbone)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]

        # Create verification head
        self.verification_head = self._create_verification_head(head_type, feature_dim)

    def _extract_feature_extractor(self, model: nn.Module) -> nn.Module:
        """Extract feature extractor from classification model"""
        # For most models, remove the last layer (classifier)
        if hasattr(model, "classifier"):
            # ResNet-style models
            modules = list(model.children())[:-1]  # Remove classifier
            return nn.Sequential(*modules, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        elif hasattr(model, "fc"):
            # Some models use 'fc' for final layer
            modules = list(model.children())[:-1]  # Remove fc
            return nn.Sequential(*modules, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        else:
            # Fallback: use the whole model and hope for the best
            return model

    def _create_verification_head(self, head_type: str, feature_dim: int) -> nn.Module:
        """Create verification head based on type"""
        if head_type == "cosine_similarity":
            return CosineSimilarityHead(feature_dim)
        elif head_type == "l2_distance":
            return L2DistanceHead(feature_dim)
        elif head_type == "learned_similarity":
            return LearnedSimilarityHead(feature_dim)
        else:
            raise ValueError(f"Unknown verification head type: {head_type}")

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for image pair

        Args:
            x1: First image in pair [batch_size, 3, H, W]
            x2: Second image in pair [batch_size, 3, H, W]

        Returns:
            Similarity scores [batch_size]
        """
        # Extract features for both images
        features1 = self.backbone(x1)
        features2 = self.backbone(x2)

        # Compute similarity using verification head
        similarity = self.verification_head(features1, features2)

        return similarity


class CosineSimilarityHead(nn.Module):
    """Cosine similarity head for verification"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        # Normalize features
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)

        # Compute cosine similarity
        similarity = torch.sum(features1 * features2, dim=1)

        # Scale to make it similar to logits (optional)
        return similarity * 10.0


class L2DistanceHead(nn.Module):
    """L2 distance head for verification"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        # Compute L2 distance
        distance = torch.norm(features1 - features2, dim=1)

        # Convert distance to similarity (negative distance)
        similarity = -distance

        return similarity


class LearnedSimilarityHead(nn.Module):
    """Learned similarity head with MLP"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.similarity_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        # Concatenate features
        combined = torch.cat([features1, features2], dim=1)

        # Compute similarity through MLP
        similarity = self.similarity_net(combined).squeeze(1)

        return similarity
