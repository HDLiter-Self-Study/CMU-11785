import json
from typing import List, Optional, Union

import torch
import torch.nn as nn
import inspect
import importlib


from ..models.common_blocks.pooling import PoolingFactory
from ..models.common_blocks.convolution_block import get_activation


def _snake_to_camel(snake_str: str) -> str:
    """Converts a snake_case string to CamelCase."""
    return "".join(word.capitalize() for word in snake_str.split("_"))


class ClassificationHead(nn.Module):
    """
    A flexible classification head that combines pooling, an MLP, and a final
    classifier layer which can be a standard Linear layer or a margin-based one
    (e.g., ArcFace, CosFace, SphereFace).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        pooling_type: str = "adaptive_avg",
        hidden_dims: Union[str, List[int], None] = None,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        classifier_type: str = "linear",
        **kwargs,
    ):
        super().__init__()
        self.classifier_type = classifier_type
        self.is_margin_based = classifier_type != "linear"

        # Parse hidden_dims if it's a string
        if isinstance(hidden_dims, str):
            try:
                hidden_dims = json.loads(hidden_dims)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid string format for hidden_dims: {hidden_dims}")
        hidden_dims = hidden_dims or []

        # --- Part 1: Feature Processor (Pooling -> Flatten -> MLP) ---
        feature_layers = []
        feature_layers.append(PoolingFactory.create(pooling_type))
        feature_layers.append(nn.Flatten())

        last_dim = in_features
        if hidden_dims:
            for h_dim in hidden_dims:
                feature_layers.append(nn.Linear(last_dim, h_dim))
                if use_batch_norm:
                    feature_layers.append(nn.BatchNorm1d(h_dim))
                feature_layers.append(get_activation(activation))
                if dropout_rate > 0:
                    feature_layers.append(nn.Dropout(p=dropout_rate))
                last_dim = h_dim

        self.feature_processor = nn.Sequential(*feature_layers)

        # --- Part 2: Final Classifier Layer ---
        if self.classifier_type == "linear":
            self.classifier = nn.Linear(last_dim, num_classes)
        else:
            # Dynamically find and instantiate the margin-based head
            self.classifier = self._create_margin_head(
                classifier_type=self.classifier_type,
                in_features=last_dim,
                out_features=num_classes,
                **kwargs,
            )

    def _create_margin_head(self, classifier_type: str, in_features: int, out_features: int, **kwargs) -> nn.Module:
        """
        Dynamically creates a margin-based head (e.g., ArcFace, CosFace)
        and populates its parameters from kwargs based on a prefix.
        """
        module_name = "src.heads.margin_based_heads"
        # Convert snake_case (e.g., 'arcface') to CamelCase ClassName (e.g., 'ArcFaceHead')
        class_name = _snake_to_camel(classifier_type) + "Head"

        try:
            module = importlib.import_module(module_name)
            constructor = getattr(module, class_name)
        except (ImportError, AttributeError):
            raise ValueError(f"Could not find classifier '{class_name}' in '{module_name}'.")

        # --- Parameter Extraction ---
        # Find parameters for the constructor by inspecting its signature
        sig = inspect.signature(constructor)
        allowed_params = set(sig.parameters.keys())

        # The prefix for kwargs is the classifier type + underscore (e.g., 'arcface_')
        prefix = f"{classifier_type}_"
        head_params = {}
        for key, value in kwargs.items():
            if key.startswith(prefix):
                param_name = key[len(prefix) :]
                if param_name in allowed_params:
                    head_params[param_name] = value

        # Inject required features and classes
        head_params["in_features"] = in_features
        head_params["out_features"] = out_features

        # Validate that all required params (excluding self, in_features, out_features) are present
        required_params = {
            p.name
            for p in sig.parameters.values()
            if p.default == inspect.Parameter.empty and p.name not in ["self", "in_features", "out_features"]
        }

        missing_params = required_params - set(head_params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters for {class_name}: {missing_params}")

        # Fail-fast: check for unused prefixed parameters
        unrecognized_params = {
            key for key in kwargs if key.startswith(prefix) and key[len(prefix) :] not in allowed_params
        }
        if unrecognized_params:
            raise ValueError(f"Unrecognized parameters for {class_name}: {unrecognized_params}")

        return constructor(**head_params)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the classification head.
        Expects a 4D tensor (B, C, H, W) from the backbone.
        Labels are required if the classifier is margin-based (e.g., 'arcface').
        """
        features = self.feature_processor(x)

        if self.is_margin_based:
            if labels is None:
                raise ValueError(f"Labels must be provided for '{self.classifier_type}' classifier.")
            return self.classifier(features, labels)
        else:
            return self.classifier(features)


__all__ = ["ClassificationHead"]
