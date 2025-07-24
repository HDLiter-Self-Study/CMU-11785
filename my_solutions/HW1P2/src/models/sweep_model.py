import torch.nn as nn
import torch
import numpy as np

from models.data_augment import SeqAugmentor, MixUp

# Define confusion pairs based on actual error analysis results
CONFUSION_PAIRS = [
    # Top misclassified pairs from error analysis
    (["IH", "AH"], "ih_ah_confusion"),  # IH<->AH: 6152+3733 times
    (["Z", "S"], "z_s_confusion"),  # Z<->S: 4494+3522 times
    (["D", "T"], "d_t_confusion"),  # D<->T: 4469+3327 times
    (["D", "N"], "d_n_confusion"),  # D->N: 3814 times
    (["ER", "R"], "er_r_confusion"),  # ER<->R: 3456+3328 times
    (["EH", "AE"], "eh_ae_confusion"),  # EH->AE: 2537 times
]

CUSTOM_SHAPES = {
    "custom_9_layers_cylinder_1": [1460, 1460, 1460, 1460, 1460, 1460, 1460, 1460, 1460],
    "custom_9_layers_cylinder_with_heads": [1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280],
    "custom_9_layers_cylinder_pyramid_1": [2048, 2048, 2048, 2048, 1024, 1024, 512, 512, 256],
    "custom_9_layers_cylinder_pyramid_2": [2048, 2048, 2048, 2048, 1024, 1024, 512, 256, 128],
    "custom_8_layers_cylinder_1": [1580, 1580, 1580, 1580, 1580, 1580, 1580, 1580],
    "custom_8_layers_cylinder_with_heads": [1320, 1320, 1320, 1320, 1320, 1320, 1320, 1320],
    "custom_7_layers_cylinder_1": [1700, 1700, 1700, 1700, 1700, 1700, 1700],
    "custom_6_layers_cylinder_1": [1850, 1850, 1850, 1850, 1850, 1850],
    "custom_10_layers_cylinder_1": [1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400],
}


class SweepModel(nn.Module):
    """
    A model for hyperparameter sweeping.
    """

    def __init__(
        self,
        input_size,
        output_size,
        batch_norm=False,
        activation="relu",
        hidden_shape="cylinder",
        input_dropout=0.1,
        hidden_dropout=0.3,
        num_hidden_layers=6,
        max_total_parameters=2e7,
        layer_wise_dropout=False,
        time_masking=0.0,
        frequency_masking=0.0,
        mixup_alpha=0.4,
        mixup_prob=0.0,
        use_confusion_heads=False,
        confusion_weight=0.2,
        phonemes_to_idx=None,
        confusion_dropout=0.2,
    ):
        """
        :param input_size: Size of the input layer.
        :param hidden: Shape of the hidden layers ('cylinder', 'pyramid', 'reverse_pyramid').
        :param output_size: Size of the output layer.
        """
        super(SweepModel, self).__init__()
        layers = []
        in_features = input_size
        hidden_sizes = self.generate_hidden_sizes(
            input_size=input_size,
            output_size=output_size,
            shape=hidden_shape,
            num_hidden_layers=num_hidden_layers,
            max_total_parameters=max_total_parameters,
        )
        # Add data augmentation
        self.seq_augmentor = SeqAugmentor(frequency_masking=frequency_masking, time_masking=time_masking)
        # Add flattening layer
        self.flatten = nn.Flatten()
        # If mixup is enabled, add it to the model
        self.mixup = MixUp(alpha=mixup_alpha, prob=mixup_prob, num_classes=output_size)

        # Initialize confusion-aware heads if enabled
        self.use_confusion_heads = use_confusion_heads
        self.confusion_heads = nn.ModuleDict()
        self.confusion_weight = confusion_weight
        # Create a mapping from phoneme to index
        self.phonemes_to_idx = phonemes_to_idx
        self.confusion_dropout = confusion_dropout
        self.activation = activation  # Store activation name for confusion heads
        if use_confusion_heads and self.phonemes_to_idx is not None:
            self._setup_confusion_heads(hidden_sizes[-2])  # The last hidden layer's in_features

        # Add input dropout
        layers.append(nn.Dropout(p=input_dropout))
        # Create the hidden layers based on the specified shape
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(
                nn.Linear(in_features, hidden_size, bias=not batch_norm)
            )  # Add bias only if batch normalization is not used
            # If batch normalization is enabled, add it after the linear layer
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            # Create a new activation function instance for each layer
            layers.append(self._get_activation_function(activation))
            if layer_wise_dropout:
                layers.append(nn.Dropout(p=hidden_dropout * hidden_size / 1024))  # Scale dropout by layer size
            else:
                layers.append(nn.Dropout(p=hidden_dropout))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def _setup_confusion_heads(self, feature_size):
        """Setup specialized heads for confused phoneme pairs."""
        # Create specialized heads for each confusion pair
        for confusion_pair, head_name in CONFUSION_PAIRS:
            # Get indices of confused phonemes
            confusion_indices = []
            for phoneme in confusion_pair:
                if phoneme in self.phonemes_to_idx:
                    confusion_indices.append(self.phonemes_to_idx[phoneme])

            if len(confusion_indices) > 1:
                # Create a specialized head for this confusion pair
                # This head will output probabilities for just the confused phonemes
                self.confusion_heads[head_name] = nn.Sequential(
                    nn.Linear(feature_size, feature_size // 2),
                    self._get_activation_function(self.activation),  # Create new instance with stored activation
                    nn.Dropout(self.confusion_dropout),
                    nn.Linear(feature_size // 2, len(confusion_indices)),
                )
                # Store the indices for this head
                setattr(self, f"{head_name}_indices", confusion_indices)

    def forward(self, x, labels=None):
        """
        Forward pass through the model.

        :param x: Input tensor.
        :param labels: Target labels (required if mixup is enabled and model is in training mode).
        :return: Output tensor or (output, labels) if mixup is applied.
        """
        # Apply data augmentation
        x = self.seq_augmentor(x)
        # Flatten the input
        x = self.flatten(x)
        # If mixup is enabled, apply it
        if self.mixup.prob > 0 and self.training:
            if labels is None:
                raise ValueError("Labels must be provided for mixup.")
            x, labels = self.mixup(x, labels)

        # Pass through the shared backbone
        features = self.model[:-1](x)  # All layers except the final classification layer

        # Main classification head
        main_output = self.model[-1](features)

        # Confusion-aware heads
        confusion_outputs = {}
        if self.use_confusion_heads:
            for head_name, head_module in self.confusion_heads.items():
                confusion_outputs[head_name] = head_module(features)

        if self.training:
            if self.use_confusion_heads:
                return main_output, confusion_outputs, labels
            else:
                return main_output, labels
        else:
            if self.use_confusion_heads:
                # Combine main output with confusion heads for inference
                combined_output = self._combine_outputs(main_output, confusion_outputs)
                return combined_output
            else:
                return main_output

    def _combine_outputs(self, main_output, confusion_outputs):
        """Combine main output with confusion-aware heads."""
        # Start with the main output
        combined = main_output.clone()

        # For each confusion head, replace the corresponding logits
        for head_name, confusion_logits in confusion_outputs.items():
            if hasattr(self, f"{head_name}_indices"):
                indices = getattr(self, f"{head_name}_indices")
                # Apply a weighted combination (you can adjust the weight)
                weight = self.confusion_weight
                combined[:, indices] = (1 - weight) * combined[:, indices] + weight * confusion_logits

        return combined

    @staticmethod
    def _get_activation_function(activation):
        """
        Get the activation function based on the specified name.
        :param activation: Name of the activation function ('relu', 'gelu', 'sigmoid').
        :return: Activation function.
        """
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def _nearest_power_of_2(self, x):
        # Find the nearest power of 2 less than or equal to x
        return 2 ** (int(np.floor(np.log2(x))))

    def _floored_sqrt(self, x):
        # Find the largest integer less than or equal to the square root of x
        return int(np.floor(np.sqrt(x)))

    def _calc_max_size_cylinder(self, max_total_parameters, num_hidden_layers):
        # The shape of the hidden layers is a cylinder, meaning all hidden layers have the same size.
        # The number of parameters in each layer is approximately max_size^2,
        max_parameters_in_layer = int(max_total_parameters / num_hidden_layers)
        max_size = self._floored_sqrt(max_parameters_in_layer)
        return self._nearest_power_of_2(max_size)

    def _calc_max_size_pyramid(self, max_total_parameters, num_hidden_layers):
        # Pyramid shape: sizes decrease by a factor of 2, so parameters decrease by a factor of 4.
        # The number of parameters in each layer is n, n/4, n/16, ..., n/(4**(num_hidden_layers-1)).
        # The total number of parameters is approximately n * (1 + 1/4 + 1/16 + ... + 1/(4**(num_hidden_layers-1)))
        # This is a geometric series with a sum of n * (1 - (1/4)**num_hidden_layers) / (1 - 1/4)
        # We can rearrange this to find n:
        # n = max_total_parameters * (3/4) / (1 - (1/4)**num_hidden_layers)
        max_parameters_in_layer = int(max_total_parameters * (3 / 4) / (1 - (1 / 4) ** num_hidden_layers))
        # The number of parameters in maximum layer is approximately max_size*(max_size/2)
        max_size = self._floored_sqrt(max_parameters_in_layer * 2)
        return self._nearest_power_of_2(max_size)

    def _generate_hidden_sizes_cylinder(self, input_size, output_size, max_total_parameters, num_hidden_layers):
        max_size = self._calc_max_size_cylinder(max_total_parameters, num_hidden_layers)
        hidden_sizes = [max_size] * num_hidden_layers
        return hidden_sizes

    def _generate_hidden_sizes_pyramid(self, input_size, output_size, max_total_parameters, num_hidden_layers):
        max_size = self._calc_max_size_pyramid(max_total_parameters, num_hidden_layers)
        hidden_sizes = [max_size // (2**i) for i in range(num_hidden_layers)]
        return hidden_sizes

    def _generate_hidden_sizes_reverse_pyramid(self, input_size, output_size, max_total_parameters, num_hidden_layers):
        hidden_sizes = self._generate_hidden_sizes_pyramid(
            input_size, output_size, max_total_parameters, num_hidden_layers
        )
        return hidden_sizes[::-1]

    def _try_splits(self, split_fn, input_size, output_size, max_total_parameters, num_hidden_layers):
        """
        Try different ways to split the total number of hidden layers and parameters between two shapes,
        and select the split that yields the largest model under the parameter constraint.

        :param split_fn: Function that generates hidden sizes given parameter and layer splits.
        :param input_size: Size of the input layer.
        :param output_size: Size of the output layer.
        :param max_total_parameters: Maximum allowed total number of parameters.
        :param num_hidden_layers: Total number of hidden layers.
        :return: List of hidden layer sizes for the best split found, or None if no valid split exists.
        """
        # Try three splits: (n//2, n-n//2), (n//2+1, n-n//2-1), (n//2-1, n-n//2+1)
        splits = [
            (num_hidden_layers // 2, num_hidden_layers - num_hidden_layers // 2),
            (num_hidden_layers // 2 + 1, num_hidden_layers - (num_hidden_layers // 2 + 1)),
            (num_hidden_layers // 2 - 1, num_hidden_layers - (num_hidden_layers // 2 - 1)),
        ]
        best_hidden_sizes = None
        best_total_params = -1
        for first_half_layers, second_half_layers in splits:
            if first_half_layers <= 0 or second_half_layers <= 0:
                continue
            first_half_parameters = max_total_parameters // 2
            second_half_parameters = max_total_parameters - first_half_parameters
            hidden_sizes = split_fn(
                input_size,
                output_size,
                first_half_parameters,
                first_half_layers,
                second_half_parameters,
                second_half_layers,
            )
            total_params = self._estimate_total_parameters(hidden_sizes, input_size, output_size)
            if total_params <= max_total_parameters and total_params > best_total_params:
                best_total_params = total_params
                best_hidden_sizes = hidden_sizes
        return best_hidden_sizes

    def _generate_hidden_sizes_cylinder_pyramid(self, input_size, output_size, max_total_parameters, num_hidden_layers):
        def split_fn(
            input_size,
            output_size,
            first_half_parameters,
            first_half_layers,
            second_half_parameters,
            second_half_layers,
        ):
            return self._generate_hidden_sizes_cylinder(
                input_size, output_size, first_half_parameters, first_half_layers
            ) + self._generate_hidden_sizes_pyramid(input_size, output_size, second_half_parameters, second_half_layers)

        hidden_sizes = self._try_splits(split_fn, input_size, output_size, max_total_parameters, num_hidden_layers)
        if hidden_sizes is None:
            raise ValueError("Could not find a valid split for cylinder_pyramid shape.")
        return hidden_sizes

    def _generate_hidden_sizes_reverse_pyramid_cylinder(
        self,
        input_size,
        output_size,
        max_total_parameters,
        num_hidden_layers,
    ):
        def split_fn(
            input_size,
            output_size,
            first_half_parameters,
            first_half_layers,
            second_half_parameters,
            second_half_layers,
        ):
            return self._generate_hidden_sizes_reverse_pyramid(
                input_size, output_size, first_half_parameters, first_half_layers
            ) + self._generate_hidden_sizes_cylinder(
                input_size, output_size, second_half_parameters, second_half_layers
            )

        hidden_sizes = self._try_splits(split_fn, input_size, output_size, max_total_parameters, num_hidden_layers)
        if hidden_sizes is None:
            raise ValueError("Could not find a valid split for pyramid_cylinder shape.")
        return hidden_sizes

    def _generate_hidden_sizes_reverse_pyramid_cylinder(
        self,
        input_size,
        output_size,
        max_total_parameters,
        num_hidden_layers,
    ):
        hidden_sizes = self._generate_hidden_sizes_cylinder_pyramid(
            input_size,
            output_size,
            max_total_parameters,
            num_hidden_layers,
        )
        hidden_sizes.reverse()
        return hidden_sizes

    def _generate_hidden_sizes_diamond(
        self,
        input_size,
        output_size,
        max_total_parameters,
        num_hidden_layers,
    ):
        def split_fn(
            input_size,
            output_size,
            first_half_parameters,
            first_half_layers,
            second_half_parameters,
            second_half_layers,
        ):
            return self._generate_hidden_sizes_reverse_pyramid(
                input_size, output_size, first_half_parameters, first_half_layers
            ) + self._generate_hidden_sizes_pyramid(input_size, output_size, second_half_parameters, second_half_layers)

        hidden_sizes = self._try_splits(split_fn, input_size, output_size, max_total_parameters, num_hidden_layers)
        if hidden_sizes is None:
            raise ValueError("Could not find a valid split for diamond shape.")
        return hidden_sizes

    def _estimate_total_parameters(self, hidden_sizes, input_size, output_size):
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        total_parameters = 0
        for i in range(len(layer_sizes) - 1):
            total_parameters += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]  # Weights + biases
        return total_parameters

    def generate_hidden_sizes(
        self,
        input_size,
        output_size,
        shape="cylinder",
        num_hidden_layers=6,
        max_total_parameters=2e7,
    ):
        """
        Generate hidden layer sizes based on the shape and number of layers.
        :param shape: Shape of the hidden layer sizes ('cylinder', 'pyramid', 'reverse_pyramid').
        :param num_hidden_layers: Number of hidden layers.
        :param max_total_parameters: Maximum total number of parameters in the model.
        :return: List of hidden layer sizes.
        """
        if shape in CUSTOM_SHAPES:
            # If the shape is a custom predefined shape, return it directly.
            hidden_sizes = CUSTOM_SHAPES[shape]
            return hidden_sizes

        generate_method = getattr(self, f"_generate_hidden_sizes_{shape}", None)
        if generate_method is None:
            raise ValueError(
                f"Unknown shape: {shape}. Supported shapes are 'cylinder', 'pyramid', 'reverse_pyramid', "
                f"'cylinder_pyramid', 'reverse_pyramid_cylinder', and 'diamond'."
            )
        hidden_sizes = generate_method(input_size, output_size, max_total_parameters, num_hidden_layers)
        if len(hidden_sizes) != num_hidden_layers:
            raise ValueError(
                f"Generated hidden sizes length {len(hidden_sizes)} does not match the number of hidden layers "
                f"{num_hidden_layers}."
            )
        # Ensure that all hidden sizes are at least min_layer_size
        # This is to ensure that the model can handle larger input and output sizes.
        min_layer_size = 128
        if input_size > 512:
            min_layer_size = 256
        if output_size > 1024:
            min_layer_size = 256
        hidden_sizes = [max(size, min_layer_size) for size in hidden_sizes]

        return hidden_sizes
