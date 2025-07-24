import json
import torch
from torch import nn


class SeqAugmentor(nn.Module):
    """
    A simple sequential data augmentation module that applies frequency and time masking.
    """

    def __init__(self, frequency_masking=0.0, time_masking=0.0):
        super(SeqAugmentor, self).__init__()
        self.frequency_masking = frequency_masking
        self.time_masking = time_masking

    def forward(self, x):
        """
        Apply frequency and time masking to the input tensor.
        :param x: Input tensor of shape (batch_size, seq_length, num_features)
        :return: Augmented tensor
        """
        if not self.training:
            return x
        if self.frequency_masking >= 1.0:
            raise ValueError("Frequency masking should be a fraction between 0 and 1.")
        elif self.frequency_masking > 0:
            x = self.apply_frequency_masking(x)

        if self.time_masking >= 1.0:
            raise ValueError("Time masking should be a fraction between 0 and 1.")
        elif self.time_masking > 0:
            x = self.apply_time_masking(x)
        return x

    def apply_frequency_masking(self, x):
        """
        Apply frequency masking to the input tensor.
        :param x: Input tensor of shape (batch_size, seq_length, num_features)
        :return: Tensor with frequency masking applied
        """
        num_frequencies = x.size(-1)
        num_frequency_masks = int(self.frequency_masking * num_frequencies)
        masked_indices = torch.randperm(num_frequencies)[:num_frequency_masks]
        mask = torch.ones(num_frequencies, device=x.device, dtype=x.dtype)
        mask[masked_indices] = 0
        mask = mask.view(1, 1, -1)  # Reshape for broadcasting
        x = x * mask  # Broadcasting applies mask to all batches and time steps
        return x

    def apply_time_masking(self, x):
        """
        Apply time masking to the input tensor.
        :param x: Input tensor of shape (batch_size, seq_length, num_features)
        :return: Tensor with time masking applied
        """
        seq_length = x.size(1)
        num_time_masks = int(self.time_masking * seq_length)
        # Randomly select indices to mask, but the center frame should not be masked (the length of the mask is always odd)
        center_index = seq_length // 2
        masked_indices = torch.randperm(seq_length)[: num_time_masks + 1]
        # Ensure the center index is not in the masked indices
        if center_index in masked_indices:
            masked_indices = masked_indices[masked_indices != center_index]
        else:
            masked_indices = masked_indices[:num_time_masks]

        mask = torch.ones(seq_length, device=x.device, dtype=x.dtype)
        mask[masked_indices] = 0
        mask = mask.view(1, -1, 1)  # Reshape for broadcasting
        x = x * mask  # Broadcasting applies mask to all batches and frequency bins
        return x


class MixUp(nn.Module):
    """
    MixUp data augmentation layer.
    """

    def __init__(self, num_classes, alpha=0.4, prob=0.3, same_only=False):
        """
        Initialize the MixUp layer.
        :param alpha: Hyperparameter for the Beta distribution
        :param prob: Probability of applying MixUp
        """
        super(MixUp, self).__init__()
        self.alpha = alpha
        self.prob = prob
        self.same_only = same_only

    def forward(self, x, labels):
        """
        Apply MixUp to the input tensor and labels.
        :param x: Input tensor of shape (batch_size, seq_length, num_features)
        :param labels: Labels tensor of shape (batch_size, num_classes)
        :return: Mixed input tensor and mixed labels
        """
        if not self.training or torch.rand(1).item() > self.prob:
            return x, labels
        # Turn the labels into one-hot encoding if they are not already
        if labels.dim() == 1 or labels.size(1) == 1:
            labels = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
        batch_size = x.size(0)
        lambda_param = torch.distributions.Beta(self.alpha, self.alpha).sample((batch_size, 1)).to(x.device)
        indices = torch.randperm(batch_size).to(x.device)
        if not self.same_only:
            # If same_only is False, allow mixing with different classes
            mixed_x = lambda_param * x + (1 - lambda_param) * x[indices]
            mixed_labels = lambda_param * labels + (1 - lambda_param) * labels[indices]
        else:
            original_indices = torch.arange(batch_size, device=x.device)
            same_class_indices = torch.where((labels == labels[indices]).squeeze(-1), indices, original_indices)
            # Mix only with the same class indices
            mixed_x = lambda_param * x + (1 - lambda_param) * x[same_class_indices]
            mixed_labels = labels  # The labels remain unchanged as we mix only within the same class

        return mixed_x, mixed_labels
