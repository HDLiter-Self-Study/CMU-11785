import torch
import torch.nn as nn
from typing import Iterable, Union


class GradientClipper(nn.Module):
    """
    A module wrapper for torch.nn.utils.clip_grad_norm_.

    This class allows gradient clipping to be treated as a configurable pipeline
    component, similar to other modules. It stores the clipping parameters
    and applies them when called.
    """

    def __init__(self, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False):
        """
        Args:
            max_norm (float): max norm of the gradients.
            norm_type (float): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.
            error_if_nonfinite (bool): if True, an error is raised if the total
                norm of the gradients from :attr:`parameters` is ``nan``,
                ``inf``, or ``-inf``. Default: ``False``
        """
        super().__init__()
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.error_if_nonfinite = error_if_nonfinite

    def forward(self, parameters: Union[torch.Tensor, Iterable[torch.Tensor]]) -> torch.Tensor:
        """
        Clips the gradients of the given parameters in-place.

        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized.

        Returns:
            Tensor: The total norm of the parameters (before clipping).
        """
        return torch.nn.utils.clip_grad_norm_(
            parameters=parameters,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            error_if_nonfinite=self.error_if_nonfinite,
        )
