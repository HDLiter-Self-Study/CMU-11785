import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from typing import Optional


class EmaModel(AveragedModel):
    """
    A wrapper for PyTorch's AveragedModel to specifically implement
    Exponential Moving Average (EMA).

    This wrapper simplifies the creation of an EMA model by handling the
    avg_fn internally, exposing only the essential `decay` parameter.
    """

    def __init__(
        self, model: torch.nn.Module, decay: float, device: Optional[torch.device] = None, use_buffers: bool = True
    ):
        """
        Args:
            model (torch.nn.Module): The model to average.
            decay (float): The decay factor for the EMA. A float between 0 and 1.
            device (torch.device, optional): The device for the averaged model.
                If None, it will be the same as the input model's device.
            use_buffers (bool): If True, it will also average the buffers of the model.
        """
        if not (0.0 <= decay <= 1.0):
            raise ValueError(f"Decay must be between 0 and 1, but got {decay}")

        self.decay = decay
        avg_fn = get_ema_multi_avg_fn(self.decay)

        super().__init__(model, avg_fn=avg_fn, device=device, use_buffers=use_buffers)

    def forward(self, *args, **kwargs):
        """
        The forward pass of the averaged model.
        """
        return super().forward(*args, **kwargs)
