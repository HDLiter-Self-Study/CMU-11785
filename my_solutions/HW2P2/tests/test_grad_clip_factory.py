import pytest
import torch
from torch import nn
from src.pipelines.factories import GradClipFactory
from src.utils.grad_clip import ClipNorm

# =============================================================================
# GradClipFactory Tests
# =============================================================================

GRAD_CLIP_CONFIG = [{"mode": "single", "instances": {"gradient_clipper": {"max_norm": 1.0}}}]


def test_grad_clip_factory_builds_gradient_clipper():
    factory = GradClipFactory()
    clipper = factory.build(GRAD_CLIP_CONFIG)
    assert isinstance(clipper, ClipNorm)
    assert clipper.max_norm == 1.0


def test_gradient_clipper_forward_pass():
    clipper = ClipNorm(max_norm=1.0)
    params = [torch.nn.Parameter(torch.randn(10, 10))]

    # Assign a gradient with a norm > 1.0
    params[0].grad = torch.full((10, 10), 10.0)
    norm_before = torch.linalg.vector_norm(params[0].grad)
    assert norm_before > 1.0

    # Apply clipping
    clipper(params)

    norm_after = torch.linalg.vector_norm(params[0].grad)
    assert norm_after == pytest.approx(1.0)
