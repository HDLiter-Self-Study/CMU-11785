import pytest
import torch
from torch import nn
from src.pipelines.factories import EmaFactory
from src.utils import EmaModel

# Mock model for tests
mock_model = nn.Linear(10, 2)

# =============================================================================
# EmaFactory Tests
# =============================================================================

EMA_CONFIG = [{"mode": "single", "instances": {"ema_model": {"decay": 0.9}}}]


def test_ema_factory_builds_ema_model():
    factory = EmaFactory()
    ema_model = factory.build(EMA_CONFIG, model=mock_model)
    assert isinstance(ema_model, EmaModel)
    assert ema_model.decay == 0.9


def test_ema_factory_requires_model_parameter():
    factory = EmaFactory()
    # BaseFactory wraps the TypeError from EmaModel into a RuntimeError
    # to provide more context, so we expect a RuntimeError here.
    with pytest.raises(RuntimeError, match="missing 1 required positional argument: 'model'"):
        factory.build(EMA_CONFIG)
