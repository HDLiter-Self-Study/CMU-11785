import pytest
import json
from pathlib import Path
from src.models.model_factory import ModelFactory
from src.models.architectures import ResNet, ConvNeXt

# Load effective config once
EFFECTIVE_CONFIG_PATH = Path(__file__).parent.parent / "effective_latest.json"
with open(EFFECTIVE_CONFIG_PATH, "r") as f:
    EFFECTIVE_CONFIG = json.load(f)

# Extract ResNet and ConvNeXt architecture configs from the loaded JSON
RESNET_ARCH_CONFIG = None
CONVNEXT_ARCH_CONFIG = None
for trial in EFFECTIVE_CONFIG["effective_data"]:
    arch_type = trial.get("model", {}).get("architectures", {}).get("type")
    if arch_type == "resnet" and RESNET_ARCH_CONFIG is None:
        RESNET_ARCH_CONFIG = trial["model"]["architectures"]
    elif arch_type == "convnext" and CONVNEXT_ARCH_CONFIG is None:
        CONVNEXT_ARCH_CONFIG = trial["model"]["architectures"]

# =============================================================================
# ModelFactory Tests
# =============================================================================


@pytest.fixture
def model_factory():
    return ModelFactory()


@pytest.fixture
def data_config():
    """Provides a mock data config for tests."""
    return {"in_channels": 3}


@pytest.mark.skipif(RESNET_ARCH_CONFIG is None, reason="No ResNet config found in effective_latest.json")
def test_model_factory_creates_resnet(model_factory, data_config):
    """
    Tests if the ModelFactory correctly creates a ResNet model from config.
    """
    model = model_factory.create(RESNET_ARCH_CONFIG, data_config)
    assert isinstance(model, ResNet)


@pytest.mark.skipif(CONVNEXT_ARCH_CONFIG is None, reason="No ConvNeXt config found in effective_latest.json")
def test_model_factory_creates_convnext(model_factory, data_config):
    """
    Tests if the ModelFactory correctly creates a ConvNeXt model from config.
    """
    model = model_factory.create(CONVNEXT_ARCH_CONFIG, data_config)
    assert isinstance(model, ConvNeXt)


def test_model_factory_raises_error_on_unknown_type(model_factory, data_config):
    """
    Tests if the ModelFactory raises a ValueError for an unknown architecture type.
    """
    bad_config = {"type": "unknown_arch", "regnet_rule": {}, "num_stages": 1, "block_type": ["basic"]}
    with pytest.raises(ValueError, match="Unsupported architecture type: unknown_arch"):
        # The error is raised by the planner, so we call create directly
        model_factory.create(bad_config, data_config)
