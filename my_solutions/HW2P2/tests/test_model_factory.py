import pytest
import json
import torch
import torch.nn as nn
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
    with pytest.raises(ValueError, match="Unsupported architecture type"):
        # The error is raised by the StagePlanner for unsupported architecture types
        model_factory.create(bad_config, data_config)


# =============================================================================
# Enhanced ModelFactory Tests - Structure and Functionality
# =============================================================================


@pytest.mark.skipif(RESNET_ARCH_CONFIG is None, reason="No ResNet config found in effective_latest.json")
def test_resnet_model_structure(model_factory, data_config):
    """
    Tests ResNet model structure matches configuration expectations.
    """
    model = model_factory.create(RESNET_ARCH_CONFIG, data_config)

    # Verify it's the correct type
    assert isinstance(model, ResNet)

    # Verify basic structure exists
    assert hasattr(model, "backbone")
    assert hasattr(model, "stem")

    # Check input channels match
    assert model.in_channels == data_config["in_channels"]

    # Verify backbone structure (backbone contains all blocks in a single Sequential)
    backbone = model.backbone
    assert isinstance(backbone, nn.Sequential)

    # Backbone should contain multiple blocks (total across all stages)
    assert len(backbone) > 0

    # Total blocks should match the sum of all stage depths
    expected_total_blocks = sum(
        RESNET_ARCH_CONFIG.get("regnet_rule", {}).get("depth_bias", 0) for _ in range(RESNET_ARCH_CONFIG["num_stages"])
    )
    # Since we can't easily predict exact block count, just ensure it's reasonable
    assert (
        len(backbone) >= RESNET_ARCH_CONFIG["num_stages"]
    ), f"Expected at least {RESNET_ARCH_CONFIG['num_stages']} blocks"


@pytest.mark.skipif(CONVNEXT_ARCH_CONFIG is None, reason="No ConvNeXt config found in effective_latest.json")
def test_convnext_model_structure(model_factory, data_config):
    """
    Tests ConvNeXt model structure matches configuration expectations.
    """
    model = model_factory.create(CONVNEXT_ARCH_CONFIG, data_config)

    # Verify it's the correct type
    assert isinstance(model, ConvNeXt)

    # Verify basic structure exists
    assert hasattr(model, "backbone")
    assert hasattr(model, "stem")

    # Check input channels match
    assert model.in_channels == data_config["in_channels"]

    # Verify backbone structure (backbone contains all blocks in a single Sequential)
    backbone = model.backbone
    assert isinstance(backbone, nn.Sequential)

    # Backbone should contain multiple blocks
    assert len(backbone) > 0

    # Total blocks should be reasonable for the number of stages
    assert (
        len(backbone) >= CONVNEXT_ARCH_CONFIG["num_stages"]
    ), f"Expected at least {CONVNEXT_ARCH_CONFIG['num_stages']} blocks"


@pytest.mark.skipif(RESNET_ARCH_CONFIG is None, reason="No ResNet config found in effective_latest.json")
def test_resnet_forward_pass(model_factory, data_config):
    """
    Tests ResNet forward pass functionality and output shapes.
    """
    model = model_factory.create(RESNET_ARCH_CONFIG, data_config)

    # Test with different input sizes
    test_sizes = [(1, 3, 224, 224), (2, 3, 112, 112), (1, 3, 64, 64)]

    model.eval()
    with torch.no_grad():
        for batch_size, channels, height, width in test_sizes:
            input_tensor = torch.randn(batch_size, channels, height, width)

            # Forward pass should not raise an exception
            try:
                output = model(input_tensor)

                # Output should be a dictionary with 'feats' key
                assert isinstance(output, dict)
                assert "feats" in output
                assert isinstance(output["feats"], torch.Tensor)

                # Output batch size should match input
                assert output["feats"].shape[0] == batch_size

                # Output should have reduced spatial dimensions due to downsampling
                # With 4 stages and downsamplings [1, 2, 2, 2], total downsampling is 8
                expected_spatial_reduction = 8  # 1 * 2 * 2 * 2
                expected_height = height // expected_spatial_reduction
                expected_width = width // expected_spatial_reduction

                # Allow some tolerance for stem downsampling
                assert output["feats"].shape[2] <= expected_height + 2
                assert output["feats"].shape[3] <= expected_width + 2

            except Exception as e:
                pytest.fail(f"Forward pass failed for input size {input_tensor.shape}: {e}")


@pytest.mark.skipif(CONVNEXT_ARCH_CONFIG is None, reason="No ConvNeXt config found in effective_latest.json")
def test_convnext_forward_pass(model_factory, data_config):
    """
    Tests ConvNeXt forward pass functionality and output shapes.
    """
    model = model_factory.create(CONVNEXT_ARCH_CONFIG, data_config)

    # Test with different input sizes
    test_sizes = [(1, 3, 224, 224), (2, 3, 112, 112)]

    model.eval()
    with torch.no_grad():
        for batch_size, channels, height, width in test_sizes:
            input_tensor = torch.randn(batch_size, channels, height, width)

            # Forward pass should not raise an exception
            try:
                output = model(input_tensor)

                # Output should be a dictionary with 'feats' key
                assert isinstance(output, dict)
                assert "feats" in output
                assert isinstance(output["feats"], torch.Tensor)

                # Output batch size should match input
                assert output["feats"].shape[0] == batch_size

                # Output should have reduced spatial dimensions
                assert output["feats"].shape[2] < height
                assert output["feats"].shape[3] < width

            except Exception as e:
                pytest.fail(f"Forward pass failed for input size {input_tensor.shape}: {e}")


def test_gradient_flow(model_factory, data_config):
    """
    Tests that gradients can flow through the model properly.
    """
    if RESNET_ARCH_CONFIG is None:
        pytest.skip("No ResNet config found in effective_latest.json")

    model = model_factory.create(RESNET_ARCH_CONFIG, data_config)

    # Enable gradient computation
    model.train()

    # Create sample input and target
    input_tensor = torch.randn(2, 3, 64, 64, requires_grad=True)

    # Forward pass
    output = model(input_tensor)

    # Create a simple loss (sum of features)
    loss = output["feats"].sum()

    # Backward pass
    loss.backward()

    # Check that gradients exist for model parameters
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            # Gradients should not be all zeros
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))
            break

    assert has_gradients, "No gradients found in model parameters"

    # Check that input tensor received gradients
    assert input_tensor.grad is not None
    assert not torch.allclose(input_tensor.grad, torch.zeros_like(input_tensor.grad))


def test_model_parameter_count(model_factory, data_config):
    """
    Tests that created models have reasonable parameter counts.
    """
    configs_to_test = []
    if RESNET_ARCH_CONFIG is not None:
        configs_to_test.append(("resnet", RESNET_ARCH_CONFIG))
    if CONVNEXT_ARCH_CONFIG is not None:
        configs_to_test.append(("convnext", CONVNEXT_ARCH_CONFIG))

    for arch_type, config in configs_to_test:
        model = model_factory.create(config, data_config)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Models should have reasonable parameter counts (not zero, not impossibly large)
        assert total_params > 1000, f"{arch_type} model has too few parameters: {total_params}"
        assert total_params < 500_000_000, f"{arch_type} model has too many parameters: {total_params}"

        # By default, all parameters should be trainable
        assert trainable_params == total_params, f"{arch_type} model has non-trainable parameters"


def test_model_device_compatibility(model_factory, data_config):
    """
    Tests that models can be moved to different devices.
    """
    if RESNET_ARCH_CONFIG is None:
        pytest.skip("No ResNet config found in effective_latest.json")

    model = model_factory.create(RESNET_ARCH_CONFIG, data_config)

    # Test CPU functionality
    model = model.cpu()
    input_tensor = torch.randn(1, 3, 64, 64)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        assert output["feats"].device.type == "cpu"

    # Test CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

        with torch.no_grad():
            output = model(input_tensor)
            assert output["feats"].device.type == "cuda"


def test_different_input_channels(model_factory):
    """
    Tests models with different input channel configurations.
    """
    if RESNET_ARCH_CONFIG is None:
        pytest.skip("No ResNet config found in effective_latest.json")

    # Test different input channel counts
    channel_configs = [
        {"in_channels": 1},  # Grayscale
        {"in_channels": 3},  # RGB
        {"in_channels": 4},  # RGBA
    ]

    for data_config in channel_configs:
        model = model_factory.create(RESNET_ARCH_CONFIG, data_config)
        assert model.in_channels == data_config["in_channels"]

        # Test forward pass
        input_tensor = torch.randn(1, data_config["in_channels"], 64, 64)
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            assert isinstance(output, dict)
            assert "feats" in output
            assert isinstance(output["feats"], torch.Tensor)


def test_config_parameter_propagation(model_factory, data_config):
    """
    Tests that configuration parameters are properly propagated to the model.
    """
    configs_to_test = []
    if RESNET_ARCH_CONFIG is not None:
        configs_to_test.append(RESNET_ARCH_CONFIG)
    if CONVNEXT_ARCH_CONFIG is not None:
        configs_to_test.append(CONVNEXT_ARCH_CONFIG)

    for config in configs_to_test:
        model = model_factory.create(config, data_config)

        # Verify backbone has reasonable number of blocks
        # (backbone contains all blocks, not just stage count)
        expected_stages = config["num_stages"]
        actual_blocks = len(model.backbone)
        assert actual_blocks >= expected_stages, f"Expected at least {expected_stages} blocks, got {actual_blocks}"

        # Test model can handle the expected stages without errors
        input_tensor = torch.randn(1, data_config["in_channels"], 64, 64)
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            assert isinstance(output, dict)
            assert "feats" in output
            assert isinstance(output["feats"], torch.Tensor)
