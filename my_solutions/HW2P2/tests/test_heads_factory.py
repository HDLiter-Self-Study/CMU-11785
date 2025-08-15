import pytest
import torch.nn as nn
from src.pipelines.factories import HeadsFactory
from src.heads import ClassificationHead

# A minimal valid config for a classification head
VALID_CONFIG = [
    {
        "mode": "single",
        "instances": {
            "classification_head": {
                "pooling_type": "adaptive_avg",
                "hidden_dims": "[256]",
                "activation": "relu",
                "classifier_type": "linear",
            }
        },
    }
]

# A config specifying a margin-based head (arcface)
ARCFACE_CONFIG = [
    {
        "mode": "single",
        "instances": {
            "classification_head": {
                "pooling_type": "adaptive_max",
                "classifier_type": "arc_face",
                "arc_face_margin": 0.6,
                "arc_face_scale": 32.0,
            }
        },
    }
]


def test_head_factory_creation():
    """Test that the factory can create a ClassificationHead instance."""
    factory = HeadsFactory()
    head = factory.build(VALID_CONFIG, in_features=512, num_classes=100)
    assert isinstance(head, ClassificationHead), "The created object should be a ClassificationHead."


def test_head_factory_with_injected_params():
    """Test that injected parameters are correctly passed to the head."""
    factory = HeadsFactory()
    head = factory.build(VALID_CONFIG, in_features=1024, num_classes=80)

    assert isinstance(head, ClassificationHead)
    # The final linear layer should have the correct dimensions
    final_layer = head.classifier
    assert final_layer.in_features == 256, "The input to the final classifier should match the last hidden dim."
    assert final_layer.out_features == 80, "The output of the final classifier should match num_classes."


def test_head_factory_arcface_creation():
    """Test creating a head with a margin-based classifier like ArcFace."""
    factory = HeadsFactory()
    head = factory.build(ARCFACE_CONFIG, in_features=512, num_classes=1000)

    assert isinstance(head, ClassificationHead)
    assert head.classifier_type == "arc_face"
    # Check if arcface-specific params were passed correctly
    arcface_layer = head.classifier
    assert arcface_layer.margin == 0.6
    assert arcface_layer.scale == 32.0


def test_head_factory_empty_config():
    """Test that the factory returns None for an empty or None config."""
    factory = HeadsFactory()
    assert factory.build(None) is None, "Should return None for None config."
    assert factory.build([]) is None, "Should return None for empty list config."


def test_head_factory_invalid_config_multiple_heads():
    """Test that the factory raises an error if more than one head is configured."""
    factory = HeadsFactory()
    invalid_config = [VALID_CONFIG[0], VALID_CONFIG[0]]  # Two head configs
    with pytest.raises(ValueError, match="A model can only have one head"):
        factory.build(invalid_config, in_features=128, num_classes=10)


def test_head_factory_unknown_head():
    """Test that the factory raises an error for an unknown head type."""
    factory = HeadsFactory()
    invalid_config = [
        {
            "mode": "single",
            "instances": {"some_unknown_head": {}},
        }
    ]
    with pytest.raises(ValueError, match="Component 'some_unknown_head' could not be found"):
        factory.build(invalid_config, in_features=128, num_classes=10)
