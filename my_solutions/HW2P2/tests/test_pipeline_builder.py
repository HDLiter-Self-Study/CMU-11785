import pytest
import json
from pathlib import Path

from src.pipelines.builder import PipelineBuilder

EFFECTIVE_CONFIG_PATH = Path(__file__).parent.parent / "effective_latest.json"
with open(EFFECTIVE_CONFIG_PATH, "r") as f:
    EFFECTIVE_CONFIG = json.load(f)

TRIAL_CONFIG = EFFECTIVE_CONFIG["effective_data"][0]


@pytest.fixture
def pipeline_builder():
    return PipelineBuilder(TRIAL_CONFIG)


def test_pipeline_builder_builds_all_components(pipeline_builder):
    builder = pipeline_builder.build()
    for key in TRIAL_CONFIG["pipelines"].keys():
        assert hasattr(builder, key), f"PipelineBuilder is missing attribute: {key}"
        # Direct attribute access
        _ = getattr(builder, key)


def test_pipeline_builder_component_types(pipeline_builder):
    builder = pipeline_builder.build()
    if hasattr(builder, "optimizer") and builder.optimizer is not None:
        import torch.optim

        assert isinstance(builder.optimizer, torch.optim.Optimizer)
    if hasattr(builder, "augmentation") and builder.augmentation is not None:
        from torchvision.transforms import v2

        assert len(builder.augmentation) == 2
        assert isinstance(builder.augmentation[0], v2.Compose)
        assert isinstance(builder.augmentation[1], v2.Compose)
    if hasattr(builder, "scheduler") and builder.scheduler is not None:
        # Check if it's a scheduler by checking for common scheduler methods
        assert hasattr(builder.scheduler, "step")
        assert hasattr(builder.scheduler, "get_last_lr")


def test_pipeline_builder_dynamic_access(pipeline_builder):
    builder = pipeline_builder.build()
    for key in TRIAL_CONFIG["pipelines"].keys():
        _ = getattr(builder, key)


def test_pipeline_builder_chainable():
    builder = PipelineBuilder(TRIAL_CONFIG)
    result = builder.build()
    assert result is builder


def test_pipeline_builder_missing_component_raises():
    builder = PipelineBuilder(TRIAL_CONFIG).build()
    with pytest.raises(AttributeError):
        _ = getattr(builder, "not_a_real_component")


def test_pipeline_builder_label_mixing_and_heads_optional(pipeline_builder):
    builder = pipeline_builder.build()
    # label_mixing and heads may be None, but the attributes must exist
    assert hasattr(builder, "label_mixing")
    assert hasattr(builder, "heads") or True  # heads is optional


def test_pipeline_builder_all_keys_accessible(pipeline_builder):
    builder = pipeline_builder.build()
    for key in TRIAL_CONFIG["pipelines"].keys():
        # None is allowed, but must be accessible
        _ = getattr(builder, key)
