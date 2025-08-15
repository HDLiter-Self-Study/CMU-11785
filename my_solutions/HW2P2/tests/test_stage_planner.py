import json
from pathlib import Path
import sys
import pytest

# Ensure repo root is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the refactored planner and its dataclass
from src.models.architecture_planner import StagePlanner, StagePlan


def _load_effective_first_arch(effective_path: str):
    p = PROJECT_ROOT / effective_path
    if not p.exists():
        pytest.skip(f"Effective JSON not found: {effective_path}, skipping test.")
    data = json.loads(p.read_text(encoding="utf-8"))
    trials = data.get("effective_data") or []
    if not trials:
        pytest.skip("No trials found in effective_data.")
    arch = trials[0]["model"]["architectures"]
    return arch


def test_stage_plan_field_correctness():
    """Verify the fields of the returned StagePlan instance using a complete mock."""
    # This test now uses a complete, manually-defined arch to avoid
    # dependency on the contents of effective_latest.json.
    mock_arch = {
        "type": "resnet",
        "num_stages": 4,
        "regnet_rule": {"width_slope": 2.0, "initial_width": 64, "depth_slope": 1, "depth_bias": 1},
        "block_type": [
            {"block_type": "basic"},
            {"block_type": "basic"},
            {"block_type": "bottleneck"},
            {"block_type": "bottleneck"},
        ],
        "activation": [{"activation": "relu"}, {"activation": "relu"}, {"activation": "gelu"}, {"activation": "gelu"}],
        "normalization": [
            {"normalization": "batch_norm"},
            {"normalization": "batch_norm"},
            {"normalization": "layer_norm"},
            {"normalization": "layer_norm"},
        ],
    }

    plan = StagePlanner.plan(mock_arch)

    num_stages = plan.num_stages
    assert isinstance(num_stages, int) and num_stages == 4

    # Core lists must be aligned with num_stages
    for key in ["stages", "out_channels", "downsamplings", "block_types", "per_stage_block_params"]:
        assert hasattr(plan, key), f"StagePlan missing field: {key}"
        attr = getattr(plan, key)
        assert isinstance(attr, list), f"plan.{key} is not a list"
        assert len(attr) == num_stages, f"plan.{key} length mismatch"

    # Check content types
    assert all(isinstance(d, int) for d in plan.stages)
    assert all(isinstance(c, int) and c % 8 == 0 for c in plan.out_channels)
    assert all(isinstance(d, int) and d >= 1 for d in plan.downsamplings)
    # The block_types list can be empty if not provided, but if provided, must contain strings.
    assert all(isinstance(bt, str) for bt in plan.block_types)
    assert all(isinstance(p, dict) for p in plan.per_stage_block_params)

    # Check downsampling pattern (first is always 1, others are >=2)
    assert plan.downsamplings[0] == 1
    if num_stages > 1:
        assert all(d >= 2 for d in plan.downsamplings[1:])

    # Check that per_stage_block_params contains expected keys
    # This now reliably checks against our mock data.
    if plan.per_stage_block_params:
        first_stage_params = plan.per_stage_block_params[0]
        assert "activation" in first_stage_params and "normalization" in first_stage_params
        assert first_stage_params["activation"] == "relu"
        assert first_stage_params["normalization"] == "batch_norm"


def test_stage_planner_with_manual_dict():
    """Test with a minimal, manually-defined architecture dict."""
    manual_arch = {
        "type": "resnet",
        "num_stages": 3,
        "regnet_rule": {
            "width_slope": 2.0,
            "initial_width": 32,
            "depth_slope": 1.0,
            "depth_bias": 1.0,
            "min_stage_depth": 1,
            "max_stage_depth": 5,
        },
        "block_type": [{"block_type": "basic"}, {"block_type": "basic"}, {"block_type": "bottleneck"}],
        "activation": [{"activation": "relu"}, {"activation": "gelu"}, {"activation": "relu"}],
        "normalization": [
            {"normalization": "batch_norm"},
            {"normalization": "batch_norm"},
            {"normalization": "batch_norm"},
        ],
    }

    plan = StagePlanner.plan(manual_arch)

    assert isinstance(plan, StagePlan)
    assert plan.num_stages == 3
    assert len(plan.per_stage_block_params) == 3
    assert plan.per_stage_block_params[0]["activation"] == "relu"
    assert plan.per_stage_block_params[1]["activation"] == "gelu"
    assert all(p["normalization"] == "batch_norm" for p in plan.per_stage_block_params)
    assert plan.block_types == ["basic", "basic", "bottleneck"]


def test_strict_validation_fails_on_scalar():
    """Test that the planner now fails when a per-stage attribute is a scalar."""
    bad_arch = {
        "type": "resnet",
        "num_stages": 2,
        "regnet_rule": {"width_slope": 2, "initial_width": 32},
        "block_type": "basic",  # This should be a list
    }
    with pytest.raises(TypeError, match="'block_type' is expected to be a list"):
        StagePlanner.plan(bad_arch)


def test_strict_validation_fails_on_wrong_length():
    """Test that the planner fails when a per-stage list has the wrong length."""
    bad_arch = {
        "type": "resnet",
        "num_stages": 3,
        "regnet_rule": {"width_slope": 2, "initial_width": 32},
        "block_type": [{"block_type": "basic"}, {"block_type": "basic"}],  # Length is 2, should be 3
    }
    with pytest.raises(ValueError, match="'block_type' list length 2 != num_stages 3"):
        StagePlanner.plan(bad_arch)
