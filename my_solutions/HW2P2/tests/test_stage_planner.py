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


def test_stage_planner_returns_stage_plan_instance():
    """Verify that StagePlanner.plan returns an instance of StagePlan."""
    arch = _load_effective_first_arch("effective_latest.json")
    plan = StagePlanner.plan(arch)
    assert isinstance(plan, StagePlan)


def test_stage_plan_field_correctness():
    """Verify the fields of the returned StagePlan instance."""
    arch = _load_effective_first_arch("effective_latest.json")
    plan = StagePlanner.plan(arch)

    num_stages = plan.num_stages
    assert isinstance(num_stages, int) and num_stages > 0

    # Core lists must be aligned with num_stages
    for key in ["depths", "out_channels", "downsamplings", "block_types", "per_stage_block_params"]:
        assert hasattr(plan, key), f"StagePlan missing field: {key}"
        attr = getattr(plan, key)
        assert isinstance(attr, list), f"plan.{key} is not a list"
        assert len(attr) == num_stages, f"plan.{key} length mismatch"

    # Check content types
    assert all(isinstance(d, int) for d in plan.depths)
    assert all(isinstance(c, int) and c % 8 == 0 for c in plan.out_channels)
    assert all(isinstance(d, bool) for d in plan.downsamplings)
    assert all(isinstance(bt, str) for bt in plan.block_types)
    assert all(isinstance(p, dict) for p in plan.per_stage_block_params)

    # Check downsampling pattern (first is always False)
    assert not plan.downsamplings[0]
    if num_stages > 1:
        assert all(plan.downsamplings[1:])

    # Check that per_stage_block_params contains expected keys
    # This assumes the first trial in effective_latest.json has these params
    if plan.per_stage_block_params:
        first_stage_params = plan.per_stage_block_params[0]
        # These keys may or may not exist depending on the sample, so this is a soft check
        assert "activation" in first_stage_params or "normalization" in first_stage_params


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
        "block_type": ["basic", "basic", "bottleneck"],
        "activation": ["relu", "gelu", "relu"],
        "normalization": "batch_norm",  # Test global string expansion in resolver's output
    }

    # Manually expand normalization to list, as resolver would
    manual_arch["normalization"] = [manual_arch["normalization"]] * manual_arch["num_stages"]

    plan = StagePlanner.plan(manual_arch)

    assert isinstance(plan, StagePlan)
    assert plan.num_stages == 3
    assert len(plan.per_stage_block_params) == 3
    assert plan.per_stage_block_params[0]["activation"] == "relu"
    assert plan.per_stage_block_params[1]["activation"] == "gelu"
    assert all(p["normalization"] == "batch_norm" for p in plan.per_stage_block_params)
    assert plan.block_types == ["basic", "basic", "bottleneck"]
