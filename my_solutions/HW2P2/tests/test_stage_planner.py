import json
from pathlib import Path
import importlib.util
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _import_stage_planner():
    planner_path = PROJECT_ROOT / "src" / "models" / "architecture_planner.py"
    spec = importlib.util.spec_from_file_location("stage_planner_module", str(planner_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module.StagePlanner


def _load_effective_first_arch(effective_path: str):
    p = Path(effective_path)
    assert p.exists(), f"Effective JSON not found: {effective_path}"
    data = json.loads(p.read_text(encoding="utf-8"))
    # Supports both older and newer shapes
    trials = data.get("effective_data") or []
    assert len(trials) > 0, "effective_data empty"
    arch = trials[0]["model"]["architectures"]
    return arch


def test_stage_planner_with_resnet_example():
    StagePlanner = _import_stage_planner()
    arch = _load_effective_first_arch("effective_latest.json")
    planned = StagePlanner.plan(arch)

    num_stages = planned["num_stages"]
    assert isinstance(num_stages, int) and num_stages > 0

    # core lists aligned
    for key in ["stages", "out_channels", "downsamplings", "block_type"]:
        assert len(planned[key]) == num_stages

    # activation/normalization may be empty (optional) or aligned
    for key in ["activation", "normalization"]:
        if planned[key]:
            assert len(planned[key]) == num_stages

    # widths are multiples of 8
    assert all((c % 8) == 0 for c in planned["out_channels"])

    # downsamplings pattern check
    assert planned["downsamplings"][0] == 1
    if num_stages > 1:
        assert all(d == 2 for d in planned["downsamplings"][1:])

    # meta presence
    assert "meta" in planned and isinstance(planned["meta"], dict)
