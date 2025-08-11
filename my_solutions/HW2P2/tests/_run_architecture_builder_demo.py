import json
from pathlib import Path
import importlib.util


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def import_stage_planner():
    planner_path = PROJECT_ROOT / "src" / "models" / "architecture_planner.py"
    spec = importlib.util.spec_from_file_location("stage_planner_module", str(planner_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module.StagePlanner


def import_arch_builder():
    builder_path = PROJECT_ROOT / "src" / "models" / "architecture_builder.py"
    spec = importlib.util.spec_from_file_location("arch_builder_module", str(builder_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module.build_spec_from_planned


def main():
    eff_path = PROJECT_ROOT / "effective_latest.json"
    data = json.loads(eff_path.read_text(encoding="utf-8"))
    trials = data.get("effective_data") or []
    assert trials, "No trials in effective_latest.json"
    arch = trials[0]["model"]["architectures"]

    StagePlanner = import_stage_planner()
    planned = StagePlanner.plan(arch)

    build_spec_from_planned = import_arch_builder()
    spec = build_spec_from_planned(planned)

    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
