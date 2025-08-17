#!/usr/bin/env python3
"""
Quick demo to show the EffectiveDataConfig produced from a template or a pre-generated JSON.

Usage examples:
  # From a minimal in-memory template (single or multiple trials)
  python tests/show_effective_data_config.py --task classification --n-trials 3

  # From an existing template file
  python tests/show_effective_data_config.py --template architecture_search_template.yaml

  # From an entry/comprehensive JSON output to isolate conversion
  python tests/show_effective_data_config.py --input test_output.json -o out/effective_data.json
"""

import sys
import os
import json
import argparse
import tempfile
from pathlib import Path


from src.sampling.generation_entry import generate_configs_from_template
from src.sampling.data_resolver import resolve_effective_data_config


def build_min_template(task: str, epochs: int = 5, batch_size: int = 64, n_trials: int = 1) -> str:
    return (
        f"task: {task}\n"
        "strategy_levels:\n"
        "  basic: [augmentation, dataset, label_mixing]\n"
        "  robust: [optimizer, scheduler]\n"
        "  custom:\n"
        "    architectures:\n"
        "      activation_params.selection.choices.custom: [global]\n"
        "shortcuts:\n"
        f"  epochs: {epochs}\n"
        f"  batch_size: {batch_size}\n"
        f"  n_trials: {n_trials}\n"
    )


def _load_from_input_json(input_path: Path):
    data = json.loads(input_path.read_text(encoding="utf-8"))
    # Detect shapes:
    # 1) Entry output: { ..., "sampled": [hier1, hier2, ...] }
    # 2) Comprehensive sampling output: { "samples": [ { "hierarchical": {...} }, ... ] }
    if isinstance(data, dict) and isinstance(data.get("sampled"), list):
        cfg = data
        sampled_list = data["sampled"]
    elif isinstance(data, dict) and isinstance(data.get("samples"), list):
        cfg = {"task": data.get("task")}
        sampled_list = [s.get("hierarchical") for s in data["samples"] if isinstance(s, dict) and s.get("hierarchical")]
    else:
        raise ValueError("Unrecognized input JSON structure. Expect 'sampled' list or 'samples' list.")
    return cfg, sampled_list


def main():
    parser = argparse.ArgumentParser(description="Show EffectiveDataConfig from one or more trials")
    parser.add_argument(
        "--input", type=str, default=None, help="Path to pre-generated JSON (entry/comprehensive output)"
    )
    parser.add_argument(
        "--template", type=str, default=None, help="Path to template YAML (ignored when --input provided)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["classification", "verification_finetune"],
        help="Task for minimal template (ignored when --input/--template provided)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--n-trials", type=int, default=1, help="Trials for minimal template (ignored with --input/--template)"
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to write EffectiveDataConfig JSON")
    args = parser.parse_args()

    effective_list = []

    if args.input:
        cfg, sampled_list = _load_from_input_json(Path(args.input))
        if not sampled_list:
            raise RuntimeError("No hierarchical samples found in input JSON.")
        for idx, sampled in enumerate(sampled_list):
            eff = resolve_effective_data_config(cfg, sampled)
            effective_list.append({"trial_index": idx, **eff})
    else:
        # Prepare template path
        if args.template:
            template_path = Path(args.template)
            if not template_path.is_file():
                raise FileNotFoundError(template_path)
        else:
            task = args.task or "classification"
            fd, p = tempfile.mkstemp(suffix=".yaml")
            os.close(fd)
            Path(p).write_text(build_min_template(task, args.epochs, args.batch_size, args.n_trials), encoding="utf-8")
            template_path = Path(p)

        # Generate config + trials
        cfg = generate_configs_from_template(str(template_path))
        sampled_list = cfg.get("sampled", [])
        if not sampled_list:
            raise RuntimeError("No sampled results found. Ensure n_trials >= 1 in template.")
        for idx, sampled in enumerate(sampled_list):
            eff = resolve_effective_data_config(cfg, sampled)
            effective_list.append({"trial_index": idx, **eff})

    output_payload = {"n_trials": len(effective_list), "effective_data": effective_list}

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(str(out_path))
    else:
        print(json.dumps(output_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
