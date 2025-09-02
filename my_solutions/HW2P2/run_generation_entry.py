#!/usr/bin/env python3
"""
CLI to run the sampling generation entry with a template and save output.

Usage:
  python run_generation_entry.py --template path/to/template.yaml --output out.json [--allow-new-paths]
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run sampling generation entry and record output")
    parser.add_argument("--template", required=True, help="Path to entry template YAML")
    parser.add_argument("--output", help="Path to output JSON file; default adds timestamp")
    parser.add_argument(
        "--allow-new-paths", action="store_true", help="Allow creating new leaf keys when applying overrides"
    )
    args = parser.parse_args()

    # Ensure src on sys.path
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))

    from src.sampling.generation_entry import ConfigTemplateProcessor, TrialConfigGenerator
    import optuna

    # Process template once
    processor = ConfigTemplateProcessor(args.template, allow_new_paths=args.allow_new_paths)

    # Create study and generate trials
    study = optuna.create_study(storage="sqlite:///:memory:", study_name="entry_trials", direction="maximize")

    # Generate all trial configs
    generator = TrialConfigGenerator(processor)
    sampled_list = []

    for _ in range(processor.get_n_trials()):
        trial = study.ask()
        trial_config = generator.generate_trial_config(trial)
        # Extract the single trial parameters from sampled list
        sampled_list.append(trial_config["sampled"][0])

    # Build final result using standard format
    final_config = generator.generate_trial_config(study.ask())
    final_config["sampled"] = sampled_list

    out_path = args.output
    if not out_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"generated_config_{ts}.json"

    out_path = Path(out_path)
    out_path.write_text(json.dumps(final_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
