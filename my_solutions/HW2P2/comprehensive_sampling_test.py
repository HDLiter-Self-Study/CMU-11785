#!/usr/bin/env python3
"""
Comprehensive Sampling Test Script
Supports strategy overrides, tests parameter sampling at all granularity levels, and saves results to JSON.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import optuna
from src.sampling.sampler import SearchSpaceSampler
import json
from datetime import datetime
from collections import defaultdict
import argparse


def test_granularity_sampling(
    strategy_overrides=None, num_samples=5, output_file=None, silent=False, task="classification"
):
    """
    Test parameter sampling at different granularity levels.

    Args:
        strategy_overrides: List of Hydra overrides, used to force specific strategies.
        num_samples: Number of samples.
        output_file: Output file name.
        silent: If True, suppress all log output from the sampler.
    """
    if not silent:
        print("🚀 Starting comprehensive sampling test")
        print("=" * 60)

    # Create sampler
    try:
        if strategy_overrides:
            if not silent:
                print(f"📋 Using strategy overrides:")
                for override in strategy_overrides:
                    print(f"   {override}")
            sampler = SearchSpaceSampler(overrides=strategy_overrides, silent=silent)
        else:
            sampler = SearchSpaceSampler(silent=silent)
        if not silent:
            print("✅ Sampler created successfully")

        # Inject task into global context for condition usage
        sampler.globals["task"] = task

        if not silent:
            print(f"📊 Search space categories to be sampled: {sampler.search_space_categories}")

    except Exception as e:
        print(f"❌ Sampler creation failed: {e}")
        return None

    # Collect sampling results
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_samples": num_samples,
            "strategy_overrides": strategy_overrides or [],
            "sampled_categories": sampler.search_space_categories,
            "test_type": "comprehensive_granularity_sampling",
        },
        "samples": [],
    }

    # Statistics
    granularity_stats = defaultdict(int)
    architecture_stats = defaultdict(int)
    stage_param_stats = defaultdict(int)
    block_stage_param_stats = defaultdict(int)

    if not silent:
        print(f"\n📊 Starting sampling ({num_samples} times)...")

    for i in range(num_samples):
        try:
            study = optuna.create_study(storage="sqlite:///:memory:", study_name=f"trial_{i}", direction="maximize")
            trial = study.ask()

            # Perform sampling
            result = sampler.sample_all_params(trial)
            flat_params = result["flat"]
            hierarchical_params = result["hierarchical"]

            # Analyze sampling result
            arch_params = flat_params.get("architectures", {})
            arch_type = arch_params.get("architecture_type")
            activation_gran = arch_params.get(f"{arch_type}_activation_granularity") if arch_type else None
            norm_gran = arch_params.get(f"{arch_type}_norm_granularity") if arch_type else None
            block_type_gran = arch_params.get(f"{arch_type}_block_type_granularity") if arch_type else None
            num_stages = arch_params.get("num_stages")

            # Identify different types of parameters (from all categories)
            stage_params = []
            block_stage_params = []
            all_flat_params = {}

            # Merge all flat parameters from all categories
            for category, category_params in flat_params.items():
                if isinstance(category_params, dict):
                    for param_name, param_value in category_params.items():
                        all_flat_params[f"{category}.{param_name}"] = param_value

                    # In architecture params, find stage and block_stage parameters
                    if category == "architectures":
                        for param_name in category_params.keys():
                            if "_stage_" in param_name and "_of_" in param_name:
                                if any(
                                    block_type in param_name
                                    for block_type in ["basic", "bottleneck", "inverted_bottleneck"]
                                ):
                                    block_stage_params.append(param_name)
                                else:
                                    stage_params.append(param_name)

            # Build sample record
            sample_record = {
                "sample_id": i + 1,
                "trial_number": trial.number,
                "architecture_type": arch_type,
                "num_stages": num_stages,
                "granularities": {
                    "activation": activation_gran,
                    "normalization": norm_gran,
                    "block_type": block_type_gran,
                },
                "parameter_counts": {
                    "total_flat_params": len(all_flat_params),
                    "categories_sampled": len(flat_params),
                    "stage_params": len(stage_params),
                    "block_stage_params": len(block_stage_params),
                },
                "category_param_counts": {
                    category: len(params) if isinstance(params, dict) else 1 for category, params in flat_params.items()
                },
                "stage_parameters": stage_params,
                "block_stage_parameters": block_stage_params,
                "flat": flat_params,
                "hierarchical": hierarchical_params,
            }

            results["samples"].append(sample_record)

            # Update statistics
            granularity_key = f"{activation_gran}|{norm_gran}|{block_type_gran}"
            granularity_stats[granularity_key] += 1
            architecture_stats[arch_type] += 1
            stage_param_stats[len(stage_params)] += 1
            block_stage_param_stats[len(block_stage_params)] += 1

            # Show progress in real time
            if not silent:
                category_summary = ", ".join(
                    [f"{cat}:{len(params) if isinstance(params, dict) else 1}" for cat, params in flat_params.items()]
                )
                arch_info = f"{arch_type}" if arch_type else "No architecture"
                granularity_info = f"Granularity({activation_gran},{norm_gran},{block_type_gran})" if arch_type else ""

                print(
                    f"   Sample #{i+1}: {arch_info}, {granularity_info}, "
                    f"Categories({category_summary}), Stage:{len(stage_params)}, Block_Stage:{len(block_stage_params)}"
                )

        except Exception as e:
            print(f"   ❌ Sample #{i+1} sampling failed: {e}")
            # Optionally log the full traceback for debugging
            # import traceback
            # traceback.print_exc()
            continue

    # Add statistics summary
    category_stats = defaultdict(int)
    for sample in results["samples"]:
        for category in sample["category_param_counts"]:
            category_stats[category] += 1

    results["statistics"] = {
        "granularity_combinations": dict(granularity_stats),
        "architecture_distribution": dict(architecture_stats),
        "stage_param_distribution": dict(stage_param_stats),
        "block_stage_param_distribution": dict(block_stage_param_stats),
        "category_distribution": dict(category_stats),
    }

    # Save results
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"sampling_results_{timestamp}.json"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if not silent:
            print(f"\n💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"\n❌ Failed to save results: {e}")

    # Show statistics summary
    if not silent:
        print(f"\n📈 Sampling statistics summary:")
        print(f"   Successful samples: {len(results['samples'])}/{num_samples}")
        print(f"   Sampled categories: {list(category_stats.keys())}")
        print(f"   Category distribution: {dict(category_stats)}")

        if architecture_stats:
            print(f"   Architecture distribution: {dict(architecture_stats)}")
            print(f"   Granularity combination distribution:")
            for combo, count in granularity_stats.items():
                if combo != "None|None|None":  # Skip no-architecture case
                    activation, norm, block_type = combo.split("|")
                    print(
                        f"     Activation: {activation}, Normalization: {norm}, Block type: {block_type} → {count} times"
                    )

        if block_stage_param_stats and any(count > 0 for count in block_stage_param_stats.values()):
            print(f"   Block_Stage parameter distribution: {dict(block_stage_param_stats)}")

    return results


def test_forced_scenarios(silent=False, task="classification"):
    """Test forced scenarios"""
    if not silent:
        print("\n\n🎯 Forced scenario tests")
        print("=" * 40)

    # Define different forced scenarios
    scenarios = [
        {
            "name": "Force Block_Stage Activation Function",
            "overrides": [
                "++search_spaces.architectures.strategy_level=custom",
                "+search_spaces.architectures.activation_params.selection.choices.custom=[block_stage]",
            ],
            "samples": 3,
        },
        {
            "name": "Force Block_Stage Normalization",
            "overrides": [
                "++search_spaces.architectures.strategy_level=custom",
                "+search_spaces.architectures.normalization_params.selection.choices.custom=[block_stage]",
            ],
            "samples": 3,
        },
        {
            "name": "Force Block_Type Normalization",
            "overrides": [
                "++search_spaces.architectures.strategy_level=custom",
                "+search_spaces.architectures.normalization_params.selection.choices.custom=[block_type]",
            ],
            "samples": 3,
        },
        {
            "name": "Force Stage-level Parameters",
            "overrides": [
                "++search_spaces.architectures.strategy_level=custom",
                "+search_spaces.architectures.activation_params.selection.choices.custom=[stage]",
                "+search_spaces.architectures.normalization_params.selection.choices.custom=[stage]",
            ],
            "samples": 3,
        },
        {
            "name": "Force ResNet Architecture",
            "overrides": [
                "++search_spaces.architectures.strategy_level=custom",
                "+search_spaces.architectures.architecture_selection.selection.choices.custom=[resnet]",
            ],
            "samples": 3,
        },
        {
            "name": "Force ConvNeXt Architecture",
            "overrides": [
                "++search_spaces.architectures.strategy_level=custom",
                "+search_spaces.architectures.architecture_selection.selection.choices.custom=[convnext]",
            ],
            "samples": 3,
        },
    ]

    all_scenario_results = {}

    for scenario in scenarios:
        if not silent:
            print(f"\n🧪 Testing scenario: {scenario['name']}")

        result = test_granularity_sampling(
            strategy_overrides=scenario["overrides"],
            num_samples=scenario["samples"],
            output_file=f"scenario_{scenario['name'].replace(' ', '_').lower()}.json",
            silent=silent,  # Pass silent flag down
            task=task,
        )

        if result:
            all_scenario_results[scenario["name"]] = result
            if not silent:
                print(f"✅ Scenario test completed")
        else:
            if not silent:
                print(f"❌ Scenario test failed")

    return all_scenario_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Comprehensive Sampling Test Script")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples (default: 10)")
    parser.add_argument("--output", type=str, help="Output file name")
    parser.add_argument("--override", action="append", help="Hydra override parameter (can be used multiple times)")
    parser.add_argument("--scenarios", action="store_true", help="Run forced scenario tests")
    parser.add_argument("--basic-only", action="store_true", help="Run only basic random sampling")
    parser.add_argument("--silent", action="store_true", help="Run in silent mode with no log output")
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "verification_finetune"],
        help="Global task type, injected into sampler.globals for conditional logic",
    )
    # categories argument removed, always sample all categories

    args = parser.parse_args()

    if not args.silent:
        print("🌟 Comprehensive Sampling Test Script")
        print("=" * 60)

    # Basic random sampling test
    if not args.scenarios or not args.basic_only:
        if not args.silent:
            print("\n📋 Basic random sampling test")
        basic_result = test_granularity_sampling(
            strategy_overrides=args.override,
            num_samples=args.samples,
            output_file=args.output,
            silent=args.silent,
            task=args.task,
        )

    # Forced scenario tests
    if args.scenarios and not args.basic_only:
        scenario_results = test_forced_scenarios(silent=args.silent, task=args.task)

        # Save summary of all scenario results
        summary_file = "all_scenarios_summary.json"
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(scenario_results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 All scenario summaries saved to: {summary_file}")
        except Exception as e:
            print(f"\n❌ Failed to save scenario summary: {e}")

    print("\n🎉 Comprehensive test completed!")


if __name__ == "__main__":
    main()
