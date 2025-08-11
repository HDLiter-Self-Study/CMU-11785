#!/usr/bin/env python3
"""
综合采样测试脚本
支持strategy覆盖，测试所有粒度级别的参数采样，并保存结果到JSON
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
    测试不同粒度级别的参数采样

    Args:
        strategy_overrides: Hydra override列表，用于强制特定strategy
        num_samples: 采样次数
        output_file: 输出文件名
        silent: If True, suppress all log output from the sampler.
    """
    if not silent:
        print("🚀 开始综合采样测试")
        print("=" * 60)

    # 创建采样器
    try:
        if strategy_overrides:
            if not silent:
                print(f"📋 使用Strategy覆盖:")
                for override in strategy_overrides:
                    print(f"   {override}")
            sampler = SearchSpaceSampler(overrides=strategy_overrides, silent=silent)
        else:
            sampler = SearchSpaceSampler(silent=silent)
        if not silent:
            print("✅ 采样器创建成功")

        # 注入 task 到全局上下文，供 condition 使用
        sampler.globals["task"] = task

        if not silent:
            print(f"📊 将采样的搜索空间类别: {sampler.search_space_categories}")

    except Exception as e:
        print(f"❌ 采样器创建失败: {e}")
        return None

    # 收集采样结果
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

    # 统计信息
    granularity_stats = defaultdict(int)
    architecture_stats = defaultdict(int)
    stage_param_stats = defaultdict(int)
    block_stage_param_stats = defaultdict(int)

    if not silent:
        print(f"\n📊 开始采样 (共{num_samples}次)...")

    for i in range(num_samples):
        try:
            study = optuna.create_study(storage="sqlite:///:memory:", study_name=f"trial_{i}", direction="maximize")
            trial = study.ask()

            # 进行采样
            result = sampler.sample_all_params(trial)
            flat_params = result["flat"]
            hierarchical_params = result["hierarchical"]

            # 分析采样结果
            arch_params = flat_params.get("architectures", {})
            arch_type = arch_params.get("architecture_type")
            activation_gran = arch_params.get(f"{arch_type}_activation_granularity") if arch_type else None
            norm_gran = arch_params.get(f"{arch_type}_norm_granularity") if arch_type else None
            block_type_gran = arch_params.get(f"{arch_type}_block_type_granularity") if arch_type else None
            num_stages = arch_params.get("num_stages")

            # 识别不同类型的参数（从所有类别中）
            stage_params = []
            block_stage_params = []
            all_flat_params = {}

            # 合并所有类别的扁平参数
            for category, category_params in flat_params.items():
                if isinstance(category_params, dict):
                    for param_name, param_value in category_params.items():
                        all_flat_params[f"{category}.{param_name}"] = param_value

                    # 在架构参数中查找stage和block_stage参数
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

            # 构建样本记录
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

            # 更新统计信息
            granularity_key = f"{activation_gran}|{norm_gran}|{block_type_gran}"
            granularity_stats[granularity_key] += 1
            architecture_stats[arch_type] += 1
            stage_param_stats[len(stage_params)] += 1
            block_stage_param_stats[len(block_stage_params)] += 1

            # 实时显示进度
            if not silent:
                category_summary = ", ".join(
                    [f"{cat}:{len(params) if isinstance(params, dict) else 1}" for cat, params in flat_params.items()]
                )
                arch_info = f"{arch_type}" if arch_type else "无架构"
                granularity_info = f"粒度({activation_gran},{norm_gran},{block_type_gran})" if arch_type else ""

                print(
                    f"   样本#{i+1}: {arch_info}, {granularity_info}, "
                    f"类别({category_summary}), Stage:{len(stage_params)}, Block_Stage:{len(block_stage_params)}"
                )

        except Exception as e:
            print(f"   ❌ 样本#{i+1}采样失败: {e}")
            # Optionally log the full traceback for debugging
            # import traceback
            # traceback.print_exc()
            continue

    # 添加统计摘要
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

    # 保存结果
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"sampling_results_{timestamp}.json"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if not silent:
            print(f"\n💾 结果已保存到: {output_file}")
    except Exception as e:
        print(f"\n❌ 保存结果失败: {e}")

    # 显示统计摘要
    if not silent:
        print(f"\n📈 采样统计摘要:")
        print(f"   成功采样: {len(results['samples'])}/{num_samples}")
        print(f"   采样类别: {list(category_stats.keys())}")
        print(f"   类别分布: {dict(category_stats)}")

        if architecture_stats:
            print(f"   架构分布: {dict(architecture_stats)}")
            print(f"   粒度组合分布:")
            for combo, count in granularity_stats.items():
                if combo != "None|None|None":  # 跳过无架构的情况
                    activation, norm, block_type = combo.split("|")
                    print(f"     激活:{activation}, 归一化:{norm}, 块类型:{block_type} → {count}次")

        if block_stage_param_stats and any(count > 0 for count in block_stage_param_stats.values()):
            print(f"   Block_Stage参数分布: {dict(block_stage_param_stats)}")

    return results


def test_forced_scenarios(silent=False, task="classification"):
    """测试强制场景"""
    if not silent:
        print("\n\n🎯 强制场景测试")
        print("=" * 40)

    # 定义不同的强制场景
    scenarios = [
        {
            "name": "强制Block_Stage激活函数",
            "overrides": [
                "++search_spaces.architectures.strategy_level=custom",
                "+search_spaces.architectures.activation_params.selection.choices.custom=[block_stage]",
            ],
            "samples": 3,
        },
        {
            "name": "强制Block_Stage归一化",
            "overrides": [
                "++search_spaces.architectures.strategy_level=custom",
                "+search_spaces.architectures.normalization_params.selection.choices.custom=[block_stage]",
            ],
            "samples": 3,
        },
        {
            "name": "强制Block_Type归一化",
            "overrides": [
                "++search_spaces.architectures.strategy_level=custom",
                "+search_spaces.architectures.normalization_params.selection.choices.custom=[block_type]",
            ],
            "samples": 3,
        },
        {
            "name": "强制Stage级参数",
            "overrides": [
                "++search_spaces.architectures.strategy_level=custom",
                "+search_spaces.architectures.activation_params.selection.choices.custom=[stage]",
                "+search_spaces.architectures.normalization_params.selection.choices.custom=[stage]",
            ],
            "samples": 3,
        },
        {
            "name": "强制ResNet架构",
            "overrides": [
                "++search_spaces.architectures.strategy_level=custom",
                "+search_spaces.architectures.architecture_selection.selection.choices.custom=[resnet]",
            ],
            "samples": 3,
        },
        {
            "name": "强制ConvNeXt架构",
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
            print(f"\n🧪 测试场景: {scenario['name']}")

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
                print(f"✅ 场景测试完成")
        else:
            if not silent:
                print(f"❌ 场景测试失败")

    return all_scenario_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="综合采样测试脚本")
    parser.add_argument("--samples", type=int, default=10, help="采样次数 (默认: 10)")
    parser.add_argument("--output", type=str, help="输出文件名")
    parser.add_argument("--override", action="append", help="Hydra覆盖参数 (可重复使用)")
    parser.add_argument("--scenarios", action="store_true", help="运行强制场景测试")
    parser.add_argument("--basic-only", action="store_true", help="只运行基础随机采样")
    parser.add_argument("--silent", action="store_true", help="Run in silent mode with no log output")
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "verification_finetune"],
        help="全局任务类型，注入到sampler.globals 用于条件判断",
    )
    # categories 参数已移除，始终全量采样

    args = parser.parse_args()

    if not args.silent:
        print("🌟 综合采样测试脚本")
        print("=" * 60)

    # 基础随机采样测试
    if not args.scenarios or not args.basic_only:
        if not args.silent:
            print("\n📋 基础随机采样测试")
        basic_result = test_granularity_sampling(
            strategy_overrides=args.override,
            num_samples=args.samples,
            output_file=args.output,
            silent=args.silent,
            task=args.task,
        )

    # 强制场景测试
    if args.scenarios and not args.basic_only:
        scenario_results = test_forced_scenarios(silent=args.silent, task=args.task)

        # 保存所有场景结果的汇总
        summary_file = "all_scenarios_summary.json"
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(scenario_results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 所有场景汇总已保存到: {summary_file}")
        except Exception as e:
            print(f"\n❌ 保存场景汇总失败: {e}")

    print("\n🎉 综合测试完成！")


if __name__ == "__main__":
    main()
