"""
Search Space Sampler for Optuna-based hyperparameter optimization.
Supports hierarchical configuration sampling with architecture-aware parameters
using dependency_order for clean parameter resolution.
"""

import optuna
from typing import Dict, Any, List, Optional, Union, Tuple
from collections import defaultdict, deque
import inspect
from omegaconf import DictConfig

from .config_manager import get_config


class SearchSpaceSampler:
    """
    Search space parser and Optuna sampler with architecture-aware parameter support.
    Uses dependency_order for clean parameter resolution without hardcoded prefixes.
    """

    # --- class-level constants to minimize hard-coding ---------------------------------
    _REQUIRED = {
        "categorical": {"choices"},
        "float": {"low", "high"},
        "int": {"low", "high"},
    }
    _CAST = {"float": float, "int": int}

    def __init__(self, config_name: str = "main", overrides: Optional[List[str]] = None):
        """
        Initialize the sampler with a config name and optional overrides.

        Args:
            config_name: Name of the config to load (default: "main")
            overrides: List of Hydra-style parameter overrides
        """
        self.config: DictConfig = get_config(config_name, overrides)
        self.search_spaces: DictConfig = self.config.search_spaces
        self._dependency_graph = None  # Will be built when needed

        # Dynamically discover all search space categories
        self.search_space_categories = list(self.search_spaces.keys())

        # Safe built-ins for eval operations
        self.safe_builtins = {
            "max": max,
            "min": min,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "abs": abs,
        }

        # ------------------------------------------------------------------
        # Helper: evaluate "$..." expressions safely with sampled parameters
        # ------------------------------------------------------------------

    def _safe_eval(self, expr: str, params: Dict[str, Any]):
        """Safely evaluate an expression string using sampled params & whitelisted builtins."""
        return eval(expr, {"__builtins__": self.safe_builtins}, dict(params))

    def _evaluate_condition(self, condition: str, sampled_params: Dict[str, Any]) -> bool:
        """Evaluate "$expr" to boolean with shared _safe_eval."""
        if not condition or not condition.startswith("$"):
            return True
        try:
            return bool(self._safe_eval(condition[1:], sampled_params))
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate condition '{condition[1:]}': {e}. Available parameters: {list(sampled_params.keys())}"
            )

    def _resolve_dynamic_value(self, value, sampled_params: Dict[str, Any]):
        """Resolve strings starting with "$" via shared _safe_eval."""
        if isinstance(value, str) and value.startswith("$"):
            try:
                return self._safe_eval(value[1:], sampled_params)
            except Exception as e:
                raise ValueError(
                    f"Failed to resolve dynamic value '{value}': {e}. Available params: {list(sampled_params.keys())}"
                )
        return value

    def _is_dict_like(self, value) -> bool:
        """Check if a value is dict-like (dict or has keys)"""
        return isinstance(value, dict) or hasattr(value, "keys")

    def _normalize_choices(self, choices, param_name: str = ""):
        """Normalize choices to a proper Python list"""
        # Convert OmegaConf ListConfig to Python list
        if hasattr(choices, "_content"):  # OmegaConf ListConfig
            choices = list(choices)
        elif not isinstance(choices, list):
            if choices is None or choices == "":
                raise ValueError(f"Empty or null choices for {param_name}")
            choices = [choices]

        # Ensure we have valid choices
        if not choices:
            raise ValueError(f"No valid choices found for {param_name}")

        return choices

    def _resolve_param_value(
        self, value, param_config: DictConfig, sampled_params: Dict[str, Any], param_name: str = ""
    ):
        """Resolve value according to dependency_order as nested dictionary keys"""
        if not self._is_dict_like(value):
            return value

        dependency_order = getattr(param_config, "dependency_order", [])
        if not dependency_order:
            raise ValueError(f"No dependency order found for {param_name}")

        current_value = value

        # Navigate through nested dict according to dependency_order
        for dep_param in dependency_order:
            key = sampled_params[dep_param]

            if not self._is_dict_like(current_value):
                raise ValueError(
                    f"Cannot use key '{dep_param}'={key} for {param_name} because {current_value} is not a dict"
                )
            if dep_param not in sampled_params:
                raise ValueError(f"Dependency parameter '{dep_param}' not found in sampled_params for {param_name}")

            if key not in current_value:
                raise ValueError(
                    f"Key '{key}' not found for {current_value} at dependency level '{dep_param}' for {param_name}"
                )

            current_value = current_value[key]

        return current_value

    def _build_suggest_kwargs(
        self,
        param_config: DictConfig,
        param_name: str,
        sampled_params: Dict[str, Any],
        param_type: str,
        accepted_keys: set,
    ) -> Dict[str, Any]:
        """
        Walk through *all* user-defined attributes in the YAML (except metadata) and
        convert them into kwargs for Optuna's suggest_* API.
        Supported keys and their implicit conversions:
        - choices : list -> handled by _normalize_choices
        - low/high: numeric conversion according to *param_type*
        - others  : resolved dynamically but kept as-is (e.g. log, step)
        """
        kw: Dict[str, Any] = {}
        for k, raw in param_config.items():
            if k != "choices" and k not in accepted_keys:
                continue
            base_val = self._resolve_param_value(raw, param_config, sampled_params, f"{k} for {param_name}")
            val = self._resolve_dynamic_value(base_val, sampled_params)
            if k in {"low", "high"}:
                try:
                    val = self._CAST[param_type](val)
                except Exception as e:
                    raise ValueError(f"{param_name}.{k}={val} not {param_type}: {e}")
            elif k == "choices":
                val = self._normalize_choices(val, param_name)
                if not val:
                    raise ValueError(f"{param_name}.choices empty")
            kw[k] = val

        missing = self._REQUIRED[param_type] - kw.keys()
        if missing:
            raise ValueError(f"{param_name} missing required key(s): {missing}")
        return kw

    def _sample_single_param(
        self, trial: optuna.Trial, param_config: DictConfig, param_name: str, sampled_params: Dict[str, Any]
    ) -> Any:
        """Universal sampler: no type-specific branches, everything driven by attribute names."""
        param_type: str = param_config.type
        suggest_fn = getattr(trial, f"suggest_{param_type}", None)
        if not callable(suggest_fn):
            raise ValueError(f"Unsupported parameter type: {param_type}")

        # keep only kwargs that exist in the Optuna function signature
        accepted = set(inspect.signature(suggest_fn).parameters) - {"name"}

        kw_all = self._build_suggest_kwargs(param_config, param_name, sampled_params, param_type, accepted)

        if param_type == "categorical":
            choices = kw_all.pop("choices")
            kw = {k: v for k, v in kw_all.items() if k in accepted}
            return suggest_fn(param_name, choices, **kw)

        kw = {k: v for k, v in kw_all.items() if k in accepted}
        return suggest_fn(param_name, **kw)

    def _sample_stage_param(
        self,
        trial: optuna.Trial,
        param_config: DictConfig,
        base_param_name: str,
        sampled_params: Dict[str, Any],
        arch_prefix: str = "",
    ) -> Dict[str, Any]:
        """
        采样stage级参数，根据num_stages展开为多个独立参数

        Args:
            trial: Optuna trial对象
            param_config: 参数配置
            base_param_name: 基础参数名（如 'activation'）
            sampled_params: 已采样的参数
            arch_prefix: 架构前缀

        Returns:
            Dict[str, Any]: stage级参数的字典 {stage_param_name: value}
        """
        # 获取stage数量
        num_stages = sampled_params.get("regnet_num_stages")
        if num_stages is None:
            raise ValueError(f"regnet_num_stages not found when sampling stage parameter {base_param_name}")

        stage_params = {}

        # 为每个stage采样参数
        for stage_idx in range(num_stages):
            # Stage计数从1开始，更符合人类习惯
            stage_number = stage_idx + 1
            # 生成stage级参数名：{arch}_{param}_stage_{i}_of_{n}
            if arch_prefix:
                stage_param_name = f"{arch_prefix}_{base_param_name}_stage_{stage_number}_of_{num_stages}"
            else:
                stage_param_name = f"{base_param_name}_stage_{stage_number}_of_{num_stages}"

            # 采样单个stage的参数值
            stage_value = self._sample_single_param(trial, param_config, stage_param_name, sampled_params)
            stage_params[stage_param_name] = stage_value

        return stage_params

    def _get_stage_block_type(
        self, stage_idx: int, num_stages: int, sampled_params: Dict[str, Any], arch_prefix: str = ""
    ) -> str:
        """
        获取指定stage的block_type

        Args:
            stage_idx: stage索引
            num_stages: 总stage数量
            sampled_params: 已采样的参数
            arch_prefix: 架构前缀

        Returns:
            str: 该stage的block_type
        """
        # 首先尝试获取stage级别的block_type (stage计数从1开始)
        stage_number = stage_idx + 1
        stage_block_type_key = f"stage_block_type_stage_{stage_number}_of_{num_stages}"
        if arch_prefix:
            stage_block_type_key = f"{arch_prefix}_{stage_block_type_key}"

        stage_block_type = sampled_params.get(stage_block_type_key)
        if stage_block_type is not None:
            return stage_block_type

        # 回退到全局block_type
        global_block_type_key = f"{arch_prefix}_block_type" if arch_prefix else "block_type"
        global_block_type = sampled_params.get(global_block_type_key)
        if global_block_type is not None:
            return global_block_type

        # 最后尝试直接查找block_type
        fallback_block_type = sampled_params.get("block_type")
        if fallback_block_type is not None:
            return fallback_block_type

        raise ValueError(
            f"无法确定stage {stage_idx}的block_type。已检查的键: "
            f"[{stage_block_type_key}, {global_block_type_key}, block_type]"
        )

    def _sample_block_stage_param(
        self,
        trial: optuna.Trial,
        param_config: DictConfig,
        base_param_name: str,
        sampled_params: Dict[str, Any],
        arch_prefix: str = "",
    ) -> Dict[str, Any]:
        """
        采样block_stage级参数，根据num_stages和block_type展开

        Args:
            trial: Optuna trial对象
            param_config: 参数配置
            base_param_name: 基础参数名
            sampled_params: 已采样的参数
            arch_prefix: 架构前缀

        Returns:
            Dict[str, Any]: block_stage级参数的字典
        """
        # 获取stage数量
        num_stages = sampled_params.get("regnet_num_stages")
        if num_stages is None:
            raise ValueError(f"regnet_num_stages not found when sampling block_stage parameter {base_param_name}")

        block_stage_params = {}

        # 为每个stage采样参数
        for stage_idx in range(num_stages):
            # Stage计数从1开始，更符合人类习惯
            stage_number = stage_idx + 1
            # 获取这个stage的block_type
            try:
                current_block_type = self._get_stage_block_type(stage_idx, num_stages, sampled_params, arch_prefix)
            except ValueError as e:
                print(f"⚠️  警告: {e}")
                print(f"已采样的参数键: {list(sampled_params.keys())}")
                raise

            # 生成block_stage级参数名：{arch}_{param}_stage_{i}_of_{n}_{block_type}
            if arch_prefix:
                block_stage_param_name = (
                    f"{arch_prefix}_{base_param_name}_stage_{stage_number}_of_{num_stages}_{current_block_type}"
                )
            else:
                block_stage_param_name = f"{base_param_name}_stage_{stage_number}_of_{num_stages}_{current_block_type}"

            # 采样参数值
            stage_value = self._sample_single_param(trial, param_config, block_stage_param_name, sampled_params)
            block_stage_params[block_stage_param_name] = stage_value

        return block_stage_params

    def _sample_block_type_param(
        self,
        trial: optuna.Trial,
        param_config: DictConfig,
        base_param_name: str,
        sampled_params: Dict[str, Any],
        arch_prefix: str = "",
    ) -> Dict[str, Any]:
        """
        为block_type粒度采样参数
        根据stage_block_type的采样结果，为每个不同的block_type生成独立参数
        """
        # 首先确保regnet_num_stages已采样
        self._ensure_regnet_num_stages_sampled(trial, sampled_params, arch_prefix)

        num_stages = sampled_params.get("regnet_num_stages", 0)
        if num_stages == 0:
            raise ValueError("无法获取regnet_num_stages参数")

        # 收集所有不同的block_type
        unique_block_types = set()

        # 首先检查全局block_type
        global_block_type_key = f"{arch_prefix}_block_type" if arch_prefix else "block_type"
        if global_block_type_key in sampled_params:
            global_block_type = sampled_params[global_block_type_key]
            unique_block_types.add(global_block_type)

        # 然后检查stage级别的block_type
        for stage_idx in range(num_stages):
            stage_number = stage_idx + 1  # 1-based indexing
            stage_block_type_key = f"{arch_prefix}_stage_block_type_stage_{stage_number}_of_{num_stages}"
            if stage_block_type_key in sampled_params:
                block_type = sampled_params[stage_block_type_key]
                unique_block_types.add(block_type)

        # 如果没有找到任何block_type，需要主动采样stage_block_type参数
        if not unique_block_types:
            # 主动采样stage级别的block_type参数
            # 主动采样stage_block_type参数
            self._ensure_stage_block_type_sampled(trial, sampled_params, arch_prefix, num_stages)

            # 重新收集block_type
            for stage_idx in range(num_stages):
                stage_number = stage_idx + 1  # 1-based indexing
                stage_block_type_key = f"{arch_prefix}_stage_block_type_stage_{stage_number}_of_{num_stages}"
                if stage_block_type_key in sampled_params:
                    block_type = sampled_params[stage_block_type_key]
                    unique_block_types.add(block_type)

            # 如果仍然没有找到，使用默认值
            if not unique_block_types:
                unique_block_types.add("basic")  # 默认block_type

        # 为每个不同的block_type生成参数
        block_type_params = {}
        for block_type in unique_block_types:
            param_name = (
                f"{arch_prefix}_{base_param_name}_{block_type}" if arch_prefix else f"{base_param_name}_{block_type}"
            )
            value = self._sample_single_param(trial, param_config, param_name, sampled_params)
            block_type_params[param_name] = value

        return block_type_params

    def _build_stage_list_for_block_type(
        self,
        param_name: str,
        sampled_params: Dict[str, Any],
        arch_prefix: str,
        block_type_params: Dict[str, Any],
    ) -> List[Any]:
        """
        为block_type粒度的参数构建stage列表

        Args:
            param_name: 参数名（如activation, normalization）
            sampled_params: 已采样的参数
            arch_prefix: 架构前缀
            block_type_params: block_type参数字典

        Returns:
            List[Any]: 每个stage对应的参数值列表
        """
        num_stages = sampled_params.get("regnet_num_stages", 0)
        stage_list = []

        for stage_idx in range(num_stages):
            stage_number = stage_idx + 1
            # 获取该stage的block_type
            stage_block_type_key = f"{arch_prefix}_stage_block_type_stage_{stage_number}_of_{num_stages}"
            if stage_block_type_key in sampled_params:
                block_type = sampled_params[stage_block_type_key]
                # 查找对应的block_type参数
                block_type_param_key = f"{arch_prefix}_{param_name}_{block_type}"
                if block_type_param_key in block_type_params:
                    stage_list.append(block_type_params[block_type_param_key])
                else:
                    stage_list.append(None)
            else:
                stage_list.append(None)

        return stage_list

    def _ensure_regnet_num_stages_sampled(
        self, trial: optuna.Trial, sampled_params: Dict[str, Any], arch_prefix: str = ""
    ) -> None:
        """
        确保regnet_num_stages参数已经被采样
        如果没有采样，则主动查找并采样
        """
        # 检查是否已经采样
        if "regnet_num_stages" in sampled_params:
            return

        # 主动查找regnet_num_stages的配置并采样
        arch_config = self.search_spaces.architectures

        def find_regnet_num_stages_config(node: DictConfig, path: str = "") -> DictConfig:
            """递归查找regnet_num_stages的配置"""
            for key, child in node.items():
                if not isinstance(child, DictConfig):
                    continue

                class_type = child.get("class", "")

                if class_type == "param":
                    param_name = child.get("param_name", key)
                    if param_name == "regnet_num_stages":
                        return child

                # 递归查找
                result = find_regnet_num_stages_config(child, f"{path}.{key}")
                if result is not None:
                    return result

            return None

        # 查找配置
        regnet_config = find_regnet_num_stages_config(arch_config)
        if regnet_config is None:
            raise ValueError("无法找到regnet_num_stages的配置定义")

        # 采样参数
        param_name = f"{arch_prefix}_regnet_num_stages" if arch_prefix else "regnet_num_stages"
        value = self._sample_single_param(trial, regnet_config, param_name, sampled_params)

        # 更新采样结果
        sampled_params["regnet_num_stages"] = value

        print(f"🔧 主动采样 regnet_num_stages = {value}")

    def _ensure_stage_block_type_sampled(
        self, trial: optuna.Trial, sampled_params: Dict[str, Any], arch_prefix: str, num_stages: int
    ) -> None:
        """
        确保stage_block_type参数已经被采样
        """
        # 检查是否所有stage的block_type都已采样
        all_sampled = True
        for stage_idx in range(num_stages):
            stage_number = stage_idx + 1
            stage_block_type_key = f"{arch_prefix}_stage_block_type_stage_{stage_number}_of_{num_stages}"
            if stage_block_type_key not in sampled_params:
                all_sampled = False
                break

        if all_sampled:
            return

        # 查找stage_block_type的配置
        arch_config = self.search_spaces.architectures

        def find_stage_block_type_config(node: DictConfig, path: str = "") -> DictConfig:
            """递归查找stage_block_type的配置"""
            for key, child in node.items():
                if not isinstance(child, DictConfig):
                    continue

                class_type = child.get("class", "")

                if class_type == "param":
                    param_name = child.get("param_name", key)
                    if param_name == "stage_block_type":
                        return child

                # 递归查找
                result = find_stage_block_type_config(child, f"{path}.{key}")
                if result is not None:
                    return result

            return None

        # 查找配置
        stage_block_type_config = find_stage_block_type_config(arch_config)
        if stage_block_type_config is None:
            # 无法找到stage_block_type的配置定义
            return

        # 为每个stage采样block_type参数
        for stage_idx in range(num_stages):
            stage_number = stage_idx + 1
            stage_param_name = f"{arch_prefix}_stage_block_type_stage_{stage_number}_of_{num_stages}"
            if stage_param_name not in sampled_params:
                value = self._sample_single_param(trial, stage_block_type_config, stage_param_name, sampled_params)
                sampled_params[stage_param_name] = value
                # 主动采样stage block_type参数

    def _sample_config_recursively(
        self,
        trial: optuna.Trial,
        cfg: DictConfig,
        sampled: Dict[str, Any],
        prefix: str = "",
        hierarchical_tree: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Single-pass recursion with unified taxonomy (strategy/technique/instance/param)."""
        out: Dict[str, Any] = {}

        # pull through strategy_level if available
        if hasattr(cfg := cfg, "strategy_level") and "strategy_level" not in sampled:
            sampled["strategy_level"] = cfg.strategy_level
            out["strategy_level"] = cfg.strategy_level

        for key, node in cfg.items():
            if key in {"strategy_level", "description"} or not isinstance(node, DictConfig):
                continue

            c = node.get("class", "")

            # ---------- strategy ----------
            if c == "strategy":
                # For strategy level, pass through without creating extra nesting
                # The technique selections will create the actual structure
                out.update(self._sample_config_recursively(trial, node, sampled, prefix, hierarchical_tree))
                continue

            # ---------- technique ----------
            if c == "technique":
                sel_attr = next((a for a in node.keys() if a.endswith("selection")), None)
                selected_instance = None

                if sel_attr:
                    sel_cfg = node[sel_attr]
                    raw = sel_cfg.param_name
                    arch = sampled.get("architecture_type", "")
                    sel_name = f"{arch}_{raw}" if arch else raw
                    if raw not in sampled:
                        sel_val = self._sample_single_param(trial, sel_cfg, sel_name, sampled)
                        sampled[raw] = sel_val
                        out[sel_name] = sel_val
                        selected_instance = sel_val

                if hasattr(node, "condition") and not self._evaluate_condition(node.condition, sampled):
                    continue

                # Process technique contents, but organize by selected instance
                for sub_key, sub_node in node.items():
                    if sub_key == sel_attr or sub_key in {"class", "condition", "description"}:
                        continue
                    if not isinstance(sub_node, DictConfig):
                        continue

                    sub_class = sub_node.get("class", "")
                    if sub_class == "instance":
                        # Check explicit condition BEFORE sampling parameters
                        if hasattr(sub_node, "condition") and not self._evaluate_condition(sub_node.condition, sampled):
                            continue

                        # Check if this instance should skip its own layer in hierarchy
                        skip_instance_layer = getattr(sub_node, "skip_instance_layer", False)

                        if skip_instance_layer:
                            # Skip instance layer: merge instance content directly into parent
                            sub_out = self._sample_config_recursively(
                                trial, sub_node, sampled, prefix, hierarchical_tree
                            )
                            out.update(sub_out)
                        else:
                            # Normal instance processing: create instance-level subtree
                            instance_tree = {} if hierarchical_tree is not None else None
                            sub_out = self._sample_config_recursively(trial, sub_node, sampled, prefix, instance_tree)
                            out.update(sub_out)

                            # Add to hierarchical tree since condition was already checked
                            if hierarchical_tree is not None:
                                # 检查重复key，避免冲突
                                if sub_key in hierarchical_tree:
                                    raise ValueError(
                                        f"层次化参数构建冲突：key '{sub_key}' 已存在。"
                                        f"这通常是因为多个technique中有相同的instance名称。"
                                        f"当前路径: {prefix}, 已存在的值: {hierarchical_tree[sub_key]}, "
                                        f"新值: {instance_tree}"
                                    )
                                hierarchical_tree[sub_key] = instance_tree
                continue

            # ---------- param ----------
            if c == "param":
                raw = node.param_name
                arch = sampled.get("architecture_type", "")
                name = f"{arch}_{raw}" if arch else raw
                if raw in sampled:
                    continue
                if hasattr(node, "condition") and not self._evaluate_condition(node.condition, sampled):
                    continue

                # 检查是否是stage级或block_stage级参数
                granularity = getattr(node, "granularity", "global")

                if granularity == "stage":
                    # Stage级参数：首先确保regnet_num_stages已采样
                    self._ensure_regnet_num_stages_sampled(trial, sampled, arch)

                    # 展开为多个stage参数
                    stage_params = self._sample_stage_param(trial, node, raw, sampled, arch)
                    sampled[raw] = "stage_expanded"  # 标记已处理
                    out.update(stage_params)

                    # Record in hierarchical tree as list
                    if hierarchical_tree is not None:
                        # 将stage参数转换为list形式
                        stage_list = []
                        num_stages = sampled.get("regnet_num_stages", 0)
                        for stage_idx in range(num_stages):
                            stage_number = stage_idx + 1
                            # 构建正确的参数键名
                            if raw == "stage_block_type":
                                # 对于stage_block_type，参数名格式是 {arch}_stage_block_type_stage_{i}_of_{n}
                                stage_param_key = f"{arch}_{raw}_stage_{stage_number}_of_{num_stages}"
                            else:
                                # 对于其他stage参数，参数名格式是 {arch}_stage_{param}_stage_{i}_of_{n}
                                stage_param_key = f"{arch}_stage_{raw}_stage_{stage_number}_of_{num_stages}"

                            if stage_param_key in stage_params:
                                stage_list.append(stage_params[stage_param_key])
                            else:
                                # 如果找不到对应的参数，使用默认值
                                stage_list.append(None)
                        hierarchical_tree[key] = stage_list

                elif granularity == "block_stage":
                    # Block_stage级参数：首先确保regnet_num_stages已采样
                    self._ensure_regnet_num_stages_sampled(trial, sampled, arch)

                    # 展开为多个stage+block_type参数
                    block_stage_params = self._sample_block_stage_param(trial, node, raw, sampled, arch)
                    sampled[raw] = "block_stage_expanded"  # 标记已处理
                    out.update(block_stage_params)

                    # Record in hierarchical tree as list
                    if hierarchical_tree is not None:
                        # 将block_stage参数转换为list形式
                        stage_list = []
                        num_stages = sampled.get("regnet_num_stages", 0)
                        for stage_idx in range(num_stages):
                            stage_number = stage_idx + 1
                            # 查找该stage的参数
                            stage_value = None
                            for param_key, param_value in block_stage_params.items():
                                if f"_stage_{stage_number}_of_{num_stages}_" in param_key:
                                    stage_value = param_value
                                    break
                            stage_list.append(stage_value)
                        hierarchical_tree[key] = stage_list

                elif granularity == "block_type":
                    # Block_type级参数：为每个不同的block_type生成独立参数
                    block_type_params = self._sample_block_type_param(trial, node, raw, sampled, arch)
                    sampled[raw] = "block_type_expanded"  # 标记已处理
                    out.update(block_type_params)

                    # Record in hierarchical tree as stage list
                    if hierarchical_tree is not None:
                        # 为block_type粒度的activation/norm构建stage列表
                        stage_list = self._build_stage_list_for_block_type(raw, sampled, arch, block_type_params)
                        hierarchical_tree[key] = stage_list

                elif granularity == "stem":
                    # Stem级参数：单个参数，但使用特殊的stem前缀
                    stem_param_name = f"{arch}_{raw}" if arch else raw
                    val = self._sample_single_param(trial, node, stem_param_name, sampled)
                    sampled[raw] = val
                    out[stem_param_name] = val

                    # Record in hierarchical tree with stem_ prefix
                    if hierarchical_tree is not None:
                        stem_key = f"stem_{key}"  # 使用stem_activation, stem_normalization等
                        hierarchical_tree[stem_key] = val

                else:
                    # 普通参数（global, block_type等）
                    val = self._sample_single_param(trial, node, name, sampled)
                    sampled[raw] = val
                    out[name] = val

                    # Record in hierarchical tree using the YAML key name
                    if hierarchical_tree is not None:
                        hierarchical_tree[key] = val
                continue

            # ---------- instance ----------
            if c == "instance":
                # Check explicit condition
                if hasattr(node, "condition") and not self._evaluate_condition(node.condition, sampled):
                    continue

                # Create instance-level subtree
                instance_tree = {} if hierarchical_tree is not None else None
                out.update(self._sample_config_recursively(trial, node, sampled, prefix, instance_tree))
                if hierarchical_tree is not None and instance_tree:
                    hierarchical_tree[key] = instance_tree
                continue

            # ---------- fallback ----------
            if any(isinstance(v, DictConfig) for v in node.values() if hasattr(node, "values")):
                out.update(self._sample_config_recursively(trial, node, sampled, prefix, hierarchical_tree))

        return out

    def _merge_min_max_pairs(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge _min and _max parameter pairs into tuples
        """
        merged_results = {}
        min_params = {}
        max_params = {}

        # First pass: collect all _min and _max parameters
        for param_name, value in results.items():
            if param_name.endswith("_min"):
                base_name = param_name[:-4]  # Remove '_min'
                min_params[base_name] = value
            elif param_name.endswith("_max"):
                base_name = param_name[:-4]  # Remove '_max'
                max_params[base_name] = value
            else:
                # Keep non-min/max parameters as is
                merged_results[param_name] = value

        # Second pass: merge min/max pairs
        for base_name in min_params:
            if base_name in max_params:
                # Both min and max exist, create tuple
                merged_results[base_name] = (min_params[base_name], max_params[base_name])
            else:
                # Only min exists, keep as is
                merged_results[f"{base_name}_min"] = min_params[base_name]

        # Add remaining max parameters that don't have min counterparts
        for base_name in max_params:
            if base_name not in min_params:
                merged_results[f"{base_name}_max"] = max_params[base_name]

        return merged_results

    def _merge_min_max_hierarchical(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge _min and _max parameter pairs in hierarchical structure
        """
        if not isinstance(tree, dict):
            return tree

        # First, recursively process all nested dictionaries
        processed_tree = {}
        for key, value in tree.items():
            if isinstance(value, dict):
                processed_tree[key] = self._merge_min_max_hierarchical(value)
            else:
                processed_tree[key] = value

        # Then apply min/max merging to the current level
        return self._merge_min_max_pairs(processed_tree)

    def sample_all_params(
        self, trial: optuna.Trial, categories: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Sample parameters for all or specified search space categories
        """
        if categories is None:
            categories = self.search_space_categories

        results = {}
        global_sampled_params = {}  # Track parameters across all categories

        # --- pre-sample architecture_type so that later params get correct prefix/dependency ---
        if "architectures" in categories and "architecture_type" not in global_sampled_params:
            try:
                arch_strategy_level = self.search_spaces.architectures.architectures.strategy_level
                arch_sel_cfg = self.search_spaces.architectures.architectures.architecture_selection.selection
                arch_val = self._sample_single_param(
                    trial, arch_sel_cfg, "architecture_type", {"strategy_level": arch_strategy_level}
                )
                global_sampled_params["architecture_type"] = arch_val
            except Exception as e:
                raise RuntimeError(f"Failed to pre-sample architecture_type: {e}")

        # Now sample each requested category
        for category in categories:
            if category not in self.search_space_categories:
                continue

            cat_cfg = getattr(self.search_spaces, category)
            cat_sampled = dict(global_sampled_params)  # local context
            cat_res = self._sample_config_recursively(trial, cat_cfg, cat_sampled, "")
            cat_res = self._merge_min_max_pairs(cat_res)
            results[category] = cat_res
            global_sampled_params.update(cat_res)

        return results

    def sample_all_params_hierarchical(
        self, trial: optuna.Trial, categories: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Sample parameters and return both flat and hierarchical representations.
        Returns: {"flat": flat_results, "hierarchical": hierarchical_results}
        """
        if categories is None:
            categories = self.search_space_categories

        flat_results = {}
        hierarchical_results = {}
        global_sampled_params = {}

        # --- pre-sample architecture_type so that later params get correct prefix/dependency ---
        if "architectures" in categories and "architecture_type" not in global_sampled_params:
            try:
                arch_strategy_level = self.search_spaces.architectures.architectures.strategy_level
                arch_sel_cfg = self.search_spaces.architectures.architectures.architecture_selection.selection
                arch_val = self._sample_single_param(
                    trial, arch_sel_cfg, "architecture_type", {"strategy_level": arch_strategy_level}
                )
                global_sampled_params["architecture_type"] = arch_val
            except Exception as e:
                raise RuntimeError(f"Failed to pre-sample architecture_type: {e}")

        # Now sample each requested category with hierarchical tree tracking
        for category in categories:
            if category not in self.search_space_categories:
                continue

            cat_cfg = getattr(self.search_spaces, category)
            cat_sampled = dict(global_sampled_params)  # local context

            # Create hierarchical tree for this category
            category_tree = {}
            cat_res = self._sample_config_recursively(trial, cat_cfg, cat_sampled, "", category_tree)
            cat_res = self._merge_min_max_pairs(cat_res)

            # Add architecture_type to both flat and hierarchical results if it exists
            if category == "architectures" and "architecture_type" in global_sampled_params:
                cat_res["architecture_type"] = global_sampled_params["architecture_type"]
                category_tree["architecture_type"] = global_sampled_params["architecture_type"]

            flat_results[category] = cat_res
            # Apply min/max merging to hierarchical structure as well
            merged_hierarchical_tree = self._merge_min_max_hierarchical(category_tree)
            hierarchical_results[category] = merged_hierarchical_tree
            global_sampled_params.update(cat_res)

        return {"flat": flat_results, "hierarchical": hierarchical_results}

    # =============================================================================
    # DEPENDENCY GRAPH AND TOPOLOGICAL SORTING
    # =============================================================================

    def _build_dependency_graph(self, cfg: DictConfig) -> Dict[str, List[str]]:
        """构建参数依赖图，返回正确的采样顺序"""
        graph = defaultdict(list)  # param -> [依赖于它的参数]
        in_degree = defaultdict(int)  # param -> 依赖数量
        all_params = set()

        def extract_dependencies(node: DictConfig, prefix: str = ""):
            """递归提取参数依赖关系"""
            for key, child in node.items():
                if not isinstance(child, DictConfig):
                    continue

                class_type = child.get("class", "")

                # 只处理param节点
                if class_type == "param":
                    param_name = child.get("param_name", key)
                    all_params.add(param_name)

                    # 检查depends_on字段
                    if hasattr(child, "depends_on"):
                        dependencies = child.depends_on
                        if isinstance(dependencies, (list, tuple)):
                            for dep in dependencies:
                                all_params.add(dep)
                                graph[dep].append(param_name)
                                in_degree[param_name] += 1
                        elif isinstance(dependencies, str):
                            all_params.add(dependencies)
                            graph[dependencies].append(param_name)
                            in_degree[param_name] += 1

                    # 特殊处理：stage级参数自动依赖regnet_num_stages
                    if hasattr(child, "granularity") and child.granularity in ["stage", "block_stage"]:
                        dep = "regnet_num_stages"
                        all_params.add(dep)
                        if param_name not in graph[dep]:  # 避免重复添加
                            graph[dep].append(param_name)
                            in_degree[param_name] += 1

                # 递归处理子节点
                extract_dependencies(child, f"{prefix}{key}_")

        # 从配置中提取所有依赖关系
        extract_dependencies(cfg)

        # 执行拓扑排序
        queue = deque([param for param in all_params if in_degree[param] == 0])
        sorted_order = []

        while queue:
            param = queue.popleft()
            sorted_order.append(param)

            for dependent in graph[param]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # 检查循环依赖
        if len(sorted_order) != len(all_params):
            remaining = all_params - set(sorted_order)
            raise ValueError(f"检测到循环依赖或孤立节点: {remaining}")

        return sorted_order

    def _get_sorted_config_items(self, cfg: DictConfig) -> List[Tuple[str, DictConfig]]:
        """根据拓扑排序返回配置项的正确顺序"""
        if self._dependency_graph is None:
            self._dependency_graph = self._build_dependency_graph(cfg)

        # 构建参数名到配置节点的映射
        param_to_config = {}
        config_items = []

        def map_params_to_configs(node: DictConfig, prefix: str = ""):
            """递归构建参数到配置的映射"""
            for key, child in node.items():
                if not isinstance(child, DictConfig):
                    continue

                class_type = child.get("class", "")

                if class_type == "param":
                    param_name = child.get("param_name", key)
                    param_to_config[param_name] = (key, child)

                # 非param节点直接加入列表（保持原顺序）
                if class_type in ["strategy", "technique", "instance"]:
                    config_items.append((key, child))

                # 递归处理子节点
                map_params_to_configs(child, f"{prefix}{key}_")

        # 构建映射
        map_params_to_configs(cfg)

        # 按拓扑排序顺序添加param节点
        sorted_param_items = []
        for param_name in self._dependency_graph:
            if param_name in param_to_config:
                sorted_param_items.append(param_to_config[param_name])

        # 返回：非param节点(原顺序) + param节点(拓扑排序)
        return config_items + sorted_param_items
