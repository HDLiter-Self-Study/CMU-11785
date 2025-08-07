"""
Main search space sampler class for Optuna-based hyperparameter optimization.

This module provides the main SearchSpaceSampler class that orchestrates the entire
parameter sampling process. It supports hierarchical configuration sampling with
architecture-aware parameters and multiple granularity levels.
"""

from typing import Dict, Any, List, Optional, Set
import inspect
import optuna
from omegaconf import DictConfig

from config.config_manager import get_config
from .parameter_naming import ParameterNaming
from .safe_evaluator import SafeEvaluator
from .granularity_handler import GranularityHandler
from .dependency_manager import DependencyManager
from .enums import GranularityLevel, ConfigClass, parse_config_class


class SearchSpaceSampler:
    """
    Main search space sampler class for Optuna-based hyperparameter optimization.
    """

    PARAM_TYPE_REQUIREMENTS = {
        "categorical": {"choices"},
        "float": {"low", "high"},
        "int": {"low", "high"},
    }

    TYPE_CAST_FUNCTIONS = {
        "float": float,
        "int": int,
    }

    def __init__(self, config_name: str = "main", overrides: Optional[List[str]] = None, silent: bool = False):
        """
        Initialize the search space sampler.
        """
        self.config: DictConfig = get_config(config_name, overrides)
        self.search_spaces: DictConfig = self.config.search_spaces
        self.search_space_categories = list(self.search_spaces.keys())
        self.silent = silent

        # Initialize helper components
        self.evaluator = SafeEvaluator()
        self.granularity_handler = GranularityHandler(self, silent=self.silent)
        self.dependency_manager = DependencyManager(self, silent=self.silent)
        self.naming = ParameterNaming()

        self.current_architecture_type: Optional[str] = None
        self.current_num_stages: Optional[int] = None
        self._trial_in_progress: bool = False

    def _log(self, message: str):
        """Prints a log message if the silent flag is not set."""
        if not self.silent:
            print(message)

    def sample_all_params(
        self, trial: optuna.Trial, categories: Optional[List[str]] = None, include_hierarchical: bool = True
    ) -> Dict[str, Any]:
        """
        Sample parameters for all or specified search space categories.
        """
        self._reset_trial_state()
        self._trial_in_progress = True

        try:
            if categories is None:
                categories = self.search_space_categories

            flat_results = {}
            hierarchical_results = {} if include_hierarchical else None
            global_sampled_params = {}

            self._pre_sample_and_set_architecture_info(trial, categories, global_sampled_params)

            for category in categories:
                if category not in self.search_space_categories:
                    continue

                cat_cfg = getattr(self.search_spaces, category)
                cat_sampled = dict(global_sampled_params)
                category_tree = {} if include_hierarchical else None

                cat_res = self._process_strategy_node(trial, cat_cfg, cat_sampled, "", category_tree)
                cat_res = self._merge_min_max_pairs(cat_res)

                if category == "architectures" and "architecture_type" in global_sampled_params:
                    cat_res["architecture_type"] = global_sampled_params["architecture_type"]
                    if include_hierarchical and category_tree is not None:
                        category_tree["architecture_type"] = global_sampled_params["architecture_type"]

                flat_results[category] = cat_res
                if include_hierarchical and category_tree is not None:
                    hierarchical_results[category] = self._merge_min_max_hierarchical(category_tree)

            if include_hierarchical:
                return {"flat": flat_results, "hierarchical": hierarchical_results}
            else:
                return flat_results
        finally:
            self._trial_in_progress = False

    def _pre_sample_and_set_architecture_info(
        self, trial: optuna.Trial, categories: List[str], global_sampled_params: Dict[str, Any]
    ) -> None:
        if "architectures" not in categories:
            return
        try:
            arch_strategy_level = self.search_spaces.architectures.strategy_level
            arch_sel_cfg = self.search_spaces.architectures.architecture_selection.selection
            arch_val = self._sample_single_param(
                trial, arch_sel_cfg, "architecture_type", {"strategy_level": arch_strategy_level}
            )
            num_stages_sel_cfg = self.search_spaces.architectures.num_stages_selection.selection
            num_stages_val = self._sample_single_param(
                trial, num_stages_sel_cfg, "num_stages", {"strategy_level": arch_strategy_level}
            )
            self.current_architecture_type = arch_val
            self.current_num_stages = num_stages_val
            global_sampled_params["architecture_type"] = arch_val
            global_sampled_params["num_stages"] = num_stages_val
        except Exception as e:
            raise RuntimeError(f"Failed to pre-sample architecture_type: {e}")

    def _reset_trial_state(self) -> None:
        self.current_architecture_type = None
        self.current_num_stages = None

    @property
    def architecture_type(self) -> str:
        if not self._trial_in_progress or self.current_architecture_type is None:
            raise RuntimeError("architecture_type is only available during an active trial")
        return self.current_architecture_type

    @property
    def num_stages(self) -> int:
        if not self._trial_in_progress or self.current_num_stages is None:
            raise RuntimeError("num_stages is only available during an active trial")
        return self.current_num_stages

    def _process_strategy_node(
        self,
        trial: optuna.Trial,
        cfg: DictConfig,
        sampled: Dict[str, Any],
        prefix: str = "",
        hierarchical_tree: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        out = {}

        # Use the DependencyManager to get a topologically sorted list of technique nodes.
        sorted_techniques = self.dependency_manager.get_sorted_config_tree(cfg)
        self._log(f"🔗 [STRATEGY] Processing {len(sorted_techniques)} techniques in dependency order.")

        if hasattr(cfg, "strategy_level") and "strategy_level" not in sampled:
            sampled["strategy_level"] = cfg.strategy_level
            out["strategy_level"] = cfg.strategy_level
        elif "strategy_level" not in sampled and hasattr(cfg, "architectures"):
            if hasattr(cfg.architectures, "strategy_level"):
                sampled["strategy_level"] = cfg.architectures.strategy_level
                out["strategy_level"] = cfg.architectures.strategy_level

        # Iterate over the sorted list of techniques.
        for technique_node in sorted_techniques:
            key = technique_node["_original_key"]
            technique_out = {}
            self._process_technique_node(trial, technique_node, key, sampled, prefix, hierarchical_tree, technique_out)

            # Accumulate sampled results for subsequent techniques
            sampled.update(technique_out)
            out.update(technique_out)

        return out

    def _process_technique_node(
        self,
        trial: optuna.Trial,
        node: DictConfig,
        key: str,
        sampled: Dict[str, Any],
        prefix: str,
        hierarchical_tree: Optional[Dict[str, Any]],
        out: Dict[str, Any],
    ) -> None:
        if "selection" in node:
            sel_cfg = node["selection"]
            raw = sel_cfg.param_name
            arch = self.architecture_type
            sel_name = self.naming.build_param_name(arch, raw)
            if raw not in sampled:
                sel_val = self._sample_single_param(trial, sel_cfg, sel_name, sampled)
                sampled[raw] = sel_val
                out[sel_name] = sel_val

        if hasattr(node, "condition"):
            if not self.evaluator.evaluate_condition(node.condition, sampled):
                return

        # Iterate over the ordered children provided by the DependencyManager.
        for sub_node in node.get("_ordered_children", []):
            sub_key = sub_node["_original_key"]

            # Skip the 'selection' node as it's a meta-property, not a child instance.
            if sub_key == "selection":
                continue

            sub_node_class = parse_config_class(sub_node["class"])

            if sub_node_class == ConfigClass.INSTANCE.value:
                self._process_instance_node(trial, sub_node, sub_key, sampled, prefix, hierarchical_tree, out)
            else:
                raise ValueError(
                    f"Only instance nodes are allowed at technique level, not {sub_node_class} for {sub_key}"
                )

    def _process_param_node(
        self,
        trial: optuna.Trial,
        node: DictConfig,
        key: str,
        sampled: Dict[str, Any],
        hierarchical_tree: Optional[Dict[str, Any]],
        out: Dict[str, Any],
    ) -> None:
        raw = node.param_name
        arch = self.architecture_type
        name = self.naming.build_param_name(arch, raw)

        if raw in sampled:
            return
        if hasattr(node, "condition") and not self.evaluator.evaluate_condition(node.condition, sampled):
            return

        granularity = getattr(node, "granularity", GranularityLevel.GLOBAL.value)

        if granularity == GranularityLevel.STAGE.value:
            stage_params = self.granularity_handler.sample_stage_params(trial, node, raw, sampled, arch)
            sampled[raw] = "stage_expanded"
            out.update(stage_params)
            if hierarchical_tree is not None:
                stage_list = self._build_stage_list(raw, sampled, arch, stage_params)
                if raw == "stage_block_type_selection":
                    hierarchical_tree["block_type"] = stage_list
                hierarchical_tree[key] = stage_list
        elif granularity == GranularityLevel.BLOCK_STAGE.value:
            block_stage_params = self.granularity_handler.sample_block_stage_params(trial, node, raw, sampled, arch)
            sampled[raw] = "block_stage_expanded"
            out.update(block_stage_params)
            if hierarchical_tree is not None:
                hierarchical_tree[key] = self._build_block_stage_list(raw, sampled, arch, block_stage_params)
        elif granularity == GranularityLevel.BLOCK_TYPE.value:
            block_type_params = self.granularity_handler.sample_block_type_params(trial, node, raw, sampled, arch)
            sampled[raw] = "block_type_expanded"
            out.update(block_type_params)
            if hierarchical_tree is not None:
                # The new GranularityHandler logic does not require hierarchical_block_type anymore.
                # It deduces the block types directly from the `sampled` dictionary.
                stage_list = self.granularity_handler.build_stage_list_for_block_type(
                    raw, sampled, arch, block_type_params
                )
                hierarchical_tree[key] = stage_list
        elif granularity == GranularityLevel.STEM.value:
            stem_param_name = self.naming.build_param_name(arch, raw)
            val = self._sample_single_param(trial, node, stem_param_name, sampled)
            sampled[raw] = val
            out[stem_param_name] = val
            if hierarchical_tree is not None:
                hierarchical_tree[f"stem_{key}"] = val
        else:  # Global
            val = self._sample_single_param(trial, node, name, sampled)
            sampled[raw] = val
            out[name] = val
            if hierarchical_tree is not None:
                if granularity == GranularityLevel.GLOBAL.value and key in ["activation", "normalization"]:
                    num_stages = self.num_stages
                    hierarchical_tree[key] = [val] * num_stages if num_stages > 0 else val
                else:
                    hierarchical_tree[key] = val

    def _process_instance_node(
        self,
        trial: optuna.Trial,
        node: DictConfig,
        key: str,
        sampled: Dict[str, Any],
        prefix: str,
        hierarchical_tree: Optional[Dict[str, Any]],
        out: Dict[str, Any],
    ) -> None:
        if hasattr(node, "condition") and not self.evaluator.evaluate_condition(node.condition, sampled):
            return

        skip_instance_layer = getattr(node, "skip_instance_layer", False)

        if skip_instance_layer:
            sub_out = self._process_instance_content(trial, node, sampled, prefix, hierarchical_tree)
            out.update(sub_out)
        else:
            instance_tree = {} if hierarchical_tree is not None else None
            sub_out = self._process_instance_content(trial, node, sampled, prefix, instance_tree)
            out.update(sub_out)
            if hierarchical_tree is not None and instance_tree:
                if key in hierarchical_tree:
                    raise ValueError(f"Hierarchical parameter conflict: key '{key}' already exists.")
                hierarchical_tree[key] = instance_tree

    def _process_instance_content(
        self,
        trial: optuna.Trial,
        node: DictConfig,
        sampled: Dict[str, Any],
        prefix: str,
        hierarchical_tree: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        out = {}
        # Iterate over the ordered children (params) provided by the DependencyManager.
        for child_node in node.get("_ordered_children", []):
            key = child_node["_original_key"]
            child_class = parse_config_class(child_node["class"])

            if child_class == ConfigClass.PARAM.value:
                self._process_param_node(trial, child_node, key, sampled, hierarchical_tree, out)
            else:
                raise ValueError(f"Only param nodes are allowed at instance level, not {child_class} for {key}")
        return out

    def _sample_single_param(
        self, trial: optuna.Trial, param_config: DictConfig, param_name: str, sampled_params: Dict[str, Any]
    ) -> Any:
        param_type = param_config.type
        suggest_fn = getattr(trial, f"suggest_{param_type}", None)
        if not callable(suggest_fn):
            raise ValueError(f"Unsupported parameter type: {param_type}")
        accepted_keys = set(inspect.signature(suggest_fn).parameters) - {"name"}
        kwargs = self._build_suggest_kwargs(param_config, param_name, sampled_params, param_type, accepted_keys)
        if param_type == "categorical":
            choices = kwargs.pop("choices")
            return suggest_fn(param_name, choices, **{k: v for k, v in kwargs.items() if k in accepted_keys})
        return suggest_fn(param_name, **{k: v for k, v in kwargs.items() if k in accepted_keys})

    def _build_suggest_kwargs(
        self,
        param_config: DictConfig,
        param_name: str,
        sampled_params: Dict[str, Any],
        param_type: str,
        accepted_keys: Set[str],
    ) -> Dict[str, Any]:
        kwargs = {}
        for key, raw_value in param_config.items():
            if key != "choices" and key not in accepted_keys:
                continue
            resolved_value = self._resolve_param_value(
                raw_value, param_config, sampled_params, f"{key} for {param_name}"
            )
            final_value = self.evaluator.resolve_dynamic_value(resolved_value, sampled_params)
            if key in {"low", "high"}:
                try:
                    final_value = self.TYPE_CAST_FUNCTIONS[param_type](final_value)
                except Exception as e:
                    raise ValueError(f"{param_name}.{key}={final_value} not {param_type}: {e}")
            elif key == "choices":
                final_value = self._normalize_choices(final_value, param_name)
                if not final_value:
                    raise ValueError(f"{param_name}.choices empty")
            kwargs[key] = final_value
        missing_keys = self.PARAM_TYPE_REQUIREMENTS.get(param_type, set()) - kwargs.keys()
        if missing_keys:
            raise ValueError(f"{param_name} missing required key(s): {missing_keys}")
        return kwargs

    def _resolve_param_value(
        self, value: Any, param_config: DictConfig, sampled_params: Dict[str, Any], param_name: str = ""
    ) -> Any:
        if not self._is_dict_like(value):
            return value
        dependency_order = getattr(param_config, "dependency_order", [])
        if not dependency_order:
            raise ValueError(f"No dependency order found for {param_name}")
        current_value = value
        for dep_param in dependency_order:
            if dep_param not in sampled_params:
                raise ValueError(f"Dependency '{dep_param}' not found for {param_name}")
            key = sampled_params[dep_param]
            if not self._is_dict_like(current_value) or key not in current_value:
                raise ValueError(f"Cannot resolve dependency for {param_name}")
            current_value = current_value[key]
        return current_value

    def _normalize_choices(self, choices: Any, param_name: str = "") -> List[Any]:
        if hasattr(choices, "_content"):
            choices = list(choices)
        elif not isinstance(choices, list):
            choices = [choices] if choices is not None else []
        if not choices:
            raise ValueError(f"No valid choices for {param_name}")
        return choices

    def _is_dict_like(self, value: Any) -> bool:
        return isinstance(value, dict) or hasattr(value, "keys")

    def _build_stage_list(
        self, param_name: str, sampled_params: Dict[str, Any], arch_prefix: str, stage_params: Dict[str, Any]
    ) -> List[Any]:
        """Builds a list of values for each stage from the given stage_params dict."""
        num_stages = self.num_stages
        stage_list = []
        for stage_idx in range(num_stages):
            stage_number = stage_idx + 1
            # Build the key exactly as it was built during sampling to find the correct value.
            stage_param_key = self.naming.build_stage_param_name(arch_prefix, param_name, stage_number, num_stages)
            stage_list.append(stage_params.get(stage_param_key))
        return stage_list

    def _build_block_stage_list(
        self, param_name: str, sampled_params: Dict[str, Any], arch_prefix: str, block_stage_params: Dict[str, Any]
    ) -> List[Any]:
        num_stages = self.num_stages
        stage_list = []
        for stage_idx in range(num_stages):
            stage_number = stage_idx + 1
            stage_value = None
            for param_key, param_value in block_stage_params.items():
                if f"_stage_{stage_number}_of_{num_stages}_" in param_key:
                    stage_value = param_value
                    break
            stage_list.append(stage_value)
        return stage_list

    def _merge_min_max_pairs(self, results: Dict[str, Any]) -> Dict[str, Any]:
        merged_results = {}
        min_params = {k[:-4]: v for k, v in results.items() if k.endswith("_min")}
        max_params = {k[:-4]: v for k, v in results.items() if k.endswith("_max")}
        for k, v in results.items():
            if not k.endswith("_min") and not k.endswith("_max"):
                merged_results[k] = v
        for base_name, min_val in min_params.items():
            if base_name in max_params:
                merged_results[base_name] = (min_val, max_params[base_name])
            else:
                merged_results[f"{base_name}_min"] = min_val
        for base_name, max_val in max_params.items():
            if base_name not in min_params:
                merged_results[f"{base_name}_max"] = max_val
        return merged_results

    def _merge_min_max_hierarchical(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(tree, dict):
            return tree
        processed_tree = {k: self._merge_min_max_hierarchical(v) if isinstance(v, dict) else v for k, v in tree.items()}
        return self._merge_min_max_pairs(processed_tree)
