"""
Main search space sampler for Optuna-based hyperparameter optimization.

Key design (Phase 2):
- Global dependency-managed execution: the top-level `search_spaces` container is
  passed to the dependency manager, which sorts STRATEGY nodes by same-level
  `depends_on`. This guarantees `architectures` runs before others.
- No pre-sampling: special keys like `architecture_type` and `num_stages` are
  sampled as normal PARAMs under `architectures`.
- Global keys: PARAMs can declare `arch_irrelevant: true` (no architecture prefix)
  and `export_to_global: true` (exposed via `self.globals`). The sampler then uses
  these to build prefixed parameter names for architecture-aware params.
- Granularity support is delegated to `GranularityHandler`.
"""

from typing import Dict, Any, List, Optional, Set
import inspect
import optuna
from omegaconf import DictConfig, OmegaConf, ListConfig

from config.config_manager import get_config
from .parameter_naming import ParameterNaming
from .safe_evaluator import SafeEvaluator
from .granularity_handler import GranularityHandler
from .dependency_manager import DependencyManager
from .enums import GranularityLevel, ConfigClass, parse_config_class


class SearchSpaceSampler:
    """
    Orchestrates dependency-ordered sampling across all strategies and techniques.

    Responsibilities:
    - Build globally sorted strategy list (via DependencyManager) and traverse it.
    - Sample PARAMs respecting per-node conditions and `depends_on` ordering.
    - Apply naming rules: `arch_irrelevant` (no prefix) vs architecture-aware (prefixed).
    - Maintain `globals` for cross-strategy keys (e.g., architecture_type, num_stages).
    - Expand stage/block-type/block-stage parameters using `GranularityHandler`.
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

    def __init__(
        self,
        config_name: str = "main",
        overrides: Optional[List[str]] = None,
        silent: bool = False,
    ):
        """
        Initialize the search space sampler.
        """
        # Compose config using Hydra overrides
        self.config: DictConfig = get_config(config_name, overrides)
        self.search_spaces: DictConfig = self.config.search_spaces
        self.search_space_categories = list(self.search_spaces.keys())
        self.silent = silent

        # Initialize helper components
        self.evaluator = SafeEvaluator()
        self.granularity_handler = GranularityHandler(self, silent=self.silent)
        self.dependency_manager = DependencyManager(self, silent=self.silent)
        self.naming = ParameterNaming()

        # Global key-value store for cross-strategy parameters (e.g., architecture_type, num_stages)
        self.globals: Dict[str, Any] = {}
        self._trial_in_progress: bool = False
        self._cached_eval_context: Optional[Dict[str, Any]] = None

    def _log(self, message: str):
        """Prints a log message if the silent flag is not set."""
        if not self.silent:
            print(message)

    def sample_all_params(self, trial: optuna.Trial, include_hierarchical: bool = True) -> Dict[str, Any]:
        """
        Sample parameters for all strategies in the order determined by global
        dependency analysis. Returns both flat and hierarchical views when requested.
        """
        self._reset_trial_state()
        self._trial_in_progress = True

        try:
            # Global sorted tree (strategies at top-level, techniques/instances/params in order)
            sorted_root = self.dependency_manager.get_sorted_config_tree(self.search_spaces)

            flat_results: Dict[str, Dict[str, Any]] = {}
            hierarchical_results: Optional[Dict[str, Any]] = {} if include_hierarchical else None
            sampled_params: Dict[str, Any] = {}
            # Seed context with any pre-populated globals (e.g., task from wrapper)
            if self.globals:
                sampled_params.update(self.globals)

            # No pre-pass: global ordering with same-level `depends_on` ensures
            # `architectures` (and its selection keys) are sampled first.

            for strategy_node in sorted_root:
                category = strategy_node["_original_key"]
                # Unified hierarchical structure: always a list of technique groups
                category_tree: Optional[List[Dict[str, Any]]] = [] if include_hierarchical else None
                category_out: Dict[str, Any] = {}

                self._process_sorted_strategy_node(
                    trial=trial,
                    strategy_node=strategy_node,
                    sampled=sampled_params,
                    hierarchical_tree=category_tree,  # type: ignore[arg-type]
                    out=category_out,
                )

                flat_results[category] = self._merge_min_max_pairs(category_out)
                if include_hierarchical and category_tree is not None:
                    hierarchical_results[category] = self._merge_min_max_hierarchical(category_tree)

            if include_hierarchical:
                return {"flat": flat_results, "hierarchical": hierarchical_results}
            return flat_results
        finally:
            self._trial_in_progress = False

    # pre-sample flow removed in global dependency-driven flow

    def _reset_trial_state(self) -> None:
        # Preserve externally injected globals (e.g., task) across trials
        preserved_keys = {"task"}
        preserved = {k: v for k, v in self.globals.items() if k in preserved_keys}
        self.globals = preserved
        self._cached_eval_context = None

    def _get_eval_context(self, sampled: Dict[str, Any]) -> Dict[str, Any]:
        """Return merged evaluator context, caching within a trial until sampled changes."""
        # Simple cache invalidation by length match; sufficient since we only ever add keys
        if self._cached_eval_context is None or len(self._cached_eval_context) != (len(self.globals) + len(sampled)):
            ctx = dict(self.globals)
            ctx.update(sampled)
            self._cached_eval_context = ctx
        return self._cached_eval_context

    @property
    def architecture_type(self) -> str:
        if not self._trial_in_progress or "architecture_type" not in self.globals:
            raise RuntimeError("architecture_type is only available after it has been sampled in the current trial")
        return str(self.globals["architecture_type"])

    @property
    def num_stages(self) -> int:
        if not self._trial_in_progress or "num_stages" not in self.globals:
            raise RuntimeError("num_stages is only available after it has been sampled in the current trial")
        return int(self.globals["num_stages"])

    def _process_sorted_strategy_node(
        self,
        trial: optuna.Trial,
        strategy_node: DictConfig,
        sampled: Dict[str, Any],
        hierarchical_tree: Optional[List[Dict[str, Any]]],
        out: Dict[str, Any],
    ) -> None:
        """Process a STRATEGY node already sorted by the dependency manager.

        - Writes `strategy_level` into outputs when present
        - Iterates over technique nodes in their dependency-sorted order
        - Accumulates sampled values into `sampled` and `out`
        """
        # Persist strategy level if present
        if hasattr(strategy_node, "strategy_level") and "strategy_level" not in sampled:
            sampled["strategy_level"] = strategy_node.strategy_level
            out["strategy_level"] = strategy_node.strategy_level

        # Process techniques in dependency order (already sorted by DM)
        grouped_mode = isinstance(hierarchical_tree, list)
        for technique_node in strategy_node.get("_ordered_children", []):
            key = technique_node["_original_key"]
            technique_out: Dict[str, Any] = {}

            if grouped_mode:
                # Build per-technique hierarchical subtree (instances collected under their names)
                group_tree: Dict[str, Any] = {}
                self._process_sorted_technique_node(
                    trial=trial,
                    node=technique_node,
                    key=key,
                    sampled=sampled,
                    hierarchical_tree=group_tree,
                    out=technique_out,
                    grouped_mode=True,
                )
                sampled.update(technique_out)
                out.update(technique_out)

                # Construct the group record
                selection_value = technique_out.get("selection") if isinstance(technique_out, dict) else None
                instances: Dict[str, Any] = {}
                for inst_name, inst_cfg in group_tree.items():
                    if inst_name == "selection":
                        continue
                    instances[inst_name] = inst_cfg if inst_cfg is not None else {}
                # Drop empty groups
                if hierarchical_tree is not None and not (selection_value is None and not instances):
                    hierarchical_tree.append({"selection": selection_value, "instances": instances})
            else:
                self._process_sorted_technique_node(
                    trial=trial,
                    node=technique_node,
                    key=key,
                    sampled=sampled,
                    hierarchical_tree=None,
                    out=technique_out,
                    grouped_mode=False,
                )
                sampled.update(technique_out)
                out.update(technique_out)

    def _process_sorted_technique_node(
        self,
        trial: optuna.Trial,
        node: DictConfig,
        key: str,
        sampled: Dict[str, Any],
        hierarchical_tree: Optional[Dict[str, Any]],
        out: Dict[str, Any],
        grouped_mode: bool,
    ) -> None:
        """Process a TECHNIQUE node and its children.

        - Handles an optional `selection` PARAM before other instances
        - Respects technique-level `condition`
        - Delegates INSTANCE processing to `_process_instance_node`
        """
        # Respect technique-level condition BEFORE any selection sampling
        if hasattr(node, "condition"):
            context = self._get_eval_context(sampled)
            if not self.evaluator.evaluate_condition(node.condition, context):
                return

        # Handle technique-level selection param if exists
        if "selection" in node:
            sel_cfg = node["selection"]
            # In grouped mode, selection is kept out of the instance subtree and surfaced via out["selection"]
            self._process_param_node(
                trial,
                sel_cfg,
                "selection",
                sampled,
                None if grouped_mode else hierarchical_tree,
                out,
            )
            if grouped_mode:
                try:
                    raw_name = sel_cfg.param_name
                    if raw_name in sampled:
                        out["selection"] = sampled[raw_name]
                except Exception:
                    pass

        # Iterate over the ordered children provided by the DependencyManager.
        for sub_node in node.get("_ordered_children", []):
            sub_key = sub_node["_original_key"]

            # Skip the 'selection' node as it's a meta-property, not a child instance.
            if sub_key == "selection":
                continue

            sub_node_class = parse_config_class(sub_node["class"])

            if sub_node_class == ConfigClass.INSTANCE.value:
                self._process_instance_node(
                    trial,
                    sub_node,
                    sub_key,
                    sampled,
                    None,
                    hierarchical_tree if grouped_mode else None,
                    out,
                )
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
        """Process a PARAM node with unified handling of naming and granularity.

        Naming rules:
        - If `arch_irrelevant: true`, use raw `param_name` without prefix
        - Else, the architecture prefix from `globals['architecture_type']` is applied
        - If `export_to_global: true`, store the sampled raw value into `self.globals`

        Granularity rules:
        - stage/block_type/block_stage/stem are expanded via `GranularityHandler`
        - global values are placed directly
        """
        raw = node.param_name
        arch_irrelevant = bool(getattr(node, "arch_irrelevant", False))
        export_to_global = bool(getattr(node, "export_to_global", False))

        if arch_irrelevant:
            name = raw
        else:
            arch = self.architecture_type
            name = self.naming.build_param_name(arch, raw)

        if raw in sampled:
            return
        if hasattr(node, "condition"):
            context = self._get_eval_context(sampled)
            if not self.evaluator.evaluate_condition(node.condition, context):
                return

        granularity = getattr(node, "granularity", GranularityLevel.GLOBAL.value)

        if granularity == GranularityLevel.STAGE.value:
            stage_params = self.granularity_handler.sample_stage_params(trial, node, raw, sampled, arch)
            sampled[raw] = "stage_expanded"
            out.update(stage_params)
            if hierarchical_tree is not None:
                # Use ParameterNaming helper to extract ordered values
                stage_list = self.naming.extract_stage_values_from_params(arch, raw, self.num_stages, stage_params)
                if raw == "stage_block_type_selection":
                    hierarchical_tree["block_type"] = stage_list
                hierarchical_tree[key] = stage_list
        elif granularity == GranularityLevel.BLOCK_STAGE.value:
            block_stage_params = self.granularity_handler.sample_block_stage_params(trial, node, raw, sampled, arch)
            sampled[raw] = "block_stage_expanded"
            out.update(block_stage_params)
            if hierarchical_tree is not None:
                stage_list = self.naming.extract_block_stage_values_from_params(
                    arch, raw, self.num_stages, block_stage_params
                )
                hierarchical_tree[key] = stage_list
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
                # When hierarchical_tree is a dict (grouped-mode per-instance subtree), write as-is
                # For global strategy-level grouping we collect values via technique_out
                hierarchical_tree[key] = val

        # Export to global namespace if requested
        if export_to_global:
            if raw in self.globals and self.globals[raw] != sampled.get(raw):
                raise ValueError(f"Global key '{raw}' already exists with a different value.")
            self.globals[raw] = sampled.get(raw)

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
        if hasattr(node, "condition"):
            context = self._get_eval_context(sampled)
            if not self.evaluator.evaluate_condition(node.condition, context):
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
            context = self._get_eval_context(sampled_params)
            final_value = self.evaluator.resolve_dynamic_value(resolved_value, context)
            # Normalize OmegaConf containers to plain Python for Optuna/storage only when needed
            if isinstance(final_value, (DictConfig, ListConfig)):
                final_value = OmegaConf.to_container(final_value, resolve=True)
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

    def _merge_min_max_hierarchical(self, tree: Any) -> Any:
        # Support lists of groups as hierarchical structure
        if isinstance(tree, list):
            return [self._merge_min_max_hierarchical(t) for t in tree]
        if isinstance(tree, dict):
            processed_tree = {
                k: self._merge_min_max_hierarchical(v) if isinstance(v, (dict, list)) else v for k, v in tree.items()
            }
            return self._merge_min_max_pairs(processed_tree)
        return tree
