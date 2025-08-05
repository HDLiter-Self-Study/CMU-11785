"""
Search Space Sampler for Optuna-based hyperparameter optimization.
Supports hierarchical configuration sampling with architecture-aware parameters
using dependency_order for clean parameter resolution.
"""

import optuna
from typing import Dict, Any, List, Optional, Union
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

    def _sample_config_recursively(
        self,
        trial: optuna.Trial,
        cfg: DictConfig,
        sampled: Dict[str, Any],
        prefix: str = "",
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
                out.update(self._sample_config_recursively(trial, node, sampled, prefix))
                continue

            # ---------- technique ----------
            if c == "technique":
                sel_attr = next((a for a in node.keys() if a.endswith("selection")), None)
                if sel_attr:
                    sel_cfg = node[sel_attr]
                    raw = sel_cfg.param_name
                    arch = sampled.get("architecture_type", "")
                    sel_name = f"{arch}_{raw}" if arch else raw
                    if raw not in sampled:
                        sel_val = self._sample_single_param(trial, sel_cfg, sel_name, sampled)
                        sampled[raw] = sel_val
                        out[sel_name] = sel_val
                if hasattr(node, "condition") and not self._evaluate_condition(node.condition, sampled):
                    continue
                out.update(self._sample_config_recursively(trial, node, sampled, prefix))
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
                val = self._sample_single_param(trial, node, name, sampled)
                sampled[raw] = val
                out[name] = val
                continue
            if c == "technique":
                sel_attr = next((a for a in node.keys() if a.endswith("selection")), None)
                if sel_attr:
                    sel_cfg = node[sel_attr]
                    raw = sel_cfg.param_name
                    arch = sampled.get("architecture_type", "")
                    sel_name = f"{arch}_{raw}" if arch else raw
                    if raw not in sampled:
                        sel_val = self._sample_single_param(trial, sel_cfg, sel_name, sampled)
                        sampled[raw] = sel_val
                        out[sel_name] = sel_val
                if hasattr(node, "condition") and not self._evaluate_condition(node.condition, sampled):
                    continue
                out.update(self._sample_config_recursively(trial, node, sampled, prefix))
                continue

            # ---------- instance ----------
            if c == "instance":
                if hasattr(node, "condition") and not self._evaluate_condition(node.condition, sampled):
                    continue
                out.update(self._sample_config_recursively(trial, node, sampled, prefix))
                continue

            # ---------- fallback ----------
            if any(isinstance(v, DictConfig) for v in node.values() if hasattr(node, "values")):
                out.update(self._sample_config_recursively(trial, node, sampled, prefix))

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
            cat_res = self._sample_config_recursively(trial, cat_cfg, cat_sampled)
            cat_res = self._merge_min_max_pairs(cat_res)
            results[category] = cat_res
            global_sampled_params.update(cat_res)

        return results
