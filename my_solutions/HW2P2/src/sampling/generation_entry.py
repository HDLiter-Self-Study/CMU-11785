"""
Generation entry: build full configs from a user template with strategy levels,
shortcuts, and arbitrary path overrides, then run n_trials sampling.

Returns one config dict with:
- all non-search_spaces config (task merged)
- sampled: List[dict] of hierarchical sampled dicts per trial
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from copy import deepcopy
import os
import json

from omegaconf import DictConfig, OmegaConf

from config.config_manager import get_config
from sampling.sampler import SearchSpaceSampler


def _load_task_config(task: str) -> DictConfig:
    """Compose base config including the selected task config via Hydra override."""
    if task not in {"classification", "verification_finetune"}:
        raise ValueError(f"Unsupported task: {task}")
    return get_config("main", overrides=[f"+task_configs={task}"])


def _path_exists(cfg: DictConfig, path: str) -> bool:
    parts = path.split(".")
    node: Any = cfg
    for key in parts:
        if not hasattr(node, key):
            return False
        node = getattr(node, key)
    return True


def _value_to_override_literal(value: Any) -> str:
    # JSON is compatible with Hydra value parsing for scalars/lists
    return json.dumps(value)


def _coerce_value(value: Any, typ: str, coerce_scalar_to_list: bool) -> Any:
    if typ == "int":
        return int(value)
    if typ == "float":
        return float(value)
    if typ == "str":
        return str(value)
    if typ == "bool":
        if isinstance(value, bool):
            return value
        if str(value).lower() in {"true", "1"}:
            return True
        if str(value).lower() in {"false", "0"}:
            return False
        raise ValueError(f"Cannot coerce to bool: {value}")
    if typ == "list[int]":
        if isinstance(value, list):
            return [int(v) for v in value]
        if coerce_scalar_to_list:
            return [int(value)]
        raise ValueError("Expected list for list[int]")
    raise ValueError(f"Unsupported shortcut type: {typ}")


def _normalize_shortcuts(template: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Expand shortcuts into concrete kv overrides and get n_trials using a registry."""
    shortcuts = template.get("shortcuts", {}) or {}
    kv: Dict[str, Any] = {}

    # load registry
    registry_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "shortcut_registry.yaml"))
    registry = OmegaConf.to_container(OmegaConf.load(registry_path), resolve=True)

    # iterate user-provided shortcuts
    for name, user_value in shortcuts.items():
        if name not in registry:
            raise KeyError(f"Unknown shortcut '{name}'. Add it to shortcut_registry.yaml")
        spec = registry[name]
        typ = spec.get("type")
        targets: List[str] = spec.get("targets", [])
        coerce_flag = bool(spec.get("coerce_scalar_to_list", False))
        coerced = _coerce_value(user_value, typ, coerce_flag)
        for path in targets:
            kv[path] = coerced

    # derive n_trials from kv (fallback 1)
    n_trials = int(kv.get("optuna.n_trials", 1))
    return kv, n_trials


def _normalize_strategy_levels(template: Dict[str, Any]) -> Dict[str, Any]:
    """Turn strategy_levels into kv overrides; support embedded custom_choices.

    Input format example:
    strategy_levels:
      architectures:
        level: custom
        custom_choices:
          activation_params.selection.choices.custom: [stage]
      training: robust
    """
    levels = template.get("strategy_levels", {}) or {}
    kv: Dict[str, Any] = {}

    for cat, spec in levels.items():
        if isinstance(spec, dict):
            level = spec.get("level")
            if level is None:
                raise ValueError(f"strategy_levels.{cat}.level missing")
            kv[f"search_spaces.{cat}.strategy_level"] = str(level)
            # embedded custom choices
            if str(level) == "custom":
                custom = spec.get("custom_choices", {}) or {}
                for rel_path, value in custom.items():
                    full_path = f"search_spaces.{cat}.{rel_path}"
                    kv[full_path] = value
        else:
            kv[f"search_spaces.{cat}.strategy_level"] = str(spec)
    return kv


def _collect_overrides(template: Dict[str, Any], allow_new_paths: bool, pre_cfg: DictConfig) -> Tuple[List[str], int]:
    """Build final Hydra overrides (with auto '+'/'++') and compute n_trials; check for conflicts and existence."""
    kv_from_levels = _normalize_strategy_levels(template)
    kv_from_shortcuts, n_trials = _normalize_shortcuts(template)
    kv_from_overrides = template.get("overrides", {}) or {}

    # conflict check: same key in different sources
    def _check_conflict(a: Dict[str, Any], b: Dict[str, Any], a_name: str, b_name: str):
        overlap = set(a.keys()) & set(b.keys())
        if overlap:
            raise ValueError(f"Conflicting keys between {a_name} and {b_name}: {sorted(overlap)}")

    _check_conflict(kv_from_levels, kv_from_shortcuts, "strategy_levels", "shortcuts")
    _check_conflict(kv_from_levels, kv_from_overrides, "strategy_levels", "overrides")
    _check_conflict(kv_from_shortcuts, kv_from_overrides, "shortcuts", "overrides")

    kv: Dict[str, Any] = {}
    kv.update(kv_from_levels)
    kv.update(kv_from_shortcuts)
    kv.update(kv_from_overrides)

    override_list: List[str] = []
    for path, value in kv.items():
        parent_path = ".".join(path.split(".")[:-1]) if "." in path else ""
        parent_exists = True if not parent_path else _path_exists(pre_cfg, parent_path)
        full_exists = _path_exists(pre_cfg, path)
        if not parent_exists:
            raise KeyError(f"Parent path does not exist: {parent_path}")
        if not full_exists and not allow_new_paths:
            raise KeyError(f"Path not found (creation disabled): {path}")
        prefix = "++" if (not full_exists and allow_new_paths) else ""
        override_list.append(f"{prefix}{path}={_value_to_override_literal(value)}")

    return override_list, n_trials


def generate_configs_from_template(template_path: str, allow_new_paths: bool = False) -> Dict[str, Any]:
    """Main entry.

    - Load base config
    - Merge task_config selected by `task`
    - Build kv overrides from strategy_levels/custom choices/shortcuts/overrides
    - Apply overrides with path checking
    - Run n_trials sampling and return one dict with sampled list
    """
    if not os.path.isfile(template_path):
        raise FileNotFoundError(template_path)

    template = OmegaConf.to_container(OmegaConf.load(template_path), resolve=True)
    if not isinstance(template, dict):
        raise ValueError("template must be a YAML mapping")

    # choose task and compose config with task included
    task = template.get("task")
    if not task:
        raise ValueError("template.task is required: one of ['classification','verification','verification_finetune']")
    # Build a preliminary config (task only) for path validation
    pre_cfg = _load_task_config(str(task))

    # collect overrides
    override_list, n_trials = _collect_overrides(template, allow_new_paths, pre_cfg)

    # Compose final config via Hydra overrides
    cfg = get_config("main", overrides=[f"+task_configs={task}", *override_list])

    # run sampling
    sampler = SearchSpaceSampler(silent=True, overrides=[f"+task_configs={task}", *override_list])
    # inject task into sampler.globals
    sampler.globals["task"] = str(task)
    sampled_list: List[Dict[str, Any]] = []
    for _ in range(n_trials):
        import optuna

        study = optuna.create_study(storage="sqlite:///:memory:", study_name="entry_trial", direction="maximize")
        trial = study.ask()
        res = sampler.sample_all_params(trial, include_hierarchical=True)
        sampled_list.append(res["hierarchical"])  # keep hierarchical view only

    # build final dict (without search_spaces), attach sampled list
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError("resolved config is not a mapping")
    cfg_dict.pop("search_spaces", None)
    cfg_dict["task"] = str(task)
    cfg_dict["sampled"] = sampled_list
    return cfg_dict
