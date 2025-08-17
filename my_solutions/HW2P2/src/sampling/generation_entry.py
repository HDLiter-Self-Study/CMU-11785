"""
Generation entry: build full configs from a user template with strategy levels,
shortcuts, and arbitrary path overrides, then run n_trials sampling.

Returns one config dict with:
- all non-search_spaces config (task merged)
- sampled: List[dict] of hierarchical sampled dicts per trial
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Set
from copy import deepcopy
import os
import json

from omegaconf import DictConfig, OmegaConf

from src.config.config_manager import get_config
from src.sampling.sampler import SearchSpaceSampler


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


def _normalize_shortcuts(template: Dict[str, Any]) -> Tuple[Dict[str, Any], int, Dict[str, Dict[str, bool]]]:
    """Expand shortcuts into concrete kv overrides and get n_trials using a registry.

    Dict-like shortcuts are driven entirely by the registry spec (type=dict):
    - accept_dict: support whole-dict assignment, e.g. shortcuts.<name>: {...}
    - allow_children: support child-key assignment, e.g. shortcuts.<name>.<key>: value
    - allow_new_keys: allow creating new child keys under the target path(s)
    Both whole-dict and child-key forms may appear together, but assigning the same
    key via both forms fast-fails.
    """
    shortcuts = template.get("shortcuts", {}) or {}
    kv: Dict[str, Any] = {}
    allowed_targets: Dict[str, Dict[str, bool]] = {}

    # load registry for regular shortcuts
    registry_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "shortcut_registry.yaml"))
    registry = OmegaConf.to_container(OmegaConf.load(registry_path), resolve=True)

    # Generic support for dict-like shortcuts that allow whole-dict and child-key forms
    # Detect capabilities from registry: accept_dict + allow_children
    dict_like_specs: Dict[str, Dict[str, Any]] = {
        name: spec
        for name, spec in registry.items()
        if isinstance(spec, dict)
        and spec.get("type") == "dict"
        and (spec.get("accept_dict") or spec.get("allow_children"))
    }
    # Buckets: base whole-dict and child-keys per shortcut name
    dict_bases: Dict[str, Dict[str, Any]] = {k: {} for k in dict_like_specs.keys()}
    dict_children: Dict[str, Dict[str, Any]] = {k: {} for k in dict_like_specs.keys()}

    # First pass: capture loader_settings-related entries and skip them from registry handling
    for name, user_value in list(shortcuts.items()):
        # Whole-dict assignment for dict-like shortcuts
        if name in dict_like_specs and dict_like_specs[name].get("accept_dict"):
            if not isinstance(user_value, dict):
                raise ValueError(f"shortcuts.{name} must be a mapping (dict)")
            dict_bases[name] = dict(user_value)
            continue
        # Child-key assignment for dict-like shortcuts
        for dict_name in dict_like_specs.keys():
            prefix = f"{dict_name}."
            if name.startswith(prefix):
                child_key = name[len(prefix) :]
                if not child_key:
                    raise ValueError(f"Invalid shortcut key: '{prefix}'")
                dict_children[dict_name][child_key] = shortcuts[name]
                break

    # Validate no overlapping keys between base dict and child keys for each dict-like shortcut
    for dict_name, base_dict in dict_bases.items():
        children = dict_children.get(dict_name, {})
        overlap_keys: Set[str] = set(base_dict.keys()) & set(children.keys())
        if overlap_keys:
            raise ValueError(f"Conflicting {dict_name} shortcuts for keys: " + ", ".join(sorted(overlap_keys)))

    # Emit overrides for each dict-like shortcut per registry targets
    for dict_name, spec in dict_like_specs.items():
        targets: List[str] = spec.get("targets", [])
        accept_dict = bool(spec.get("accept_dict"))
        allow_children = bool(spec.get("allow_children"))
        allow_new_keys = bool(spec.get("allow_new_keys", False))
        if not targets:
            continue
        # record allowed targets and capabilities for collector
        for t in targets:
            allowed_targets[t] = {"allow_children": allow_children, "allow_new_keys": allow_new_keys}
        # Whole-dict assignment
        if accept_dict and dict_bases.get(dict_name):
            for path in targets:
                kv[path] = dict_bases[dict_name]
        # Child-key assignment
        if allow_children and dict_children.get(dict_name):
            for ck, cv in dict_children[dict_name].items():
                for path in targets:
                    kv[f"{path}.{ck}"] = cv

    # iterate remaining user-provided shortcuts through registry
    for name, user_value in shortcuts.items():
        # Skip dict-like shortcuts already handled above
        if name in dict_like_specs:
            continue
        for dict_name in dict_like_specs.keys():
            if name.startswith(dict_name + "."):
                break
        else:
            pass
        if name in dict_like_specs or any(name.startswith(dn + ".") for dn in dict_like_specs.keys()):
            continue
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
    return kv, n_trials, allowed_targets


def _normalize_strategy_levels(template: Dict[str, Any]) -> Dict[str, Any]:
    """Turn strategy_levels into kv overrides using the NEW format only.

    New format (level-centric):
      strategy_levels:
        basic: [augmentation, dataset]
        robust: [losses, training]
        custom:
          architectures:
            activation_params.selection.choices.custom: [stage]

    Semantics:
      - Keys are level names (e.g., basic/robust/comprehensive/custom)
      - A list value applies that level to the listed categories
      - For custom: a mapping of {category: {custom_choice_overrides}}
        is allowed; the resolver will set the level to 'custom' for those
        categories and apply the embedded override keys under the category.
    """
    levels = template.get("strategy_levels", {}) or {}
    if not isinstance(levels, dict):
        raise ValueError("strategy_levels must be a mapping of level -> categories")

    kv: Dict[str, Any] = {}

    for level_name, value in levels.items():
        # Case 1: list of categories
        if isinstance(value, list):
            for cat in value:
                if not isinstance(cat, str):
                    raise ValueError("strategy_levels entries must be category names (strings)")
                kv[f"search_spaces.{cat}.strategy_level"] = str(level_name)
            continue

        # Case 2: mapping of category -> embedded overrides (primarily for custom)
        if isinstance(value, dict):
            for cat, overrides in value.items():
                if not isinstance(cat, str):
                    raise ValueError("strategy_levels category keys must be strings")
                kv[f"search_spaces.{cat}.strategy_level"] = str(level_name)
                if overrides is None:
                    continue
                if not isinstance(overrides, dict):
                    raise ValueError(
                        f"strategy_levels.{level_name}.{cat} must be a mapping of relative override paths -> values"
                    )
                for rel_path, v in overrides.items():
                    full_path = f"search_spaces.{cat}.{rel_path}"
                    kv[full_path] = v
            continue

        raise ValueError(
            "strategy_levels values must be either a list of categories or a mapping of category -> overrides"
        )

    return kv


def _collect_overrides(template: Dict[str, Any], allow_new_paths: bool, pre_cfg: DictConfig) -> Tuple[List[str], int]:
    """Build final Hydra overrides (with auto '+'/'++') and compute n_trials; check for conflicts and existence."""
    kv_from_levels = _normalize_strategy_levels(template)
    kv_from_shortcuts, n_trials, allowed_targets = _normalize_shortcuts(template)
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
    # Ensure parent-before-child ordering for stable application
    for path, value in sorted(kv.items(), key=lambda kvp: kvp[0].count(".")):
        parent_path = ".".join(path.split(".")[:-1]) if "." in path else ""
        parent_exists = True if not parent_path else _path_exists(pre_cfg, parent_path)
        full_exists = _path_exists(pre_cfg, path)

        # Check if this path falls under any dict-like target and what is allowed
        matched_target = None
        for t in allowed_targets.keys():
            if path == t or path.startswith(t + "."):
                matched_target = t
                break

        if not parent_exists:
            if matched_target is not None:
                # ensure the ancestor right before the target exists (e.g., task_configs.data)
                if "." in matched_target:
                    ancestor = matched_target.rsplit(".", 1)[0]
                    if not _path_exists(pre_cfg, ancestor):
                        raise KeyError(f"Parent path does not exist: {parent_path}")
            else:
                raise KeyError(f"Parent path does not exist: {parent_path}")

        if not full_exists:
            allowed = allow_new_paths
            if matched_target is not None:
                # Creating the target dict itself is always allowed for dict-like shortcuts
                if path == matched_target:
                    allowed = True
                else:
                    # Creating a new child key: require allow_children and allow_new_keys
                    caps = allowed_targets[matched_target]
                    if caps.get("allow_children") and caps.get("allow_new_keys"):
                        allowed = True
            if allowed:
                prefix = "++"
            else:
                raise KeyError(f"Path not found (creation disabled): {path}")
        else:
            prefix = ""

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

    # run sampling (reuse a single in-memory study to reduce logs/noise)
    sampler = SearchSpaceSampler(silent=True, overrides=[f"+task_configs={task}", *override_list])
    sampler.globals["task"] = str(task)
    sampled_list: List[Dict[str, Any]] = []
    import optuna

    study = optuna.create_study(storage="sqlite:///:memory:", study_name="entry_trials", direction="maximize")
    for _ in range(n_trials):
        trial = study.ask()
        res = sampler.sample_all_params(trial, include_hierarchical=True)
        sampled_list.append(res["hierarchical"])  # keep hierarchical view only

    # build final dict, extract policies, drop heavy search_spaces
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError("resolved config is not a mapping")
    # Extract lightweight policies from search_spaces; enforce presence
    try:
        ss = cfg.search_spaces
        policies: Dict[str, Any] = {}
        for cat, node in ss.items():
            if not hasattr(node, "policy"):
                raise ValueError(f"search_spaces.{cat}.policy is required")
            pol = node.policy
            if pol is not None:
                policies[str(cat)] = OmegaConf.to_container(pol, resolve=True)
        # All categories must have policies
        if not policies:
            raise ValueError("No policies extracted from search_spaces; each category must define a policy block")
        cfg_dict["policies"] = policies
    except Exception:
        raise
    cfg_dict.pop("search_spaces", None)
    cfg_dict["task"] = str(task)
    cfg_dict["sampled"] = sampled_list
    return cfg_dict
