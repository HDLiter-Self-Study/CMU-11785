"""
Effective data configuration resolver.

Takes the merged config dict (from generation entry) and one hierarchical
sampled result, and produces a normalized EffectiveDataConfig used by
data factories (augmentation/label_mixing/sampler/datasets/dataloaders).

Key policy for data-pipeline categories:
- The sampler outputs a list of technique-groups for each category, each of shape:
    { "selection": str | None, "instances": { instance_name: params_dict | True } }
- We validate and normalize each group using the three-case rule:
    1) selection == "none" AND instances == {}  → drop the group
    2) len(instances) == 1 (selection any or missing) → mode = "single"
    3) len(instances) >= 2 → require selection == "random_choice"
- Boolean True and None instance params are normalized to empty dicts.
- Duplicate instance names across groups in the same category are disallowed.

This resolver no longer reads legacy mixed 'training' groups. It expects split
strategies for optimizer/scheduler/ema/grad_clip/loader.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Set
from .group_utils import parse_groups_with_policy, CategoryPolicy
from .mode_handlers import get_mode_handler


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = d
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _resolve_default(
    category: str, cfg: Dict[str, Any], sampled: Dict[str, Any], policies: Dict[str, Any]
) -> List[Dict[str, Any]]:
    pol = policies.get(category)
    if not isinstance(pol, dict):
        raise ValueError(f"Missing policy for '{category}'")
    pol_obj = CategoryPolicy(**pol)
    if category not in sampled:
        if pol_obj.required:
            raise ValueError(f"Missing required category '{category}' in sampled config")
        return []
    groups = sampled[category]
    if not isinstance(groups, list):
        raise ValueError(f"'{category}' must be a list of groups")
    normalized_groups = parse_groups_with_policy(category=category, node=groups, policy=pol_obj)
    out_groups: List[Dict[str, Any]] = []
    for idx, parsed in enumerate(normalized_groups):
        mode = parsed.get("mode")
        instances = parsed.get("instances", {})
        handler = get_mode_handler(category, mode)
        orig = groups[idx] if idx < len(groups) else {}
        out_groups.append(handler(instances, orig))
    return out_groups


def _resolve_category(
    category: str, cfg: Dict[str, Any], sampled: Dict[str, Any], policies: Dict[str, Any]
) -> List[Dict[str, Any]]:
    fn_name = f"_resolve_{category}"
    fn = globals().get(fn_name)
    if callable(fn):
        # All specialized resolvers use signature (cfg, sampled, policies)
        return fn(cfg, sampled, policies)
    return _resolve_default(category, cfg, sampled, policies)


def _resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Optional[str]]:
    data_cfg = _get(cfg, "task_configs.data", {})
    if not isinstance(data_cfg, dict):
        data_cfg = {}
    out: Dict[str, Optional[str]] = {
        "train_dir": data_cfg.get("train_dir"),
        "val_dir": data_cfg.get("val_dir"),
        "train_pairs": data_cfg.get("train_pairs"),
        "val_pairs": data_cfg.get("val_pairs"),
        "images_dir": data_cfg.get("images_dir"),
    }
    return out


def _resolve_image_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve image settings with override semantics.

    - Base from cfg.data.image_settings
    - Override from task_configs.data.image_settings
    """
    # Base image settings from data.image_settings
    base_settings = _get(cfg, "data.image_settings", {}) or {}
    if not isinstance(base_settings, dict):
        base_settings = {}

    # Override from task_configs.data.image_settings
    override_settings = _get(cfg, "task_configs.data.image_settings", {}) or {}
    if not isinstance(override_settings, dict):
        override_settings = {}

    # Merge with override semantics
    settings = {**base_settings, **override_settings}

    return settings


def _resolve_loader(cfg: Dict[str, Any], sampled: Dict[str, Any], policies: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Resolve loader via default path, then inject loader_settings into instance params.

    - Reuses _resolve_default for grouped validation and mode handling.
    - Injects `task_configs.data.loader_settings` into each instance params.
    - Fast-fails on overlapping keys.
    """
    # First, resolve using the default pipeline mechanism
    base_groups = _resolve_default("loader", cfg, sampled, policies)

    # Retrieve loader settings with override semantics:
    # base from cfg.data.loader_settings, then updated by task_configs.data.loader_settings
    base_settings = _get(cfg, "data.loader_settings", {}) or {}
    if not isinstance(base_settings, dict):
        base_settings = {}
    override_settings = _get(cfg, "task_configs.data.loader_settings", {}) or {}
    if not isinstance(override_settings, dict):
        override_settings = {}
    settings = {**base_settings, **override_settings}

    # Merge settings into each group's instances and collapse into a single canonical instance
    merged_groups: List[Dict[str, Any]] = []
    for rendered in base_groups:
        if isinstance(rendered, dict) and isinstance(rendered.get("instances"), dict):
            accumulated_params: Dict[str, Any] = {}
            for inst_name, params in rendered["instances"].items():
                if not isinstance(params, dict):
                    raise ValueError("loader instance params must be a dict after rendering")
                # Flatten scalar wrapper: {"value": v} -> {inst_name: v}
                if set(params.keys()) == {"value"}:
                    params = {inst_name: params["value"]}
                # Merge this instance's params into accumulated param dict
                dup = set(accumulated_params.keys()) & set(params.keys())
                if dup:
                    raise ValueError(f"Duplicate loader param keys across instances: {sorted(dup)}")
                accumulated_params.update(params)

            # Merge loader_settings; error on overlap
            overlap: Set[str] = set(accumulated_params.keys()) & set(settings.keys())
            if overlap:
                raise ValueError(f"Conflicting keys between loader params and data.loader_settings: {sorted(overlap)}")
            final_params = {**accumulated_params, **settings} if settings else dict(accumulated_params)

            new_group = dict(rendered)
            new_group["instances"] = {"batch_configuration": final_params}
            merged_groups.append(new_group)
        else:
            merged_groups.append(rendered)

    return merged_groups


def _merge_wandb(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge wandb configs with task override semantics.

    - Scalar keys: override wins when present
    - tags: if both present and are lists -> concat and de-duplicate preserving order
    """
    if not isinstance(base, dict):
        base = {}
    if not isinstance(override, dict):
        override = {}
    merged = dict(base)
    for k, v in override.items():
        if k == "tags" and isinstance(v, list) and isinstance(merged.get("tags"), list):
            merged["tags"] = list(dict.fromkeys(list(merged["tags"]) + list(v)))
        else:
            merged[k] = v
    return merged


def _normalize_architectures(sampled: Dict[str, Any]) -> Dict[str, Any]:
    """Convert sampler's grouped architectures output into a compact model spec.

    This function is now a pure passthrough and restructuring utility. It iterates
    through the sampler's output groups and unpacks all instances directly into the
    output spec. It no longer contains complex routing logic based on 'selection'
    values, as the new sampler provides a consistent hierarchical structure.
    """
    if "architectures" not in sampled:
        return {}
    groups = sampled.get("architectures")
    if not isinstance(groups, list):
        raise ValueError("'architectures' must be a list of groups")

    out: Dict[str, Any] = {"extras": {}}

    for group in groups:
        if not isinstance(group, dict):
            continue

        # Handle top-level selection keys that are not instances
        selection = group.get("selection")
        if isinstance(selection, str) and selection in {"resnet", "convnext"}:
            out["type"] = selection
        elif isinstance(selection, int):
            out["num_stages"] = selection

        # Unpack all instances directly
        instances = group.get("instances", {})
        if not isinstance(instances, dict):
            continue

        for key, value in instances.items():
            if key in {"activation", "normalization", "block_type"}:
                # Flatten nested dicts like {'activation': 'relu'} that can be produced
                # by the sampler for global-granularity parameters.
                if isinstance(value, dict) and key in value and len(value) == 1:
                    out[key] = value[key]
                else:
                    out[key] = value
            elif key == "regnet_rule":
                out[key] = value
            elif key == "stem_block" and isinstance(value, dict):
                out.setdefault("stem", {}).update(value)
            elif key.endswith("_block") and isinstance(value, dict):
                out.setdefault("blocks", {})[key] = value
            elif key in {
                "projection_type",
                "stochastic_depth_prob",
                "conv_drop_prob",
                "se_pooling",
                "layer_scale_init_value",
                "pre_activation",
            }:
                if key == "pre_activation" and isinstance(value, dict):
                    # Flatten pre_activation selection
                    out[key] = value.get("selection")
                else:
                    out[key] = value
            else:
                out["extras"][key] = value

    # Final cleanup: drop empty fields and ensure required fields exist
    cleaned = {k: v for k, v in out.items() if v is not None and v != {}}
    if "extras" in cleaned and not cleaned["extras"]:
        del cleaned["extras"]

    return cleaned


def resolve_effective_data_config(cfg: Dict[str, Any], sampled_hierarchical: Dict[str, Any]) -> Dict[str, Any]:
    """Build a normalized EffectiveDataConfig from final cfg and one sampled result.

    Args:
        cfg: Final config dict returned by generation entry (without search_spaces)
        sampled_hierarchical: One element from the entry "sampled" list

    Returns:
        A plain Python dict with normalized data config for factories.
    """
    task = cfg.get("task", "classification")

    # Split strategies only
    policies = cfg.get("policies") if isinstance(cfg, dict) else None
    if not isinstance(policies, dict):
        raise ValueError("Missing 'policies' in cfg. Ensure each search_spaces.<category> defines a policy block.")
    architectures_eff = _normalize_architectures(sampled_hierarchical)

    # Top-level meta
    # Enforce that epochs must come from task_configs (no fallback to main)
    epochs_val = _get(cfg, "task_configs.training.epochs")
    if epochs_val is None:
        raise ValueError("task_configs.training.epochs is required (no fallback)")
    run = {"epochs": epochs_val}
    checkpoints = _get(cfg, "training.checkpoints", {}) or {}
    wandb_base = _get(cfg, "wandb", {}) or {}
    wandb_override = _get(cfg, "task_configs.wandb", {}) or {}
    wandb = _merge_wandb(wandb_base, wandb_override)

    # Build pipelines dynamically from policies (excluding 'architectures')
    pipelines: Dict[str, Any] = {}
    for category in policies.keys():
        if category == "architectures":
            continue
        pipelines[category] = _resolve_category(category, cfg, sampled_hierarchical, policies)

    eff: Dict[str, Any] = {
        "task": task,
        "device": cfg.get("device"),
        "seed": cfg.get("seed"),
        "resume_from": cfg.get("resume_from"),
        "run": run,
        "checkpoints": checkpoints,
        "wandb": wandb,
        "paths": _resolve_paths(cfg),
        "image_settings": _resolve_image_settings(cfg),
        "model": {"architectures": architectures_eff},
        # Pipelines (isolated unified list semantics)
        "pipelines": pipelines,
        # Pairing strategy for online pairs (placeholder; can be filled from task_config)
        "pairing": {},
    }
    return eff
