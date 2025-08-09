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

from typing import Any, Dict, Optional, List
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

    Returns a dict with keys like: type, num_stages, stem, blocks, stage_block_type,
    stagewise, global, regnet_rule, width_multiplier, projection_type, stochastic_depth_prob,
    conv_drop_prob, se_pooling, layer_scale_init_value, extras.
    """
    if "architectures" not in sampled:
        return {}
    groups = sampled["architectures"]
    if not isinstance(groups, list):
        raise ValueError("'architectures' must be a list of groups")

    out: Dict[str, Any] = {
        "type": None,
        "num_stages": None,
        "stem": {},
        "blocks": {},
        "block_type": None,
        # activation/normalization will be set directly at top-level as str (global) or list (stage-wise)
        "activation": None,
        "normalization": None,
        "extras": {},
    }

    def put_scalar(name: str, value: Any) -> None:
        if value is not None:
            out[name] = value

    for idx, g in enumerate(groups):
        if not isinstance(g, dict):
            raise ValueError(f"architectures[{idx}] must be a dict")
        selection = g.get("selection")
        instances = g.get("instances", {})
        if not isinstance(instances, dict):
            raise ValueError(f"architectures[{idx}].instances must be a dict")

        # Selection-based routing
        if isinstance(selection, str) and selection in {"resnet", "convnext"}:
            out["type"] = selection
            continue
        if isinstance(selection, int):
            out["num_stages"] = selection
            continue
        if selection == "stage":
            # stage-wise values like activation/normalization and block_type list
            for k, v in instances.items():
                if k == "block_type" and isinstance(v, list):
                    out["block_type"] = v
                elif k in {"activation", "normalization"}:
                    out[k] = v
                elif k.startswith("stem_"):
                    sub_key = k[len("stem_") :]
                    out.setdefault("stem", {})[sub_key] = v
                else:
                    out["extras"][k] = v
            continue
        if selection == "global":
            for k, v in instances.items():
                if k in {"activation", "normalization"}:
                    out[k] = v
                elif k.startswith("stem_"):
                    sub_key = k[len("stem_") :]
                    out.setdefault("stem", {})[sub_key] = v
                else:
                    out["extras"][k] = v
            continue
        if selection == "regnet_rule":
            if "regnet_rule" in instances and isinstance(instances["regnet_rule"], dict):
                out["regnet_rule"] = instances["regnet_rule"]
            else:
                out["extras"].update(instances)
            continue

        # Selection is None or other: route by instance key
        for k, v in instances.items():
            if k == "stem_block" and isinstance(v, dict):
                out.setdefault("stem", {}).update(v)
            elif k.endswith("_block") and isinstance(v, dict):
                out.setdefault("blocks", {})[k] = v
            elif k == "block_type":
                out["block_type"] = v
            elif k in {
                "width_multiplier",
                "projection_type",
                "stochastic_depth_prob",
                "conv_drop_prob",
                "se_pooling",
                "layer_scale_init_value",
            }:
                put_scalar(k, v)
            elif k == "pre_activation":
                # Some archs expose a toggle here; flatten to top-level
                out["pre_activation"] = v.get("selection") if isinstance(v, dict) else v
            else:
                out["extras"][k] = v

    # Expand string-valued fields to per-stage lists when needed
    num_stages = out.get("num_stages")
    if isinstance(num_stages, int) and num_stages > 0:
        for key in ("activation", "normalization", "block_type"):
            val = out.get(key)
            if isinstance(val, str):
                out[key] = [val] * num_stages
            elif isinstance(val, list) and len(val) != num_stages:
                raise ValueError(f"architectures.{key} list length {len(val)} != num_stages {num_stages}")

    # Final cleanup: drop empty fields
    cleaned = {k: v for k, v in out.items() if v not in (None, {}, [])}
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
        "model": {"architectures": architectures_eff},
        # Pipelines (isolated unified list semantics)
        "pipelines": pipelines,
        # Pairing strategy for online pairs (placeholder; can be filled from task_config)
        "pairing": {},
    }
    return eff
