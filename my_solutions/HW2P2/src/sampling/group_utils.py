"""
Utilities for parsing grouped technique lists across data categories.

Centralizes common validation and normalization to eliminate duplication
in resolvers. Follows the project's fast-fail policy: invalid shapes or
unsupported modes raise immediately.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Union
from .enums import SelectionMode


class CategoryPolicy:
    """Policy for validating grouped category parsing.

    Args:
        allowed_multi_modes: Iterable of allowed multi-instance modes (e.g., 'random_choice').
        require_selection_when_multi: Whether a selection is required when multiple instances are present.
        required: Whether the category must be present in sampled output.
    """

    def __init__(
        self,
        allowed_multi_modes: Union[Set[str], List[str], tuple],
        require_selection_when_multi: bool = True,
        required: bool = False,
    ) -> None:
        if not isinstance(allowed_multi_modes, (set, list, tuple)):
            raise ValueError("allowed_multi_modes must be a set/list/tuple of strings")
        # Normalize to set[str]
        normalized_modes: Set[str] = set(str(m) for m in allowed_multi_modes)
        # Validate against SelectionMode
        valid_modes = {m.value for m in SelectionMode}
        unknown = normalized_modes - valid_modes
        if unknown:
            raise ValueError(f"Unknown selection modes in policy: {sorted(unknown)}")

        self.allowed_multi_modes: Set[str] = normalized_modes
        self.require_selection_when_multi: bool = bool(require_selection_when_multi)
        self.required: bool = bool(required)


def normalize_instances(instances: Dict[str, Any], allow_scalar: bool = True) -> Dict[str, Any]:
    """Normalize instance map to {name: params_dict}.

    - True/None → {}
    - dict → as-is
    - scalar (int/float/str) → {"value": scalar} when allow_scalar
    - otherwise → error
    """
    if not isinstance(instances, dict):
        raise ValueError("instances must be a mapping")
    normalized: Dict[str, Any] = {}
    for name, cfg in instances.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Instance name must be a non-empty string")
        if isinstance(cfg, bool) and cfg is True:
            normalized[name] = {}
        elif cfg is None:
            normalized[name] = {}
        elif allow_scalar and isinstance(cfg, (int, float, str)):
            normalized[name] = {"value": cfg}
        elif isinstance(cfg, dict):
            normalized[name] = cfg
        else:
            raise ValueError(f"Instance '{name}' must be a dict/True/None or scalar, got {type(cfg)}")
    return normalized


def parse_groups_with_policy(
    category: str, node: Any, policy: Union[CategoryPolicy, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Validate and normalize list of technique groups using a CategoryPolicy.

    Input node shape: [ { selection: str|None, instances: {name: dict|True|None|scalar} }, ... ]
    Output: [ { mode: 'single'|mode_from_selection, instances: {name: dict} }, ... ]
    """
    # Accept dict or CategoryPolicy, construct policy object here to reduce caller boilerplate
    if isinstance(policy, dict):
        try:
            policy = CategoryPolicy(**policy)
        except TypeError as e:
            raise ValueError(f"Invalid policy for category '{category}': {e}") from e
    elif not isinstance(policy, CategoryPolicy):
        raise ValueError(f"Policy for category '{category}' must be a dict or CategoryPolicy")
    if not isinstance(node, list):
        raise ValueError(f"'{category}' must be a list of groups: received {type(node)}")

    seen_instances: set = set()
    groups_out: List[Dict[str, Any]] = []

    for idx, group in enumerate(node):
        if not isinstance(group, dict):
            raise ValueError(f"{category}[{idx}] must be a dict with 'selection' and 'instances'")
        selection = group.get("selection", None)
        if selection is not None and not isinstance(selection, str):
            raise ValueError(f"{category}[{idx}].selection must be str or None")
        instances_raw = group.get("instances", None)
        if not isinstance(instances_raw, dict):
            raise ValueError(f"{category}[{idx}].instances must be a dict")

        instances = normalize_instances(instances_raw, allow_scalar=True)
        num_instances = len(instances)

        if selection == "none" and num_instances == 0:
            continue
        if num_instances == 0:
            raise ValueError(f"{category}[{idx}] has no instances. Use selection='none' with empty instances to skip")

        if num_instances == 1:
            mode = SelectionMode.SINGLE.value
        else:
            if selection is None and policy.require_selection_when_multi:
                raise ValueError(
                    f"{category}[{idx}] has multiple instances but no selection. "
                    f"Supported modes: {sorted(policy.allowed_multi_modes)}"
                )
            if selection not in policy.allowed_multi_modes:
                raise ValueError(
                    f"{category}[{idx}] unsupported selection='{selection}'. "
                    f"Allowed: {sorted(policy.allowed_multi_modes)}"
                )
            mode = selection

        duplicate = seen_instances.intersection(instances.keys())
        if duplicate:
            raise ValueError(f"{category}[{idx}] duplicates instance(s) across groups: {sorted(duplicate)}")
        seen_instances.update(instances.keys())

        groups_out.append({"mode": mode, "instances": instances})

    return groups_out
