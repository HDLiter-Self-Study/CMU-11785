"""
Mode handler registry for turning parsed groups into effective pipeline records.

Handlers are category-agnostic by default. When a category needs special
semantics for a mode, add a category-specific override in CATEGORY_MODE_HANDLERS.

Each handler receives normalized instances and the original group dict (for
accessing optional extras like weights), and returns the final group record.
"""

from __future__ import annotations

from typing import Any, Dict, Callable, List

from .enums import SelectionMode


def _build_components_for_sum(instances: Dict[str, Any]) -> List[Dict[str, Any]]:
    components: List[Dict[str, Any]] = []
    for inst_name, params in instances.items():
        components.append({"type": inst_name, "params": params, "weight": 1.0})
    return components


def _build_components_for_weighted(
    instances: Dict[str, Any], weights_map: Dict[str, Any] | None
) -> List[Dict[str, Any]]:
    if not isinstance(weights_map, dict):
        weights_map = {}
    components: List[Dict[str, Any]] = []
    for inst_name, params in instances.items():
        weight_value = None
        if inst_name in weights_map:
            weight_value = float(weights_map[inst_name])
        elif isinstance(params, dict) and "weight" in params:
            weight_value = float(params.pop("weight"))
        if weight_value is None:
            raise ValueError("weighted_sum requires a 'weights' map or per-instance 'weight'")
        components.append({"type": inst_name, "params": params, "weight": weight_value})
    return components


# ---------- Generic mode handlers (category-agnostic) ----------


def generic_single(instances: Dict[str, Any], group: Dict[str, Any]) -> Dict[str, Any]:
    return {"mode": SelectionMode.SINGLE.value, "instances": instances}


def generic_random_choice(instances: Dict[str, Any], group: Dict[str, Any]) -> Dict[str, Any]:
    return {"mode": SelectionMode.RANDOM_CHOICE.value, "instances": instances}


def generic_sum(instances: Dict[str, Any], group: Dict[str, Any]) -> Dict[str, Any]:
    return {"mode": SelectionMode.SUM.value, "components": _build_components_for_sum(instances)}


def generic_weighted_sum(instances: Dict[str, Any], group: Dict[str, Any]) -> Dict[str, Any]:
    weights_map = group.get("weights")
    return {
        "mode": SelectionMode.WEIGHTED_SUM.value,
        "components": _build_components_for_weighted(instances, weights_map),
    }


GENERIC_MODE_HANDLERS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {
    SelectionMode.SINGLE.value: generic_single,
    SelectionMode.RANDOM_CHOICE.value: generic_random_choice,
    SelectionMode.SUM.value: generic_sum,
    SelectionMode.WEIGHTED_SUM.value: generic_weighted_sum,
}


CATEGORY_MODE_HANDLERS: Dict[str, Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]]] = {
    # Place category-specific overrides here if any category diverges from generic behavior
}


def get_mode_handler(category: str, mode: str) -> Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]:
    """Return a handler for (category, mode), falling back to generic for simple modes.

    Raises when no handler is available (fast-fail).
    """
    cat_map = CATEGORY_MODE_HANDLERS.get(category, {})
    if mode in cat_map:
        return cat_map[mode]
    if mode in GENERIC_MODE_HANDLERS:
        return GENERIC_MODE_HANDLERS[mode]
    raise ValueError(f"No handler for category='{category}', mode='{mode}'")
