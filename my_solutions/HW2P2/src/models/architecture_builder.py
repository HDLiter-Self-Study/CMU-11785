"""
ArchitectureBuilder: Transform StagePlanner output into a build specification
that runtime model constructors can consume. This module is framework-agnostic
(no torch import) and focuses on deriving per-stage parameters consistently.

Inputs
------
- planned: dict produced by StagePlanner.plan(), containing:
  - num_stages, stages, out_channels, downsamplings
  - block_type (list), activation (list or empty), normalization (list or empty)
  - meta: { type, stem, blocks, extras }

Outputs
-------
Plain dict spec with keys:
- type: architecture type (e.g., "resnet", "convnext")
- num_stages, stages, out_channels, downsamplings
- block_types: per-stage block type list
- block_params: list[dict] of per-stage parameter dicts
- stem: dict
- extras: dict (optional)

Notes
-----
- This builder does not instantiate modules; it prepares a normalized spec
  so that a separate runtime factory can map it to specific classes.
- For per-stage params, we merge in architecture-level block params when the
  block type matches (e.g., use "bottleneck_block" params for block_type=="bottleneck").
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_spec_from_planned(planned: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(planned, dict):
        raise ValueError("planned must be a dict produced by StagePlanner.plan()")

    num_stages = planned.get("num_stages")
    stages = planned.get("stages")
    out_channels = planned.get("out_channels")
    downsamplings = planned.get("downsamplings")
    block_types = planned.get("block_types")
    activations = planned.get("activation") or []
    normalizations = planned.get("normalization") or []
    meta = planned.get("meta") or {}

    # Validate lengths
    for key, arr in {
        "stages": stages,
        "out_channels": out_channels,
        "downsamplings": downsamplings,
        "block_types": block_types,
    }.items():
        if not isinstance(arr, list) or len(arr) != num_stages:
            raise ValueError(f"{key} must be a list of length num_stages")

    if activations and len(activations) != num_stages:
        raise ValueError("activation length must equal num_stages when provided")
    if normalizations and len(normalizations) != num_stages:
        raise ValueError("normalization length must equal num_stages when provided")

    arch_type = meta.get("type")
    stem = meta.get("stem", {}) or {}
    blocks_cfg = meta.get("blocks", {}) or {}
    extras = meta.get("extras", {}) or {}
    # Only allow extras that are accepted by BaseResNetBlock or common blocks
    allowed_extra_keys = {
        "projection_type",
        "stochastic_depth_prob",
        "conv_drop_prob",
        "layer_scale_init_value",
        "layer_scale",
        "use_se",
        "pre_activation",
    }
    filtered_extras = {k: v for k, v in extras.items() if k in allowed_extra_keys}

    # Build per-stage params
    block_params: List[Dict[str, Any]] = []
    for i in range(num_stages):
        bt_entry = block_types[i]
        # Support string or dict entry for block type
        if isinstance(bt_entry, dict):
            bt = bt_entry.get("block_type") or bt_entry.get("type") or bt_entry.get("name")
        else:
            bt = bt_entry
        # Pick block-level defaults based on naming convention "<type>_block"
        block_defaults = blocks_cfg.get(f"{bt}_block", {}) or {}
        # activation/normalization entries may be str or dict; normalize to str
        act_entry = activations[i] if activations else None
        if isinstance(act_entry, dict):
            act_entry = act_entry.get("activation")
        norm_entry = normalizations[i] if normalizations else None
        if isinstance(norm_entry, dict):
            norm_entry = norm_entry.get("normalization")
        stage_params: Dict[str, Any] = {}
        if act_entry is not None:
            stage_params["activation"] = act_entry
        if norm_entry is not None:
            stage_params["norm"] = norm_entry
        # Drop None values for cleanliness
        stage_params = {k: v for k, v in stage_params.items() if v is not None}

        # Merge block defaults and general extras in a predictable order
        if block_defaults:
            stage_params.update(block_defaults)
        if filtered_extras:
            stage_params.update(filtered_extras)

        block_params.append(stage_params)

    return {
        "type": arch_type,
        "num_stages": num_stages,
        "stages": stages,
        "out_channels": out_channels,
        "downsamplings": downsamplings,
        "block_types": [bt.get("block_type") if isinstance(bt, dict) else bt for bt in block_types],
        "block_params": block_params,
        "stem": stem,
        "extras": filtered_extras if filtered_extras else {},
    }
