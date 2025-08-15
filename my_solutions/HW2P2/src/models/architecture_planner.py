"""
StagePlanner: Plan per-stage topology (depths, widths, downsamplings) and expand
per-stage attributes from a normalized architecture spec produced by the resolver.

Inputs
------
- A dict under `model.architectures` from Effective JSON with keys like:
  - type: "resnet" | "convnext"
  - num_stages: int
  - regnet_rule: { width_slope, initial_width, depth_slope, depth_bias, min_stage_depth, max_stage_depth }
  - width_multiplier: float|int (optional)
  - block_type: List[Dict[str, str]] (e.g., [{"block_type": "basic"}, {"block_type": "bottleneck"}])
  - activation: List[Dict[str, Any]] (e.g., [{"activation": "relu"}, {"activation": "gelu", "activation_params": {...}}])
  - normalization: List[Dict[str, Any]] (e.g., [{"normalization": "batch_norm"}, {"normalization": "group_norm", "num_groups": 4}])
  - stem: dict (optional)
  - blocks: dict (optional)
  - extras at top-level (projection_type, stochastic_depth_prob, conv_drop_prob, etc.)

Outputs
-------
StagePlan dataclass with keys:
- num_stages: int
- stages: List[int] (depths per stage)
- out_channels: List[int] (output channels per stage)
- downsamplings: List[int] (downsampling factors per stage: [1, 2, 2, ...])
- block_types: List[str] (extracted block type names: ["basic", "bottleneck", ...])
- per_stage_block_params: List[Dict[str, Any]] (merged activation/normalization params per stage)
- meta: Dict[str, Any] (pass-through items like stem/blocks/extras for the builder)

Design Notes
------------
- This planner does not build modules; it only derives stage-level shapes and expands
  per-stage attributes to lists of length `num_stages`. It fast-fails on schema errors.
- Formulas are adapted from RegNet-style progression already present in the codebase.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from typing import Any, Dict, List


@dataclass
class StagePlan:
    """Dataclass to hold the planned architecture specification."""

    num_stages: int
    stages: List[int]
    out_channels: List[int]
    downsamplings: List[int]
    block_types: List[str]
    per_stage_block_params: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class StagePlanner:
    """Plan stage topology and expand per-stage fields from architecture spec.

    This class is intentionally small and pure (no side effects, no framework deps)
    so it can be unit-tested in isolation and reused by builders.
    """

    @staticmethod
    def plan(arch: Dict[str, Any]) -> StagePlan:
        """Plan per-stage configuration from normalized architecture spec.

        Args:
            arch: Architecture dict under `model.architectures` from Effective JSON.

        Returns:
            StagePlan: A dataclass containing the planned stage configuration.

        Raises:
            ValueError: On missing keys, invalid types, or inconsistent lengths.
        """
        if not isinstance(arch, dict):
            raise ValueError("architecture spec must be a dict")

        # Required keys
        arch_type = arch.get("type")
        num_stages = arch.get("num_stages")
        regnet_rule = arch.get("regnet_rule")

        if arch_type not in {"resnet", "convnext"}:
            raise ValueError(f"Unsupported architecture type: {arch_type}")
        if not isinstance(num_stages, int) or num_stages <= 0:
            raise ValueError(f"num_stages must be a positive integer, but got {num_stages}")
        if not isinstance(regnet_rule, dict):
            raise ValueError("regnet_rule must be provided as a dict")

        # Stricly validate per-stage attributes. If they exist, they must be lists
        # of the correct length. No automatic expansion is performed.
        for name in ["activation", "normalization", "block_type"]:
            attr = arch.get(name)
            if attr is not None:
                if not isinstance(attr, list):
                    raise TypeError(f"'{name}' is expected to be a list, but got {type(attr)}")
                if len(attr) != num_stages:
                    raise ValueError(f"'{name}' list length {len(attr)} != num_stages {num_stages}")

        stage_depths, stage_widths = StagePlanner._plan_shapes(arch, num_stages)
        downsamplings = StagePlanner._plan_downsamplings(arch, num_stages)

        # Optional width multiplier (apply to widths, round to multiple of 8)
        # Now extracted from regnet_rule.width_multiplier
        width_multiplier = None
        if regnet_rule and "width_multiplier" in regnet_rule:
            width_multiplier = regnet_rule["width_multiplier"]
        if width_multiplier is not None:
            if not isinstance(width_multiplier, (int, float)):
                raise ValueError("regnet_rule.width_multiplier must be int or float")
            stage_widths = [
                StagePlanner._round_to_multiple_of_8(int(max(1, w * width_multiplier))) for w in stage_widths
            ]

        # Optional clamp by a max stage width cap (acts as tail guard only)
        max_stage_width = regnet_rule.get("max_stage_width") or regnet_rule.get("regnet_max_stage_width")
        if max_stage_width is not None:
            try:
                cap = int(max_stage_width)
                if cap > 0:
                    stage_widths = [min(w, cap) for w in stage_widths]
            except (ValueError, TypeError):
                pass  # Ignore if max_stage_width is not a valid number

        # Combine all per-stage parameters into a single list of dicts
        per_stage_params = []
        for i in range(num_stages):
            params = {}
            if arch.get("activation") and i < len(arch["activation"]):
                # Activation entry must be a dict in new format
                act_entry = arch["activation"][i]
                if isinstance(act_entry, dict):
                    params.update(act_entry)
                else:
                    raise ValueError(f"activation items must be dict, got: {type(act_entry)}")

            if arch.get("normalization") and i < len(arch["normalization"]):
                # Normalization entry must be a dict in new format
                norm_entry = arch["normalization"][i]
                if isinstance(norm_entry, dict):
                    params.update(norm_entry)
                else:
                    raise ValueError(f"normalization items must be dict, got: {type(norm_entry)}")

            per_stage_params.append(params)

        # Collect meta for builders (pass-through fields)
        meta: Dict[str, Any] = {
            "type": arch_type,
            "stem": arch.get("stem", {}) or {},
            "blocks": arch.get("blocks", {}) or {},
        }

        # Extras: everything else that is not already consumed
        consumed_keys = {
            "type",
            "num_stages",
            "regnet_rule",
            "block_type",
            "activation",
            "normalization",
            "stem",
            "blocks",
        }
        extras = {k: v for k, v in arch.items() if k not in consumed_keys}
        if extras:
            meta["extras"] = extras

        # Extract block_type list from List[Dict] format
        block_type_raw = arch.get("block_type") or []
        block_types_list = []
        if block_type_raw:
            for item in block_type_raw:
                if isinstance(item, dict):
                    # Force use of "block_type" key for fast fail
                    if "block_type" not in item:
                        raise ValueError(f"block_type dict must contain 'block_type' key, got: {item}")
                    block_types_list.append(item["block_type"])
                else:
                    raise ValueError(f"block_type items must be dict, got: {type(item)}")

        return StagePlan(
            num_stages=num_stages,
            stages=stage_depths,
            out_channels=stage_widths,
            downsamplings=downsamplings,
            block_types=block_types_list,
            per_stage_block_params=per_stage_params,
            meta=meta,
        )

    @staticmethod
    def _round_to_multiple_of_8(x: int) -> int:
        x = max(1, int(x))
        return max(8, (x + 7) // 8 * 8)

    @staticmethod
    def _plan_shapes(spec: Dict[str, Any], num_stages: int) -> tuple[List[int], List[int]]:
        regnet_rule = spec.get("regnet_rule")
        if not isinstance(regnet_rule, dict):
            raise ValueError("regnet_rule must be provided as a dict")

        width_slope = regnet_rule.get("width_slope")
        initial_width = regnet_rule.get("initial_width")
        if not isinstance(width_slope, (int, float)) or not isinstance(initial_width, (int, float)):
            raise ValueError("regnet_rule.width_slope and initial_width must be numeric")
        stage_widths: List[int] = []
        for i in range(num_stages):
            w = initial_width * (width_slope**i)
            stage_widths.append(StagePlanner._round_to_multiple_of_8(int(round(w))))

        depth_slope = regnet_rule.get("depth_slope", 0.0)
        depth_bias = regnet_rule.get("depth_bias", 0.0)
        min_stage_depth = int(regnet_rule.get("min_stage_depth", 1))
        max_stage_depth = int(regnet_rule.get("max_stage_depth", 10))
        if not isinstance(depth_slope, (int, float)):
            raise ValueError("regnet_rule.depth_slope must be numeric")
        stage_depths: List[int] = []
        for i in range(num_stages):
            d = depth_bias + depth_slope * i
            d_int = int(round(d))
            d_int = max(min_stage_depth, min(max_stage_depth, d_int))
            stage_depths.append(max(1, d_int))

        return stage_depths, stage_widths

    @staticmethod
    def _plan_downsamplings(spec: Dict[str, Any], num_stages: int) -> List[int]:
        # First stage has no downsampling (factor=1), other stages downsample by 2x
        downsamplings = [1] + [2] * (num_stages - 1)
        return downsamplings
