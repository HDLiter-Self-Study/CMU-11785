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
  - block_type: str | List[str]
  - activation: str | List[str]
  - normalization: str | List[str]
  - stem: dict (optional)
  - blocks: dict (optional)
  - extras at top-level (projection_type, stochastic_depth_prob, conv_drop_prob, etc.)

Outputs
-------
Dict with keys:
- stages: List[int]
- out_channels: List[int]
- downsamplings: List[int]
- block_type: List[str]
- activation: List[str]
- normalization: List[str]
- num_stages: int
- meta: Dict[str, Any] (pass-through items like stem/blocks/extras for the builder)

Design Notes
------------
- This planner does not build modules; it only derives stage-level shapes and expands
  per-stage attributes to lists of length `num_stages`. It fast-fails on schema errors.
- Formulas are adapted from RegNet-style progression already present in the codebase.
"""

from __future__ import annotations

from typing import Any, Dict, List


class StagePlanner:
    """Plan stage topology and expand per-stage fields from architecture spec.

    This class is intentionally small and pure (no side effects, no framework deps)
    so it can be unit-tested in isolation and reused by builders.
    """

    @staticmethod
    def plan(arch: Dict[str, Any]) -> Dict[str, Any]:
        """Plan per-stage configuration from normalized architecture spec.

        Args:
            arch: Architecture dict under `model.architectures` from Effective JSON.

        Returns:
            Dict[str, Any]: Planned stage configuration and expanded attributes.

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
            raise ValueError(f"num_stages must be positive int, got: {num_stages}")
        if not isinstance(regnet_rule, dict):
            raise ValueError("regnet_rule must be provided as a dict")

        # Compute stage widths and depths
        stage_widths = StagePlanner._compute_widths(num_stages, regnet_rule)
        stage_depths = StagePlanner._compute_depths(num_stages, regnet_rule)
        downsamplings = [1] + [2] * (num_stages - 1)

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

        # Expand per-stage fields. Support flexible shapes from resolver output:
        # - scalar str → broadcast
        # - list[str] → as-is
        # - dict (e.g., {"activation": "relu"}) → broadcast dict
        # - list[dict] (e.g., [{"block_type": "basic"}, ...]) → as-is
        block_type = StagePlanner._expand_to_list(arch.get("block_type"), num_stages, required=True, name="block_type")
        # activation may come as top-level or under extras; handle both
        raw_activation = arch.get("activation")
        if raw_activation is None:
            raw_activation = (arch.get("extras", {}) or {}).get("activation")
        activation = StagePlanner._expand_to_list(raw_activation, num_stages, required=False, name="activation")
        # normalization may come as top-level or under extras; handle both
        raw_norm = arch.get("normalization")
        if raw_norm is None:
            raw_norm = (arch.get("extras", {}) or {}).get("normalization")
        normalization = StagePlanner._expand_to_list(raw_norm, num_stages, required=False, name="normalization")

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
            "width_multiplier",
            "block_type",
            "activation",
            "normalization",
            "stem",
            "blocks",
        }
        extras = {k: v for k, v in arch.items() if k not in consumed_keys}
        if extras:
            meta["extras"] = extras

        return {
            "num_stages": num_stages,
            "stages": stage_depths,
            "out_channels": stage_widths,
            "downsamplings": downsamplings,
            "block_type": block_type,
            "activation": activation,
            "normalization": normalization,
            "meta": meta,
        }

    @staticmethod
    def _round_to_multiple_of_8(x: int) -> int:
        x = max(1, int(x))
        return max(8, (x + 7) // 8 * 8)

    @staticmethod
    def _compute_widths(num_stages: int, rule: Dict[str, Any]) -> List[int]:
        width_slope = rule.get("width_slope")
        initial_width = rule.get("initial_width")
        if not isinstance(width_slope, (int, float)) or not isinstance(initial_width, (int, float)):
            raise ValueError("regnet_rule.width_slope and initial_width must be numeric")
        widths: List[int] = []
        for i in range(num_stages):
            w = initial_width * (width_slope**i)
            widths.append(StagePlanner._round_to_multiple_of_8(int(round(w))))
        return widths

    @staticmethod
    def _compute_depths(num_stages: int, rule: Dict[str, Any]) -> List[int]:
        depth_slope = rule.get("depth_slope", 0.0)
        depth_bias = rule.get("depth_bias", 0.0)
        min_stage_depth = int(rule.get("min_stage_depth", 1))
        max_stage_depth = int(rule.get("max_stage_depth", 10))
        if not isinstance(depth_slope, (int, float)):
            raise ValueError("regnet_rule.depth_slope must be numeric")
        depths: List[int] = []
        for i in range(num_stages):
            d = depth_bias + depth_slope * i
            d_int = int(round(d))
            d_int = max(min_stage_depth, min(max_stage_depth, d_int))
            depths.append(max(1, d_int))
        return depths

    @staticmethod
    def _expand_to_list(value: Any, length: int, required: bool, name: str) -> List[Any]:
        """Normalize a possibly scalar/mapping value to a list of length `length`.

        Accepts the following shapes:
        - None → [] when not required, else error
        - str → broadcast to list[str]
        - dict → broadcast to list[dict]
        - list[Any] → length must equal `length` (elements may be str or dict)
        """
        if value is None:
            if required:
                raise ValueError(f"'{name}' is required and must be str or list with length {length}")
            return []
        if isinstance(value, list):
            if len(value) != length:
                raise ValueError(f"'{name}' list length {len(value)} != num_stages {length}")
            return value
        if isinstance(value, str):
            return [value] * length
        if isinstance(value, dict):
            return [value] * length
        raise ValueError(f"'{name}' must be a str, dict, or list of length {length}")
