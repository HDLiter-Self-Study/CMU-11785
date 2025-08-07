"""
Safe expression evaluator for dynamic parameter resolution.

This module provides safe evaluation of dynamic expressions in configuration files,
using a whitelist of allowed built-in functions to prevent code injection while
enabling flexible parameter dependencies and conditional logic.

Features:
- Safe evaluation of conditional expressions (e.g., "$param_name == 'value'")
- Dynamic value resolution for expressions starting with '$'
- Whitelist-based security model to prevent code injection
- Comprehensive error reporting with available parameter context
"""

from typing import Dict, Any


class SafeEvaluator:
    """
    Safe expression evaluator for dynamic parameter resolution.

    This class provides safe evaluation of dynamic expressions in configuration files,
    using a whitelist of allowed built-in functions to prevent code injection.
    """

    # Whitelist of safe built-in functions that can be used in expressions
    SAFE_BUILTINS = {
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

    @classmethod
    def evaluate_condition(cls, condition: str, params: Dict[str, Any]) -> bool:
        """
        Evaluate a conditional expression safely.

        Args:
            condition: Condition string starting with '$' (e.g., '$param_name == "value"')
            params: Available parameters for evaluation

        Returns:
            Boolean result of condition evaluation

        Raises:
            ValueError: If condition evaluation fails
        """
        if not condition or not condition.startswith("$"):
            return True

        try:
            expr = condition[1:]  # Remove '$' prefix
            return bool(eval(expr, {"__builtins__": cls.SAFE_BUILTINS}, dict(params)))
        except Exception as e:
            available_params = list(params.keys())
            raise ValueError(
                f"Failed to evaluate condition '{expr}': {e}. " f"Available parameters: {available_params}"
            )

    @classmethod
    def resolve_dynamic_value(cls, value: Any, params: Dict[str, Any]) -> Any:
        """
        Resolve dynamic values (strings starting with '$').

        Args:
            value: Value to resolve (may be dynamic expression)
            params: Available parameters for evaluation

        Returns:
            Resolved value

        Raises:
            ValueError: If dynamic value resolution fails
        """
        if isinstance(value, str) and value.startswith("$"):
            try:
                expr = value[1:]  # Remove '$' prefix
                return eval(expr, {"__builtins__": cls.SAFE_BUILTINS}, dict(params))
            except Exception as e:
                available_params = list(params.keys())
                raise ValueError(
                    f"Failed to resolve dynamic value '{expr}': {e}. " f"Available parameters: {available_params}"
                )
        return value
