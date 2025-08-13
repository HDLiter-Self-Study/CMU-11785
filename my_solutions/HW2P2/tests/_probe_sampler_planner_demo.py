import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Add StagePlan to imports
# add src to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))
from models.architecture_planner import StagePlan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
# Ensure `src` is importable as top-level (for `sampling`, `config`, `models` packages)
sys.path.insert(0, str(SRC_DIR))

# ---------------- Parameter estimation & summary helpers ---------------- #


def _dump_trial_summary(idx: int, arch_spec: Dict[str, Any], planned: StagePlan) -> Dict[str, Any]:
    """Build a compact summary for analysis with parameter estimates.

    The parameter estimates are coarse and derived from stage/block shapes only.
    """
    regnet_rule = arch_spec.get("regnet_rule") or {}
    width_multiplier = arch_spec.get("width_multiplier")
    arch_shape = (arch_spec.get("extras") or {}).get("arch_shape") or arch_spec.get("arch_shape")

    param_est = _estimate_params_coarse(planned)

    return {
        "trial": idx,
        "arch_type": arch_spec.get("type"),
        "arch_shape": arch_shape,
        "num_stages": planned.num_stages,
        "out_channels": planned.out_channels,
        "stages": planned.depths,
        "downsamplings": planned.downsamplings,
        "regnet": {
            "width_multiplier": width_multiplier,
            "rule": regnet_rule,
        },
        "param_estimate": param_est,
    }


# ---------------- Template override helpers ---------------- #


def _inject_shape_override(content: Dict[str, Any], shape: str) -> Dict[str, Any]:
    """Inject a custom override into the template content to fix arch_shape choice.

    This writes under strategy_levels.custom.architectures:
      shape_search.selection.choices.custom: [shape]
    """
    strategy_levels = content.setdefault("strategy_levels", {})
    custom = strategy_levels.setdefault("custom", {})
    arches = custom.setdefault("architectures", {})
    arches["shape_search.selection.choices.custom"] = [str(shape)]
    return content


def _inject_num_stages_override(content: Dict[str, Any], num_stages: int) -> Dict[str, Any]:
    """Inject a custom override to fix num_stages choice."""
    strategy_levels = content.setdefault("strategy_levels", {})
    custom = strategy_levels.setdefault("custom", {})
    arches = custom.setdefault("architectures", {})
    arches["num_stages_selection.selection.choices.custom"] = [int(num_stages)]
    return content


def _make_temp_template(
    template_path: Path,
    n_trials: Optional[int] = None,
    shape: Optional[str] = None,
    num_stages: Optional[int] = None,
) -> Path:
    """Create a temporary template with optional trial count, shape, and num_stages injected."""
    original_text = template_path.read_text(encoding="utf-8")
    try:
        from omegaconf import OmegaConf

        content = OmegaConf.to_container(OmegaConf.load(str(template_path)), resolve=True)
        if not isinstance(content, dict):
            raise ValueError("template must be a mapping")
        shortcuts = content.get("shortcuts") or {}
        if not isinstance(shortcuts, dict):
            shortcuts = {}
        if n_trials is not None:
            shortcuts["n_trials"] = int(n_trials)
            shortcuts["optuna.n_trials"] = int(n_trials)
        content["shortcuts"] = shortcuts
        if shape is not None:
            content = _inject_shape_override(content, shape)
        if num_stages is not None:
            content = _inject_num_stages_override(content, int(num_stages))
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        tmp_path = Path(tmp.name)
        tmp.close()
        tmp_path.write_text(OmegaConf.to_yaml(content), encoding="utf-8")
        return tmp_path
    except Exception:
        # Fallback: append simple YAML – only supports n_trials without complex nested edits
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        tmp_path = Path(tmp.name)
        tmp.close()
        with tmp_path.open("w", encoding="utf-8") as f:
            f.write(original_text)
            f.write("\nshortcuts:\n")
            if n_trials is not None:
                f.write(f"  n_trials: {int(n_trials)}\n")
                f.write(f"  optuna.n_trials: {int(n_trials)}\n")
            # Note: without structured YAML editing, shape/num_stages override is skipped in fallback
        return tmp_path


# ---------------- Estimation helpers ---------------- #


def _extract_block_type(bt: Any) -> str:
    if isinstance(bt, str):
        return bt
    if isinstance(bt, dict):
        if "block_type" in bt and isinstance(bt["block_type"], str):
            return bt["block_type"]
        if "name" in bt and isinstance(bt["name"], str):
            return bt["name"]
    return "unknown"


def _estimate_params_coarse(planned: StagePlan) -> Dict[str, Any]:
    """Coarse parameter estimation per stage and total.

    Assumptions:
    - ResNet basic: each block has two 3x3 convs. Projection 1x1 at stage head if channels change.
    - ResNet bottleneck: expansion ratio 4, with 1x1-3x3-1x1. Projection when needed.
    - ConvNeXt inverted bottleneck: expansion ratio 4, k=7 depthwise with pw-dw-pw.
    - Norms approximated as 2*C per conv output.
    - Downsample projection added when channels change between stages.
    """
    num_stages: int = planned.num_stages
    out_channels: List[int] = planned.out_channels
    depths: List[int] = planned.depths
    block_types_raw: List[Any] = planned.block_types
    block_types: List[str] = [_extract_block_type(x) for x in block_types_raw]
    arch_type: str = str(planned.meta.get("type") or "")

    if not num_stages or len(out_channels) != num_stages or len(depths) != num_stages:
        return {"total": 0, "stages": []}

    stem = planned.meta.get("stem") or {}
    stem_out = stem.get("out_channels")
    prev_c = int(stem_out) if isinstance(stem_out, int) else (out_channels[0] if out_channels else 3)

    total_params = 0
    stage_summaries: List[Dict[str, Any]] = []

    for i in range(num_stages):
        ci = int(out_channels[i])
        di = int(depths[i])
        bt = block_types[i] if i < len(block_types) else "basic"

        stage_params = 0
        proj_params = 0
        norm_params = 0

        if arch_type == "resnet":
            if bt == "bottleneck":
                cb = max(1, ci // 4)
                per_block_conv = ci * cb + cb * cb * 9 + cb * ci
                per_block_norm = 2 * cb + 2 * cb + 2 * ci
            else:
                per_block_conv = 2 * ci * ci * 9
                per_block_norm = 2 * 2 * ci
            stage_params += di * (per_block_conv + per_block_norm)
            if prev_c != ci:
                proj_params = prev_c * ci
                norm_params += 2 * ci
        elif arch_type == "convnext":
            er = 4
            k = 7
            ce = ci * er
            per_block_conv = ci * ce + ce * (k * k) + ce * ci
            per_block_norm = 2 * (2 * ce + ci)
            stage_params += di * (per_block_conv + per_block_norm)
            if prev_c != ci:
                proj_params = prev_c * ci * 4
                norm_params += 2 * ci
        else:
            per_block_conv = 2 * ci * ci * 9
            per_block_norm = 2 * 2 * ci
            stage_params += di * (per_block_conv + per_block_norm)
            if prev_c != ci:
                proj_params = prev_c * ci
                norm_params += 2 * ci

        stage_total = stage_params + proj_params + norm_params
        total_params += stage_total

        stage_summaries.append(
            {
                "stage": i + 1,
                "in_channels": prev_c,
                "out_channels": ci,
                "num_blocks": di,
                "block_type": bt,
                "params_conv+norm": int(stage_params),
                "params_projection": int(proj_params + norm_params),
                "stage_total": int(stage_total),
            }
        )

        prev_c = ci

    # ---------------- Head (classification) params ---------------- #
    # Assumptions match src/models/common_blocks/head.py default behavior:
    # - Global avg pool: no parameters
    # - 2D normalization with affine: 2 * C_last
    # - Optional hidden MLP is disabled by default (hidden_dims is None)
    # - Final Linear: (C_last -> num_classes)
    # If future runs provide num_classes in meta.extras, prefer it; else use default.
    DEFAULT_NUM_CLASSES = 8631
    meta = planned.meta or {}
    extras = meta.get("extras") or {}
    num_classes = int(extras.get("num_classes") or DEFAULT_NUM_CLASSES)
    hidden_dims = extras.get("head_hidden_dims")  # optional; None by default
    try:
        hidden_dims = int(hidden_dims) if hidden_dims is not None else None
    except Exception:
        hidden_dims = None

    last_c = int(out_channels[-1]) if out_channels else 0
    head_norm_params = 2 * last_c if last_c > 0 else 0
    head_fc1_params = (last_c * hidden_dims + hidden_dims) if (hidden_dims and last_c > 0) else 0
    head_in_to_fc2 = hidden_dims if hidden_dims else last_c
    head_fc2_params = (head_in_to_fc2 * num_classes + num_classes) if head_in_to_fc2 > 0 else 0
    head_total = head_norm_params + head_fc1_params + head_fc2_params

    return {"total": int(total_params + head_total), "stages": stage_summaries, "head_total": int(head_total)}


# ---------------- Stats computation & rendering ---------------- #


def _compute_depth_stats(results: List[Dict[str, Any]]) -> Dict[int, Dict[int, Dict[int, int]]]:
    grouped: Dict[int, Dict[int, Dict[int, int]]] = {}
    for item in results:
        ns = int(item.get("num_stages") or 0)
        depths = item.get("stages") or []
        if not ns or not isinstance(depths, list):
            continue
        grouped.setdefault(ns, {})
        for i, d in enumerate(depths, start=1):
            by_stage = grouped[ns].setdefault(i, {})
            by_stage[int(d)] = by_stage.get(int(d), 0) + 1
    return grouped


def _compute_total_depth_stats(results: List[Dict[str, Any]]) -> Dict[int, Dict[int, int]]:
    totals: Dict[int, Dict[int, int]] = {}
    for item in results:
        ns = int(item.get("num_stages") or 0)
        depths = item.get("stages") or []
        if not ns or not isinstance(depths, list):
            continue
        tot = int(sum(int(x) for x in depths))
        bucket = totals.setdefault(ns, {})
        bucket[tot] = bucket.get(tot, 0) + 1
    return totals


def _compute_depth_stats_by_shape(results: List[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[int, Dict[int, int]]]]:
    """Per-shape depth histogram per stage position.

    Returns a nested mapping:
      shape -> num_stages -> stage_index -> { depth_value -> count }
    """
    grouped: Dict[str, Dict[int, Dict[int, Dict[int, int]]]] = {}
    for item in results:
        shape = str(item.get("_group_shape") or item.get("arch_shape") or "<none>")
        ns = int(item.get("num_stages") or 0)
        depths = item.get("stages") or []
        if not ns or not isinstance(depths, list):
            continue
        grouped.setdefault(shape, {})
        grouped[shape].setdefault(ns, {})
        for i, d in enumerate(depths, start=1):
            by_stage = grouped[shape][ns].setdefault(i, {})
            by_stage[int(d)] = by_stage.get(int(d), 0) + 1
    return grouped


def _compute_total_depth_stats_by_shape(results: List[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[int, int]]]:
    """Per-shape total depth histogram.

    Returns a nested mapping:
      shape -> num_stages -> { total_depth -> count }
    """
    totals: Dict[str, Dict[int, Dict[int, int]]] = {}
    for item in results:
        shape = str(item.get("_group_shape") or item.get("arch_shape") or "<none>")
        ns = int(item.get("num_stages") or 0)
        depths = item.get("stages") or []
        if not ns or not isinstance(depths, list):
            continue
        tot = int(sum(int(x) for x in depths))
        totals.setdefault(shape, {})
        bucket = totals[shape].setdefault(ns, {})
        bucket[tot] = bucket.get(tot, 0) + 1
    return totals


def _compute_width_summary(results: List[Dict[str, Any]]) -> Dict[int, Dict[int, Dict[str, int]]]:
    """Compute per-stage width summary grouped by num_stages.

    Returns:
      num_stages -> stage_index -> summary dict (count/min/mean/median/p90/p95/max)
    """
    from collections import defaultdict

    buckets: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    for item in results:
        ns = int(item.get("num_stages") or 0)
        widths = item.get("out_channels") or []
        if not ns or not isinstance(widths, list) or len(widths) < ns:
            continue
        for i in range(ns):
            try:
                w = int(widths[i])
            except Exception:
                continue
            buckets[ns][i + 1].append(w)

    summaries: Dict[int, Dict[int, Dict[str, int]]] = {}
    for ns, stage_map in buckets.items():
        summaries[ns] = {}
        for stage_idx, vals in stage_map.items():
            summaries[ns][stage_idx] = _summarize(vals)
    return summaries


def _compute_width_summary_by_shape(results: List[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[int, Dict[str, int]]]]:
    """Compute per-stage width summary grouped by shape and num_stages.

    Returns:
      shape -> num_stages -> stage_index -> summary dict
    """
    from collections import defaultdict

    buckets: Dict[str, Dict[int, Dict[int, List[int]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in results:
        shape = str(item.get("_group_shape") or item.get("arch_shape") or "<none>")
        ns = int(item.get("num_stages") or 0)
        widths = item.get("out_channels") or []
        if not ns or not isinstance(widths, list) or len(widths) < ns:
            continue
        for i in range(ns):
            try:
                w = int(widths[i])
            except Exception:
                continue
            buckets[shape][ns][i + 1].append(w)

    summaries: Dict[str, Dict[int, Dict[int, Dict[str, int]]]] = {}
    for shape, ns_map in buckets.items():
        summaries[shape] = {}
        for ns, stage_map in ns_map.items():
            summaries[shape][ns] = {}
            for stage_idx, vals in stage_map.items():
                summaries[shape][ns][stage_idx] = _summarize(vals)
    return summaries


def _compute_max_width_summary(results: List[Dict[str, Any]]) -> Dict[int, Dict[str, int]]:
    """Compute max stage width summary per trial grouped by num_stages."""
    from collections import defaultdict

    buckets: Dict[int, List[int]] = defaultdict(list)
    for item in results:
        ns = int(item.get("num_stages") or 0)
        widths = item.get("out_channels") or []
        if not ns or not isinstance(widths, list) or len(widths) == 0:
            continue
        try:
            m = max(int(w) for w in widths)
        except Exception:
            continue
        buckets[ns].append(m)

    summaries: Dict[int, Dict[str, int]] = {}
    for ns, vals in buckets.items():
        summaries[ns] = _summarize(vals)
    return summaries


def _compute_max_width_summary_by_shape(results: List[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[str, int]]]:
    """Compute max stage width summary per trial grouped by shape and num_stages."""
    from collections import defaultdict

    buckets: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    for item in results:
        shape = str(item.get("_group_shape") or item.get("arch_shape") or "<none>")
        ns = int(item.get("num_stages") or 0)
        widths = item.get("out_channels") or []
        if not ns or not isinstance(widths, list) or len(widths) == 0:
            continue
        try:
            m = max(int(w) for w in widths)
        except Exception:
            continue
        buckets[shape][ns].append(m)

    summaries: Dict[str, Dict[int, Dict[str, int]]] = {}
    for shape, ns_map in buckets.items():
        summaries[shape] = {}
        for ns, vals in ns_map.items():
            summaries[shape][ns] = _summarize(vals)
    return summaries


def _compute_breakdown(
    results: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], Dict[str, Dict[int, int]], Dict[str, Dict[str, int]]]:
    shape_counts: Dict[str, int] = {}
    shape_by_num: Dict[str, Dict[int, int]] = {}
    type_by_shape: Dict[str, Dict[str, int]] = {}
    for item in results:
        shape = str(item.get("arch_shape") or "<none>")
        ns = int(item.get("num_stages") or 0)
        at = str(item.get("arch_type") or "<none>")
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
        sbn = shape_by_num.setdefault(shape, {})
        sbn[ns] = sbn.get(ns, 0) + 1
        tbs = type_by_shape.setdefault(at, {})
        tbs[shape] = tbs.get(shape, 0) + 1
    return shape_counts, shape_by_num, type_by_shape


def _percentile(sorted_vals: List[int], p: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, int(round(p * (len(sorted_vals) - 1)))))
    return int(sorted_vals[idx])


def _summarize(vals: List[int]) -> Dict[str, int]:
    vals2 = sorted(vals)
    n = len(vals2)
    if n == 0:
        return {"count": 0, "min": 0, "mean": 0, "median": 0, "p90": 0, "p95": 0, "max": 0}
    mean = int(sum(vals2) / n)
    return {
        "count": n,
        "min": int(vals2[0]),
        "mean": mean,
        "median": _percentile(vals2, 0.5),
        "p90": _percentile(vals2, 0.9),
        "p95": _percentile(vals2, 0.95),
        "max": int(vals2[-1]),
    }


def _compute_param_stats(
    results: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, Dict[int, int]]]]:
    """Compute parameter estimate stats.

    Returns:
      - overall: {"count","min","mean","median","p90","p95","max"}
      - by_shape: shape -> { summary: same fields, hist: bins dict }
    """

    def _bins(v: int) -> str:
        # bins in millions
        m = v / 1_000_000.0
        if m < 5:
            return "[0,5)"
        if m < 10:
            return "[5,10)"
        if m < 15:
            return "[10,15)"
        if m < 20:
            return "[15,20)"
        if m < 30:
            return "[20,30)"
        return "[30,+)"

    totals_all: List[int] = []
    by_shape_vals: Dict[str, List[int]] = {}
    for item in results:
        est = item.get("param_estimate") or {}
        tot = int(est.get("total") or 0)
        totals_all.append(tot)
        shape = str(item.get("arch_shape") or "<none>")
        by_shape_vals.setdefault(shape, []).append(tot)

    overall = {"overall": _summarize(totals_all)}

    by_shape_hist: Dict[str, Dict[str, Dict[int, int]]] = {}
    for shape, vals in by_shape_vals.items():
        sm = _summarize(vals)
        hist: Dict[str, int] = {}
        for v in vals:
            b = _bins(v)
            hist[b] = hist.get(b, 0) + 1
        if shape not in by_shape_hist:
            by_shape_hist[shape] = {}
        by_shape_hist[shape]["summary"] = sm
        by_shape_hist[shape]["hist"] = hist

    return overall, by_shape_hist


def _compute_param_stats_by_shape_stage_with_threshold(
    results: List[Dict[str, Any]], threshold: int
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Param estimate summary grouped by shape and num_stages; includes <=threshold ratio."""
    buckets: Dict[str, Dict[int, List[int]]] = {}
    group_buckets: Dict[str, List[int]] = {}
    for item in results:
        # Prefer explicit group tags if present
        shape = str(item.get("_group_shape") or item.get("arch_shape") or "<none>")
        ns = int(item.get("_group_ns") or item.get("num_stages") or 0)
        est = item.get("param_estimate") or {}
        tot = int(est.get("total") or 0)
        buckets.setdefault(shape, {}).setdefault(ns, []).append(tot)
        key = f"{shape}_{ns}"
        group_buckets.setdefault(key, []).append(tot)
    summaries: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for shape, ns_map in buckets.items():
        summaries[shape] = {}
        for ns, vals in ns_map.items():
            sm = _summarize(vals)
            lte = sum(1 for v in vals if v <= threshold)
            ratio = 0.0 if sm["count"] == 0 else lte / sm["count"]
            sm2: Dict[str, Any] = dict(sm)
            sm2["lte_threshold_count"] = lte
            sm2["lte_threshold_ratio"] = ratio
            summaries[shape][ns] = sm2
    return summaries


def _render_ascii_table(
    per_stage: Dict[int, Dict[int, Dict[int, int]]],
    total_stats: Dict[int, Dict[int, int]],
    breakdown: Tuple[Dict[str, int], Dict[str, Dict[int, int]], Dict[str, Dict[str, int]]],
    param_stats: Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, Dict[int, int]]]],
    param_by_shape_stage: Dict[str, Dict[int, Dict[str, Any]]],
    threshold_millions: float,
    per_stage_by_shape: Optional[Dict[str, Dict[int, Dict[int, Dict[int, int]]]]] = None,
    total_stats_by_shape: Optional[Dict[str, Dict[int, Dict[int, int]]]] = None,
) -> str:
    lines: List[str] = []

    # Shape & type breakdowns
    shape_counts, shape_by_num, type_by_shape = breakdown
    lines.append("Shape breakdown (counts)")
    lines.append("==========================")
    for shape in sorted(shape_counts.keys()):
        lines.append(f"{shape:14s} : {shape_counts[shape]:4d}")
    lines.append("")

    lines.append("Shape by num_stages (counts)")
    lines.append("=============================")
    for shape in sorted(shape_by_num.keys()):
        parts = [f"ns={ns}:{cnt}" for ns, cnt in sorted(shape_by_num[shape].items())]
        lines.append(f"{shape:14s} : " + ", ".join(parts))
    lines.append("")

    lines.append("Arch type by shape (counts)")
    lines.append("===========================")
    for at in sorted(type_by_shape.keys()):
        parts = [f"{shape}:{cnt}" for shape, cnt in sorted(type_by_shape[at].items())]
        lines.append(f"{at:10s} : " + ", ".join(parts))
    lines.append("")

    # Parameter estimate stats
    overall, by_shape_hist = param_stats
    lines.append("Parameter estimate (total params) - overall")
    lines.append("==========================================")
    ov = overall.get("overall", {})
    lines.append(
        f"count={ov.get('count',0)}  min={ov.get('min',0):,}  median={ov.get('median',0):,}  "
        f"mean={ov.get('mean',0):,}  p90={ov.get('p90',0):,}  p95={ov.get('p95',0):,}  max={ov.get('max',0):,}"
    )
    lines.append("")

    lines.append("Parameter estimate by shape (summary + histogram bins in millions)")
    lines.append("=================================================================")
    for shape in sorted(by_shape_hist.keys()):
        sm = by_shape_hist[shape].get("summary", {})
        lines.append(
            f"{shape:14s} : count={sm.get('count',0)}  min={sm.get('min',0):,}  median={sm.get('median',0):,}  "
            f"mean={sm.get('mean',0):,}  p90={sm.get('p90',0):,}  p95={sm.get('p95',0):,}  max={sm.get('max',0):,}"
        )
        hist = by_shape_hist[shape].get("hist", {})
        bin_order = ["[0,5)", "[5,10)", "[10,15)", "[15,20)", "[20,30)", "[30,+)"]
        parts = [f"{b}:{hist.get(b,0)}" for b in bin_order]
        lines.append("  bins: " + ", ".join(parts))
    lines.append("")

    # Parameter estimate by shape x num_stages with <= threshold
    lines.append(f"Parameter estimate by shape x num_stages (summary, <= {threshold_millions:.0f}M ratio)")
    lines.append("================================================================================")
    for shape in sorted(param_by_shape_stage.keys()):
        lines.append(f"{shape}:")
        ns_map = param_by_shape_stage[shape]
        for ns in sorted(ns_map.keys()):
            sm = ns_map[ns]
            ratio = sm.get("lte_threshold_ratio", 0.0)
            lte_count = sm.get("lte_threshold_count", 0)
            lines.append(
                f"  ns={ns}  count={sm.get('count',0):3d}  min={sm.get('min',0):,}  median={sm.get('median',0):,}  "
                f"mean={sm.get('mean',0):,}  p90={sm.get('p90',0):,}  p95={sm.get('p95',0):,}  max={sm.get('max',0):,}  "
                f"<= {threshold_millions:.0f}M: {lte_count}/{sm.get('count',0)} ({ratio*100:.1f}%)"
            )
    lines.append("")

    # Flat group summaries: {shape}_{num_stages}
    lines.append("Grouped summaries by {shape}_{num_stages}")
    lines.append("=========================================")
    flat_items: List[Tuple[str, Dict[str, Any]]] = []
    for shape in sorted(param_by_shape_stage.keys()):
        for ns in sorted(param_by_shape_stage[shape].keys()):
            key = f"{shape}_{ns}"
            flat_items.append((key, param_by_shape_stage[shape][ns]))
    for key, sm in sorted(flat_items, key=lambda kv: kv[0]):
        ratio = sm.get("lte_threshold_ratio", 0.0)
        lte_count = sm.get("lte_threshold_count", 0)
        lines.append(
            f"{key:20s}  count={sm.get('count',0):3d}  min={sm.get('min',0):,}  median={sm.get('median',0):,}  "
            f"mean={sm.get('mean',0):,}  p90={sm.get('p90',0):,}  p95={sm.get('p95',0):,}  max={sm.get('max',0):,}  "
            f"<= {threshold_millions:.0f}M: {lte_count}/{sm.get('count',0)} ({ratio*100:.1f}%)"
        )
    lines.append("")

    # Flat group summaries: {arch_type}_{shape}_{num_stages}
    lines.append("Grouped summaries by {arch_type}_{shape}_{num_stages}")
    lines.append("====================================================")
    typed: Dict[str, Dict[str, Dict[int, List[int]]]] = {}
    # build typed buckets from original results in breakdown context is not available here;
    # we approximate by using param_by_shape_stage for shapes/ns and assume both arch types appear similarly.
    # For accurate split, we rely on group tags stored in JSON; read them back from the latest JSON output if present.
    try:
        data_path = Path(PROJECT_ROOT / "sampling_probe.json")
        if data_path.is_file():
            arr = json.loads(data_path.read_text(encoding="utf-8"))
            typed_buckets: Dict[str, Dict[str, Dict[int, List[int]]]] = {}
            for item in arr:
                at = str(item.get("arch_type") or "<none>")
                shape = str(item.get("_group_shape") or item.get("arch_shape") or "<none>")
                ns = int(item.get("_group_ns") or item.get("num_stages") or 0)
                tot = int((item.get("param_estimate") or {}).get("total") or 0)
                typed_buckets.setdefault(at, {}).setdefault(shape, {}).setdefault(ns, []).append(tot)
            for at in sorted(typed_buckets.keys()):
                lines.append(f"{at}:")
                for shape in sorted(typed_buckets[at].keys()):
                    for ns in sorted(typed_buckets[at][shape].keys()):
                        vals = typed_buckets[at][shape][ns]
                        sm = _summarize(vals)
                        lte = sum(1 for v in vals if v <= int(threshold_millions * 1_000_000))
                        ratio = 0.0 if sm["count"] == 0 else lte / sm["count"]
                        lines.append(
                            f"  {shape}_{ns:>1d}  count={sm.get('count',0):3d}  min={sm.get('min',0):,}  median={sm.get('median',0):,}  "
                            f"mean={sm.get('mean',0):,}  p90={sm.get('p90',0):,}  p95={sm.get('p95',0):,}  max={sm.get('max',0):,}  "
                            f"<= {threshold_millions:.0f}M: {lte}/{sm.get('count',0)} ({ratio*100:.1f}%)"
                        )
                lines.append("")
    except Exception:
        pass

    # Per-stage depth distribution
    header = "Depth distribution by num_stages (per stage position)"
    lines.append(header)
    lines.append("=" * len(header))

    for ns in sorted(per_stage.keys()):
        lines.append("")
        lines.append(f"num_stages = {ns}")
        lines.append("-" * (14 + len(str(ns))))
        by_stage = per_stage[ns]
        max_count = 1
        for stage_idx, freq in by_stage.items():
            if freq:
                max_count = max(max_count, max(freq.values()))
        scale = max(1, max_count // 50)

        for stage_idx in range(1, ns + 1):
            freq = by_stage.get(stage_idx, {})
            if not freq:
                lines.append(f"Stage {stage_idx}: (no data)")
                continue
            lines.append(f"Stage {stage_idx}:")
            for depth in sorted(freq.keys()):
                count = freq[depth]
                bar_len = max(1, count // scale)
                bar = "#" * bar_len
                lines.append(f"  depth={depth:2d}  count={count:4d}  {bar}")

        totals = total_stats.get(ns, {})
        if totals:
            lines.append("")
            lines.append(f"Total depth distribution (num_stages={ns})")
            lines.append("~" * (29 + len(str(ns))))
            t_max = max(totals.values()) if totals else 1
            t_scale = max(1, t_max // 50)
            for total in sorted(totals.keys()):
                cnt = totals[total]
                bar_len = max(1, cnt // t_scale)
                bar = "#" * bar_len
                lines.append(f"  total={total:3d}  count={cnt:4d}  {bar}")

    # Optional: Per-shape per-stage depth distributions
    if per_stage_by_shape and total_stats_by_shape:
        header2 = "Depth distribution by shape and num_stages (per stage position)"
        lines.append("")
        lines.append(header2)
        lines.append("=" * len(header2))

        for shape in sorted(per_stage_by_shape.keys()):
            lines.append("")
            lines.append(f"Shape: {shape}")
            ns_map = per_stage_by_shape[shape]
            for ns in sorted(ns_map.keys()):
                lines.append("")
                lines.append(f"num_stages = {ns}")
                lines.append("-" * (14 + len(str(ns))))
                stage_map = ns_map[ns]
                max_count = 1
                for _, freq in stage_map.items():
                    if freq:
                        max_count = max(max_count, max(freq.values()))
                scale = max(1, max_count // 50)
                for stage_idx in range(1, ns + 1):
                    freq = stage_map.get(stage_idx, {})
                    if not freq:
                        lines.append(f"Stage {stage_idx}: (no data)")
                        continue
                    lines.append(f"Stage {stage_idx}:")
                    for depth in sorted(freq.keys()):
                        count = freq[depth]
                        bar_len = max(1, count // scale)
                        bar = "#" * bar_len
                        lines.append(f"  depth={depth:2d}  count={count:4d}  {bar}")

                # Totals per shape+ns
                totals2 = (total_stats_by_shape.get(shape) or {}).get(ns) or {}
                if totals2:
                    lines.append("")
                    lines.append(f"Total depth distribution (shape={shape}, num_stages={ns})")
                    lines.append("~" * (29 + len(str(ns)) + len(shape)))
                    t_max = max(totals2.values()) if totals2 else 1
                    t_scale = max(1, t_max // 50)
                    for total in sorted(totals2.keys()):
                        cnt = totals2[total]
                        bar_len = max(1, cnt // t_scale)
                        bar = "#" * bar_len
                        lines.append(f"  total={total:3d}  count={cnt:4d}  {bar}")

    return "\n".join(lines)


# ---------------- Main flow ---------------- #


def _run_generation(template_path: Path, n_trials: int) -> List[Dict[str, Any]]:
    from sampling.generation_entry import generate_configs_from_template
    from sampling.data_resolver import resolve_effective_data_config
    from models.architecture_planner import StagePlanner

    cfg_dict = generate_configs_from_template(str(template_path), allow_new_paths=False)
    sampled = cfg_dict.get("sampled") or []
    results: List[Dict[str, Any]] = []
    for idx, hierarchical in enumerate(sampled):
        eff = resolve_effective_data_config(cfg_dict, hierarchical)
        arch_spec = eff["model"]["architectures"]
        planned: StagePlan = StagePlanner.plan(arch_spec)
        results.append(_dump_trial_summary(idx=idx, arch_spec=arch_spec, planned=planned))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sample models, resolve, plan, and save per-trial summary plus depth/param stats."
    )
    parser.add_argument(
        "--template",
        "-t",
        type=str,
        default=str(PROJECT_ROOT / "architecture_search_template.yaml"),
        help="Path to search template YAML (defaults to architecture_search_template.yaml)",
    )
    parser.add_argument(
        "--n_trials",
        "-n",
        type=int,
        default=None,
        help="Override total number of trials for this probe run (ignored when using --ns_list/--shapes with --group_trials)",
    )
    parser.add_argument(
        "--group_trials",
        type=int,
        default=0,
        help="Trials per (shape,num_stages) group. When >0 and both --ns_list and --shapes are set, runs cartesian groups.",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default="",
        help="Comma-separated list of arch_shape values to run in separate batches (avoids dynamic Optuna choices)",
    )
    parser.add_argument(
        "--ns_list",
        type=str,
        default="",
        help="Comma-separated list of num_stages values to run. Use with --shapes and --group_trials to form groups.",
    )
    parser.add_argument(
        "--lte_threshold_millions",
        type=float,
        default=18.0,
        help="Threshold in millions for <= threshold ratio stats (default 18.0).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(PROJECT_ROOT / "sampling_probe.json"),
        help="Write all per-trial results to this JSON file (array).",
    )
    parser.add_argument(
        "--stats_out",
        "-s",
        type=str,
        default=str(PROJECT_ROOT / "sampling_depth_stats.txt"),
        help="Write depth/param distribution report to this text file.",
    )

    args = parser.parse_args()
    template_path = Path(args.template).resolve()
    if not template_path.is_file():
        raise FileNotFoundError(f"Template not found: {template_path}")

    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    ns_values = [int(x.strip()) for x in args.ns_list.split(",") if x.strip()]

    all_results: List[Dict[str, Any]] = []

    if shapes and ns_values and args.group_trials > 0:
        # Cartesian groups: for each (shape, ns) run group_trials
        for shape in shapes:
            for ns in ns_values:
                tmp_tpl = _make_temp_template(
                    template_path, n_trials=int(args.group_trials), shape=shape, num_stages=ns
                )
                batch_results = _run_generation(tmp_tpl, int(args.group_trials))
                # tag results with explicit group labels (do not rely on resolver)
                for r in batch_results:
                    r["_group_shape"] = shape
                    r["_group_ns"] = int(ns)
                    r["_group_key"] = f"{shape}_{int(ns)}"
                all_results.extend(batch_results)
                try:
                    os.remove(tmp_tpl)
                except Exception:
                    pass
    else:
        # Fallback: previous modes
        total_trials = int(args.n_trials or 30)
        if shapes:
            per = max(1, total_trials // len(shapes))
            remainder = total_trials - per * len(shapes)
            for i, shape in enumerate(shapes):
                n_this = per + (1 if i < remainder else 0)
                tmp_tpl = _make_temp_template(template_path, n_trials=n_this, shape=shape)
                batch_results = _run_generation(tmp_tpl, n_this)
                all_results.extend(batch_results)
                try:
                    os.remove(tmp_tpl)
                except Exception:
                    pass
        else:
            tmp_tpl = _make_temp_template(template_path, n_trials=total_trials, shape=None)
            all_results = _run_generation(tmp_tpl, total_trials)
            try:
                os.remove(tmp_tpl)
            except Exception:
                pass

    # Write JSON
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Write stats report
    per_stage_stats = _compute_depth_stats(all_results)
    total_stats = _compute_total_depth_stats(all_results)
    breakdown = _compute_breakdown(all_results)
    param_stats = _compute_param_stats(all_results)
    threshold = int(args.lte_threshold_millions * 1_000_000)
    param_by_shape_stage = _compute_param_stats_by_shape_stage_with_threshold(all_results, threshold=threshold)
    # Extended shape-specific depth stats
    per_stage_by_shape = _compute_depth_stats_by_shape(all_results)
    total_stats_by_shape = _compute_total_depth_stats_by_shape(all_results)

    # Width summaries
    width_per_stage = _compute_width_summary(all_results)
    width_per_stage_by_shape = _compute_width_summary_by_shape(all_results)
    max_width_by_ns = _compute_max_width_summary(all_results)
    max_width_by_shape_ns = _compute_max_width_summary_by_shape(all_results)

    report = _render_ascii_table(
        per_stage_stats,
        total_stats,
        breakdown,
        param_stats,
        param_by_shape_stage,
        threshold_millions=float(args.lte_threshold_millions),
        per_stage_by_shape=per_stage_by_shape,
        total_stats_by_shape=total_stats_by_shape,
    )
    # Append width summaries to report
    lines_extra: List[str] = []
    lines_extra.append("")
    lines_extra.append("Width summary by num_stages (per stage)")
    lines_extra.append("======================================")
    for ns in sorted(width_per_stage.keys()):
        lines_extra.append("")
        lines_extra.append(f"num_stages = {ns}")
        for stage_idx in sorted(width_per_stage[ns].keys()):
            sm = width_per_stage[ns][stage_idx]
            lines_extra.append(
                f"  stage={stage_idx}  count={sm.get('count',0):3d}  min={sm.get('min',0):,}  median={sm.get('median',0):,}  "
                f"mean={sm.get('mean',0):,}  p90={sm.get('p90',0):,}  p95={sm.get('p95',0):,}  max={sm.get('max',0):,}"
            )

    lines_extra.append("")
    lines_extra.append("Width summary by shape x num_stages (per stage)")
    lines_extra.append("===============================================")
    for shape in sorted(width_per_stage_by_shape.keys()):
        lines_extra.append("")
        lines_extra.append(f"{shape}:")
        ns_map = width_per_stage_by_shape[shape]
        for ns in sorted(ns_map.keys()):
            for stage_idx in sorted(ns_map[ns].keys()):
                sm = ns_map[ns][stage_idx]
                lines_extra.append(
                    f"  ns={ns} stage={stage_idx}  count={sm.get('count',0):3d}  min={sm.get('min',0):,}  median={sm.get('median',0):,}  "
                    f"mean={sm.get('mean',0):,}  p90={sm.get('p90',0):,}  p95={sm.get('p95',0):,}  max={sm.get('max',0):,}"
                )

    lines_extra.append("")
    lines_extra.append("Max stage width summary by num_stages")
    lines_extra.append("=====================================")
    for ns in sorted(max_width_by_ns.keys()):
        sm = max_width_by_ns[ns]
        lines_extra.append(
            f"ns={ns}  count={sm.get('count',0):3d}  min={sm.get('min',0):,}  median={sm.get('median',0):,}  "
            f"mean={sm.get('mean',0):,}  p90={sm.get('p90',0):,}  p95={sm.get('p95',0):,}  max={sm.get('max',0):,}"
        )

    lines_extra.append("")
    lines_extra.append("Max stage width summary by shape x num_stages")
    lines_extra.append("=============================================")
    for shape in sorted(max_width_by_shape_ns.keys()):
        lines_extra.append(f"{shape}:")
        ns_map = max_width_by_shape_ns[shape]
        for ns in sorted(ns_map.keys()):
            sm = ns_map[ns]
            lines_extra.append(
                f"  ns={ns}  count={sm.get('count',0):3d}  min={sm.get('min',0):,}  median={sm.get('median',0):,}  "
                f"mean={sm.get('mean',0):,}  p90={sm.get('p90',0):,}  p95={sm.get('p95',0):,}  max={sm.get('max',0):,}"
            )

    report = report + "\n" + "\n".join(lines_extra)
    stats_path = Path(args.stats_out).resolve()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
