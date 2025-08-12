import json
from pathlib import Path
import sys

import pytest

# Ensure repo root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_create_model_smoke():
    # Import lazily to ensure torch is available
    from src.models.architecture_factory import ArchitectureFactory

    eff_path = PROJECT_ROOT / "effective_latest.json"
    data = json.loads(eff_path.read_text(encoding="utf-8"))
    trials = data.get("effective_data") or []
    assert trials, "No trials present"
    # Pick a trial whose stem normalization does not require extra params
    # Prefer convnext (its stem uses layer_norm approximation by default)
    chosen = None
    for t in trials:
        arch = t["model"]["architectures"]
        if arch.get("type") == "convnext":
            chosen = arch
            break
    if chosen is None:
        chosen = trials[0]["model"]["architectures"]
    arch_cfg = chosen

    factory = ArchitectureFactory()
    model = factory.create_model(arch_cfg, in_channels=3, num_classes=2)

    # forward smoke test with dummy data (small spatial size)
    import torch

    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert isinstance(out, dict) and "out" in out and "feats" in out
    assert out["out"].shape[0] == 2
