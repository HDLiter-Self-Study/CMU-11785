from __future__ import annotations

import pytest
import torch

from src.pipelines.factories import OptimizerFactory


def test_adam_betas_merge_and_conflict() -> None:
    params = [torch.nn.Parameter(torch.randn(3, requires_grad=True))]
    fac = OptimizerFactory()

    # Merge beta1/beta2 into betas
    opt = fac.create({"adam": {"lr": 1e-3, "beta1": 0.9, "beta2": 0.999}}, params=params)
    for pg in opt.param_groups:
        assert pg["betas"] == (0.9, 0.999)

    # Conflict: betas together with beta1/beta2 should fail
    with pytest.raises(ValueError):
        fac.create({"adam": {"lr": 1e-3, "betas": (0.9, 0.999), "beta1": 0.9}}, params=params)


def test_adamw_betas_merge() -> None:
    params = [torch.nn.Parameter(torch.randn(3, requires_grad=True))]
    fac = OptimizerFactory()
    opt = fac.create({"adamw": {"lr": 1e-3, "beta1": 0.85, "beta2": 0.995}}, params=params)
    for pg in opt.param_groups:
        assert pg["betas"] == (0.85, 0.995)


def test_rmsprop_available_from_torch() -> None:
    params = [torch.nn.Parameter(torch.randn(3, requires_grad=True))]
    fac = OptimizerFactory()
    # Ensure RMSprop path works via reflection (no custom wrapper needed)
    opt = fac.create({"rmsprop": {"lr": 1e-3, "alpha": 0.99, "eps": 1e-8}}, params=params)
    assert opt is not None
