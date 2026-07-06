from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from MPC4plus.experiments.training.train_vertex_gnn import resolve_device


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto() -> None:
    device = resolve_device("auto")
    assert device in {"cpu", "cuda"}


def test_resolve_device_cuda_behaviour() -> None:
    if torch.cuda.is_available():
        assert resolve_device("cuda") == "cuda"
    else:
        with pytest.raises(ValueError):
            resolve_device("cuda")