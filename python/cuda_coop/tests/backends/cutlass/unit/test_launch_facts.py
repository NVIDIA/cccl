# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import sys
from types import ModuleType, SimpleNamespace

import pytest

from ....support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT

from cuda.coop._core import LaunchFacts
from cuda.coop.cutlass._dsl import _launch


class KernelOp:
    def __init__(self, attributes, parent_op=None):
        self.attributes = attributes
        self.parent_op = parent_op


def _install_fake_cutlass(monkeypatch):
    cutlass = ModuleType("cutlass")
    cutlass.__path__ = []
    cute = ModuleType("cutlass.cute")
    mlir = ModuleType("cutlass._mlir")
    mlir.__path__ = []
    ir = ModuleType("cutlass._mlir.ir")
    cutlass.cute = cute
    cutlass._mlir = mlir
    mlir.ir = ir
    for name, module in (
        ("cutlass", cutlass),
        ("cutlass.cute", cute),
        ("cutlass._mlir", mlir),
        ("cutlass._mlir.ir", ir),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return cute, ir


def test_launch_metadata_reconciles_all_exact_aliases():
    facts = _launch.launch_facts_from_launch_metadata(
        {
            "threads_per_block": (8, 4),
            "block_dim": (8, 4, 1),
            "grid": 9,
            "cluster_dim_x": 2,
            "cluster_dim_y": 1,
            "cluster_dim_z": 1,
            "cooperative_launch": True,
            "cluster_launch": True,
        },
        source="test_metadata",
    )

    assert facts.exact_block_dim == (8, 4, 1)
    assert facts.exact_grid_dim == (9, 1, 1)
    assert facts.exact_cluster_dim == (2, 1, 1)
    assert facts.cooperative_launch is True
    assert facts.cluster_launch is True
    assert {origin.fact for origin in facts.provenance} == {
        "exact_block_dim",
        "exact_grid_dim",
        "exact_cluster_dim",
        "cooperative_launch",
        "cluster_launch",
    }
    assert {origin.source for origin in facts.provenance} == {"test_metadata"}


def test_launch_metadata_rejects_conflicting_or_partial_shapes():
    with pytest.raises(ValueError, match="conflicting exact_block_dim"):
        _launch.launch_facts_from_launch_metadata({"block": 32, "block_dim": 64})
    with pytest.raises(ValueError, match="requires x, y, and z"):
        _launch.launch_facts_from_launch_metadata({"block_dim_x": 8, "block_dim_y": 4})
    with pytest.raises(TypeError, match="must be a bool"):
        _launch.launch_facts_from_launch_metadata(
            {"block": 32, "cooperative_launch": 1}
        )


def test_kernel_attributes_keep_reqntid_and_maxntid_distinct():
    operation = KernelOp(
        {
            "nvvm.reqntid": "attr : 8, 4, 2>",
            "nvvm.maxntid": "attr : 16, 4, 2>",
        }
    )
    facts = _launch.launch_facts_from_kernel_op(operation)

    assert facts.exact_block_dim == (8, 4, 2)
    assert facts.max_block_dim == (16, 4, 2)
    assert [origin.fact for origin in facts.provenance] == [
        "exact_block_dim",
        "max_block_dim",
    ]
    assert _launch.block_dim_from_kernel_op(
        operation,
        allow_maxntid=False,
    ) == (8, 4, 2)

    max_only = KernelOp({"nvvm.maxntid": "attr : 256>"})
    max_facts = _launch.launch_facts_from_kernel_op(max_only)
    assert max_facts.exact_block_dim is None
    assert max_facts.max_block_dim == (256, 1, 1)
    assert _launch.block_dim_from_kernel_op(max_only, allow_maxntid=False) is None
    assert _launch.block_dim_from_kernel_op(
        max_only,
        allow_maxntid=True,
    ) == (256, 1, 1)


def test_kernel_fallback_ignores_private_cutlass_launch_fact_attributes():
    operation = KernelOp(
        {
            "cutlass_launch_facts": {
                "exact_block_dim": "array<i64: 64, 1, 1>",
                "exact_grid_dim": "array<i64: 8, 2, 1>",
                "exact_cluster_dim": "array<i64: 2, 1, 1>",
                "cooperative_launch": "true",
                "cluster_launch": "true",
            },
            "nvvm.reqntid": "attr : 64, 1, 1>",
        }
    )

    facts = _launch.launch_facts_from_kernel_op(operation)

    assert facts.exact_block_dim == (64, 1, 1)
    assert facts.exact_grid_dim is None
    assert facts.exact_cluster_dim is None
    assert facts.cooperative_launch is None
    assert facts.cluster_launch is None
    assert {origin.source for origin in facts.provenance} == {"kernel_attribute"}


def test_private_cutlass_launch_facts_api_is_preferred_and_verified():
    facts = _launch.launch_facts_from_cutlass_api(
        SimpleNamespace(
            exact_block_dim=(64, 1, 1),
            exact_grid_dim=(8, 2, 1),
            exact_cluster_dim=(2, 1, 1),
            cooperative_launch=True,
            cluster_launch=False,
        )
    )

    assert facts.exact_block_dim == (64, 1, 1)
    assert facts.exact_grid_dim == (8, 2, 1)
    assert facts.exact_cluster_dim == (2, 1, 1)
    assert facts.cooperative_launch is True
    assert facts.cluster_launch is False
    assert {origin.source for origin in facts.provenance} == {"cutlass_provider_api"}
    assert all(origin.verified for origin in facts.provenance)


def test_private_cutlass_launch_facts_api_rejects_invalid_values():
    with pytest.raises(ValueError, match="exact_block_dim"):
        _launch.launch_facts_from_cutlass_api(
            SimpleNamespace(exact_block_dim=(64, 0, 1))
        )
    with pytest.raises(ValueError, match="cooperative_launch"):
        _launch.launch_facts_from_cutlass_api(SimpleNamespace(cooperative_launch=1))


def test_current_launch_facts_merge_private_api_with_maxntid(monkeypatch):
    cute, ir = _install_fake_cutlass(monkeypatch)

    operation = KernelOp({"nvvm.maxntid": "attr : 128>"})
    monkeypatch.setattr(
        ir,
        "InsertionPoint",
        SimpleNamespace(
            current=SimpleNamespace(block=SimpleNamespace(owner=operation))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cute,
        "_get_launch_facts",
        lambda: SimpleNamespace(exact_grid_dim=(8, 1, 1)),
        raising=False,
    )

    facts = _launch.current_kernel_launch_facts()

    assert facts.exact_grid_dim == (8, 1, 1)
    assert facts.max_block_dim == (128, 1, 1)
    assert {origin.source for origin in facts.provenance} == {
        "cutlass_provider_api",
        "kernel_attribute",
    }


def test_current_launch_facts_reject_private_reqntid_conflict(monkeypatch):
    cute, ir = _install_fake_cutlass(monkeypatch)

    operation = KernelOp({"nvvm.reqntid": "attr : 32>"})
    monkeypatch.setattr(
        ir,
        "InsertionPoint",
        SimpleNamespace(
            current=SimpleNamespace(block=SimpleNamespace(owner=operation))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cute,
        "_get_launch_facts",
        lambda: SimpleNamespace(exact_block_dim=(64, 1, 1)),
        raising=False,
    )

    with pytest.raises(ValueError, match="conflicting exact_block_dim"):
        _launch.current_kernel_launch_facts()


def test_current_launch_facts_falls_back_only_when_private_facts_are_unavailable(
    monkeypatch,
):
    cute, ir = _install_fake_cutlass(monkeypatch)
    operation = KernelOp({"nvvm.reqntid": "attr : 32>"})
    monkeypatch.setattr(
        ir,
        "InsertionPoint",
        SimpleNamespace(
            current=SimpleNamespace(block=SimpleNamespace(owner=operation))
        ),
        raising=False,
    )

    def unavailable():
        raise RuntimeError("launch facts are unavailable on the enclosing kernel")

    monkeypatch.setattr(cute, "_get_launch_facts", unavailable, raising=False)

    facts = _launch.current_kernel_launch_facts()

    assert facts.exact_block_dim == (32, 1, 1)
    assert {origin.source for origin in facts.provenance} == {"kernel_attribute"}


def test_kernel_attributes_reject_conflicting_exact_requirements():
    operation = KernelOp(
        {"nvvm.reqntid": "attr : 32>"},
        KernelOp({"reqntid": "attr : 64>"}),
    )

    with pytest.raises(ValueError, match="conflicting exact_block_dim"):
        _launch.launch_facts_from_kernel_op(operation)

    with pytest.raises(ValueError, match="does not contain a valid"):
        _launch.launch_facts_from_kernel_op(KernelOp({"nvvm.reqntid": "malformed"}))


def test_infer_launch_facts_reconciles_call_and_kernel_sources(monkeypatch):
    monkeypatch.setattr(
        _launch,
        "current_kernel_launch_facts",
        lambda: LaunchFacts(exact_block_dim=(8, 4, 1), max_block_dim=(16, 8, 1)),
    )

    facts = _launch.infer_launch_facts(
        {"launch_metadata": {"block": (8, 4)}},
        scope="cuda.coop.cutlass",
        primitive_name="scan",
    )
    assert facts.exact_block_dim == (8, 4, 1)
    assert facts.max_block_dim == (16, 8, 1)

    with pytest.raises(ValueError, match="conflicting exact_block_dim"):
        _launch.infer_launch_facts(
            {"launch_metadata": {"block": 64}},
            scope="cuda.coop.cutlass",
            primitive_name="scan",
        )
