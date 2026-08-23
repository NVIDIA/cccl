# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import sys

import cuda.coop.numba_mlir as coop

_FOUNDATION_EXPORTS = [
    "Hierarchy",
    "StatefulFunction",
    "TempStorage",
    "ThreadData",
    "ThreadGroup",
    "ThreadHierarchy",
    "gpu_dataclass",
    "gpu_dataclass_argument_handler",
    "local",
    "shared",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]


def test_foundation_exports_only_group_and_storage_building_blocks():
    assert coop.__all__ == _FOUNDATION_EXPORTS
    assert sorted(name for name in dir(coop) if not name.startswith("_")) == (
        _FOUNDATION_EXPORTS
    )

    for name in (
        "load",
        "store",
        "reduce",
        "scan",
        "exchange",
        "radix_sort_keys",
        "_block",
        "_warp",
    ):
        assert not hasattr(coop, name)

    loaded = set(sys.modules)
    assert "cuda.coop.numba_mlir._group_ops" not in loaded
    assert "cuda.coop.numba_mlir._single_phase_rewrites" in loaded
    assert not any(name.startswith("cuda.coop.numba_mlir._block") for name in loaded)
    assert not any(name.startswith("cuda.coop.numba_mlir._warp") for name in loaded)


def test_compiler_hooks_are_registered_exactly_once_and_idempotently():
    group_rewrites = importlib.import_module("cuda.coop.numba_mlir._group_rewrites")
    storage_rewrites = importlib.import_module(
        "cuda.coop.numba_mlir._single_phase_rewrites"
    )
    planner_registry = importlib.import_module(
        "numba_cuda_mlir._whole_function_planners"
    )._planner_registry
    rewrite_registry = importlib.import_module(
        "numba_cuda_mlir.numba_cuda.core.rewrites"
    ).rewrite_registry

    def counts():
        with planner_registry._lock:
            planner_counts = (
                planner_registry._planners.count(
                    group_rewrites.CoopGroupHierarchyPlanner
                ),
                planner_registry._planners.count(
                    storage_rewrites.CoopWholeFunctionPlanner
                ),
            )
        return (
            *planner_counts,
            rewrite_registry.rewrites["before-inference"].count(
                storage_rewrites.CoopSinglePhaseRewrite
            ),
        )

    assert counts() == (1, 1, 1)
    coop._initialize_runtime_hooks()
    coop._initialize_runtime_hooks()
    assert counts() == (1, 1, 1)


def test_lazy_storage_and_stateful_helpers_resolve_from_public_runtime():
    assert coop.local is importlib.import_module("numba_cuda_mlir.cuda").local
    assert coop.shared is importlib.import_module("numba_cuda_mlir.cuda").shared

    def add(lhs, rhs):
        return lhs + rhs

    stateful = coop.StatefulFunction(add, "state_type", name="add")
    assert stateful.op is add
    assert stateful.dtype == "state_type"
    assert stateful.name == "add"
    assert callable(coop.gpu_dataclass)
    assert callable(coop.gpu_dataclass_argument_handler.prepare_args)
