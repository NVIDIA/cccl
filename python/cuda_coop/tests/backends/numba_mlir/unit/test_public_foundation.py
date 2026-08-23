# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import sys

import cuda.coop.numba_mlir as coop
from cuda.coop.numba_mlir import (
    _stateful_function,
    _temp_storage,
    _thread_data,
    _types,
)
from cuda.coop.numba_mlir._compiler import _activation

_PUBLIC_EXPORTS = [
    "BlockLoadAlgorithm",
    "BlockScanAlgorithm",
    "BlockStoreAlgorithm",
    "Hierarchy",
    "StatefulFunction",
    "TempStorage",
    "ThreadData",
    "ThreadGroup",
    "ThreadHierarchy",
    "WarpLoadAlgorithm",
    "WarpStoreAlgorithm",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "gpu_dataclass",
    "gpu_dataclass_argument_handler",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "local",
    "reduce",
    "scan",
    "shared",
    "shuffle",
    "store",
    "sum",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]


def test_public_exports_cover_the_incremental_primitive_families():
    assert coop.__all__ == _PUBLIC_EXPORTS
    assert sorted(name for name in dir(coop) if not name.startswith("_")) == (
        _PUBLIC_EXPORTS
    )

    assert "_block" not in coop.__all__
    assert "_warp" not in coop.__all__

    loaded = set(sys.modules)
    assert "cuda.coop.numba_mlir._group_load_store" in loaded
    assert "cuda.coop.numba_mlir._group_reduce" in loaded
    assert "cuda.coop.numba_mlir._group_scan" in loaded
    assert "cuda.coop.numba_mlir._compiler._rewrite" in loaded
    assert importlib.import_module("cuda.coop.numba_mlir._lowering").__all__ == ()


def test_group_markers_use_exact_callable_identity():
    from cuda.coop.numba_mlir._compiler._operations import group_operation_name

    assert group_operation_name(coop.reduce) == "reduce"

    def reduce(*args, **kwargs):
        del args, kwargs

    reduce.__module__ = coop.reduce.__module__
    reduce.__name__ = coop.reduce.__name__
    reduce.__cuda_coop_backend_member__ = "reduce"
    assert group_operation_name(reduce) is None


def test_lowering_factories_use_exact_callable_identity():
    from cuda.coop.numba_mlir import _lowering
    from cuda.coop.numba_mlir._compiler._operations import factory_operation

    assert factory_operation(_lowering.scan) is not None

    def scan(*args, **kwargs):
        del args, kwargs

    scan.__module__ = _lowering.scan.__module__
    scan.__name__ = _lowering.scan.__name__
    assert factory_operation(scan) is None


def test_compiler_hooks_are_registered_exactly_once_and_idempotently():
    group_rewrites = importlib.import_module(
        "cuda.coop.numba_mlir._compiler._group_planner"
    )
    storage_rewrites = importlib.import_module(
        "cuda.coop.numba_mlir._compiler._rewrite"
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
    _activation._initialize_runtime_hooks()
    _activation._initialize_runtime_hooks()
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


def test_public_runtime_helpers_have_semantic_module_owners():
    assert coop.ThreadData is _thread_data.ThreadData
    assert coop.TempStorage is _temp_storage.TempStorage
    assert coop.StatefulFunction is _stateful_function.StatefulFunction
    assert _types.StatefulFunction is _stateful_function.StatefulFunction
    assert coop.ThreadData.__module__ == "cuda.coop.numba_mlir._thread_data"
    assert coop.TempStorage.__module__ == "cuda.coop.numba_mlir._temp_storage"
    assert coop.StatefulFunction.__module__ == "cuda.coop.numba_mlir._stateful_function"
