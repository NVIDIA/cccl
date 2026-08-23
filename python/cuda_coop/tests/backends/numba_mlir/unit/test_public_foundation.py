# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import sys

import cuda.coop.numba_mlir as coop

_PUBLIC_EXPORTS = [
    "BlockHistogramAlgorithm",
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
    "adjacent_difference",
    "discontinuity",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "gpu_dataclass",
    "gpu_dataclass_argument_handler",
    "histogram",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "local",
    "merge_sort_keys",
    "merge_sort_pairs",
    "radix_rank",
    "radix_sort_keys",
    "radix_sort_pairs",
    "reduce",
    "run_length_decode",
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
    "topk_max_keys",
    "topk_max_pairs",
    "topk_min_keys",
    "topk_min_pairs",
]


def test_public_exports_cover_the_incremental_primitive_families():
    assert coop.__all__ == _PUBLIC_EXPORTS
    assert sorted(name for name in dir(coop) if not name.startswith("_")) == (
        _PUBLIC_EXPORTS
    )

    assert "_block" not in coop.__all__
    assert "_warp" not in coop.__all__

    loaded = set(sys.modules)
    assert "cuda.coop.numba_mlir._group_ops" in loaded
    assert "cuda.coop.numba_mlir._single_phase_rewrites" in loaded
    assert importlib.import_module("cuda.coop.numba_mlir._block").__all__ == ()
    assert importlib.import_module("cuda.coop.numba_mlir._warp").__all__ == ()


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
