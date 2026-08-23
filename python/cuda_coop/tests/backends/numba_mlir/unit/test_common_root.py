# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest


def _frontend_calls(func_ir, ir):
    return [
        inst.value
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "call"
    ]


def test_common_root_group_calls_are_detected_without_compiler_scope(optional_backend):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda import coop
    from cuda.coop._core import root_api
    from cuda.coop.numba_mlir._group_rewrites import has_group_markers

    def common_root_reduce(value):
        return coop.reduce(coop.this_block(), value)

    assert root_api._backend_module_name() is None
    assert has_group_markers(run_frontend(common_root_reduce))
    assert root_api._backend_module_name() is None


def test_common_root_storage_markers_are_detected_without_compiler_scope(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    from cuda import coop
    from cuda.coop._core import root_api
    from cuda.coop.numba_mlir import _single_phase_rewrites as rewrites

    def common_root_storage():
        payload = coop.ThreadData(2)
        scratch = coop.TempStorage()
        return payload, scratch

    func_ir = run_frontend(common_root_storage)
    rewrite = object.__new__(rewrites.CoopSinglePhaseRewrite)
    rewrite._func_ir = func_ir
    rewrite._block_defs = {}
    calls = _frontend_calls(func_ir, ir)

    assert root_api._backend_module_name() is None
    assert sum(rewrite._is_thread_data_ctor_call(call) for call in calls) == 1
    assert sum(rewrite._is_temp_storage_ctor_call(call) for call in calls) == 1
    assert root_api._backend_module_name() is None


def test_common_root_identity_rewrite_supports_logical_warp_merge_sort(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda import coop
    from cuda.coop.numba_mlir import _group_ops
    from cuda.coop.numba_mlir._group_rewrites import (
        _GroupCallPlanner,
        _is_common_root_operation,
        has_group_markers,
    )

    def common_logical_merge_sort(value):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        keys[1] = value - 1
        group = coop.this_warp().group_by(8)
        return coop.merge_sort_keys(group, keys)

    assert _is_common_root_operation(coop.merge_sort_keys, "merge_sort_keys")
    assert not _is_common_root_operation(
        _group_ops.merge_sort_keys,
        "merge_sort_keys",
    )
    state = SimpleNamespace(
        func_ir=run_frontend(common_logical_merge_sort),
        args=(types.int32,),
    )
    planner = _GroupCallPlanner(
        state,
        {
            "block": (32, 1, 1),
            "grid": (1, 1, 1),
            "cluster": None,
        },
    )
    assert has_group_markers(state.func_ir)
    assert planner.run()
    assert not has_group_markers(state.func_ir)
