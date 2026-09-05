# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest


def test_group_marker_detection_distinguishes_group_ir(optional_backend):
    coop = optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._group_rewrites import has_group_markers

    def group_marker_function(value):
        group = coop.this_block()
        return coop.reduce(group, value)

    def plain_function(value):
        return value + 1

    assert has_group_markers(run_frontend(group_marker_function))
    assert not has_group_markers(run_frontend(plain_function))


# Regression: https://github.com/NVIDIA/numba-cuda-mlir/pull/239
@pytest.mark.parametrize("lanes_per_group", [8, 16])
def test_scalar_group_by_requests_and_consumes_literal(
    optional_backend, lanes_per_group
):
    coop = optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir.errors import ForceLiteralArg
    from numba_cuda_mlir.numba_cuda import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    def scalar_group_by(_values, _sums, width, exhaustive):
        group = coop.this_warp().group_by(width, exhaustive=exhaustive)
        return group.count()

    def make_planner(width_type, exhaustive_type, value_type=types.int32):
        func_ir = run_frontend(scalar_group_by)
        state = SimpleNamespace(
            func_ir=func_ir,
            args=(value_type, types.int32, width_type, exhaustive_type),
        )
        planner = _GroupCallPlanner(
            state,
            {"block": (32, 1, 1), "grid": (1, 1, 1), "cluster": None},
        )
        group_var = next(
            inst.target
            for block in func_ir.blocks.values()
            for inst in block.body
            if isinstance(inst, ir.Assign) and inst.target.name == "group"
        )
        runtime_var = next(
            inst.target
            for block in func_ir.blocks.values()
            for inst in block.body
            if isinstance(inst, ir.Assign)
            and isinstance(inst.value, ir.Arg)
            and inst.value.index == 0
        )
        return planner, group_var, runtime_var

    planner, group_var, runtime_var = make_planner(types.int64, types.boolean)
    assert planner._try_constant(runtime_var) == (False, None)
    assert not planner._is_none(runtime_var)
    with pytest.raises(ForceLiteralArg) as exc_info:
        planner._group(group_var)
    assert exc_info.value.requested_args == frozenset({2})

    planner, group_var, _ = make_planner(
        types.IntegerLiteral(lanes_per_group),
        types.boolean,
    )
    with pytest.raises(ForceLiteralArg) as exc_info:
        planner._group(group_var)
    assert exc_info.value.requested_args == frozenset({3})

    planner, group_var, _ = make_planner(
        types.IntegerLiteral(lanes_per_group),
        types.literal(True),
    )
    group = planner._group(group_var)
    assert group is not None
    assert group.mapping is not None
    assert group.mapping.count == lanes_per_group
    assert planner.run()

    planner, _, none_var = make_planner(
        types.IntegerLiteral(lanes_per_group),
        types.literal(True),
        value_type=types.none,
    )
    assert planner._try_constant(none_var) == (True, None)
    assert planner._is_none(none_var)


def test_mapped_warps_reduce_fails_during_whole_function_planning(
    optional_backend,
):
    coop = optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    def mapped_warps_sum(value):
        group = coop.this_block().group_by(1)
        return coop.sum(group, value)

    state = SimpleNamespace(func_ir=run_frontend(mapped_warps_sum))
    planner = _GroupCallPlanner(
        state,
        {
            "block": (64, 1, 1),
            "grid": (1, 1, 1),
            "cluster": None,
        },
    )
    expected = (
        "cuda.coop.numba_mlir reduce/sum does not support "
        "warps_within_block groups because the current CUDAX mapping does not "
        "preserve independent mapped-group reduction semantics"
    )

    with pytest.raises(NotImplementedError) as error_info:
        planner.run()

    assert str(error_info.value) == expected


@pytest.mark.parametrize("group_kind", ["thread", "warp", "block", "grid"])
@pytest.mark.parametrize("operation", ["rank", "count"])
@pytest.mark.parametrize(("block_threads", "is_supported"), [(48, False), (64, True)])
def test_warp_queries_require_complete_physical_warps_during_planning(
    optional_backend,
    group_kind,
    operation,
    block_threads,
    is_supported,
):
    coop = optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    if group_kind == "thread" and operation == "rank":

        def warp_query():
            return coop.this_thread().rank("warp")

    elif group_kind == "thread":

        def warp_query():
            return coop.this_thread().count("warp")

    elif group_kind == "warp" and operation == "rank":

        def warp_query():
            return coop.this_warp().rank("warp")

    elif group_kind == "warp":

        def warp_query():
            return coop.this_warp().count("warp")

    elif group_kind == "block" and operation == "rank":

        def warp_query():
            return coop.this_block().rank("warp")

    elif group_kind == "block":

        def warp_query():
            return coop.this_block().count("warp")

    elif group_kind == "grid" and operation == "rank":

        def warp_query():
            return coop.this_grid().rank("warp")

    elif group_kind == "grid" and operation == "count":

        def warp_query():
            return coop.this_grid().count("warp")

    else:
        raise AssertionError(f"unhandled warp query: {group_kind}.{operation}")

    state = SimpleNamespace(func_ir=run_frontend(warp_query))
    planner = _GroupCallPlanner(
        state,
        {
            "block": (block_threads, 1, 1),
            "grid": (1, 1, 1),
            "cluster": None,
        },
    )

    if is_supported:
        assert planner.run()
    else:
        with pytest.raises(NotImplementedError, match="complete 32-thread warps"):
            planner.run()
