# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest
from numba_cuda_mlir.errors import ForceLiteralArg
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.compiler import run_frontend
from numba_cuda_mlir.numbair_transforms import ir

import cuda.coop.numba_mlir as coop
from cuda.coop._core import LaunchFacts, resolve_thread_group
from cuda.coop.numba_mlir._compiler import _nvrtc
from cuda.coop.numba_mlir._compiler._group_planner import (
    GroupRewriteError,
    _GroupCallPlanner,
    has_group_markers,
)
from cuda.coop.numba_mlir._lowering import _thread_group as _thread_group_lowering


def _state(function, args=()):
    return SimpleNamespace(func_ir=run_frontend(function), args=args)


def _launch(block=(64, 1, 1), grid=(1, 1, 1), cluster=None):
    return {"block": block, "grid": grid, "cluster": cluster}


def test_group_marker_detection_distinguishes_group_ir():
    def group_marker_function():
        return coop.this_block().count()

    def plain_function(value):
        return value + 1

    assert has_group_markers(run_frontend(group_marker_function))
    assert not has_group_markers(run_frontend(plain_function))


@pytest.mark.parametrize("lanes_per_group", [8, 16])
def test_scalar_group_by_requests_and_consumes_literals(lanes_per_group):
    def scalar_group_by(width, exhaustive):
        group = coop.this_warp().group_by(width, exhaustive=exhaustive)
        return group.count()

    def make_planner(width_type, exhaustive_type):
        state = _state(scalar_group_by, (width_type, exhaustive_type))
        planner = _GroupCallPlanner(state, _launch(block=(32, 1, 1)))
        group_var = next(
            inst.target
            for block in state.func_ir.blocks.values()
            for inst in block.body
            if isinstance(inst, ir.Assign) and inst.target.name == "group"
        )
        return planner, group_var

    planner, group_var = make_planner(types.int64, types.boolean)
    with pytest.raises(ForceLiteralArg) as exc_info:
        planner._group(group_var)
    assert exc_info.value.requested_args == frozenset({0})

    planner, group_var = make_planner(
        types.IntegerLiteral(lanes_per_group),
        types.boolean,
    )
    with pytest.raises(ForceLiteralArg) as exc_info:
        planner._group(group_var)
    assert exc_info.value.requested_args == frozenset({1})

    planner, group_var = make_planner(
        types.IntegerLiteral(lanes_per_group),
        types.literal(True),
    )
    group = planner._group(group_var)
    assert group is not None
    assert group.mapping is not None
    assert group.mapping.count == lanes_per_group


def test_planner_shares_compile_context_across_group_methods(monkeypatch):
    captured = []
    context_resolutions = []
    context = _nvrtc.CompileContext(
        nvrtc_path="/toolkit/lib/libnvrtc.so.13",
        nvrtc_version=_nvrtc.version(13, 2),
        include_dirs=("/toolkit/include",),
        header_identity="headers-a",
    )

    def query():
        group = coop.this_block()
        return group.count("thread") + group.rank("thread")

    class _Invocable:
        def __call__(self):
            raise AssertionError("compile-time invocable executed by Python")

    def make_invocable(**kwargs):
        captured.append(kwargs)
        return _Invocable()

    monkeypatch.setattr(
        _thread_group_lowering,
        "make_group_method_invocable",
        make_invocable,
    )
    monkeypatch.setattr(
        _nvrtc,
        "resolve_compile_context",
        lambda: context_resolutions.append(True) or context,
    )
    state = _state(query)
    planner = _GroupCallPlanner(state, _launch())

    assert planner.run()
    assert len(captured) == 2
    assert {item["operation"] for item in captured} == {"count", "rank"}
    assert all(item["level"] == "thread" for item in captured)
    assert all(item["group"].kind == "block" for item in captured)
    assert all(item["group"].thread_count == 64 for item in captured)
    assert all(item["compile_context"] is context for item in captured)
    assert context_resolutions == [True]


def test_mapped_parent_queries_render_the_parent_group():
    mapped = resolve_thread_group(
        coop.this_warp().group_by(8),
        LaunchFacts(exact_block_dim=64),
        through_level="warp",
    ).require_supported()

    assert _thread_group_lowering._query_expr(mapped, "rank", "warp") == (
        "group.rank(group_parent)"
    )
    assert _thread_group_lowering._query_expr(mapped, "count", "warp") == (
        "group.count(group_parent)"
    )
    assert "#define _CUDAX_DISABLE_COOPERATIVE_GROUPS_INTEROP" not in (
        _thread_group_lowering._INCLUDE_LINES
    )


@pytest.mark.parametrize("kind", ("thread", "warp", "block", "cluster", "grid"))
def test_current_physical_group_rendering_uses_implicit_hierarchy(kind):
    group = getattr(coop, f"this_{kind}")()

    assert _thread_group_lowering._group_prelude(group) == [
        f"  ::cuda::experimental::this_{kind} "
        "group{::cuda::experimental::implicit_hierarchy()};"
    ]


def test_descriptor_values_cannot_escape_to_runtime():
    def escapes():
        return coop.this_block()

    with pytest.raises(GroupRewriteError, match="would escape to runtime"):
        _GroupCallPlanner(_state(escapes), _launch()).run()


@pytest.mark.parametrize("operation", ["sync", "sync_aligned"])
def test_grid_synchronization_fails_during_planning(operation):
    if operation == "sync":

        def grid_sync():
            coop.this_grid().sync()

    else:

        def grid_sync():
            coop.this_grid().sync_aligned()

    with pytest.raises(NotImplementedError, match="cooperative launch"):
        _GroupCallPlanner(_state(grid_sync), _launch()).run()
