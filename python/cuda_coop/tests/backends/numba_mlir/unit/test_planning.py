# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types
from numba_cuda_mlir.numba_cuda.compiler import run_frontend
from numba_cuda_mlir.numba_cuda.core.ir_utils import build_definitions
from numba_cuda_mlir.numbair_transforms import ir

import cuda.coop.numba_mlir as coop
from cuda.coop.numba_mlir._compiler import _group_reduce
from cuda.coop.numba_mlir._compiler._group_planner import (
    CoopGroupHierarchyPlanner,
    GroupRewriteError,
    _GroupCallPlanner,
)
from cuda.coop.numba_mlir._compiler._operations import factory_operation
from cuda.coop.numba_mlir._compiler._rewrite import CoopSinglePhaseRewrite
from cuda.coop.numba_mlir._compiler._rewrite_support import _factory_from_call
from cuda.coop.numba_mlir._lowering import _reduce


def _kernel(source):
    return coop.sum(
        coop.this_block(),
        source[0],
        algorithm="raking",
    )


def _kernel_with_optional_valid_items(source, valid_items=None):
    return coop.sum(
        coop.this_block(),
        source[0],
        valid_items=valid_items,
    )


def _kernel_with_static_valid_items(source):
    return coop.sum(
        coop.this_block(),
        source[0],
        valid_items=17,
    )


def _kernel_with_invalid_static_valid_items(source):
    return coop.sum(
        coop.this_block(),
        source[0],
        valid_items=33,
    )


def _kernel_with_runtime_binary_op(source, binary_op):
    return coop.reduce(
        coop.this_block(),
        source[0],
        binary_op=binary_op,
    )


def _kernel_with_invalid_group(source):
    return coop.sum(source[0], source[0])


def _state(function, *args):
    return SimpleNamespace(
        func_ir=run_frontend(function),
        args=args,
        typemap={},
        calltypes={},
        metadata={"targetoptions": {}},
    )


def test_hierarchy_planning_selects_factory_before_dtype_materialization(
    monkeypatch,
):
    func_ir = run_frontend(_kernel)
    refreshed = []
    state = SimpleNamespace(
        func_ir=func_ir,
        args=(types.Array(types.int32, 1, "C"),),
        typemap={},
        calltypes={},
        typingctx=SimpleNamespace(refresh=lambda: refreshed.append(True)),
        metadata={"targetoptions": {}},
    )
    monkeypatch.setattr(
        _reduce._nvrtc,
        "resolve_compile_context",
        lambda state: pytest.fail("hierarchy planning must not compile a provider"),
    )

    assert _GroupCallPlanner(state, {"grid": 1, "block": (8, 4)}).run()
    func_ir._definitions = build_definitions(func_ir.blocks)

    selected = []
    for block in func_ir.blocks.values():
        for inst in block.body:
            if not isinstance(inst, ir.Assign):
                continue
            call = inst.value
            if not isinstance(call, ir.Expr) or call.op != "call":
                continue
            if (target := _factory_from_call(func_ir, call)) is not None:
                selected.append(target)

    assert len(selected) == 1
    factory, metadata = selected[0]
    assert factory is _reduce.sum
    assert factory_operation(factory) is metadata
    assert metadata.operation == "sum"

    materialized = []

    def materialize(**kwargs):
        materialized.append(kwargs)

        def marker(value):
            return value

        return marker

    monkeypatch.setattr(_reduce, "_materialize", materialize)
    rewrite = CoopSinglePhaseRewrite(state)
    block = next(iter(func_ir.blocks.values()))

    assert rewrite.match(func_ir, block, state.typemap, state.calltypes)
    func_ir.blocks[next(iter(func_ir.blocks))] = rewrite.apply()

    assert materialized == [
        {
            "threads_per_block": (8, 4, 1),
            "operation": "sum",
            "binary_op": "sum",
            "algorithm": "raking",
            "num_valid": False,
            "state": state,
        }
    ]
    assert refreshed == [True]


@pytest.mark.parametrize("none_type", [types.none, types.Omitted(None)])
def test_group_planner_treats_post_inline_optional_none_as_static(none_type):
    func_ir = run_frontend(_kernel_with_optional_valid_items)
    state = SimpleNamespace(
        func_ir=func_ir,
        args=(types.Array(types.int32, 1, "C"), none_type),
        typemap={},
        calltypes={},
        metadata={"targetoptions": {}},
    )

    assert _GroupCallPlanner(state, {"grid": 1, "block": 32}).run()
    func_ir._definitions = build_definitions(func_ir.blocks)

    factory_calls = []
    for block in func_ir.blocks.values():
        for inst in block.body:
            if not isinstance(inst, ir.Assign):
                continue
            call = inst.value
            if not isinstance(call, ir.Expr) or call.op != "call":
                continue
            if _factory_from_call(func_ir, call) is not None:
                factory_calls.append(call)

    assert len(factory_calls) == 1
    assert "num_valid" not in dict(factory_calls[0].kws)


def test_group_planner_routes_static_controls_through_portable_contract(
    monkeypatch,
):
    state = _state(
        _kernel_with_static_valid_items,
        types.Array(types.int32, 1, "C"),
    )
    portable_plans = []
    original = _group_reduce.plan_group_primitive

    def plan(call, launch):
        result = original(call, launch)
        portable_plans.append(result)
        return result

    monkeypatch.setattr(_group_reduce, "plan_group_primitive", plan)

    assert _GroupCallPlanner(state, {"grid": 1, "block": 32}).run()

    assert len(portable_plans) == 1
    portable = portable_plans[0].require_supported()
    assert portable.implementation is not None
    assert portable.implementation.block_dim == (32, 1, 1)
    assert portable.implementation.valid_items
    assert portable.call.operation.valid_items.value == 17


def test_portable_contract_rejects_out_of_range_static_valid_items():
    state = _state(
        _kernel_with_invalid_static_valid_items,
        types.Array(types.int32, 1, "C"),
    )

    with pytest.raises(ValueError, match="valid_items must be at most 32"):
        _GroupCallPlanner(state, {"grid": 1, "block": 32}).run()


def test_group_planner_rejects_runtime_binary_operator():
    state = _state(
        _kernel_with_runtime_binary_op,
        types.Array(types.int32, 1, "C"),
        types.unicode_type,
    )

    with pytest.raises(GroupRewriteError, match="binary_op must be a compile-time"):
        _GroupCallPlanner(state, {"grid": 1, "block": 32}).run()


def test_group_planner_rejects_non_descriptor_group():
    state = _state(
        _kernel_with_invalid_group,
        types.Array(types.int32, 1, "C"),
    )

    with pytest.raises(GroupRewriteError, match="group must come from this_block"):
        _GroupCallPlanner(state, {"grid": 1, "block": 32}).run()


def test_group_planner_requires_configured_launch_metadata():
    state = _state(
        _kernel,
        types.Array(types.int32, 1, "C"),
    )

    with pytest.raises(
        RuntimeError,
        match="requires metadata from a configured kernel launch",
    ):
        CoopGroupHierarchyPlanner(state).run()
