# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter
from types import SimpleNamespace

import pytest


def _planned_factories(func_ir, ir):
    globals_by_name = {
        inst.target.name: inst.value.value
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Global)
    }
    return Counter(
        globals_by_name.get(inst.value.func.name)
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "call"
    )


def _plan(function, *, arg_types):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    func_ir = run_frontend(function)
    state = SimpleNamespace(func_ir=func_ir, args=arg_types)
    planner = _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    )
    return func_ir, planner


@pytest.mark.evidence_for("group.shuffle", backend="numba_mlir", evidence="lowering")
def test_common_and_qualified_shuffle_lower_to_the_same_array_factories(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    import cuda.coop.numba_mlir._block as scoped_block
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import has_group_markers

    def common(value):
        items = coop.ThreadData(3, dtype=types.int32)
        for index in range(3):
            items[index] = value + index
        group = coop.this_block()
        down = coop.shuffle(group, items)
        up = coop.shuffle(group, items, mode="up", distance=1)
        return items[0], down[0], up[1]

    def qualified(value):
        items = numba_coop.ThreadData(3, dtype=types.int32)
        for index in range(3):
            items[index] = value + index
        group = numba_coop.this_block()
        down = numba_coop.shuffle(group, items)
        up = numba_coop.shuffle(group, items, mode="up", distance=1)
        return items[0], down[0], up[1]

    for function in (common, qualified):
        func_ir, planner = _plan(function, arg_types=(types.int32,))
        assert has_group_markers(func_ir)
        assert planner.run()
        assert not has_group_markers(func_ir)
        assert _planned_factories(func_ir, ir)[scoped_block.shuffle] == 2


@pytest.mark.evidence_for("group.shuffle", backend="numba_mlir", evidence="lowering")
def test_common_shuffle_rejects_scalar_but_qualified_scalar_remains_available(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    def common(value):
        return coop.shuffle(coop.this_block(), value)

    _, common_planner = _plan(common, arg_types=(types.int32,))
    with pytest.raises(
        TypeError,
        match=r"cuda\.coop\.shuffle requires a fixed-size ThreadData",
    ):
        common_planner.run()

    def qualified(value):
        return numba_coop.shuffle(
            numba_coop.this_block(),
            value,
            mode="offset",
            distance=2,
        )

    _, qualified_planner = _plan(qualified, arg_types=(types.int32,))
    assert qualified_planner.run()


@pytest.mark.evidence_for("group.shuffle", backend="numba_mlir", evidence="lowering")
def test_common_shuffle_requires_thread_data_but_qualified_keeps_local_arrays(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    def common(value):
        items = cuda.local.array(2, types.int32)
        items[0] = value
        items[1] = value + 1
        return coop.shuffle(coop.this_block(), items)

    _, common_planner = _plan(common, arg_types=(types.int32,))
    with pytest.raises(
        TypeError,
        match=(
            r"cuda\.coop\.shuffle requires a fixed-size ThreadData payload "
            r"in common V1"
        ),
    ):
        common_planner.run()

    def qualified(value):
        items = cuda.local.array(2, types.int32)
        items[0] = value
        items[1] = value + 1
        return numba_coop.shuffle(numba_coop.this_block(), items)

    _, qualified_planner = _plan(qualified, arg_types=(types.int32,))
    assert qualified_planner.run()


@pytest.mark.evidence_for("group.shuffle", backend="numba_mlir", evidence="lowering")
@pytest.mark.parametrize(
    ("mode", "distance", "message"),
    [
        ("offset", 1, r"mode must be one of: down, up"),
        ("rotate", 1, r"mode must be one of: down, up"),
        ("down", 2, r"distance must be exactly 1 in common V1"),
        ("up", 0, r"distance must be exactly 1 in common V1"),
    ],
)
def test_common_shuffle_rejects_nonportable_modes_and_distances(
    optional_backend,
    mode,
    distance,
    message,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda import coop

    def common(value):
        items = coop.ThreadData(2, dtype=types.int32)
        items[0] = value
        items[1] = value + 1
        return coop.shuffle(
            coop.this_block(),
            items,
            mode=mode,
            distance=distance,
        )

    _, planner = _plan(common, arg_types=(types.int32,))
    with pytest.raises(ValueError, match=message):
        planner.run()
