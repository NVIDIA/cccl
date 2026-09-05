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


@pytest.mark.evidence_for(
    "group.adjacent_difference", backend="numba_mlir", evidence="lowering"
)
@pytest.mark.evidence_for(
    "group.discontinuity", backend="numba_mlir", evidence="lowering"
)
def test_common_and_qualified_adjacent_discontinuity_lower_to_same_factories(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    import cuda.coop.numba_mlir._block as scoped_block
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import (
        _GroupCallPlanner,
        has_group_markers,
    )

    def common(value, valid_items, predecessor, successor):
        group = coop.this_block()
        storage = coop.TempStorage()
        items = coop.ThreadData(2, dtype=types.int32)
        items[0] = value
        items[1] = value + 1
        left = coop.adjacent_difference(
            group,
            items,
            valid_items=valid_items,
            tile_predecessor_item=predecessor,
            temp_storage=storage,
        )
        right = coop.adjacent_difference(
            group,
            items,
            direction="right",
            tile_successor_item=successor,
            temp_storage=storage,
        )
        heads = coop.discontinuity(
            group,
            items,
            mode="heads",
            tile_predecessor_item=predecessor,
            temp_storage=storage,
        )
        tails = coop.discontinuity(
            group,
            items,
            mode="tails",
            tile_successor_item=successor,
            temp_storage=storage,
        )
        return items[0], left[0], right[0], heads[0], tails[0]

    def qualified(value, valid_items, predecessor, successor):
        group = numba_coop.this_block()
        storage = numba_coop.TempStorage()
        items = numba_coop.ThreadData(2, dtype=types.int32)
        items[0] = value
        items[1] = value + 1
        left = numba_coop.adjacent_difference(
            group,
            items,
            valid_items=valid_items,
            tile_predecessor_item=predecessor,
            temp_storage=storage,
        )
        right = numba_coop.adjacent_difference(
            group,
            items,
            direction="right",
            tile_successor_item=successor,
            temp_storage=storage,
        )
        heads = numba_coop.discontinuity(
            group,
            items,
            mode="heads",
            tile_predecessor_item=predecessor,
            temp_storage=storage,
        )
        tails = numba_coop.discontinuity(
            group,
            items,
            mode="tails",
            tile_successor_item=successor,
            temp_storage=storage,
        )
        pair = numba_coop.discontinuity(
            group,
            items,
            mode="heads_and_tails",
            tile_predecessor_item=predecessor,
            tile_successor_item=successor,
            temp_storage=storage,
        )
        return (
            items[0],
            left[0],
            right[0],
            heads[0],
            tails[0],
            pair[0][0],
            pair[1][0],
        )

    args = (types.int32, types.int32, types.int32, types.int32)
    launch = {"block": (32, 1, 1), "grid": (1, 1, 1), "cluster": None}
    expected_counts = ((common, 2, 2), (qualified, 2, 3))
    for function, adjacent_count, discontinuity_count in expected_counts:
        func_ir = run_frontend(function)
        assert has_group_markers(func_ir)
        state = SimpleNamespace(func_ir=func_ir, args=args)
        assert _GroupCallPlanner(state, launch).run()
        assert not has_group_markers(func_ir)
        factories = _planned_factories(func_ir, ir)
        assert factories[scoped_block.adjacent_difference] == adjacent_count
        assert factories[scoped_block.discontinuity] == discontinuity_count


@pytest.mark.evidence_for(
    "group.discontinuity", backend="numba_mlir", evidence="lowering"
)
def test_common_discontinuity_rejects_qualified_heads_and_tails_mode(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    def common(value):
        items = coop.ThreadData(2, dtype=types.int32)
        items[0] = value
        return coop.discontinuity(
            coop.this_block(),
            items,
            mode="heads_and_tails",
        )

    common_ir = run_frontend(common)
    common_state = SimpleNamespace(func_ir=common_ir, args=(types.int32,))
    with pytest.raises(
        ValueError,
        match=(
            r"cuda\.coop\.discontinuity mode must be one of: heads, tails; "
            r"use a backend-qualified import"
        ),
    ):
        _GroupCallPlanner(
            common_state,
            {"block": (32, 1, 1), "grid": (1, 1, 1), "cluster": None},
        ).run()

    def qualified(value):
        items = numba_coop.ThreadData(2, dtype=types.int32)
        items[0] = value
        return numba_coop.discontinuity(
            numba_coop.this_block(),
            items,
            mode="heads_and_tails",
        )

    qualified_ir = run_frontend(qualified)
    qualified_state = SimpleNamespace(func_ir=qualified_ir, args=(types.int32,))
    assert _GroupCallPlanner(
        qualified_state,
        {"block": (32, 1, 1), "grid": (1, 1, 1), "cluster": None},
    ).run()


@pytest.mark.parametrize(
    "operation",
    ["adjacent_difference", "discontinuity"],
)
def test_common_comparison_rejects_scalar_but_qualified_scalar_remains_available(
    optional_backend,
    operation,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import (
        _GroupCallPlanner,
        has_group_markers,
    )

    if operation == "adjacent_difference":

        def common(value):
            return coop.adjacent_difference(coop.this_block(), value)

        def qualified(value):
            return numba_coop.adjacent_difference(
                numba_coop.this_block(),
                value,
            )

    else:

        def common(value):
            return coop.discontinuity(coop.this_block(), value)

        def qualified(value):
            return numba_coop.discontinuity(numba_coop.this_block(), value)

    launch = {"block": (32, 1, 1), "grid": (1, 1, 1), "cluster": None}
    common_ir = run_frontend(common)
    common_state = SimpleNamespace(func_ir=common_ir, args=(types.int32,))
    with pytest.raises(
        TypeError,
        match=(
            rf"cuda\.coop\.{operation} requires a fixed-size ThreadData payload "
            r"in common V1; use cuda\.coop\.numba_mlir for backend-qualified "
            r"scalar or local-array support"
        ),
    ):
        _GroupCallPlanner(common_state, launch).run()

    qualified_ir = run_frontend(qualified)
    qualified_state = SimpleNamespace(func_ir=qualified_ir, args=(types.int32,))
    assert has_group_markers(qualified_ir)
    assert _GroupCallPlanner(qualified_state, launch).run()
    assert not has_group_markers(qualified_ir)


@pytest.mark.parametrize(
    "operation",
    ["adjacent_difference", "discontinuity"],
)
def test_common_comparison_requires_thread_data_but_qualified_keeps_local_arrays(
    optional_backend,
    operation,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import cuda, types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    common_operation = getattr(coop, operation)
    qualified_operation = getattr(numba_coop, operation)

    def common(value):
        items = cuda.local.array(2, types.int32)
        items[0] = value
        items[1] = value + 1
        return common_operation(coop.this_block(), items)

    launch = {"block": (32, 1, 1), "grid": (1, 1, 1), "cluster": None}
    common_ir = run_frontend(common)
    common_state = SimpleNamespace(func_ir=common_ir, args=(types.int32,))
    with pytest.raises(
        TypeError,
        match=(
            rf"cuda\.coop\.{operation} requires a fixed-size ThreadData payload "
            r"in common V1"
        ),
    ):
        _GroupCallPlanner(common_state, launch).run()

    def qualified(value):
        items = cuda.local.array(2, types.int32)
        items[0] = value
        items[1] = value + 1
        return qualified_operation(numba_coop.this_block(), items)

    qualified_ir = run_frontend(qualified)
    qualified_state = SimpleNamespace(func_ir=qualified_ir, args=(types.int32,))
    assert _GroupCallPlanner(qualified_state, launch).run()
