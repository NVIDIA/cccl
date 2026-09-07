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


@pytest.mark.evidence_for("group.exchange", backend="numba_mlir", evidence="lowering")
@pytest.mark.parametrize("group_kind", ["block", "warp", "logical_warp"])
def test_common_root_exchange_lowers_both_modes_out_of_place(
    optional_backend,
    group_kind,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir._block as scoped_block
    import cuda.coop.numba_mlir._warp as scoped_warp
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import (
        _GroupCallPlanner,
        has_group_markers,
    )

    if group_kind == "block":
        group = coop.this_block()
    elif group_kind == "warp":
        group = coop.this_warp()
    else:
        group = coop.this_warp().group_by(8)

    def cohort(value):
        items = coop.ThreadData(5, dtype=types.int32)
        for index in range(5):
            items[index] = value + index
        blocked = coop.exchange(group, items)
        striped = coop.exchange(group, blocked, mode="blocked_to_striped")
        return items[0], blocked[0], striped[0]

    func_ir = run_frontend(cohort)
    assert has_group_markers(func_ir)
    state = SimpleNamespace(func_ir=func_ir, args=(types.int32,))
    assert _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    ).run()
    assert not has_group_markers(func_ir)

    expected_factory = (
        scoped_block.exchange if group_kind == "block" else scoped_warp.exchange
    )
    assert _planned_factories(func_ir, ir)[expected_factory] == 2


@pytest.mark.parametrize(
    ("group_kind", "mode", "time_slicing", "expected_args"),
    [
        (
            "block",
            "scatter_to_striped_flagged",
            True,
            4,
        ),
        ("logical_warp", "scatter_to_striped", False, 3),
    ],
)
def test_qualified_advanced_exchange_lowers_runtime_side_inputs(
    optional_backend,
    group_kind,
    mode,
    time_slicing,
    expected_args,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop.numba_mlir._group_rewrites import (
        _GroupCallPlanner,
        has_group_markers,
    )

    if group_kind == "block":

        def cohort(value):
            items = numba_coop.ThreadData(2, dtype=types.int32)
            ranks = numba_coop.ThreadData(2, dtype=types.int32)
            flags = numba_coop.ThreadData(2, dtype=types.uint8)
            items[0] = value
            ranks[0] = 0
            flags[0] = 1
            return numba_coop.exchange(
                numba_coop.this_block(),
                items,
                mode=mode,
                ranks=ranks,
                valid_flags=flags,
                warp_time_slicing=time_slicing,
            )

    else:

        def cohort(value):
            items = numba_coop.ThreadData(2, dtype=types.int32)
            ranks = numba_coop.ThreadData(2, dtype=types.int32)
            items[0] = value
            ranks[0] = 0
            return numba_coop.exchange(
                numba_coop.this_warp().group_by(8),
                items,
                mode=mode,
                ranks=ranks,
            )

    func_ir = run_frontend(cohort)
    assert has_group_markers(func_ir)
    state = SimpleNamespace(func_ir=func_ir, args=(types.int32,))
    assert _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    ).run()
    assert not has_group_markers(func_ir)

    expected_factory = (
        numba_coop._block.exchange
        if group_kind == "block"
        else numba_coop._warp.exchange
    )
    globals_by_name = {
        inst.target.name: inst.value.value
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Global)
    }
    calls = [
        inst.value
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "call"
        and globals_by_name.get(inst.value.func.name) is expected_factory
    ]
    assert len(calls) == 1
    assert len(calls[0].args) == expected_args
    keyword_names = tuple(name for name, _ in calls[0].kws)
    if group_kind == "block":
        assert keyword_names == (
            "threads_per_block",
            "block_exchange_type",
            "warp_time_slicing",
        )
    else:
        assert keyword_names == (
            "threads_in_warp",
            "threads_per_block",
            "warp_exchange_type",
        )
        constants_by_name = {
            inst.target.name: inst.value.value
            for block_ir in func_ir.blocks.values()
            for inst in block_ir.body
            if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Const)
        }
        width_var = dict(calls[0].kws)["threads_in_warp"]
        assert constants_by_name[width_var.name] == 8


@pytest.mark.evidence_for("group.exchange", backend="numba_mlir", evidence="lowering")
@pytest.mark.parametrize("group_kind", ["block", "warp"])
def test_common_root_exchange_rejects_six_items_but_qualified_api_retains_them(
    optional_backend,
    group_kind,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    common_constructor = coop.this_block if group_kind == "block" else coop.this_warp
    qualified_constructor = (
        numba_coop.this_block if group_kind == "block" else numba_coop.this_warp
    )

    def common(value):
        items = coop.ThreadData(6, dtype=types.int32)
        items[0] = value
        return coop.exchange(common_constructor(), items)

    common_ir = run_frontend(common)
    common_state = SimpleNamespace(func_ir=common_ir, args=(types.int32,))
    with pytest.raises(
        NotImplementedError,
        match=r"cuda\.coop\.exchange supports at most 5 items per thread",
    ):
        _GroupCallPlanner(
            common_state,
            {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
        ).run()

    def qualified(value):
        items = numba_coop.ThreadData(6, dtype=types.int32)
        items[0] = value
        return numba_coop.exchange(qualified_constructor(), items)

    qualified_ir = run_frontend(qualified)
    qualified_state = SimpleNamespace(func_ir=qualified_ir, args=(types.int32,))
    assert _GroupCallPlanner(
        qualified_state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    ).run()


@pytest.mark.evidence_for("group.exchange", backend="numba_mlir", evidence="lowering")
@pytest.mark.parametrize("group_kind", ["block", "warp"])
def test_common_exchange_requires_thread_data_but_qualified_keeps_local_arrays(
    optional_backend,
    group_kind,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import cuda, types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    common_constructor = coop.this_block if group_kind == "block" else coop.this_warp
    qualified_constructor = (
        numba_coop.this_block if group_kind == "block" else numba_coop.this_warp
    )

    def common(value):
        items = cuda.local.array(2, types.int32)
        items[0] = value
        items[1] = value + 1
        return coop.exchange(common_constructor(), items)

    common_state = SimpleNamespace(
        func_ir=run_frontend(common),
        args=(types.int32,),
    )
    with pytest.raises(
        TypeError,
        match=(
            r"cuda\.coop\.exchange requires a fixed-size ThreadData payload "
            r"in common V1"
        ),
    ):
        _GroupCallPlanner(
            common_state,
            {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
        ).run()

    def qualified(value):
        items = cuda.local.array(2, types.int32)
        items[0] = value
        items[1] = value + 1
        return numba_coop.exchange(qualified_constructor(), items)

    qualified_state = SimpleNamespace(
        func_ir=run_frontend(qualified),
        args=(types.int32,),
    )
    assert _GroupCallPlanner(
        qualified_state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    ).run()
