# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter
from types import SimpleNamespace

import pytest

_BLOCK = (8, 4, 2)
_ITEMS_PER_THREAD = 2


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
        {"block": _BLOCK, "grid": (1, 1, 1), "cluster": None},
    )
    return func_ir, planner


@pytest.mark.evidence_for(
    "group.merge_sort_keys", backend="numba_mlir", evidence="lowering"
)
@pytest.mark.parametrize("group_kind", ["block", "warp"])
def test_common_and_qualified_merge_sort_lower_to_identical_factories(
    optional_backend,
    group_kind,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    import cuda.coop.numba_mlir._block as scoped_block
    import cuda.coop.numba_mlir._warp as scoped_warp
    from cuda import coop
    from cuda.coop.numba_mlir._block import _block_merge_sort
    from cuda.coop.numba_mlir._group_rewrites import has_group_markers
    from cuda.coop.numba_mlir._warp import _warp_merge_sort

    common_constructor = coop.this_block if group_kind == "block" else coop.this_warp
    qualified_constructor = (
        numba_coop.this_block if group_kind == "block" else numba_coop.this_warp
    )
    valid_items = 117 if group_kind == "block" else 53
    expected_factory = (
        scoped_block.merge_sort_keys
        if group_kind == "block"
        else scoped_warp.merge_sort_keys
    )
    expected_common_factory = (
        _block_merge_sort._common_merge_sort_keys
        if group_kind == "block"
        else _warp_merge_sort._common_warp_merge_sort_keys
    )

    def common(value):
        keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        keys[0] = value
        keys[1] = value - 1
        group = common_constructor()
        full = coop.merge_sort_keys(group, keys, descending=True)
        partial = coop.merge_sort_keys(
            group,
            keys,
            valid_items=valid_items,
            oob_default=2_147_483_647,
        )
        return keys[0], full[0], partial[0]

    def qualified(value):
        keys = numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        keys[0] = value
        keys[1] = value - 1
        group = qualified_constructor()
        full = numba_coop.merge_sort_keys(group, keys, descending=True)
        partial = numba_coop.merge_sort_keys(
            group,
            keys,
            valid_items=valid_items,
            oob_default=2_147_483_647,
        )
        return keys[0], full[0], partial[0]

    counts = []
    for function, factory in (
        (common, expected_common_factory),
        (qualified, expected_factory),
    ):
        func_ir, planner = _plan(function, arg_types=(types.int32,))
        assert has_group_markers(func_ir)
        assert planner.run()
        assert not has_group_markers(func_ir)
        counts.append(_planned_factories(func_ir, ir)[factory])

    assert counts == [2, 2]
