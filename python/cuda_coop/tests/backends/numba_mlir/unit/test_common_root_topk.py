# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter
from types import SimpleNamespace

import pytest

_BLOCK = (64, 1, 1)


def _plan(function, *, arg_types, block=_BLOCK):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    func_ir = run_frontend(function)
    planner = _GroupCallPlanner(
        SimpleNamespace(func_ir=func_ir, args=arg_types),
        {"block": block, "grid": (1, 1, 1), "cluster": None},
    )
    return func_ir, planner


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


def _planned_factory_calls(func_ir, ir):
    globals_by_name = {
        inst.target.name: inst.value.value
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Global)
    }
    return tuple(
        (globals_by_name.get(inst.value.func.name), inst.value)
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "call"
    )


@pytest.mark.evidence_for(
    "group.topk_max_keys",
    backend="numba_mlir",
    evidence="lowering",
)
@pytest.mark.evidence_for(
    "group.topk_min_keys",
    backend="numba_mlir",
    evidence="lowering",
)
@pytest.mark.evidence_for(
    "group.topk_max_pairs",
    backend="numba_mlir",
    evidence="lowering",
)
@pytest.mark.evidence_for(
    "group.topk_min_pairs",
    backend="numba_mlir",
    evidence="lowering",
)
def test_common_and_qualified_topk_lower_through_separate_contract_factories(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._block import _block_topk

    def cohort(value, k, valid_items, begin_bit, end_bit):
        common_keys = coop.ThreadData(2, dtype=types.int32)
        qualified_keys = numba_coop.ThreadData(2, dtype=types.int32)
        common_values = coop.ThreadData(2, dtype=types.float64)
        qualified_values = numba_coop.ThreadData(2, dtype=types.float64)
        for item in range(2):
            common_keys[item] = value
            qualified_keys[item] = value
            common_values[item] = value
            qualified_values[item] = value
        common_max = coop.topk_max_keys(
            coop.this_block(),
            common_keys,
            k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )
        qualified_max = numba_coop.topk_max_keys(
            numba_coop.this_block(),
            qualified_keys,
            k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )
        common_min = coop.topk_min_keys(
            coop.this_block(),
            common_keys,
            k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )
        qualified_min = numba_coop.topk_min_keys(
            numba_coop.this_block(),
            qualified_keys,
            k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )
        common_pair_keys, common_pair_values = coop.topk_max_pairs(
            coop.this_block(), common_keys, common_values, k
        )
        common_min_pair_keys, common_min_pair_values = coop.topk_min_pairs(
            coop.this_block(), common_keys, common_values, k
        )
        qualified_pair_keys, qualified_pair_values = numba_coop.topk_max_pairs(
            numba_coop.this_block(), qualified_keys, qualified_values, k
        )
        qualified_min_pair_keys, qualified_min_pair_values = numba_coop.topk_min_pairs(
            numba_coop.this_block(), qualified_keys, qualified_values, k
        )
        return (
            common_max[0]
            + qualified_max[0]
            + common_min[0]
            + qualified_min[0]
            + common_pair_keys[0]
            + common_pair_values[0]
            + common_min_pair_keys[0]
            + common_min_pair_values[0]
            + qualified_pair_keys[0]
            + qualified_pair_values[0]
            + qualified_min_pair_keys[0]
            + qualified_min_pair_values[0]
        )

    func_ir, planner = _plan(
        cohort,
        arg_types=(types.int32,) * 5,
    )
    assert planner.run()

    factories = _planned_factories(func_ir, ir)
    assert factories[_block_topk._common_topk_max_keys] == 1
    assert factories[_block_topk._common_topk_min_keys] == 1
    assert factories[_block_topk.topk_max_keys] == 1
    assert factories[_block_topk.topk_min_keys] == 1
    assert factories[_block_topk._common_topk_max_pairs] == 1
    assert factories[_block_topk._common_topk_min_pairs] == 1
    assert factories[_block_topk.topk_max_pairs] == 1
    assert factories[_block_topk.topk_min_pairs] == 1


def test_common_topk_requires_thread_data_but_qualified_keeps_local_arrays(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    def common(value):
        keys = cuda.local.array(2, types.int32)
        keys[0] = value
        return coop.topk_max_keys(coop.this_block(), keys, 1)

    _common_ir, common_planner = _plan(common, arg_types=(types.int32,))
    with pytest.raises(
        TypeError,
        match=r"cuda\.coop\.topk_max_keys requires keys to be coop\.ThreadData",
    ):
        common_planner.run()

    def qualified(value):
        keys = cuda.local.array(2, types.float32)
        keys[0] = value
        return numba_coop.topk_max_keys(numba_coop.this_block(), keys, 1)

    _qualified_ir, qualified_planner = _plan(
        qualified,
        arg_types=(types.float32,),
    )
    assert qualified_planner.run()


def test_qualified_topk_keys_and_pairs_allow_begin_bit_without_end_bit(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop.numba_mlir._block import _block_topk

    def cohort(key, value, k):
        keys = numba_coop.ThreadData(2, dtype=types.uint32)
        values = numba_coop.ThreadData(2, dtype=types.int32)
        for item in range(2):
            keys[item] = key
            values[item] = value
        selected_keys = numba_coop.topk_max_keys(
            numba_coop.this_block(),
            keys,
            k,
            begin_bit=4,
        )
        pair_keys, pair_values = numba_coop.topk_min_pairs(
            numba_coop.this_block(),
            keys,
            values,
            k,
            begin_bit=4,
        )
        return selected_keys[0] + pair_keys[0] + pair_values[0]

    func_ir, planner = _plan(
        cohort,
        arg_types=(types.uint32, types.int32, types.int32),
    )
    assert planner.run()

    factories = _planned_factories(func_ir, ir)
    assert factories[_block_topk._qualified_group_topk_max_keys] == 1
    assert factories[_block_topk._qualified_group_topk_min_pairs] == 1


def test_qualified_topk_forwards_one_reusable_temp_storage_to_all_factories(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop as common_coop
    from cuda.coop.numba_mlir._block import _block_topk
    from cuda.coop.numba_mlir._single_phase_rewrites import CoopSinglePhaseRewrite

    def cohort(key, value, k):
        storage = numba_coop.TempStorage()
        keys = numba_coop.ThreadData(2, dtype=types.int32)
        values = numba_coop.ThreadData(2, dtype=types.float32)
        for item in range(2):
            keys[item] = key
            values[item] = value
        max_keys = numba_coop.topk_max_keys(
            numba_coop.this_block(), keys, k, temp_storage=storage
        )
        min_keys = numba_coop.topk_min_keys(
            numba_coop.this_block(), keys, k, temp_storage=storage
        )
        max_pair_keys, max_pair_values = numba_coop.topk_max_pairs(
            numba_coop.this_block(), keys, values, k, temp_storage=storage
        )
        min_pair_keys, min_pair_values = numba_coop.topk_min_pairs(
            numba_coop.this_block(), keys, values, k, temp_storage=storage
        )
        qualified_max_keys = numba_coop.topk_max_keys(
            numba_coop.this_block(),
            keys,
            k,
            begin_bit=1,
            temp_storage=storage,
        )
        qualified_min_keys = numba_coop.topk_min_keys(
            numba_coop.this_block(),
            keys,
            k,
            begin_bit=1,
            temp_storage=storage,
        )
        qualified_max_pair_keys, qualified_max_pair_values = numba_coop.topk_max_pairs(
            numba_coop.this_block(),
            keys,
            values,
            k,
            begin_bit=1,
            temp_storage=storage,
        )
        qualified_min_pair_keys, qualified_min_pair_values = numba_coop.topk_min_pairs(
            numba_coop.this_block(),
            keys,
            values,
            k,
            begin_bit=1,
            temp_storage=storage,
        )
        common_max_keys = common_coop.topk_max_keys(
            common_coop.this_block(),
            keys,
            k,
            temp_storage=storage,
        )
        common_min_keys = common_coop.topk_min_keys(
            common_coop.this_block(),
            keys,
            k,
            temp_storage=storage,
        )
        return (
            max_keys[0]
            + min_keys[0]
            + max_pair_keys[0]
            + max_pair_values[0]
            + min_pair_keys[0]
            + min_pair_values[0]
            + qualified_max_keys[0]
            + qualified_min_keys[0]
            + qualified_max_pair_keys[0]
            + qualified_max_pair_values[0]
            + qualified_min_pair_keys[0]
            + qualified_min_pair_values[0]
            + common_max_keys[0]
            + common_min_keys[0]
        )

    func_ir, planner = _plan(
        cohort,
        arg_types=(types.int32, types.float32, types.int32),
    )
    assert planner.run()

    expected_factories = {
        _block_topk.topk_max_keys,
        _block_topk.topk_min_keys,
        _block_topk.topk_max_pairs,
        _block_topk.topk_min_pairs,
        _block_topk._qualified_group_topk_max_keys,
        _block_topk._qualified_group_topk_min_keys,
        _block_topk._qualified_group_topk_max_pairs,
        _block_topk._qualified_group_topk_min_pairs,
        _block_topk._common_topk_max_keys,
        _block_topk._common_topk_min_keys,
    }
    topk_calls = tuple(
        (factory, call)
        for factory, call in _planned_factory_calls(func_ir, ir)
        if factory in expected_factories
    )
    assert {factory for factory, _call in topk_calls} == expected_factories
    calls = tuple(call for _factory, call in topk_calls)
    storage_vars = [dict(call.kws)["temp_storage"] for call in calls]
    assert len({storage_var.name for storage_var in storage_vars}) == 1
    planned_factory_names = {factory.__name__ for factory, _call in topk_calls}
    assert planned_factory_names <= CoopSinglePhaseRewrite._TEMP_STORAGE_RUNTIME_KW_OPS


@pytest.mark.parametrize("dtype_name", ["int32", "uint32", "int64", "uint64"])
def test_common_topk_factories_accept_portable_integer_dtypes(
    optional_backend,
    dtype_name,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._block import _block_topk
    from cuda.coop.numba_mlir._types import collect_specializations

    with collect_specializations() as collected:
        specialization = _block_topk._common_topk_max_keys(
            dtype=getattr(types, dtype_name),
            threads_per_block=64,
            items_per_thread=2,
            begin_bit=4,
        )

    assert specialization is collected[0][0]
    assert specialization.method_name == "max_keys_full"


@pytest.mark.parametrize("dtype_name", ["boolean", "int16", "float32"])
def test_common_topk_factories_reject_nonportable_key_dtypes(
    optional_backend,
    dtype_name,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._block import _block_topk

    with pytest.raises(
        TypeError,
        match=(
            r"cuda\.coop\.topk_max_keys common V1 supports key dtypes int32, "
            r"uint32, int64, uint64"
        ),
    ):
        _block_topk._common_topk_max_keys(
            dtype=getattr(types, dtype_name),
            threads_per_block=64,
            items_per_thread=2,
        )
