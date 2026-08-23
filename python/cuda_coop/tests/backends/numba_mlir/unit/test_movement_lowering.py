# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


def _plan(function, *, arg_types, block=(64, 1, 1)):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    func_ir = run_frontend(function)
    state = SimpleNamespace(func_ir=func_ir, args=arg_types)
    planner = _GroupCallPlanner(
        state,
        {"block": block, "grid": (1, 1, 1), "cluster": None},
    )
    return func_ir, planner


def _planned_factory_calls(func_ir, ir):
    globals_by_name = {
        inst.target.name: inst.value.value
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Global)
    }
    return [
        (globals_by_name.get(inst.value.func.name), inst.value)
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "call"
    ]


@pytest.mark.parametrize("group_kind", ["block", "warp", "logical_warp"])
def test_common_load_store_lowers_to_planner_private_factories(group_kind):
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir._block as private_block
    import cuda.coop.numba_mlir._warp as private_warp
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import has_group_markers

    if group_kind == "block":
        group = coop.this_block()
    elif group_kind == "warp":
        group = coop.this_warp()
    else:
        group = coop.this_warp().group_by(8)

    if group_kind == "block":

        def memory(source, destination):
            storage = coop.TempStorage()
            output = coop.ThreadData(2, dtype=types.int32)
            loaded = coop.load(
                group,
                source,
                output,
                algorithm="striped",
                valid_items=31,
                oob_default=-1,
                offset=3,
                temp_storage=storage,
            )
            coop.store(
                group,
                destination,
                loaded,
                algorithm="striped",
                valid_items=31,
                offset=3,
                temp_storage=storage,
            )

    else:

        def memory(source, destination):
            output = coop.ThreadData(2, dtype=types.int32)
            loaded = coop.load(
                group,
                source,
                output,
                algorithm="striped",
                valid_items=31,
                oob_default=-1,
                offset=3,
            )
            coop.store(
                group,
                destination,
                loaded,
                algorithm="striped",
                valid_items=31,
                offset=3,
            )

    array_type = types.Array(types.int32, 1, "C")
    func_ir, planner = _plan(
        memory,
        arg_types=(array_type, array_type),
    )
    assert has_group_markers(func_ir)
    assert planner.run()
    assert not has_group_markers(func_ir)

    expected = (
        {private_block.load, private_block.store}
        if group_kind == "block"
        else {private_warp.load, private_warp.store}
    )
    calls = [
        (factory, call)
        for factory, call in _planned_factory_calls(func_ir, ir)
        if factory in expected
    ]
    assert Counter(factory for factory, _ in calls) == Counter(expected)
    if group_kind == "logical_warp":
        constants = {
            inst.target.name: inst.value.value
            for block in func_ir.blocks.values()
            for inst in block.body
            if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Const)
        }
        assert [
            constants[dict(call.kws)["threads_in_warp"].name] for _, call in calls
        ] == [8, 8]


def test_common_and_qualified_warp_load_reject_explicit_temp_storage():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    def common(source):
        storage = coop.TempStorage()
        output = coop.ThreadData(2, dtype=types.int32)
        return coop.load(
            coop.this_warp().group_by(8),
            source,
            output,
            temp_storage=storage,
        )

    def qualified(source):
        storage = numba_coop.TempStorage()
        output = numba_coop.ThreadData(2, dtype=types.int32)
        return numba_coop.load(
            numba_coop.this_warp().group_by(8),
            source,
            output,
            temp_storage=storage,
        )

    array_type = types.Array(types.int32, 1, "C")
    for function in (common, qualified):
        _, planner = _plan(function, arg_types=(array_type,))
        with pytest.raises(NotImplementedError, match="only for block groups"):
            planner.run()


def test_fixed_capacity_temp_storage_is_forwarded_to_load_and_store():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import cuda, types
    from numba_cuda_mlir.numbair_transforms import ir

    from cuda import coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewrite,
    )

    class TypingContext:
        def refresh(self):
            pass

    class FakeInvocable:
        files = ("movement-test.ltoir",)
        temp_storage_bytes = 64
        temp_storage_alignment = 16

        def __call__(self, *args):
            del args

    def memory(source, destination):
        storage = coop.TempStorage(4096, alignment=16)
        output = coop.ThreadData(2, dtype=types.int32)
        loaded = coop.load(
            coop.this_block(),
            source,
            output,
            temp_storage=storage,
        )
        coop.store(
            coop.this_block(),
            destination,
            loaded,
            temp_storage=storage,
        )

    array_type = types.Array(types.int32, 1, "C")
    func_ir, planner = _plan(
        memory,
        arg_types=(array_type, array_type),
    )
    assert planner.run()
    state = SimpleNamespace(
        func_ir=func_ir,
        args=(array_type, array_type),
        typingctx=TypingContext(),
        typemap={},
        calltypes={},
        metadata={},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    invocable = FakeInvocable()
    rewrite._prepare_ltoir_bundle_for_matches = lambda _matches: None
    rewrite._materialize_invocable = lambda _match: (invocable, False)
    rewrite._record_invocable_specialization = lambda _invocable: None
    for label in sorted(func_ir.blocks):
        block = func_ir.blocks[label]
        while rewrite.match(func_ir, block, state.typemap, state.calltypes):
            block = rewrite.apply()
            func_ir.blocks[label] = block

    calls = _planned_factory_calls(func_ir, ir)
    invocable_calls = [call for factory, call in calls if factory is invocable]
    assert len(invocable_calls) == 2
    assert all(len(call.args) == 3 for call in invocable_calls)
    resolver = object.__new__(CoopSinglePhaseRewrite)
    resolver._func_ir = func_ir
    shared_array_calls = []
    for block in func_ir.blocks.values():
        resolver._block_defs = {
            inst.target.name: inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
        }
        shared_array_calls.extend(
            inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
            and isinstance(inst.value, ir.Expr)
            and inst.value.op == "call"
            and resolver._resolve_python_value(inst.value.func) is cuda.shared.array
        )
    assert len(shared_array_calls) == 1
    assert tuple(name for name, _ in shared_array_calls[0].kws) == ("alignment",)
    resolver._block_defs = {
        inst.target.name: inst.value
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign)
    }
    assert resolver._infer_constant(shared_array_calls[0].args[0]) == 4096
    assert resolver._infer_constant(dict(shared_array_calls[0].kws)["alignment"]) == 16


@pytest.mark.parametrize("group_kind", ["block", "warp", "logical_warp"])
def test_common_exchange_lowers_without_truncating_logical_warp(group_kind):
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir._block as private_block
    import cuda.coop.numba_mlir._warp as private_warp
    from cuda import coop

    if group_kind == "block":
        group = coop.this_block()
    elif group_kind == "warp":
        group = coop.this_warp()
    else:
        group = coop.this_warp().group_by(8)

    def movement(value):
        items = coop.ThreadData(5, dtype=types.int32)
        for index in range(5):
            items[index] = value + index
        blocked = coop.exchange(group, items)
        return coop.exchange(group, blocked, mode="blocked_to_striped")

    func_ir, planner = _plan(movement, arg_types=(types.int32,))
    assert planner.run()
    expected = (
        private_block.exchange if group_kind == "block" else private_warp.exchange
    )
    calls = [
        call
        for factory, call in _planned_factory_calls(func_ir, ir)
        if factory is expected
    ]
    assert len(calls) == 2
    if group_kind == "logical_warp":
        constants = {
            inst.target.name: inst.value.value
            for block in func_ir.blocks.values()
            for inst in block.body
            if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Const)
        }
        assert [
            constants[dict(call.kws)["threads_in_warp"].name] for call in calls
        ] == [8, 8]


def test_common_and_qualified_exchange_accept_eight_items_per_thread():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    def common(value):
        items = coop.ThreadData(8, dtype=types.int32)
        items[0] = value
        return coop.exchange(coop.this_block(), items)

    _, common_planner = _plan(common, arg_types=(types.int32,))
    assert common_planner.run()

    def qualified(value):
        items = numba_coop.ThreadData(8, dtype=types.int32)
        items[0] = value
        return numba_coop.exchange(numba_coop.this_block(), items)

    _, qualified_planner = _plan(qualified, arg_types=(types.int32,))
    assert qualified_planner.run()


def test_common_and_qualified_shuffle_lower_to_block_factory():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    import cuda.coop.numba_mlir._block as private_block
    from cuda import coop

    def common(value):
        items = coop.ThreadData(3, dtype=types.int32)
        for index in range(3):
            items[index] = value + index
        return coop.shuffle(coop.this_block(), items, mode="up")

    def qualified(value):
        items = numba_coop.ThreadData(3, dtype=types.int32)
        for index in range(3):
            items[index] = value + index
        return numba_coop.shuffle(numba_coop.this_block(), items, mode="up")

    for function in (common, qualified):
        func_ir, planner = _plan(function, arg_types=(types.int32,))
        assert planner.run()
        factories = [factory for factory, _ in _planned_factory_calls(func_ir, ir)]
        assert factories.count(private_block.shuffle) == 1


def test_qualified_shuffle_rejects_boundary_outputs():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    def qualified(value):
        items = coop.ThreadData(3, dtype=types.int32)
        suffix = coop.ThreadData(1, dtype=types.int32)
        for index in range(3):
            items[index] = value + index
        return coop.shuffle(
            coop.this_block(),
            items,
            mode="up",
            block_suffix=suffix,
        )

    _, planner = _plan(qualified, arg_types=(types.int32,))
    with pytest.raises(NotImplementedError, match="without boundary outputs"):
        planner.run()


@pytest.mark.parametrize(
    ("factory", "keyword"),
    [
        ("block_exchange", "block_exchange_type"),
        ("block_shuffle", "block_shuffle_type"),
        ("warp_exchange", "warp_exchange_type"),
    ],
)
def test_integer_selectors_use_index_and_reject_bool(factory, keyword):
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    if factory == "block_exchange":
        from cuda.coop.numba_mlir._block._block_exchange import exchange as operation

        kwargs = {"threads_per_block": 32}
    elif factory == "block_shuffle":
        from cuda.coop.numba_mlir._block._block_shuffle import shuffle as operation

        kwargs = {"threads_per_block": 32}
    else:
        from cuda.coop.numba_mlir._warp._warp_exchange import (
            warp_exchange as operation,
        )

        kwargs = {"threads_in_warp": 8}

    with pytest.raises(TypeError, match="must not be bool"):
        operation(dtype=types.int32, **kwargs, **{keyword: True})

    # NumPy integers implement __index__; accepting them avoids lossy int(...)
    # coercions while preserving exact selector validation.
    try:
        operation(dtype=types.int32, **kwargs, **{keyword: np.int64(1)})
    except RuntimeError:
        # Materialization can require a CUDA toolkit in a host-only test. The
        # selector has already passed validation by this point.
        pass


def test_warp_exchange_rejects_fractional_width_instead_of_truncating():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._warp._warp_exchange import warp_exchange

    with pytest.raises(TypeError, match="threads_in_warp must be an integer"):
        warp_exchange(dtype=types.int32, threads_in_warp=8.5)


def test_shuffle_boundary_none_and_omitted_types_are_preserved():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewrite,
    )

    rewrite = object.__new__(CoopSinglePhaseRewrite)
    rewrite._block_defs = {}
    rewrite._func_ir = SimpleNamespace(
        get_definition=lambda _value: (_ for _ in ()).throw(KeyError())
    )
    rewrite._state = SimpleNamespace(args=())
    none_var = ir.Var(ir.Scope(None, ir.Loc("test", 1)), "none", ir.Loc("test", 1))
    omitted_var = ir.Var(
        none_var.scope,
        "omitted",
        none_var.loc,
    )
    rewrite._arg_type_map = {
        none_var.name: types.none,
        omitted_var.name: types.Omitted(None),
    }

    assert rewrite._resolve_factory_kwarg_value("block_prefix", none_var) is None
    assert rewrite._resolve_factory_kwarg_value("block_suffix", omitted_var) is None
