# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


def _plan(function, *, arg_types, block=(64, 1, 1)):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._compiler._group_planner import _GroupCallPlanner

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

    import cuda.coop.numba_mlir._lowering as lowering
    from cuda import coop
    from cuda.coop.numba_mlir._compiler._group_planner import has_group_markers

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
        {lowering.load, lowering.store}
        if group_kind == "block"
        else {lowering.warp_load, lowering.warp_store}
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
    from cuda.coop.numba_mlir._compiler._rewrite import (
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


def _parameter_names(specialization):
    return [
        [type(parameter).__name__ for parameter in overload]
        for overload in specialization.parameters
    ]


def test_block_load_store_adapters_preserve_offset_overloads():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    from cuda.coop._core.block import make_block_load_spec, make_block_store_spec
    from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter

    load = NumbaMlirCoreAdapter().materialize(
        make_block_load_spec(
            dtype=types.int32,
            block_dim=(16, 2, 1),
            items_per_thread=3,
            algorithm="striped",
            valid_items=True,
            oob_default=True,
            include_full_tile=True,
            include_pointer_offset=True,
        ).specialization
    )
    assert _parameter_names(load) == [
        ["Pointer", "Pointer", "Array"],
        ["Pointer", "Pointer", "Array", "Value", "Value"],
        [
            "Pointer",
            "Pointer",
            "Array",
            "Value",
            "Value",
            "PointerOffset",
        ],
        ["Pointer", "Pointer", "Array", "PointerOffset"],
    ]

    store = NumbaMlirCoreAdapter().materialize(
        make_block_store_spec(
            dtype=types.int32,
            block_dim=(32, 1, 1),
            items_per_thread=2,
            algorithm="direct",
            include_pointer_offset=True,
        ).specialization
    )
    assert _parameter_names(store) == [
        ["Pointer", "Pointer", "Array"],
        ["Pointer", "Pointer", "Array", "PointerOffset"],
    ]


def test_static_movement_controls_are_absent_from_numba_runtime_abi():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    from cuda.coop._core import ArgumentBinding
    from cuda.coop._core.block import make_block_load_spec, make_block_store_spec
    from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter
    from cuda.coop.numba_mlir._types import PointerOffset, algo_coalesce_key

    adapter = NumbaMlirCoreAdapter()
    load = adapter.materialize(
        make_block_load_spec(
            dtype=types.int32,
            block_dim=(32, 1, 1),
            items_per_thread=2,
            algorithm="direct",
            valid_items=ArgumentBinding.static(17),
            oob_default=ArgumentBinding.static(-1),
            include_pointer_offset=ArgumentBinding.static(3),
        ).specialization
    )
    assert len(load.parameters) == 1
    method = load.parameters[0]
    offset = method[-1]
    assert isinstance(offset, PointerOffset)
    assert offset.static_value == 3
    assert not offset.is_provided_by_user()
    assert sum(parameter.is_provided_by_user() for parameter in method) == 3

    source = load._source_code()[0]
    assert ".Load((param_0 + 3)," in source
    assert "::cuda::std::int64_t param_2" not in source
    assert ", 17, -1);" in source

    static_four = adapter.materialize(
        make_block_store_spec(
            dtype=types.int32,
            block_dim=(32, 1, 1),
            items_per_thread=2,
            algorithm="direct",
            include_pointer_offset=ArgumentBinding.static(4),
        ).specialization
    )
    static_five = adapter.materialize(
        make_block_store_spec(
            dtype=types.int32,
            block_dim=(32, 1, 1),
            items_per_thread=2,
            algorithm="direct",
            include_pointer_offset=ArgumentBinding.static(5),
        ).specialization
    )
    assert static_four.mangled_name(static_four.parameters[0]) != (
        static_five.mangled_name(static_five.parameters[0])
    )
    assert algo_coalesce_key(static_four) != algo_coalesce_key(static_five)

    runtime = adapter.materialize(
        make_block_store_spec(
            dtype=types.int32,
            block_dim=(32, 1, 1),
            items_per_thread=2,
            algorithm="direct",
            include_pointer_offset=ArgumentBinding.runtime(),
        ).specialization
    )
    runtime_source = runtime._source_code()[0]
    assert "::cuda::std::int64_t param_2" in runtime_source
    assert ".Store((param_0 + param_2)," in runtime_source


def test_single_phase_rewrite_preserves_static_block_movement_bindings():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop._core import ArgumentBinding, BindingKind
    from cuda.coop.numba_mlir._compiler._rewrite import CoopSinglePhaseRewrite

    def memory(source, destination, dynamic_offset):
        output = coop.ThreadData(2, dtype=types.int32)
        loaded = coop.load(
            coop.this_block(),
            source,
            output,
            valid_items=31,
            oob_default=-1,
            offset=3,
        )
        coop.store(
            coop.this_block(),
            destination,
            loaded,
            offset=dynamic_offset,
        )

    array_type = types.Array(types.int32, 1, "C")
    arg_types = (array_type, array_type, types.int64)
    func_ir, planner = _plan(memory, arg_types=arg_types)
    assert planner.run()

    class TypingContext:
        def refresh(self):
            pass

    state = SimpleNamespace(
        func_ir=func_ir,
        args=arg_types,
        typingctx=TypingContext(),
        typemap={},
        calltypes={},
        metadata={},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    matches = []
    for label in sorted(func_ir.blocks):
        block = func_ir.blocks[label]
        if rewrite.match(func_ir, block, state.typemap, state.calltypes):
            matches.extend(rewrite._matches.values())

    load_match = next(match for match in matches if match.op_name == "load")
    assert len(load_match.runtime_args) == 2
    assert load_match.factory_kwargs["num_valid_items"] == ArgumentBinding.static(31)
    assert load_match.factory_kwargs["oob_default"] == ArgumentBinding.static(-1)
    assert load_match.factory_kwargs["offset"] == ArgumentBinding.static(3)

    store_match = next(match for match in matches if match.op_name == "store")
    assert len(store_match.runtime_args) == 3
    assert "offset" not in store_match.factory_kwargs
    assert all(
        not isinstance(value, ArgumentBinding) or value.kind is not BindingKind.RUNTIME
        for value in load_match.factory_kwargs.values()
    )
