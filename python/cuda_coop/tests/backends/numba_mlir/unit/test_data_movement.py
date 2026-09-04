# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


@pytest.fixture(autouse=True)
def _fixed_provider_compute_capability(monkeypatch):
    from cuda.coop.numba_mlir import _types

    monkeypatch.setattr(
        _types.cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )


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


def _run_single_phase_to_provider_boundary(function, *, arg_types, monkeypatch):
    from cuda.coop.numba_mlir._compiler._rewrite import CoopSinglePhaseRewrite

    func_ir, planner = _plan(function, arg_types=arg_types, block=(32, 1, 1))
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
    monkeypatch.setattr(
        rewrite,
        "_prepare_ltoir_bundle_for_matches",
        lambda _matches: pytest.fail(
            "invalid movement arguments reached provider bundling"
        ),
    )
    monkeypatch.setattr(
        rewrite,
        "_materialize_invocable",
        lambda _match: pytest.fail(
            "invalid movement arguments reached provider materialization"
        ),
    )
    for label in sorted(func_ir.blocks):
        rewrite.match(
            func_ir,
            func_ir.blocks[label],
            state.typemap,
            state.calltypes,
        )


def test_provider_memory_parameters_require_contiguous_arrays():
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir import _types

    assert _types.Pointer(types.int32).dtype() == types.Array(types.int32, 1, "C")


def test_common_direct_block_load_store_lowers_to_private_factories():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir._lowering as lowering
    from cuda import coop
    from cuda.coop.numba_mlir._compiler._group_planner import has_group_markers

    def memory(source, destination):
        storage = coop.TempStorage()
        output = coop.ThreadData(2, dtype=types.int32)
        loaded = coop.load(
            coop.this_block(),
            source,
            output,
            algorithm="direct",
            valid_items=31,
            oob_default=-1,
            offset=3,
            temp_storage=storage,
        )
        coop.store(
            coop.this_block(),
            destination,
            loaded,
            algorithm="direct",
            valid_items=31,
            offset=3,
            temp_storage=storage,
        )

    array_type = types.Array(types.int32, 1, "C")
    func_ir, planner = _plan(
        memory,
        arg_types=(array_type, array_type),
    )
    assert has_group_markers(func_ir)
    assert planner.run()
    assert not has_group_markers(func_ir)

    expected = {lowering.load, lowering.store}
    calls = [
        (factory, call)
        for factory, call in _planned_factory_calls(func_ir, ir)
        if factory in expected
    ]
    assert Counter(factory for factory, _ in calls) == Counter(expected)


@pytest.mark.parametrize("qualified", [False, True], ids=["portable", "qualified"])
@pytest.mark.parametrize(
    "group_factory",
    [
        lambda module: module.this_thread(),
        lambda module: module.this_warp(),
        lambda module: module.this_cluster(),
        lambda module: module.this_grid(),
    ],
    ids=["thread", "warp", "cluster", "grid"],
)
def test_nonblock_load_returns_typed_unsupported_plan_before_compile(
    monkeypatch, qualified, group_factory
):
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    module = numba_coop if qualified else coop
    group = group_factory(module)

    def memory(source):
        output = module.ThreadData(2, dtype=types.int32)
        return module.load(group, source, output)

    array_type = types.Array(types.int32, 1, "C")
    _, planner = _plan(memory, arg_types=(array_type,))
    from cuda.coop.numba_mlir._compiler import _group_load_store

    monkeypatch.setattr(
        _group_load_store._LoadStorePlanning,
        "_scope_factory",
        lambda *_args, **_kwargs: pytest.fail(
            "unsupported group reached provider compilation"
        ),
    )
    with pytest.raises(
        NotImplementedError,
        match=(
            r"does not support group kind|"
            r"currently lowers only this_block\(\) groups through CUB"
        ),
    ):
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
        storage_abi = "leading_pointer"
        execution_scope = "block"
        synchronization_scope = "block"

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


def _block_provider_metadata():
    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir._compiler._operations import StorageABI

    return {
        "storage_abi": StorageABI.LEADING_POINTER,
        "execution_scope": SynchronizationScope.BLOCK,
        "synchronization_scope": SynchronizationScope.BLOCK,
    }


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
            algorithm="direct",
            valid_items=True,
            oob_default=True,
            include_full_tile=True,
            include_pointer_offset=True,
        ).specialization,
        **_block_provider_metadata(),
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
        ).specialization,
        **_block_provider_metadata(),
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
        ).specialization,
        **_block_provider_metadata(),
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
        ).specialization,
        **_block_provider_metadata(),
    )
    static_five = adapter.materialize(
        make_block_store_spec(
            dtype=types.int32,
            block_dim=(32, 1, 1),
            items_per_thread=2,
            algorithm="direct",
            include_pointer_offset=ArgumentBinding.static(5),
        ).specialization,
        **_block_provider_metadata(),
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
        ).specialization,
        **_block_provider_metadata(),
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


@pytest.mark.parametrize("qualified", [False, True], ids=["root", "qualified"])
@pytest.mark.parametrize(
    ("default_type_name", "error"),
    [
        ("boolean", "supports oob_default dtypes"),
        ("float16", "supports oob_default dtypes"),
        ("complex64", "supports oob_default dtypes"),
        ("optional", "supports oob_default dtypes"),
        ("int64", "does not match payload dtype"),
    ],
    ids=["bool", "float16", "complex", "optional", "mismatched"],
)
def test_runtime_oob_default_rejects_before_provider_materialization(
    monkeypatch,
    qualified,
    default_type_name,
    error,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as root_coop
    from cuda.coop.numba_mlir._compiler._rewrite import (
        CoopSinglePhaseRewriteError,
    )

    module = qualified_coop if qualified else root_coop

    def memory(source, valid_items, oob_default):
        output = module.ThreadData(2, dtype=types.int32)
        return module.load(
            module.this_block(),
            source,
            output,
            valid_items=valid_items,
            oob_default=oob_default,
        )

    default_type = (
        types.Optional(types.int32)
        if default_type_name == "optional"
        else getattr(types, default_type_name)
    )
    array_type = types.Array(types.int32, 1, "C")
    with pytest.raises((CoopSinglePhaseRewriteError, TypeError), match=error):
        _run_single_phase_to_provider_boundary(
            memory,
            arg_types=(array_type, types.int32, default_type),
            monkeypatch=monkeypatch,
        )


@pytest.mark.parametrize("qualified", [False, True], ids=["root", "qualified"])
@pytest.mark.parametrize(
    "oob_default",
    [
        pytest.param(True, id="bool"),
        pytest.param(np.float16(1), id="float16"),
        pytest.param(1 + 2j, id="complex"),
        pytest.param(object(), id="object"),
        pytest.param(float("inf"), id="nonfinite"),
        pytest.param(1 << 65, id="outside-64-bit"),
    ],
)
def test_static_oob_default_rejects_before_provider_materialization(
    monkeypatch,
    qualified,
    oob_default,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as root_coop
    from cuda.coop.numba_mlir._compiler._rewrite import (
        CoopSinglePhaseRewriteError,
    )

    module = qualified_coop if qualified else root_coop

    def memory(source):
        output = module.ThreadData(2, dtype=types.int32)
        return module.load(
            module.this_block(),
            source,
            output,
            valid_items=31,
            oob_default=oob_default,
        )

    array_type = types.Array(types.int32, 1, "C")
    with pytest.raises(
        (CoopSinglePhaseRewriteError, TypeError, ValueError),
        match="oob_default",
    ):
        _run_single_phase_to_provider_boundary(
            memory,
            arg_types=(array_type,),
            monkeypatch=monkeypatch,
        )


@pytest.mark.parametrize("operation", ["load", "store"])
@pytest.mark.parametrize(
    "algorithm",
    [
        "striped",
        "vectorize",
        "transpose",
        "warp_transpose",
        "warp_transpose_timesliced",
    ],
)
def test_non_direct_block_algorithms_fail_before_provider_materialization(
    monkeypatch, operation, algorithm
):
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._lowering import _load_store

    monkeypatch.setattr(
        _load_store.NumbaMlirCoreAdapter,
        "materialize",
        lambda *args, **kwargs: pytest.fail(
            "unsupported algorithm reached provider materialization"
        ),
    )
    factory = getattr(_load_store, operation)

    with pytest.raises(NotImplementedError, match="only 'direct'"):
        factory(types.int32, threads_per_block=32, algorithm=algorithm)


@pytest.mark.parametrize("operation", ["load", "store"])
@pytest.mark.parametrize("algorithm", [True, "stripd", object()])
def test_invalid_block_algorithm_values_fail_before_provider_materialization(
    monkeypatch, operation, algorithm
):
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._lowering import _load_store

    monkeypatch.setattr(
        _load_store.NumbaMlirCoreAdapter,
        "materialize",
        lambda *args, **kwargs: pytest.fail(
            "invalid algorithm reached provider materialization"
        ),
    )
    factory = getattr(_load_store, operation)
    error_type = TypeError if algorithm is True else ValueError

    with pytest.raises(error_type):
        factory(types.int32, threads_per_block=32, algorithm=algorithm)


@pytest.mark.parametrize("operation", ["load", "store"])
@pytest.mark.parametrize("dtype_name", ["boolean", "float16", "complex64"])
def test_unsupported_dtypes_fail_before_provider_materialization(
    monkeypatch, operation, dtype_name
):
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._lowering import _load_store

    monkeypatch.setattr(
        _load_store.NumbaMlirCoreAdapter,
        "materialize",
        lambda *args, **kwargs: pytest.fail(
            "unsupported dtype reached provider materialization"
        ),
    )
    factory = getattr(_load_store, operation)

    with pytest.raises(TypeError, match="supports dtypes"):
        factory(getattr(types, dtype_name), threads_per_block=32)


@pytest.mark.parametrize(
    "oob_default",
    [
        pytest.param(True, id="bool"),
        pytest.param(np.float16(1), id="float16"),
        pytest.param(1 + 2j, id="complex"),
        pytest.param(object(), id="object"),
        pytest.param(float("inf"), id="nonfinite"),
        pytest.param(1 << 65, id="outside-64-bit"),
    ],
)
def test_direct_provider_rejects_invalid_static_oob_default_before_materialization(
    monkeypatch,
    oob_default,
):
    from numba_cuda_mlir import types

    from cuda.coop._core import ArgumentBinding
    from cuda.coop.numba_mlir._lowering import _load_store

    monkeypatch.setattr(
        _load_store.NumbaMlirCoreAdapter,
        "materialize",
        lambda *args, **kwargs: pytest.fail(
            "invalid oob_default reached provider materialization"
        ),
    )

    with pytest.raises((TypeError, ValueError), match="oob_default"):
        _load_store.load(
            types.int32,
            threads_per_block=32,
            num_valid_items=ArgumentBinding.static(31),
            oob_default=ArgumentBinding.static(oob_default),
        )


@pytest.mark.parametrize("operation", ["load", "store"])
def test_static_valid_items_beyond_the_exact_tile_fail_before_nvrtc(
    monkeypatch, operation
):
    from numba_cuda_mlir import types

    from cuda.coop._core import ArgumentBinding
    from cuda.coop.numba_mlir._lowering import _load_store

    monkeypatch.setattr(
        _load_store.NumbaMlirCoreAdapter,
        "materialize",
        lambda *args, **kwargs: pytest.fail("invalid tile reached materialization"),
    )
    factory = getattr(_load_store, operation)

    with pytest.raises(ValueError, match="block tile size"):
        factory(
            types.int32,
            threads_per_block=32,
            items_per_thread=2,
            num_valid_items=ArgumentBinding.static(65),
        )


@pytest.mark.parametrize("operation", ["load", "store"])
def test_negative_static_offsets_fail_before_provider_materialization(
    monkeypatch, operation
):
    from numba_cuda_mlir import types

    from cuda.coop._core import ArgumentBinding
    from cuda.coop.numba_mlir._lowering import _load_store

    monkeypatch.setattr(
        _load_store.NumbaMlirCoreAdapter,
        "materialize",
        lambda *args, **kwargs: pytest.fail(
            "negative offset reached provider materialization"
        ),
    )
    factory = getattr(_load_store, operation)

    with pytest.raises(ValueError, match="offset must be nonnegative"):
        factory(
            types.int32,
            threads_per_block=32,
            offset=ArgumentBinding.static(-1),
        )
