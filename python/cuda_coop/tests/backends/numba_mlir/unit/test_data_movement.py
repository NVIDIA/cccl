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


def _run_single_phase_to_provider_boundary(
    function,
    *,
    arg_types,
    monkeypatch,
    allow_provider_bundling=False,
):
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
    if allow_provider_bundling:
        monkeypatch.setattr(
            rewrite,
            "_compute_func_temp_storage_requirements",
            lambda _func_ir: None,
        )
    monkeypatch.setattr(
        rewrite,
        "_prepare_ltoir_bundle_for_matches",
        (
            (lambda _matches: None)
            if allow_provider_bundling
            else lambda _matches: pytest.fail(
                "invalid movement arguments reached provider bundling"
            )
        ),
    )
    monkeypatch.setattr(
        rewrite,
        "_materialize_invocable",
        lambda _match: pytest.fail(
            "invalid movement arguments reached provider materialization"
        ),
    )
    matched_group_call = False
    for label in sorted(func_ir.blocks):
        rewrite.match(
            func_ir,
            func_ir.blocks[label],
            state.typemap,
            state.calltypes,
        )
        matched_group_call |= bool(rewrite._matches)
    assert matched_group_call


class _FailingAttribute:
    def __init__(self, exception_type):
        self._exception_type = exception_type

    def __getattr__(self, name):
        raise self._exception_type(name)


@pytest.mark.parametrize(
    "exception_type",
    (AttributeError, ImportError, KeyError, TypeError, ValueError),
)
def test_call_result_dtype_treats_attribute_failure_as_unresolved(exception_type):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    from cuda.coop.numba_mlir._compiler._rewrite import CoopSinglePhaseRewrite

    failing_attribute = _FailingAttribute(exception_type)

    def kernel(value):
        return failing_attribute.cast(value)

    func_ir = run_frontend(kernel)
    state = SimpleNamespace(func_ir=func_ir, args=(), typemap={})
    rewrite = CoopSinglePhaseRewrite(state)
    call = next(
        inst.value
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "call"
    )

    assert rewrite._resolve_call_result_dtype(call) is None


@pytest.mark.parametrize("positional_value", (31, None))
def test_positional_static_runtime_control_cannot_be_repeated_by_keyword(
    positional_value,
):
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler._rewrite import (
        CoopSinglePhaseRewrite,
        CoopSinglePhaseRewriteError,
    )
    from cuda.coop.numba_mlir._lowering._load_store import load as provider_load

    def kernel(source, dynamic_valid_items):
        output = coop.ThreadData(2, dtype=types.int32)
        return provider_load(
            source,
            output,
            positional_value,
            num_valid_items=dynamic_valid_items,
            dtype=types.int32,
            threads_per_block=32,
            items_per_thread=2,
        )

    func_ir = run_frontend(kernel)
    state = SimpleNamespace(
        func_ir=func_ir,
        args=(types.Array(types.int32, 1, "C"), types.int32),
        typingctx=SimpleNamespace(refresh=lambda: None),
        typemap={},
        calltypes={},
        metadata={},
    )
    rewrite = CoopSinglePhaseRewrite(state)

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="duplicate runtime argument 'num_valid_items'",
    ):
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


def test_direct_load_store_accept_temp_storage_without_using_it():
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
        temp_storage_bytes = 0
        temp_storage_alignment = 1
        storage_abi = "none"
        execution_scope = "block"
        synchronization_scope = "none"

        def __call__(self, *args):
            del args

    def memory(source, destination):
        shared_storage = coop.TempStorage(sharing="shared")
        exclusive_storage = coop.TempStorage(sharing="exclusive")
        oversized_storage = coop.TempStorage(128 * 1024, alignment=16)
        output = coop.ThreadData(2, dtype=types.int32)
        extra_output = coop.ThreadData(2, dtype=types.int32)
        loaded = coop.load(
            coop.this_block(),
            source,
            output,
            temp_storage=shared_storage,
        )
        coop.store(
            coop.this_block(),
            destination,
            loaded,
            temp_storage=exclusive_storage,
        )
        coop.load(
            coop.this_block(),
            source,
            extra_output,
            temp_storage=oversized_storage,
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
    assert len(invocable_calls) == 3
    assert all(len(call.args) == 2 for call in invocable_calls)
    resolver = object.__new__(CoopSinglePhaseRewrite)
    resolver._func_ir = func_ir
    resolved_calls = []
    for block in func_ir.blocks.values():
        resolver._block_defs = {
            inst.target.name: inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
        }
        resolved_calls.extend(
            resolver._resolve_python_value(inst.value.func)
            for inst in block.body
            if isinstance(inst, ir.Assign)
            and isinstance(inst.value, ir.Expr)
            and inst.value.op == "call"
        )
    assert cuda.shared.array not in resolved_calls
    assert cuda.syncthreads not in resolved_calls
    assert cuda.syncwarp not in resolved_calls
    assert rewrite._temp_storage_global_plan is None
    assert rewrite._temp_storage_backing_var is None
    assert rewrite._temp_storage_plans == {}


def _parameter_names(specialization):
    return [
        [type(parameter).__name__ for parameter in overload]
        for overload in specialization.parameters
    ]


def _block_provider_metadata():
    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir._compiler._operations import StorageABI

    return {
        "storage_abi": StorageABI.NONE,
        "execution_scope": SynchronizationScope.BLOCK,
        "synchronization_scope": SynchronizationScope.NONE,
    }


def test_block_load_store_adapters_preserve_offset_overloads():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    from cuda.coop._core import ArgumentBinding
    from cuda.coop._core.block import make_block_load_spec, make_block_store_spec
    from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter
    from cuda.coop.numba_mlir._lowering._load_store import _load_store_value_abis

    load = NumbaMlirCoreAdapter(
        value_abis=_load_store_value_abis(
            dtype=types.int32,
            block_dim=(16, 2, 1),
            items_per_thread=3,
            valid_items=ArgumentBinding.runtime(),
            oob_default=ArgumentBinding.runtime(),
        )
    ).materialize(
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
        ["Pointer", "Array"],
        [
            "Pointer",
            "Array",
            "BoundedInteger",
            "ExactValue",
        ],
        [
            "Pointer",
            "Array",
            "BoundedInteger",
            "ExactValue",
            "PointerOffset",
        ],
        ["Pointer", "Array", "PointerOffset"],
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
        ["Pointer", "Array"],
        ["Pointer", "Array", "PointerOffset"],
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
    assert sum(parameter.is_provided_by_user() for parameter in method) == 2

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


def test_runtime_valid_items_is_checked_before_cub_integer_narrowing():
    from numba_cuda_mlir import types

    from cuda.coop._core import ArgumentBinding
    from cuda.coop._core.block import make_block_load_spec
    from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter
    from cuda.coop.numba_mlir._lowering._load_store import _load_store_value_abis
    from cuda.coop.numba_mlir._types import BoundedInteger, algo_coalesce_key

    adapter = NumbaMlirCoreAdapter(
        value_abis=_load_store_value_abis(
            dtype=types.int32,
            block_dim=(16, 2, 1),
            items_per_thread=3,
            valid_items=ArgumentBinding.runtime(),
        )
    )
    load = adapter.materialize(
        make_block_load_spec(
            dtype=types.int32,
            block_dim=(16, 2, 1),
            items_per_thread=3,
            algorithm="direct",
            valid_items=ArgumentBinding.runtime(),
        ).specialization,
        **_block_provider_metadata(),
    )
    valid_items = load.parameters[0][-1]
    assert isinstance(valid_items, BoundedInteger)
    assert valid_items.dtype() == types.int64
    assert valid_items.provider_dtype == types.int32
    assert (valid_items.minimum, valid_items.maximum) == (0, 96)
    assert all(
        valid_items.accepts_actual_type(dtype, None)
        for dtype in (
            types.int8,
            types.int16,
            types.int32,
            types.int64,
            types.uint8,
            types.uint16,
            types.uint32,
        )
    )
    assert all(
        not valid_items.accepts_actual_type(dtype, None)
        for dtype in (types.boolean, types.uint64, types.float32, types.float64)
    )

    source = load._source_code()[0]
    declaration = "::cuda::std::int64_t param_2"
    guard = "if (param_2 < 0 || param_2 > 96)"
    trap = 'asm volatile("trap;" : : :);'
    narrowing = (
        "::cuda::std::int32_t checked_param_2 = "
        "static_cast<::cuda::std::int32_t>(param_2);"
    )
    invocation = ".Load(param_0, *reinterpret_cast<"
    assert source.index(declaration) < source.index(guard)
    assert source.index(guard) < source.index(trap) < source.index(narrowing)
    assert source.index(narrowing) < source.index(invocation)
    assert "checked_param_2);" in source

    narrower_adapter = NumbaMlirCoreAdapter(
        value_abis=_load_store_value_abis(
            dtype=types.int32,
            block_dim=(16, 1, 1),
            items_per_thread=3,
            valid_items=ArgumentBinding.runtime(),
        )
    )
    narrower_tile = narrower_adapter.materialize(
        make_block_load_spec(
            dtype=types.int32,
            block_dim=(16, 1, 1),
            items_per_thread=3,
            algorithm="direct",
            valid_items=ArgumentBinding.runtime(),
        ).specialization,
        **_block_provider_metadata(),
    )
    assert algo_coalesce_key(load) != algo_coalesce_key(narrower_tile)
    assert load.mangled_name(load.parameters[0]) != narrower_tile.mangled_name(
        narrower_tile.parameters[0]
    )


def test_arbitrary_primitive_declares_bounded_and_exact_value_abis():
    from numba_cuda_mlir import types

    from cuda.coop._core import (
        FLOAT32,
        INT32,
        Algorithm,
        SynchronizationScope,
        Value,
    )
    from cuda.coop.numba_mlir._compiler._operations import StorageABI
    from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter
    from cuda.coop.numba_mlir._types import (
        BoundedInteger,
        ExactValue,
        algo_coalesce_key,
    )
    from cuda.coop.numba_mlir._types import (
        Value as BackendValue,
    )

    specialization = Algorithm(
        struct_name="FuturePrimitive",
        method_name="Run",
        c_name="test_future_primitive",
        includes=(),
        template_parameters=(),
        parameters=(
            (
                Value(INT32, name="remaining"),
                Value(FLOAT32, name="fill"),
            ),
        ),
    ).specialize({}, metadata={"primitive": "not_load_or_store"})
    bounded = BoundedInteger(types.int32, minimum=-2, maximum=7)
    exact = ExactValue(types.float32)
    algorithm = NumbaMlirCoreAdapter(
        value_abis={"remaining": bounded, "fill": exact}
    ).materialize(
        specialization,
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.NONE,
        synchronization_scope=SynchronizationScope.NONE,
    )

    assert algorithm.parameters == [[bounded, exact]]
    assert bounded.provider_dtype == types.int32
    assert bounded.dtype() == types.int64
    assert bounded.accepts_actual_type(types.uint32, None)
    assert not bounded.accepts_actual_type(types.uint64, None)
    assert exact.accepts_actual_type(types.float32, None)
    assert not exact.accepts_actual_type(types.float64, None)

    source = algorithm._source_code()[0]
    guard = "if (param_0 < -2 || param_0 > 7)"
    trap = 'asm volatile("trap;" : : :);'
    narrowing = (
        "::cuda::std::int32_t checked_param_0 = "
        "static_cast<::cuda::std::int32_t>(param_0);"
    )
    assert source.index(guard) < source.index(trap) < source.index(narrowing)
    assert "checked_param_0, param_1);" in source

    wider = NumbaMlirCoreAdapter(
        value_abis={
            "remaining": BoundedInteger(types.int32, minimum=-2, maximum=8),
            "fill": exact,
        }
    ).materialize(
        specialization,
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.NONE,
        synchronization_scope=SynchronizationScope.NONE,
    )
    assert algo_coalesce_key(algorithm) != algo_coalesce_key(wider)
    assert algorithm.mangled_name(algorithm.parameters[0]) != wider.mangled_name(
        wider.parameters[0]
    )

    load_like = Algorithm(
        struct_name="LegacyLikePrimitive",
        method_name="Run",
        c_name="test_load_like_primitive",
        includes=(),
        template_parameters=(),
        parameters=(
            (
                Value(INT32, name="num_valid_items"),
                Value(FLOAT32, name="oob_default"),
            ),
        ),
    ).specialize({}, metadata={"primitive": "load"})
    without_declarations = NumbaMlirCoreAdapter().materialize(
        load_like,
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.NONE,
        synchronization_scope=SynchronizationScope.NONE,
    )
    assert all(
        type(parameter) is BackendValue
        for parameter in without_declarations.parameters[0]
    )


def test_value_abi_declarations_fail_closed_during_materialization():
    from numba_cuda_mlir import types

    from cuda.coop._core import INT32, Algorithm, Pointer, SynchronizationScope, Value
    from cuda.coop.numba_mlir._compiler._operations import StorageABI
    from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter
    from cuda.coop.numba_mlir._types import ExactValue

    def specialization(parameter):
        return Algorithm(
            struct_name="FuturePrimitive",
            method_name="Run",
            c_name="test_value_abi_validation",
            includes=(),
            template_parameters=(),
            parameters=((parameter,),),
        ).specialize({})

    def materialize(parameter, value_abis):
        return NumbaMlirCoreAdapter(value_abis=value_abis).materialize(
            specialization(parameter),
            storage_abi=StorageABI.NONE,
            execution_scope=SynchronizationScope.NONE,
            synchronization_scope=SynchronizationScope.NONE,
        )

    with pytest.raises(ValueError, match="unknown .* value ABI"):
        materialize(Value(INT32, name="value"), {"missing": ExactValue(types.int32)})
    with pytest.raises(ValueError, match="require scalar Value parameters"):
        materialize(
            Pointer(INT32, name="value"),
            {"value": ExactValue(types.int32)},
        )
    with pytest.raises(ValueError, match="provider dtypes do not match"):
        materialize(
            Value(INT32, name="value"),
            {"value": ExactValue(types.float32)},
        )
    with pytest.raises(ValueError, match="output roles do not match"):
        materialize(
            Value(INT32, name="value", is_output=True),
            {"value": ExactValue(types.int32)},
        )
    with pytest.raises(TypeError, match="backend Value instances"):
        materialize(Value(INT32, name="value"), {"value": object()})


def test_runtime_offsets_reject_bool_float_and_uint64_types():
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._types import PointerOffset

    offset = PointerOffset(types.int64)
    assert all(
        offset.accepts_actual_type(dtype, None)
        for dtype in (
            types.int8,
            types.int16,
            types.int32,
            types.int64,
            types.uint8,
            types.uint16,
            types.uint32,
        )
    )
    assert all(
        not offset.accepts_actual_type(dtype, None)
        for dtype in (types.boolean, types.uint64, types.float32, types.float64)
    )


_INTEGER_LITERAL_DTYPES = (
    pytest.param("int8", np.int8, id="int8"),
    pytest.param("uint8", np.uint8, id="uint8"),
    pytest.param("int16", np.int16, id="int16"),
    pytest.param("uint16", np.uint16, id="uint16"),
    pytest.param("int32", np.int32, id="int32"),
    pytest.param("uint32", np.uint32, id="uint32"),
    pytest.param("int64", np.int64, id="int64"),
    pytest.param("uint64", np.uint64, id="uint64"),
)


@pytest.mark.parametrize(
    ("operation", "parameter"),
    [("store", "value"), ("load", "oob_default")],
)
@pytest.mark.parametrize(("dtype_name", "numpy_dtype"), _INTEGER_LITERAL_DTYPES)
def test_python_integer_literal_boundaries_are_checked_against_payload_dtype(
    operation,
    parameter,
    dtype_name,
    numpy_dtype,
):
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._compiler._parameters import coerce_static_scalar

    dtype = getattr(types, dtype_name)
    bounds = np.iinfo(numpy_dtype)
    for value in (int(bounds.min), int(bounds.max)):
        result = coerce_static_scalar(
            value,
            dtype,
            operation=operation,
            parameter=parameter,
        )
        assert isinstance(result, numpy_dtype)
        assert int(result) == value

    for value in (int(bounds.min) - 1, int(bounds.max) + 1):
        with pytest.raises(ValueError, match="outside the range"):
            coerce_static_scalar(
                value,
                dtype,
                operation=operation,
                parameter=parameter,
            )


@pytest.mark.parametrize(
    ("operation", "parameter"),
    [("store", "value"), ("load", "oob_default")],
)
@pytest.mark.parametrize(
    ("dtype_name", "numpy_dtype"),
    [("float32", np.float32), ("float64", np.float64)],
)
def test_python_float_literal_boundaries_are_checked_against_payload_dtype(
    operation,
    parameter,
    dtype_name,
    numpy_dtype,
):
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._compiler._parameters import coerce_static_scalar

    dtype = getattr(types, dtype_name)
    maximum = float(np.finfo(numpy_dtype).max)
    for value in (-maximum, 0, 1, 1.25, maximum):
        result = coerce_static_scalar(
            value,
            dtype,
            operation=operation,
            parameter=parameter,
        )
        assert isinstance(result, numpy_dtype)
        assert np.isfinite(result)

    for value in (float("-inf"), float("inf"), float("nan"), 1 << 1024):
        with pytest.raises(ValueError, match="finite"):
            coerce_static_scalar(
                value,
                dtype,
                operation=operation,
                parameter=parameter,
            )


@pytest.mark.parametrize(
    ("operation", "parameter"),
    [("store", "value"), ("load", "oob_default")],
)
def test_contextual_scalar_conversion_rejects_bool_float_to_int_and_typed_casts(
    operation,
    parameter,
):
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._compiler._parameters import coerce_static_scalar

    with pytest.raises(TypeError, match="must not be bool"):
        coerce_static_scalar(
            True,
            types.int32,
            operation=operation,
            parameter=parameter,
        )
    with pytest.raises(TypeError, match="float-to-integer"):
        coerce_static_scalar(
            1.0,
            types.int32,
            operation=operation,
            parameter=parameter,
        )
    assert coerce_static_scalar(
        np.int32(1),
        types.int32,
        operation=operation,
        parameter=parameter,
    ).dtype == np.dtype(np.int32)
    with pytest.raises(TypeError, match="does not match payload dtype"):
        coerce_static_scalar(
            np.int64(1),
            types.int32,
            operation=operation,
            parameter=parameter,
        )
    assert coerce_static_scalar(
        1,
        types.int32,
        operation=operation,
        parameter=parameter,
        source_dtype=types.int32,
    ).dtype == np.dtype(np.int32)
    with pytest.raises(TypeError, match="does not match payload dtype"):
        coerce_static_scalar(
            1,
            types.int32,
            operation=operation,
            parameter=parameter,
            source_dtype=types.int64,
        )


@pytest.mark.parametrize("qualified", [False, True], ids=["root", "qualified"])
@pytest.mark.parametrize(
    ("dtype_name", "value"),
    [
        pytest.param("uint8", 255, id="uint8-upper"),
        pytest.param("float32", 1, id="int-to-float"),
        pytest.param("float32", 1.25, id="float-rounding"),
        pytest.param("int32", np.int32(-7), id="typed-exact"),
    ],
)
def test_scalar_store_literals_are_typed_from_the_destination(
    qualified,
    dtype_name,
    value,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as root_coop

    module = qualified_coop if qualified else root_coop

    def memory(destination):
        module.store(module.this_block(), destination, value)

    array_type = types.Array(getattr(types, dtype_name), 1, "C")
    _, planner = _plan(memory, arg_types=(array_type,))
    assert planner.run()


@pytest.mark.parametrize("qualified", [False, True], ids=["root", "qualified"])
@pytest.mark.parametrize(
    ("dtype_name", "value", "error"),
    [
        pytest.param("uint8", 256, "outside the range", id="out-of-range"),
        pytest.param("int32", 1.0, "float-to-integer", id="float-to-int"),
        pytest.param("int32", True, "bool", id="bool"),
        pytest.param(
            "int32",
            np.int64(1),
            "does not match payload dtype",
            id="typed-mismatch",
        ),
    ],
)
def test_scalar_store_literals_fail_before_provider_selection(
    monkeypatch,
    qualified,
    dtype_name,
    value,
    error,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as root_coop
    from cuda.coop.numba_mlir._compiler import _group_load_store

    module = qualified_coop if qualified else root_coop

    def memory(destination):
        module.store(module.this_block(), destination, value)

    monkeypatch.setattr(
        _group_load_store._LoadStorePlanning,
        "_scope_factory",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid scalar Store reached provider selection"
        ),
    )
    array_type = types.Array(getattr(types, dtype_name), 1, "C")
    with pytest.raises((TypeError, ValueError), match=error):
        _plan(memory, arg_types=(array_type,))[1].run()


@pytest.mark.parametrize("qualified", [False, True], ids=["root", "qualified"])
@pytest.mark.parametrize(
    ("value_type_name", "accepted"),
    [("int32", True), ("int64", False)],
)
def test_runtime_scalar_store_requires_exact_destination_dtype(
    monkeypatch,
    qualified,
    value_type_name,
    accepted,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as root_coop

    module = qualified_coop if qualified else root_coop

    def memory(destination, value):
        module.store(module.this_block(), destination, value)

    array_type = types.Array(types.int32, 1, "C")
    arg_types = (array_type, getattr(types, value_type_name))
    if accepted:
        _run_single_phase_to_provider_boundary(
            memory,
            arg_types=arg_types,
            monkeypatch=monkeypatch,
            allow_provider_bundling=True,
        )
    else:
        with pytest.raises(TypeError, match="does not match payload dtype"):
            _plan(memory, arg_types=arg_types)[1].run()


@pytest.mark.parametrize("qualified", [False, True], ids=["root", "qualified"])
@pytest.mark.parametrize(
    "source_kind",
    ["index", "element", "numpy-cast", "compiler-cast"],
)
def test_cuda_and_array_scalars_keep_compiler_dtypes_for_store(
    qualified,
    source_kind,
):
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as root_coop

    module = qualified_coop if qualified else root_coop

    def memory(source, destination):
        index = cuda.threadIdx.x
        if source_kind == "index":
            value = index
        elif source_kind == "element":
            value = source[index]
        elif source_kind == "numpy-cast":
            value = np.int32(index + 1)
        else:
            value = types.int32(index + 1)
        module.store(module.this_block(), destination, value)

    array_type = types.Array(types.int32, 1, "C")
    _, planner = _plan(memory, arg_types=(array_type, array_type))
    assert planner.run()


@pytest.mark.parametrize("qualified", [False, True], ids=["root", "qualified"])
def test_runtime_scalar_expression_cannot_narrow_into_store(qualified):
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as root_coop

    module = qualified_coop if qualified else root_coop

    def memory(destination):
        value = cuda.threadIdx.x + 1
        module.store(module.this_block(), destination, value)

    array_type = types.Array(types.int32, 1, "C")
    with pytest.raises(TypeError, match="does not match payload dtype"):
        _plan(memory, arg_types=(array_type,))[1].run()


@pytest.mark.parametrize("qualified", [False, True], ids=["root", "qualified"])
@pytest.mark.parametrize("parameter", ["valid_items", "offset"])
@pytest.mark.parametrize(
    "dtype_name",
    ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32"],
)
def test_runtime_control_integer_domain_is_accepted_before_materialization(
    monkeypatch,
    qualified,
    parameter,
    dtype_name,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as root_coop

    module = qualified_coop if qualified else root_coop

    if parameter == "valid_items":

        def memory(source, control):
            output = module.ThreadData(2, dtype=types.int32)
            return module.load(module.this_block(), source, output, valid_items=control)

    else:

        def memory(source, control):
            output = module.ThreadData(2, dtype=types.int32)
            return module.load(module.this_block(), source, output, offset=control)

    array_type = types.Array(types.int32, 1, "C")
    _run_single_phase_to_provider_boundary(
        memory,
        arg_types=(array_type, getattr(types, dtype_name)),
        monkeypatch=monkeypatch,
        allow_provider_bundling=True,
    )


@pytest.mark.parametrize("qualified", [False, True], ids=["root", "qualified"])
@pytest.mark.parametrize("parameter", ["valid_items", "offset"])
@pytest.mark.parametrize(
    "dtype_name",
    ["boolean", "uint64", "float32", "float64"],
)
def test_runtime_control_invalid_types_fail_before_materialization(
    monkeypatch,
    qualified,
    parameter,
    dtype_name,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as root_coop
    from cuda.coop.numba_mlir._compiler._rewrite import (
        CoopSinglePhaseRewriteError,
    )

    module = qualified_coop if qualified else root_coop

    if parameter == "valid_items":

        def memory(source, control):
            output = module.ThreadData(2, dtype=types.int32)
            return module.load(module.this_block(), source, output, valid_items=control)

    else:

        def memory(source, control):
            output = module.ThreadData(2, dtype=types.int32)
            return module.load(module.this_block(), source, output, offset=control)

    array_type = types.Array(types.int32, 1, "C")
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=r"must be (?:an integer|a signed integer)",
    ):
        _run_single_phase_to_provider_boundary(
            memory,
            arg_types=(array_type, getattr(types, dtype_name)),
            monkeypatch=monkeypatch,
        )


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
    assert load_match.factory_kwargs["oob_default"] == ArgumentBinding.static(
        np.int32(-1)
    )
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
@pytest.mark.parametrize(
    ("algorithm", "error_type"),
    [(True, TypeError), ("stripd", ValueError), (object(), TypeError)],
)
def test_invalid_block_algorithm_values_fail_before_provider_materialization(
    monkeypatch, operation, algorithm, error_type
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
