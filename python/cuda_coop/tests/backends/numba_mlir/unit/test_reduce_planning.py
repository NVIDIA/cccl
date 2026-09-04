# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from inspect import signature
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


def _plan(function, *, arg_types=(), block=(64, 1, 1), grid=(1, 1, 1), cluster=None):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._compiler._group_planner import _GroupCallPlanner

    func_ir = run_frontend(function)
    planner = _GroupCallPlanner(
        SimpleNamespace(func_ir=func_ir, args=arg_types),
        {"block": block, "grid": grid, "cluster": cluster},
    )
    return func_ir, planner


def _planned_factory_calls(func_ir):
    from numba_cuda_mlir.numbair_transforms import ir

    globals_by_name = {
        statement.target.name: statement.value.value
        for block in func_ir.blocks.values()
        for statement in block.body
        if isinstance(statement, ir.Assign) and isinstance(statement.value, ir.Global)
    }
    return [
        (globals_by_name.get(statement.value.func.name), statement.value)
        for block in func_ir.blocks.values()
        for statement in block.body
        if isinstance(statement, ir.Assign)
        and isinstance(statement.value, ir.Expr)
        and statement.value.op == "call"
    ]


def _provider_call(func_ir, provider):
    calls = [
        call for target, call in _planned_factory_calls(func_ir) if target is provider
    ]
    assert len(calls) == 1
    return calls[0]


def _kwarg_value(func_ir, call, name):
    from numba_cuda_mlir.numbair_transforms import ir

    variable = dict(call.kws)[name]
    definitions = [
        statement.value
        for block in func_ir.blocks.values()
        for statement in block.body
        if isinstance(statement, ir.Assign) and statement.target.name == variable.name
    ]
    assert len(definitions) == 1
    definition = definitions[0]
    assert isinstance(definition, (ir.Const, ir.Global))
    return definition.value


def _match_before_inference(func_ir, *, arg_types):
    from cuda.coop.numba_mlir._compiler._rewrite import CoopSinglePhaseRewrite

    state = SimpleNamespace(
        func_ir=func_ir,
        args=arg_types,
        typemap={},
        calltypes={},
        metadata={},
        typingctx=SimpleNamespace(refresh=lambda: None),
    )
    rewrite = CoopSinglePhaseRewrite(state)
    rewrite._prepare_ltoir_bundle_for_matches = lambda _matches: None

    def materialize(match):
        metadata = match.factory_metadata
        storage = metadata.storage_abi.value == "leading_pointer"
        return (
            SimpleNamespace(
                files=("reduce-test.ltoir",),
                temp_storage_bytes=64 if storage else 0,
                temp_storage_alignment=16 if storage else 1,
                storage_abi=metadata.storage_abi,
                execution_scope=metadata.execution_scope,
                synchronization_scope=metadata.synchronization_scope,
            ),
            False,
        )

    rewrite._materialize_invocable = materialize
    matched = False
    for label in sorted(func_ir.blocks):
        matched |= rewrite.match(
            func_ir,
            func_ir.blocks[label],
            state.typemap,
            state.calltypes,
        )
    assert matched
    return rewrite


def test_reduce_registers_scalar_result_and_all_provider_abis():
    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir._compiler import _group_reduce
    from cuda.coop.numba_mlir._compiler._operations import (
        GroupResultSource,
        StorageABI,
        factory_operation,
        group_primitive,
        rewrite_operation,
    )
    from cuda.coop.numba_mlir._lowering import _reduce

    del _group_reduce
    assert group_primitive("reduce").results == (GroupResultSource("value", None),)
    assert group_primitive("sum").results == (GroupResultSource("value", None),)
    expected = {
        _reduce.sum: ("block_sum", "block", StorageABI.LEADING_POINTER, "block"),
        _reduce.block_reduce_builtin: (
            "block_reduce_builtin",
            "block",
            StorageABI.LEADING_POINTER,
            "block",
        ),
        _reduce.reduce: (
            "block_reduce_callback",
            "block",
            StorageABI.LEADING_POINTER,
            "block",
        ),
        _reduce.warp_sum: (
            "warp_sum",
            "warp",
            StorageABI.LEADING_POINTER,
            "warp",
        ),
        _reduce.warp_reduce_builtin: (
            "warp_reduce_builtin",
            "warp",
            StorageABI.LEADING_POINTER,
            "warp",
        ),
        _reduce.warp_reduce: (
            "warp_reduce_callback",
            "warp",
            StorageABI.LEADING_POINTER,
            "warp",
        ),
        _reduce.group_reduce_none: (
            "group_reduce",
            "cudax_none",
            StorageABI.NONE,
            "none",
        ),
        _reduce.group_reduce_warp: (
            "group_reduce",
            "cudax_warp",
            StorageABI.NONE,
            "warp",
        ),
        _reduce.group_reduce_block: (
            "group_reduce",
            "cudax_block",
            StorageABI.NONE,
            "block",
        ),
        _reduce.group_reduce_group: (
            "group_reduce",
            "cudax_group",
            StorageABI.NONE,
            "group",
        ),
    }
    for factory, (operation, namespace, storage_abi, scope) in expected.items():
        metadata = factory_operation(factory)
        assert metadata.operation == operation
        assert metadata.namespace == namespace
        assert metadata.storage_abi is storage_abi
        assert metadata.execution_scope is SynchronizationScope(scope)
        assert metadata.synchronization_scope is (
            SynchronizationScope(scope)
            if storage_abi is StorageABI.LEADING_POINTER
            else SynchronizationScope.NONE
        )

    assert rewrite_operation("block_sum").runtime_arg_counts == frozenset({1, 2})
    assert rewrite_operation("warp_sum").runtime_arg_counts == frozenset({1, 2})
    assert rewrite_operation("group_reduce").factory_namespaces == frozenset(
        {"cudax_block", "cudax_group", "cudax_none", "cudax_warp"}
    )


def test_public_reduce_markers_have_group_first_signatures():
    import cuda.coop.numba_mlir as qualified
    from cuda import coop as portable

    expected_reduce = (
        "group",
        "value",
        "binary_op",
        "broadcast",
        "valid_items",
        "algorithm",
    )
    expected_sum = ("group", "value", "broadcast", "valid_items", "algorithm")
    assert tuple(signature(qualified.reduce).parameters) == expected_reduce
    assert tuple(signature(portable.reduce).parameters) == expected_reduce
    assert tuple(signature(qualified.sum).parameters) == expected_sum
    assert tuple(signature(portable.sum).parameters) == expected_sum


@pytest.mark.parametrize(
    ("group", "provider_name", "launch"),
    [
        pytest.param("thread", "group_reduce_none", {}, id="thread"),
        pytest.param("warp", "group_reduce_warp", {}, id="warp"),
        pytest.param("logical_warp", "group_reduce_warp", {}, id="logical-warp"),
        pytest.param("block", "group_reduce_block", {}, id="block"),
        pytest.param("mapped_warps", "group_reduce_group", {}, id="mapped-warps"),
        pytest.param(
            "cluster",
            "group_reduce_group",
            {"grid": (2, 1, 1), "cluster": (2, 1, 1)},
            id="cluster",
        ),
    ],
)
def test_full_builtin_reduce_selects_fixed_scope_cudax_provider(
    group,
    provider_name,
    launch,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._lowering import _reduce

    descriptor = {
        "thread": coop.this_thread(),
        "warp": coop.this_warp(),
        "logical_warp": coop.this_warp().group_by(8),
        "block": coop.this_block(),
        "mapped_warps": coop.this_block().group_by(2),
        "cluster": coop.this_cluster(),
    }[group]

    def kernel(value):
        return coop.reduce(descriptor, value, binary_op="max", broadcast=False)

    func_ir, planner = _plan(kernel, arg_types=(types.int32,), **launch)
    assert planner.run()
    provider = getattr(_reduce, provider_name)
    call = _provider_call(func_ir, provider)
    assert len(call.args) == 1
    assert _kwarg_value(func_ir, call, "value_kind") == "scalar"
    assert _kwarg_value(func_ir, call, "binary_op") == "max"
    assert _kwarg_value(func_ir, call, "broadcast") is False


def test_extent_one_thread_data_preserves_array_abi_through_factory_boundary(
    monkeypatch,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop._core import Array as CoreArray
    from cuda.coop.numba_mlir._lowering import _reduce

    def kernel(value):
        items = coop.ThreadData(1, dtype=types.int32)
        items[0] = value
        return coop.sum(
            coop.this_block(),
            items,
            broadcast=False,
            algorithm="raking",
        )

    func_ir, planner = _plan(kernel, arg_types=(types.int32,))
    assert planner.run()
    call = _provider_call(func_ir, _reduce.sum)
    assert _kwarg_value(func_ir, call, "items_per_thread") == 1
    assert _kwarg_value(func_ir, call, "value_kind") == "array"

    materialized = []
    monkeypatch.setattr(
        _reduce.NumbaMlirCoreAdapter,
        "materialize",
        lambda _self, specialization, **kwargs: (
            materialized.append((specialization, kwargs)) or specialization
        ),
    )
    monkeypatch.setattr(
        _reduce,
        "make_invocable_from_specialization",
        lambda specialization: specialization,
    )
    specialization = _reduce.sum(
        types.int32,
        threads_per_block=(64, 1, 1),
        items_per_thread=1,
        value_kind="array",
        algorithm="raking",
    )
    assert any(
        isinstance(parameter, CoreArray)
        for method in specialization.parameters
        for parameter in method
    )
    assert len(materialized) == 1


@pytest.mark.parametrize(
    ("kind", "operation", "provider_name"),
    [
        ("block", "sum", "sum"),
        ("warp", "sum", "warp_sum"),
        ("block", "max", "block_reduce_builtin"),
        ("warp", "max", "warp_reduce_builtin"),
        ("block", "callback", "reduce"),
        ("warp", "callback", "warp_reduce"),
    ],
)
def test_direct_cub_reduce_selects_operation_and_scope(
    kind,
    operation,
    provider_name,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._lowering import _reduce

    descriptor = coop.this_block() if kind == "block" else coop.this_warp()

    def callback(lhs, rhs):
        return lhs + rhs

    binary_op = callback if operation == "callback" else operation

    def kernel(value):
        return coop.reduce(
            descriptor,
            value,
            binary_op=binary_op,
            broadcast=False,
            valid_items=7,
        )

    func_ir, planner = _plan(kernel, arg_types=(types.int32,))
    assert planner.run()
    provider = getattr(_reduce, provider_name)
    call = _provider_call(func_ir, provider)
    assert len(call.args) == 1
    if kind == "warp":
        assert "items_per_thread" not in dict(call.kws)
        assert "value_kind" not in dict(call.kws)
    valid_name = "num_valid" if kind == "block" else "valid_items"
    assert _kwarg_value(func_ir, call, valid_name).value == 7


@pytest.mark.parametrize("fallback", ("valid-prefix", "custom-operator"))
def test_complete_nonexhaustive_logical_warp_materializes_cub_storage(fallback):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir._lowering import _reduce

    descriptor = coop.this_warp().group_by(8, exhaustive=False)

    def callback(lhs, rhs):
        return lhs + rhs

    if fallback == "valid-prefix":

        def kernel(value):
            return coop.sum(
                descriptor,
                value,
                broadcast=False,
                valid_items=5,
            )

    else:

        def kernel(value):
            return coop.reduce(
                descriptor,
                value,
                binary_op=callback,
                broadcast=False,
            )

    func_ir, planner = _plan(kernel, arg_types=(types.int32,))
    assert planner.run()
    provider = _reduce.warp_sum if fallback == "valid-prefix" else _reduce.warp_reduce
    assert len(_provider_call(func_ir, provider).args) == 1
    rewrite = _match_before_inference(func_ir, arg_types=(types.int32,))

    requirements = rewrite._implicit_temp_storage_requirements
    assert len(requirements.uses) == 1
    lowering_plan = requirements.uses[0].lowering_plan
    assert lowering_plan.resolved_group.complete_membership is True
    assert lowering_plan.topology.instance_index == "linear_thread_rank / 8"
    assert lowering_plan.topology.thread_rank == "linear_thread_rank % 8"
    assert lowering_plan.temp_storage.instances == 8
    assert lowering_plan.temp_storage.instance_index == "linear_thread_rank / 8"
    assert (
        requirements.uses[0].lowering_plan.synchronization.storage_reuse_barrier
        is SynchronizationScope.WARP
    )

    storage_plan = rewrite._ensure_temp_storage_global_plan()
    implicit_plan = rewrite._implicit_temp_storage_plan
    assert storage_plan.total_size == 8 * 64
    assert implicit_plan.size_in_bytes == 8 * 64
    assert implicit_plan.alignment == 16
    storage_slice = next(iter(implicit_plan.slices_by_call_id.values()))
    assert storage_slice.instances == 8
    assert storage_slice.stride == 64


def test_runtime_valid_items_is_checked_before_an_int64_provider_cast():
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._lowering import _reduce

    def kernel(value, valid_items):
        return coop.sum(
            coop.this_block(),
            value,
            broadcast=False,
            valid_items=valid_items,
        )

    func_ir, planner = _plan(
        kernel,
        arg_types=(types.int32, types.uint32),
    )
    assert planner.run()
    call = _provider_call(func_ir, _reduce.sum)
    assert "reduce_valid_items_i64" in dict(call.kws)["num_valid"].name
    assert any(target is types.int64 for target, _ in _planned_factory_calls(func_ir))
    _match_before_inference(
        func_ir,
        arg_types=(types.int32, types.uint32),
    )


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("boolean", id="bool"),
        pytest.param("float32", id="float"),
        pytest.param("uint64", id="uint64"),
    ],
)
def test_runtime_valid_items_rejects_invalid_dtype_before_provider(dtype, monkeypatch):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler import _group_reduce

    def kernel(value, valid_items):
        return coop.sum(
            coop.this_block(),
            value,
            broadcast=False,
            valid_items=valid_items,
        )

    monkeypatch.setattr(
        _group_reduce._ReducePlanning,
        "_provider",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid valid_items reached provider selection"
        ),
    )
    _, planner = _plan(
        kernel,
        arg_types=(types.int32, getattr(types, dtype)),
    )
    with pytest.raises(TypeError, match="must be an integer|unsigned integer"):
        planner.run()


@pytest.mark.parametrize("valid_items", [True, 0, 65])
def test_static_valid_items_rejects_before_provider(valid_items, monkeypatch):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler import _group_reduce

    def kernel(value):
        return coop.sum(
            coop.this_block(),
            value,
            broadcast=False,
            valid_items=valid_items,
        )

    monkeypatch.setattr(
        _group_reduce._ReducePlanning,
        "_provider",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid valid_items reached provider selection"
        ),
    )
    _, planner = _plan(kernel, arg_types=(types.int32,))
    with pytest.raises((TypeError, ValueError), match="valid_items"):
        planner.run()


def test_float_bitwise_reduce_rejects_before_provider(monkeypatch):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler import _group_reduce

    def kernel(value):
        return coop.reduce(coop.this_block(), value, binary_op="bit_or")

    monkeypatch.setattr(
        _group_reduce._ReducePlanning,
        "_provider",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid bitwise dtype reached provider selection"
        ),
    )
    _, planner = _plan(kernel, arg_types=(types.float32,))
    with pytest.raises(TypeError, match="requires an integer dtype"):
        planner.run()


def test_grid_reduce_has_stable_workspace_diagnostic():
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    def kernel(value):
        return coop.sum(coop.this_grid(), value)

    _, planner = _plan(kernel, arg_types=(types.int32,))
    with pytest.raises(NotImplementedError, match="hidden per-launch workspace"):
        planner.run()


def test_qualified_local_array_is_supported_but_portable_rejects_it():
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as qualified
    from cuda import coop as portable

    def qualified_kernel(value):
        items = cuda.local.array(2, dtype=types.int32)
        items[0] = value
        items[1] = value
        return qualified.sum(qualified.this_block(), items)

    def portable_kernel(value):
        items = cuda.local.array(2, dtype=types.int32)
        items[0] = value
        items[1] = value
        return portable.sum(portable.this_block(), items)

    _, qualified_planner = _plan(qualified_kernel, arg_types=(types.int32,))
    assert qualified_planner.run()
    _, portable_planner = _plan(portable_kernel, arg_types=(types.int32,))
    with pytest.raises(TypeError, match="ThreadData value payload"):
        portable_planner.run()


def test_cub_factories_declare_leading_storage_and_scope(monkeypatch):
    from numba_cuda_mlir import types

    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir._compiler._operations import StorageABI
    from cuda.coop.numba_mlir._lowering import _reduce

    materializations = []
    monkeypatch.setattr(
        _reduce.NumbaMlirCoreAdapter,
        "materialize",
        lambda _self, specialization, **kwargs: (
            materializations.append(kwargs) or specialization
        ),
    )
    monkeypatch.setattr(
        _reduce,
        "make_invocable_from_specialization",
        lambda specialization, **kwargs: (specialization, kwargs),
    )
    _, block_invocation = _reduce.sum(types.int32, threads_per_block=64)
    _, warp_invocation = _reduce.warp_sum(
        types.int32,
        threads_in_warp=8,
        threads_per_block=(64, 1, 1),
    )
    assert materializations == [
        {
            "storage_abi": StorageABI.LEADING_POINTER,
            "execution_scope": SynchronizationScope.BLOCK,
            "synchronization_scope": SynchronizationScope.BLOCK,
            "extra_type_definitions": materializations[0]["extra_type_definitions"],
        },
        {
            "storage_abi": StorageABI.LEADING_POINTER,
            "execution_scope": SynchronizationScope.WARP,
            "synchronization_scope": SynchronizationScope.WARP,
            "extra_type_definitions": materializations[1]["extra_type_definitions"],
        },
    ]
    assert block_invocation == {}
    assert warp_invocation == {
        "threads": 8,
        "block_threads": (64, 1, 1),
    }


def test_cudax_source_has_required_macros_and_no_external_barrier():
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop._core import LaunchFacts, resolve_thread_group
    from cuda.coop.numba_mlir._lowering import _reduce

    group = resolve_thread_group(
        coop.this_block().group_by(2),
        LaunchFacts(exact_block_dim=(64, 1, 1)),
    ).group
    source = _reduce.render_group_reduce_source(
        group=group,
        dtype=types.int32,
        items_per_thread=1,
        value_kind="array",
        operation="sum",
        broadcast=False,
        symbol="mapped_reduce",
    )
    assert source.index("_CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX") < source.index(
        "#include"
    )
    assert source.index("_CUDAX_DISABLE_COOPERATIVE_GROUPS_INTEROP") < source.index(
        "#include"
    )
    assert "::cuda::experimental::group group{" in source
    assert "reinterpret_cast<::cuda::std::int32_t (*)[1]>" in source
    assert "value_or" in source
    assert "group.sync" not in source
    assert "bar.sync" not in source
    assert "TempStorage" not in source


def test_mapped_cudax_factory_uses_compiler_cache_not_module_cache(monkeypatch):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop._core import LaunchFacts, resolve_thread_group
    from cuda.coop.numba_mlir._lowering import _reduce

    group = resolve_thread_group(
        coop.this_block().group_by(2),
        LaunchFacts(exact_block_dim=(64, 1, 1)),
    ).group
    created = []
    monkeypatch.setattr(
        _reduce.cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )
    monkeypatch.setattr(
        _reduce._nvrtc,
        "resolve_compile_context",
        lambda: SimpleNamespace(symbol_suffix="test"),
    )
    monkeypatch.setattr(
        _reduce,
        "RawCAbiInvocable",
        lambda **kwargs: created.append(kwargs) or SimpleNamespace(**kwargs),
    )

    first = _reduce.group_reduce_group(
        dtype=types.int32,
        group=group,
        value_kind="scalar",
    )
    second = _reduce.group_reduce_group(
        dtype=types.int32,
        group=group,
        value_kind="scalar",
    )
    array = _reduce.group_reduce_group(
        dtype=types.int32,
        group=group,
        value_kind="array",
    )
    assert first is not second
    assert array is not first
    assert len(created) == 3
    assert created[0]["symbol"] == created[1]["symbol"]
    assert created[0]["symbol"] != created[2]["symbol"]
    assert "warps_within_block" in created[0]["symbol"]
    assert not hasattr(_reduce, "_GROUP_REDUCE_INVOCABLE_CACHE")


def test_planned_cudax_and_cub_calls_match_before_inference():
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    def full(value):
        return coop.sum(coop.this_block(), value)

    def prefix(value, valid_items):
        return coop.sum(
            coop.this_warp(),
            value,
            broadcast=False,
            valid_items=valid_items,
        )

    for function, arg_types in (
        (full, (types.int32,)),
        (prefix, (types.int32, types.int32)),
    ):
        func_ir, planner = _plan(function, arg_types=arg_types)
        assert planner.run()
        _match_before_inference(func_ir, arg_types=arg_types)
