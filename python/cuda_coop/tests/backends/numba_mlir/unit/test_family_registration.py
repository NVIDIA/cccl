# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


@pytest.fixture(autouse=True)
def _restore_private_registries():
    from cuda.coop._core.api import _dispatch as portable_dispatch
    from cuda.coop._core.group import _dispatch as core_dispatch
    from cuda.coop.numba_mlir._compiler import _operations

    registries = (
        core_dispatch._GROUP_OPERATION_FAMILIES,
        portable_dispatch._PORTABLE_GROUP_OPERATIONS_BY_NAME,
        portable_dispatch._PORTABLE_GROUP_OPERATIONS_BY_FUNCTION,
        _operations._GROUP_OPERATIONS,
        _operations._GROUP_FAMILY_MODULES,
        _operations._FACTORY_OPERATIONS,
        _operations._GROUP_PRIMITIVES,
        _operations._REWRITE_OPERATIONS,
    )
    snapshots = tuple(dict(registry) for registry in registries)
    try:
        yield
    finally:
        for registry, snapshot in zip(registries, snapshots):
            registry.clear()
            registry.update(snapshot)


@dataclass(frozen=True)
class _FakeSemantics:
    token: str

    @property
    def semantic_key(self):
        return ("fake-family", self.token)

    @property
    def result_visibility(self):
        from cuda.coop._core import ResultVisibility

        return ResultVisibility.PER_MEMBER

    @property
    def returns_value(self):
        return True


def test_core_family_registration_dispatches_classification_and_planning():
    from cuda.coop._core import (
        ArgumentKind,
        LaunchFacts,
        ParameterClassification,
        ParameterRole,
        make_group_primitive_call,
        plan_group_primitive,
        this_block,
    )
    from cuda.coop._core.group import _dispatch

    classification = ParameterClassification(
        "value",
        ArgumentKind.RUNTIME,
        ParameterRole.INPUT,
    )
    semantics = _FakeSemantics("registered")
    planned = object()
    events = []

    def classify(operation):
        events.append(("classify", operation))
        return (classification,)

    def plan(call, resolved_group, launch, operation):
        events.append(("plan", call, resolved_group, launch, operation))
        return planned

    _dispatch._register_group_operation_family(
        _FakeSemantics,
        classifications=classify,
        planner=plan,
        group_kinds=frozenset({"block"}),
        unsupported_group_message="fake family requires a block",
    )

    call = make_group_primitive_call(this_block(), semantics)
    launch = LaunchFacts(exact_block_dim=(32, 1, 1))

    assert call.argument_classifications == (classification,)
    assert plan_group_primitive(call, launch) is planned
    assert events[0] == ("classify", semantics)
    _, planned_call, resolved_group, planned_launch, planned_operation = events[1]
    assert planned_call is call
    assert resolved_group.kind == "block"
    assert resolved_group.static_size == 32
    assert resolved_group.hierarchy.block_dim == (32, 1, 1)
    assert planned_launch is launch
    assert planned_operation is semantics


def _register_fake_group_frontends(operation):
    from cuda.coop._core.api import _dispatch as portable_dispatch
    from cuda.coop.numba_mlir._compiler import _operations

    @portable_dispatch._portable_group_operation(
        operation,
        group_kinds=("block",),
    )
    def portable_marker(group, value):
        del group, value

    @_operations.group_operation(operation, family_module=__name__)
    def qualified_marker(group, value):
        del group, value

    return portable_marker, qualified_marker


def test_group_frontend_registries_require_exact_callable_identity():
    from cuda.coop._core.api import _dispatch as portable_dispatch
    from cuda.coop.numba_mlir._compiler import _operations
    from cuda.coop.numba_mlir._compiler._group_planner_support import (
        _group_operation_name,
    )

    operation = "_test_exact_family"
    portable_marker, qualified_marker = _register_fake_group_frontends(operation)

    assert (
        portable_dispatch._portable_group_operation_name(portable_marker) == operation
    )
    assert _operations.group_operation_name(qualified_marker) == operation
    assert _group_operation_name(portable_marker) == operation
    assert _group_operation_name(qualified_marker) == operation

    for marker in (portable_marker, qualified_marker):

        def impostor(group, value):
            del group, value

        impostor.__module__ = marker.__module__
        impostor.__name__ = marker.__name__
        impostor.__cuda_coop_backend_member__ = operation
        assert portable_dispatch._portable_group_operation_name(impostor) is None
        assert _operations.group_operation_name(impostor) is None
        assert _group_operation_name(impostor) is None


@pytest.mark.parametrize("portable", [True, False], ids=["portable", "qualified"])
def test_group_call_planner_selects_registered_family_lowerer(portable):
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop as portable_coop
    import cuda.coop.numba_mlir as qualified_coop
    from cuda.coop.numba_mlir._compiler import _operations
    from cuda.coop.numba_mlir._compiler._group_planner import (
        _GroupCallPlanner,
        has_group_markers,
    )
    from cuda.coop.numba_mlir._compiler._group_planning import (
        GroupPlanningContext,
    )

    operation = "_test_planned_family"
    portable_marker, qualified_marker = _register_fake_group_frontends(operation)
    marker = portable_marker if portable else qualified_marker
    group_factory = portable_coop.this_block if portable else qualified_coop.this_block
    lowered = []

    def lower(
        context,
        inst,
        *,
        operation,
        group,
        bound,
        is_common_root,
    ):
        lowered.append(
            {
                "context": context,
                "operation": operation,
                "group": group,
                "bound": bound,
                "is_common_root": is_common_root,
            }
        )
        return [ir.Assign(ir.Const(17, inst.loc), inst.target, inst.loc)]

    _operations.register_group_primitive(operation, lower=lower)

    def kernel(value):
        return marker(group_factory(), value)

    func_ir = run_frontend(kernel)
    state = SimpleNamespace(func_ir=func_ir, args=(types.int32,))
    planner = _GroupCallPlanner(
        state,
        {"block": (32, 1, 1), "grid": (1, 1, 1), "cluster": None},
    )

    assert has_group_markers(func_ir)
    assert planner.run()
    assert not has_group_markers(func_ir)
    assert len(lowered) == 1
    invocation = lowered[0]
    assert isinstance(invocation["context"], GroupPlanningContext)
    assert invocation["context"] is planner.context
    assert invocation["context"].launch.exact_block_dim == (32, 1, 1)
    assert not hasattr(invocation["context"], "_group_cache")
    assert invocation["operation"] == operation
    assert invocation["group"].kind == "block"
    assert invocation["group"].static_size == 32
    assert invocation["bound"].arguments["group"].name
    assert invocation["bound"].arguments["value"].name == "value"
    assert invocation["is_common_root"] is portable


@pytest.mark.parametrize("portable", [True, False], ids=["portable", "qualified"])
@pytest.mark.parametrize(
    ("result_index", "expected_dtype", "expected_array", "expected_extent"),
    [
        (0, "int32", False, 1),
        (1, "float32", True, 3),
    ],
    ids=["scalar", "array"],
)
def test_registered_result_sources_drive_tuple_provenance(
    portable,
    result_index,
    expected_dtype,
    expected_array,
    expected_extent,
):
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop as portable_coop
    import cuda.coop.numba_mlir as qualified_coop
    from cuda.coop._core.api import _dispatch as portable_dispatch
    from cuda.coop.numba_mlir._compiler import _operations
    from cuda.coop.numba_mlir._compiler._group_planner import _GroupCallPlanner

    operation = "_test_result_family"

    @portable_dispatch._portable_group_operation(
        operation,
        group_kinds=("block",),
    )
    def portable_marker(group, keys, values):
        del group, keys, values

    @_operations.group_operation(operation, family_module=__name__)
    def qualified_marker(group, keys, values):
        del group, keys, values

    _operations.register_group_primitive(
        operation,
        lower=lambda *args, **kwargs: [],
        results=(
            _operations.GroupResultSource("keys", None),
            _operations.GroupResultSource("values", "values"),
        ),
    )

    marker = portable_marker if portable else qualified_marker
    module = portable_coop if portable else qualified_coop

    def kernel():
        keys = module.ThreadData(2, dtype=types.int32)
        values = module.ThreadData(3, dtype=types.float32)
        pair = marker(module.this_block(), keys, values)
        return pair[result_index]

    func_ir = run_frontend(kernel)
    planner = _GroupCallPlanner(
        SimpleNamespace(func_ir=func_ir, args=()),
        {"block": (32, 1, 1), "grid": (1, 1, 1), "cluster": None},
    )
    return_value = next(
        statement.value
        for block in func_ir.blocks.values()
        for statement in block.body
        if isinstance(statement, ir.Return)
    )

    assert str(planner.context.dtype(return_value)) == expected_dtype
    assert planner.context.is_array(operation, return_value) is expected_array
    assert planner.context.array_extent(return_value) == expected_extent


def test_group_result_source_rejects_invalid_parameter_names():
    from cuda.coop.numba_mlir._compiler._operations import GroupResultSource

    with pytest.raises(ValueError, match="dtype_parameter"):
        GroupResultSource("", None)
    with pytest.raises(ValueError, match="array_parameter"):
        GroupResultSource(None, 7)


@pytest.mark.parametrize(
    ("struct_name", "scope_name", "sync_token"),
    [
        ("WarpNamedButBlockScoped", "block", "__syncthreads();"),
        ("BlockNamedButWarpScoped", "warp", "__syncwarp();"),
        ("NeutralProvider", "none", None),
    ],
)
def test_generated_synchronization_uses_metadata_not_struct_names(
    struct_name,
    scope_name,
    sync_token,
):
    from numba_cuda_mlir import types

    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir import _types
    from cuda.coop.numba_mlir._compiler._operations import StorageABI

    scope = SynchronizationScope(scope_name)
    algorithm = _types.Algorithm(
        struct_name=struct_name,
        method_name="Run",
        c_name="test_declarative_provider_metadata",
        includes=(),
        template_parameters=(),
        parameters=((_types.Pointer(types.uint8), _types.Value(types.int32)),),
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=scope,
        synchronization_scope=scope,
    )
    if scope is SynchronizationScope.WARP:
        algorithm.threads = 32
        algorithm.block_threads = 64

    source = algorithm._source_code(
        compile_identity=(90, True, "lto", (), "test-toolchain")
    )[0]

    for token in ("__syncthreads();", "__syncwarp();"):
        assert (token in source) is (token == sync_token)
    if scope is SynchronizationScope.WARP:
        assert "temp_storages[2]" in source
        assert "[__coop_thread_rank / 32]" in source
    assert scope.value in repr(
        algorithm._make_lto_ir_cache_key(
            compile_identity=(90, True, "lto", (), "test-toolchain")
        )
    )


def test_group_synchronization_scope_fails_with_stable_diagnostic():
    from numba_cuda_mlir import types

    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir import _types
    from cuda.coop.numba_mlir._compiler._operations import StorageABI

    algorithm = _types.Algorithm(
        struct_name="Provider",
        method_name="Run",
        c_name="test_unsupported_group_synchronization",
        includes=(),
        template_parameters=(),
        parameters=((_types.Pointer(types.uint8), _types.Value(types.int32)),),
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.GROUP,
        synchronization_scope=SynchronizationScope.GROUP,
    )

    with pytest.raises(NotImplementedError, match="scope 'group' has no emitter"):
        algorithm._source_code(compile_identity=(90, True, "lto", (), "test-toolchain"))


def test_storage_free_provider_uses_default_constructor_and_zero_storage():
    from numba_cuda_mlir import types

    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir import _types
    from cuda.coop.numba_mlir._compiler._operations import StorageABI

    algorithm = _types.Algorithm(
        struct_name="StorageFreeProvider",
        method_name="Run",
        c_name="test_storage_free_provider",
        includes=(),
        template_parameters=(),
        parameters=((_types.Value(types.int32),),),
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.NONE,
    )
    source, _, storage_symbols, _ = algorithm._source_code(
        compile_identity=(90, True, "lto", (), "test-toolchain")
    )

    assert "StorageFreeProvider().Run" not in source
    assert "algorithm_t_" in source
    assert "().Run(param_0);" in source
    assert "TempStorage" not in source
    assert "temp_storage" not in source
    assert storage_symbols == ()


class _FakeInvocable:
    files = ("family-registration-test.ltoir",)
    specialization = None
    temp_storage_bytes = 24
    temp_storage_alignment = 8

    def __init__(self, scope):
        from cuda.coop.numba_mlir._compiler._operations import StorageABI

        self.storage_abi = StorageABI.LEADING_POINTER
        self.execution_scope = scope
        self.synchronization_scope = scope

    def __call__(self, *args):
        del args


class _StorageFreeInvocable:
    files = ("storage-free-family-registration-test.ltoir",)
    specialization = None
    temp_storage_bytes = 0
    temp_storage_alignment = 1
    storage_abi = "none"
    execution_scope = "none"
    synchronization_scope = "none"

    def __call__(self, *args):
        del args


class _TypingContext:
    def __init__(self):
        self.refresh_count = 0

    def refresh(self):
        self.refresh_count += 1


def _resolved_calls(func_ir):
    from numba_cuda_mlir.numbair_transforms import ir

    from cuda.coop.numba_mlir._compiler._rewrite import CoopSinglePhaseRewrite

    resolver = object.__new__(CoopSinglePhaseRewrite)
    resolver._func_ir = func_ir
    calls = []
    for block in func_ir.blocks.values():
        resolver._block_defs = {
            inst.target.name: inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
        }
        for inst in block.body:
            value = getattr(inst, "value", None)
            if isinstance(value, ir.Expr) and value.op == "call":
                calls.append((resolver._resolve_python_value(value.func), value))
    return calls


@pytest.mark.parametrize(
    ("scope_name", "sync_name"),
    [
        ("block", "syncthreads"),
        ("warp", "syncwarp"),
        ("none", None),
    ],
)
def test_registered_rewrite_callbacks_drive_generic_storage_rewrite(
    scope_name,
    sync_name,
):
    import numpy as np
    from numba_cuda_mlir import cuda, types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir._compiler import _operations
    from cuda.coop.numba_mlir._compiler._group_rewriting import (
        GroupRewriteContext,
    )
    from cuda.coop.numba_mlir._compiler._rewrite import CoopSinglePhaseRewrite

    operation = "_test_rewrite_family"
    synchronization_scope = SynchronizationScope(scope_name)
    events = []
    contexts = []
    family_metadata = object()
    invocable = _FakeInvocable(synchronization_scope)

    def provider(*runtime_args, **factory_kwargs):
        assert not runtime_args
        events.append(("factory", dict(factory_kwargs)))
        return invocable

    def record_context(context):
        assert isinstance(context, GroupRewriteContext)
        assert not hasattr(context, "_matches")
        assert not hasattr(context, "_temp_storage_global_plan")
        contexts.append(context)

    def infer_payload(context, inference):
        record_context(context)
        events.append(("infer", tuple(inference.runtime_args)))
        inference.infer_kwarg("inferred", "from-callback")
        inference.infer_kwarg("element_type", np.dtype("int32"))

    def analyze_match(
        context,
        *,
        op_name,
        runtime_args,
        factory_kwargs,
    ):
        record_context(context)
        events.append(
            (
                "analyze",
                op_name,
                tuple(runtime_args),
                dict(factory_kwargs),
            )
        )
        return family_metadata

    def prepare_runtime_args(
        context,
        block,
        *,
        match,
        runtime_args,
        scope,
        loc,
    ):
        record_context(context)
        assert match.family_metadata is family_metadata
        events.append(("prepare", tuple(runtime_args)))
        prepared = ir.Var(scope, "__family_prepared_value", loc)
        block.append(ir.Assign(ir.Const(29, loc), prepared, loc))
        return [*runtime_args, prepared]

    def validate_runtime_controls(
        context,
        *,
        op_name,
        runtime_args,
        factory_kwargs,
    ):
        record_context(context)
        events.append(
            (
                "validate",
                op_name,
                tuple(runtime_args),
                dict(factory_kwargs),
            )
        )

    _operations.register_factory(
        provider,
        operation=operation,
        namespace="alternate",
        storage_abi=_operations.StorageABI.LEADING_POINTER,
        execution_scope=synchronization_scope,
        synchronization_scope=synchronization_scope,
    )
    _operations.register_rewrite_operation(
        operation,
        _operations.RewriteOperationSpec(
            factory_namespaces=frozenset({"block", "alternate"}),
            dtype_factory_kwargs=frozenset({"element_type"}),
            runtime_arg_counts=frozenset({1}),
            runtime_factory_kwargs=(),
            runtime_factory_kw_prerequisites=(),
            allowed_factory_kwargs=frozenset({"element_type", "inferred", "token"}),
            required_factory_kwargs=frozenset({"element_type", "inferred", "token"}),
            accepts_temp_storage=False,
            scalar_binding_kwargs=frozenset(),
            runtime_offset_kwarg=None,
            infer_payload=infer_payload,
            analyze_match=analyze_match,
            prepare_runtime_args=prepare_runtime_args,
            validate_runtime_controls=validate_runtime_controls,
        ),
    )

    def kernel(value):
        return provider(value, token=7, element_type=value.dtype)

    func_ir = run_frontend(kernel)
    typingctx = _TypingContext()
    state = SimpleNamespace(
        func_ir=func_ir,
        args=(types.Array(types.int32, 1, "C"),),
        typingctx=typingctx,
        typemap={},
        calltypes={},
        metadata={},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    for label in sorted(func_ir.blocks):
        block = func_ir.blocks[label]
        while rewrite.match(func_ir, block, state.typemap, state.calltypes):
            block = rewrite.apply()
            func_ir.blocks[label] = block

    calls = _resolved_calls(func_ir)
    invocable_calls = [call for target, call in calls if target is invocable]
    assert len(invocable_calls) == 1
    assert len(invocable_calls[0].args) == 3
    assert sum(target is cuda.shared.array for target, _ in calls) == 1
    sync_targets = {
        target for target, _ in calls if target in {cuda.syncthreads, cuda.syncwarp}
    }
    assert sync_targets == (set() if sync_name is None else {getattr(cuda, sync_name)})
    assert rewrite._temp_storage_global_plan.total_size == 24
    assert rewrite._temp_storage_global_plan.max_alignment == 8
    assert rewrite._implicit_temp_storage_plan.size_in_bytes == 24
    assert rewrite._implicit_temp_storage_plan.alignment == 8
    assert typingctx.refresh_count == 1

    factory_events = [event for event in events if event[0] == "factory"]
    infer_events = [event for event in events if event[0] == "infer"]
    analyze_events = [event for event in events if event[0] == "analyze"]
    prepare_events = [event for event in events if event[0] == "prepare"]
    validate_events = [event for event in events if event[0] == "validate"]
    assert factory_events == [
        (
            "factory",
            {
                "token": 7,
                "inferred": "from-callback",
                "element_type": types.int32,
            },
        )
    ]
    assert len(infer_events) == 2
    assert all(len(event[1]) == 1 for event in infer_events)
    assert len(analyze_events) == 2
    assert all(event[1] == operation for event in analyze_events)
    assert all(
        event[3]
        == {
            "token": 7,
            "inferred": "from-callback",
            "element_type": types.int32,
        }
        for event in analyze_events
    )
    assert len(prepare_events) == 1
    assert len(validate_events) == 2
    assert all(event[1] == operation for event in validate_events)
    assert len(contexts) == 7

    resolver = object.__new__(CoopSinglePhaseRewrite)
    resolver._func_ir = func_ir
    resolver._block_defs = {
        inst.target.name: inst.value
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign)
    }
    assert resolver._infer_constant(invocable_calls[0].args[-1]) == 29


def test_storage_free_provider_accepts_unused_temp_storage_descriptor():
    from numba_cuda_mlir import cuda, types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    import cuda.coop.numba_mlir as coop
    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir._compiler import _operations
    from cuda.coop.numba_mlir._compiler._rewrite import CoopSinglePhaseRewrite

    operation = "_test_storage_free_family"
    invocable = _StorageFreeInvocable()

    def provider(*runtime_args, **factory_kwargs):
        assert not runtime_args
        assert not factory_kwargs
        return invocable

    _operations.register_factory(
        provider,
        operation=operation,
        namespace="alternate",
        storage_abi=_operations.StorageABI.NONE,
        execution_scope=SynchronizationScope.NONE,
        synchronization_scope=SynchronizationScope.NONE,
    )
    _operations.register_rewrite_operation(
        operation,
        _operations.RewriteOperationSpec(
            factory_namespaces=frozenset({"alternate"}),
            dtype_factory_kwargs=frozenset(),
            runtime_arg_counts=frozenset({1}),
            runtime_factory_kwargs=(),
            runtime_factory_kw_prerequisites=(),
            allowed_factory_kwargs=frozenset(),
            required_factory_kwargs=frozenset(),
            accepts_temp_storage=True,
            scalar_binding_kwargs=frozenset(),
            runtime_offset_kwarg=None,
            infer_payload=lambda *_args: None,
        ),
    )

    def kernel(value):
        storage = coop.TempStorage()
        return provider(value, temp_storage=storage)

    func_ir = run_frontend(kernel)
    typingctx = _TypingContext()
    state = SimpleNamespace(
        func_ir=func_ir,
        args=(types.int32,),
        typingctx=typingctx,
        typemap={},
        calltypes={},
        metadata={},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    for label in sorted(func_ir.blocks):
        block = func_ir.blocks[label]
        while rewrite.match(func_ir, block, state.typemap, state.calltypes):
            block = rewrite.apply()
            func_ir.blocks[label] = block

    calls = _resolved_calls(func_ir)
    invocable_calls = [call for target, call in calls if target is invocable]
    assert len(invocable_calls) == 1
    assert len(invocable_calls[0].args) == 1
    assert all(target is not cuda.shared.array for target, _ in calls)
    assert all(target not in {cuda.syncthreads, cuda.syncwarp} for target, _ in calls)
    assert rewrite._temp_storage_global_plan is None
    assert rewrite._temp_storage_backing_var is None


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("storage_abi", "leading_pointer"),
        ("execution_scope", "block"),
        ("synchronization_scope", "block"),
    ],
)
def test_provider_metadata_must_match_registered_rewrite_contract(
    attribute,
    value,
):
    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir._compiler import _operations
    from cuda.coop.numba_mlir._compiler._rewrite_invocables import _InvocableRewrite
    from cuda.coop.numba_mlir._compiler._rewrite_support import (
        CoopSinglePhaseRewriteError,
    )

    provider_metadata = _operations.FactoryOperation(
        operation="_test_provider_contract_mismatch",
        namespace="alternate",
        storage_abi=_operations.StorageABI.NONE,
        execution_scope=SynchronizationScope.NONE,
        synchronization_scope=SynchronizationScope.NONE,
    )
    invocable = _StorageFreeInvocable()
    setattr(invocable, attribute, value)

    with pytest.raises(CoopSinglePhaseRewriteError, match=attribute):
        _InvocableRewrite._validate_invocable(invocable, provider_metadata)
