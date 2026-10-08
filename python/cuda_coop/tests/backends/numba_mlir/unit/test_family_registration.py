# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise family registration and rewriting with small test providers.

The fixtures isolate private registries so test families cannot affect later
operations. Fake invocables expose storage and call metadata without compiling
C++ providers.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


@pytest.fixture(autouse=True)
def _restore_private_registries():
    """Restore all registries changed by the test, even after a failure."""
    from cuda.coop._core.api import _dispatch as common_dispatch
    from cuda.coop._core.group import _dispatch as core_dispatch
    from cuda.coop.numba_mlir._compiler import _operations

    registries = (
        core_dispatch._GROUP_OPERATION_FAMILIES,
        common_dispatch._COMMON_GROUP_OPERATIONS_BY_NAME,
        common_dispatch._COMMON_GROUP_OPERATIONS_BY_FUNCTION,
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
    """Provide semantic identity for a temporary operation family."""

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


@pytest.mark.parametrize(
    ("override", "error", "message"),
    [
        (
            {"classifications": None},
            TypeError,
            "classifications must be callable",
        ),
        ({"planner": None}, TypeError, "planner must be callable"),
        (
            {"group_kinds": frozenset()},
            ValueError,
            "group_kinds must not be empty",
        ),
        (
            {"group_kinds": frozenset({"not_a_group"})},
            ValueError,
            "group_kinds contains unsupported values",
        ),
        (
            {"unsupported_group_message": " "},
            ValueError,
            "unsupported_group_message must be a non-empty string",
        ),
    ],
)
def test_core_family_registration_rejects_invalid_contracts(
    override,
    error,
    message,
):
    from cuda.coop._core.group import _dispatch

    arguments = {
        "classifications": lambda operation: (),
        "planner": lambda call, group, launch, operation: object(),
        "group_kinds": frozenset({"block"}),
        "unsupported_group_message": "test family requires a block",
    }
    arguments.update(override)

    with pytest.raises(error, match=message):
        _dispatch._register_group_operation_family(_FakeSemantics, **arguments)


def test_group_primitive_registration_rejects_noncallable_hooks():
    from cuda.coop.numba_mlir._compiler._operations import (
        GroupPrimitiveRegistration,
    )

    with pytest.raises(TypeError, match="lower must be callable"):
        GroupPrimitiveRegistration(lower=None)
    with pytest.raises(
        TypeError,
        match="validate_common_arguments must be callable or None",
    ):
        GroupPrimitiveRegistration(
            lower=lambda: None,
            validate_common_arguments=object(),
        )


def test_factory_registration_rejects_noncallable_provider():
    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir._compiler._operations import (
        StorageABI,
        register_factory,
    )

    with pytest.raises(TypeError, match="lowering factory must be callable"):
        register_factory(
            object(),
            operation="_test_invalid_factory",
            namespace="test",
            storage_abi=StorageABI.NONE,
            execution_scope=SynchronizationScope.NONE,
            synchronization_scope=SynchronizationScope.NONE,
        )


def _rewrite_specification(**overrides):
    """Build a valid baseline so each case can vary one registration rule."""
    from cuda.coop.numba_mlir._compiler._operations import (
        RewriteOperationSpecification,
    )

    arguments = {
        "factory_namespaces": frozenset({"test_namespace"}),
        "dtype_factory_kwargs": frozenset({"value_type"}),
        "runtime_arg_counts": frozenset({1, 2}),
        "runtime_factory_kwargs": ("tail",),
        "runtime_factory_kw_prerequisites": (),
        "allowed_factory_kwargs": frozenset(
            {"guard", "offset", "tail", "value_type"}
        ),
        "required_factory_kwargs": frozenset({"value_type"}),
        "accepts_temp_storage": False,
        "scalar_binding_kwargs": frozenset({"tail"}),
        "runtime_offset_kwarg": "offset",
        "infer_payload": lambda context, inference: None,
    }
    arguments.update(overrides)
    return RewriteOperationSpecification(**arguments)


@pytest.mark.parametrize(
    ("override", "error", "message"),
    [
        (
            {"runtime_arg_counts": frozenset()},
            ValueError,
            "runtime_arg_counts must not be empty",
        ),
        (
            {"runtime_arg_counts": frozenset({-1})},
            ValueError,
            "runtime_arg_counts must contain non-negative integers",
        ),
        (
            {"runtime_arg_counts": frozenset({True})},
            ValueError,
            "runtime_arg_counts must contain non-negative integers",
        ),
        (
            {"runtime_arg_counts": frozenset({1, 3})},
            ValueError,
            "require more trailing runtime arguments",
        ),
        (
            {"runtime_factory_kwargs": ("tail", "tail")},
            ValueError,
            "runtime_factory_kwargs must be unique",
        ),
        (
            {"runtime_factory_kwargs": ("unknown",)},
            ValueError,
            "runtime_factory_kwargs must be allowed factory kwargs",
        ),
        (
            {"runtime_factory_kw_prerequisites": (("tail",),)},
            TypeError,
            "must contain name pairs",
        ),
        (
            {
                "runtime_factory_kw_prerequisites": (
                    ("tail", "guard"),
                    ("tail", "value_type"),
                )
            },
            ValueError,
            "prerequisite names must be unique",
        ),
        (
            {"runtime_factory_kw_prerequisites": (("guard", "tail"),)},
            ValueError,
            "targets must be runtime factory kwargs",
        ),
        (
            {"runtime_factory_kw_prerequisites": (("tail", "unknown"),)},
            ValueError,
            "requirements must be known factory kwargs",
        ),
        (
            {"runtime_factory_kw_prerequisites": (("tail", "tail"),)},
            ValueError,
            "cannot require themselves",
        ),
        (
            {"scalar_binding_kwargs": frozenset({"guard"})},
            ValueError,
            "scalar_binding_kwargs must be runtime factory kwargs",
        ),
        (
            {"runtime_offset_kwarg": ""},
            ValueError,
            "runtime_offset_kwarg must be a non-empty string or None",
        ),
        (
            {"runtime_offset_kwarg": "unknown"},
            ValueError,
            "runtime_offset_kwarg must be an allowed factory kwarg",
        ),
        (
            {
                "runtime_arg_counts": frozenset({1, 2, 3}),
                "runtime_factory_kwargs": ("tail", "offset"),
            },
            ValueError,
            "must not also be a runtime factory kwarg",
        ),
        (
            {"infer_payload": None},
            TypeError,
            "infer_payload must be callable",
        ),
        (
            {"analyze_match": object()},
            TypeError,
            "analyze_match must be callable or None",
        ),
    ],
)
def test_rewrite_operation_specification_rejects_inconsistent_contracts(
    override,
    error,
    message,
):
    with pytest.raises(error, match=message):
        _rewrite_specification(**override)


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
    from cuda.coop.numba_mlir._compiler._nvrtc import CompilerIdentity
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
        template_arguments={},
    )
    if scope is SynchronizationScope.WARP:
        algorithm.logical_warp_threads = 32
        algorithm.block_threads = 64

    source = algorithm._source_code(
        compile_identity=CompilerIdentity(
            cc=90, rdc=True, code="lto", compiler_options=()
        )
    )[0]
    mangled_name = algorithm.mangled_name(algorithm.parameters[0])
    alloc_body = source.split(f"void {mangled_name}_alloc(", 1)[1].split(
        "\n}\n", 1
    )[0]
    pointer_body = source.split(f"void {mangled_name}(", 1)[1].split(
        "\n}\n", 1
    )[0]

    for token in ("__syncthreads();", "__syncwarp();"):
        assert (token in source) is (token == sync_token)
        assert (token in alloc_body) is (token == sync_token)
        assert token not in pointer_body
    if scope is SynchronizationScope.WARP:
        assert "temp_storages[2]" in source
        assert "[__coop_thread_rank / 32]" in source
    assert scope.value in repr(
        algorithm._make_lto_ir_cache_key(
            compile_identity=CompilerIdentity(
                cc=90, rdc=True, code="lto", compiler_options=()
            )
        )
    )


def test_group_synchronization_scope_fails_with_stable_diagnostic():
    from numba_cuda_mlir import types

    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir import _types
    from cuda.coop.numba_mlir._compiler._nvrtc import CompilerIdentity
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
        template_arguments={},
    )

    with pytest.raises(
        NotImplementedError, match="scope 'group' has no emitter"
    ):
        algorithm._source_code(
            compile_identity=CompilerIdentity(
                cc=90, rdc=True, code="lto", compiler_options=()
            )
        )


def test_storage_free_provider_uses_default_constructor_and_zero_storage():
    from numba_cuda_mlir import types

    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir import _types
    from cuda.coop.numba_mlir._compiler._nvrtc import CompilerIdentity
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
        template_arguments={},
    )
    source, _, storage_symbols, _ = algorithm._source_code(
        compile_identity=CompilerIdentity(
            cc=90, rdc=True, code="lto", compiler_options=()
        )
    )

    assert "StorageFreeProvider().Run" not in source
    assert "algorithm_t_" in source
    assert "().Run(param_0);" in source
    assert "TempStorage" not in source
    assert "temp_storage" not in source
    assert storage_symbols == ()


class _FakeInvocable:
    """Expose scratch and synchronization metadata for one group scope."""

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
    """Model a provider that needs neither scratch nor a reuse barrier."""

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
    """Count refreshes without a complete compiler typing context."""

    def __init__(self):
        self.refresh_count = 0

    def refresh(self):
        self.refresh_count += 1


def _resolved_calls(func_ir):
    """Resolve callable objects to inspect rewritten IR."""
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
                calls.append(
                    (resolver._resolve_python_value(value.func), value)
                )
    return calls


def test_registered_rewrite_callbacks_drive_generic_storage_rewrite():
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
    synchronization_scope = SynchronizationScope.BLOCK
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
        _operations.RewriteOperationSpecification(
            factory_namespaces=frozenset({"block", "alternate"}),
            dtype_factory_kwargs=frozenset({"element_type"}),
            runtime_arg_counts=frozenset({1}),
            runtime_factory_kwargs=(),
            runtime_factory_kw_prerequisites=(),
            allowed_factory_kwargs=frozenset(
                {"element_type", "inferred", "token"}
            ),
            required_factory_kwargs=frozenset(
                {"element_type", "inferred", "token"}
            ),
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
    assert rewrite.prepare_calls_and_storage(func_ir)
    rewrite.begin_rewrite()
    for label in sorted(func_ir.blocks):
        block = func_ir.blocks[label]
        while rewrite.match(func_ir, block, state.typemap, state.calltypes):
            block = rewrite.apply()
            func_ir.blocks[label] = block
    rewrite.finish_rewrite()

    calls = _resolved_calls(func_ir)
    invocable_calls = [call for target, call in calls if target is invocable]
    assert len(invocable_calls) == 1
    assert len(invocable_calls[0].args) == 3
    assert sum(target is cuda.shared.array for target, _ in calls) == 1
    sync_targets = {
        target
        for target, _ in calls
        if target in {cuda.syncthreads, cuda.syncwarp}
    }
    assert sync_targets == {cuda.syncthreads}
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
    assert len(infer_events) == 1
    assert all(len(event[1]) == 1 for event in infer_events)
    assert len(analyze_events) == 1
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
    assert len(validate_events) == 1
    assert all(event[1] == operation for event in validate_events)
    assert len(contexts) == 4

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
        _operations.RewriteOperationSpecification(
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
    assert rewrite.prepare_calls_and_storage(func_ir)
    rewrite.begin_rewrite()
    for label in sorted(func_ir.blocks):
        block = func_ir.blocks[label]
        while rewrite.match(func_ir, block, state.typemap, state.calltypes):
            block = rewrite.apply()
            func_ir.blocks[label] = block
    rewrite.finish_rewrite()

    calls = _resolved_calls(func_ir)
    invocable_calls = [call for target, call in calls if target is invocable]
    assert len(invocable_calls) == 1
    assert len(invocable_calls[0].args) == 1
    assert all(target is not cuda.shared.array for target, _ in calls)
    assert all(
        target not in {cuda.syncthreads, cuda.syncwarp} for target, _ in calls
    )
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
    from cuda.coop.numba_mlir._compiler._rewrite_invocables import (
        _InvocableRewrite,
    )
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


def test_group_family_loads_lazily_and_registers_additively(monkeypatch):
    from cuda.coop.numba_mlir._compiler import _operations

    operation = "_test_lazy_registration"
    module_name = "_test_lazy_registration_module"
    loads = []

    def lower(*args):
        return []

    @_operations.group_operation(operation, family_module=module_name)
    def marker(group, value):
        raise AssertionError("compiler marker must not execute")

    def load_family(name):
        loads.append(name)
        _operations.register_group_primitive(operation, lower=lower)

    original = dict(_operations._GROUP_PRIMITIVES)
    monkeypatch.setattr(_operations, "import_module", load_family)
    registration = _operations.group_primitive(operation)

    assert _operations.group_operation_name(marker) == operation
    assert registration.lower is lower
    assert _operations.group_primitive(operation) is registration
    assert loads == [module_name]
    assert all(
        _operations.group_primitive(name) is existing
        for name, existing in original.items()
    )
    _operations.register_group_primitive(operation, lower=lower)
    with pytest.raises(RuntimeError, match="already registered"):
        _operations.register_group_primitive(operation, lower=lambda: None)
