# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]

_LAZY_FAKE_FAMILY_MODULE = f"{__package__}._lazy_fake_family"


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
    missing = object()
    family_module = sys.modules.get(_LAZY_FAKE_FAMILY_MODULE, missing)
    try:
        yield
    finally:
        for registry, snapshot in zip(registries, snapshots):
            registry.clear()
            registry.update(snapshot)
        if family_module is missing:
            sys.modules.pop(_LAZY_FAKE_FAMILY_MODULE, None)
        else:
            sys.modules[_LAZY_FAKE_FAMILY_MODULE] = family_module


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


@pytest.mark.parametrize(
    ("override", "error", "message"),
    [
        ({"classifications": None}, TypeError, "classifications must be callable"),
        ({"planner": None}, TypeError, "planner must be callable"),
        ({"group_kinds": frozenset()}, ValueError, "group_kinds must not be empty"),
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


def _rewrite_spec(**overrides):
    from cuda.coop.numba_mlir._compiler._operations import RewriteOperationSpec

    arguments = {
        "factory_namespaces": frozenset({"test_namespace"}),
        "dtype_factory_kwargs": frozenset({"value_type"}),
        "runtime_arg_counts": frozenset({1, 2}),
        "runtime_factory_kwargs": ("tail",),
        "runtime_factory_kw_prerequisites": (),
        "allowed_factory_kwargs": frozenset({"guard", "offset", "tail", "value_type"}),
        "required_factory_kwargs": frozenset({"value_type"}),
        "accepts_temp_storage": False,
        "scalar_binding_kwargs": frozenset({"tail"}),
        "runtime_offset_kwarg": "offset",
        "infer_payload": lambda context, inference: None,
    }
    arguments.update(overrides)
    return RewriteOperationSpec(**arguments)


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
def test_rewrite_operation_spec_rejects_inconsistent_contracts(
    override,
    error,
    message,
):
    with pytest.raises(error, match=message):
        _rewrite_spec(**override)


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
    mangled_name = algorithm.mangled_name(algorithm.parameters[0])
    alloc_body = source.split(f"void {mangled_name}_alloc(", 1)[1].split("\n}\n", 1)[0]
    pointer_body = source.split(f"void {mangled_name}(", 1)[1].split("\n}\n", 1)[0]

    for token in ("__syncthreads();", "__syncwarp();"):
        assert (token in source) is (token == sync_token)
        assert (token in alloc_body) is (token == sync_token)
        assert token not in pointer_body
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


def _register_lazy_fake_frontends():
    from cuda.coop._core.api import _dispatch as portable_dispatch
    from cuda.coop.numba_mlir._compiler import _operations

    operations = {
        "scalar": "_test_lazy_family_scalar",
        "array": "_test_lazy_family_array",
        "pair": "_test_lazy_family_pair",
    }

    @portable_dispatch._portable_group_operation(
        operations["scalar"],
        group_kinds=("thread",),
    )
    def portable_scalar(group, value):
        del group, value

    @_operations.group_operation(
        operations["scalar"],
        family_module=_LAZY_FAKE_FAMILY_MODULE,
    )
    def qualified_scalar(group, value):
        del group, value

    @portable_dispatch._portable_group_operation(
        operations["array"],
        group_kinds=("block",),
    )
    def portable_array(group, values):
        del group, values

    @_operations.group_operation(
        operations["array"],
        family_module=_LAZY_FAKE_FAMILY_MODULE,
    )
    def qualified_array(group, values):
        del group, values

    @portable_dispatch._portable_group_operation(
        operations["pair"],
        group_kinds=("warp",),
    )
    def portable_pair(group, key, values):
        del group, key, values

    @_operations.group_operation(
        operations["pair"],
        family_module=_LAZY_FAKE_FAMILY_MODULE,
    )
    def qualified_pair(group, key, values):
        del group, key, values

    return operations, {
        "portable": {
            "scalar": portable_scalar,
            "array": portable_array,
            "pair": portable_pair,
        },
        "qualified": {
            "scalar": qualified_scalar,
            "array": qualified_array,
            "pair": qualified_pair,
        },
    }


def _lazy_fake_family_frontend(module, marker, shape, *, result_index=None):
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    if shape == "scalar":

        def kernel(value):
            return marker(module.this_thread(), value)

        args = (types.int32,)
    elif shape == "array":

        def kernel():
            values = module.ThreadData(3, dtype=types.float32)
            return marker(module.this_block(), values)

        args = ()
    elif result_index is None:

        def kernel(key):
            values = module.ThreadData(3, dtype=types.float32)
            return marker(module.this_warp(), key, values)

        args = (types.int32,)
    else:

        def kernel(key):
            values = module.ThreadData(3, dtype=types.float32)
            return marker(module.this_warp(), key, values)[result_index]

        args = (types.int32,)
    return run_frontend(kernel), args


def _return_value(func_ir):
    from numba_cuda_mlir.numbair_transforms import ir

    return next(
        statement.value
        for block in func_ir.blocks.values()
        for statement in block.body
        if isinstance(statement, ir.Return)
    )


def _run_lazy_fake_family_pipeline(func_ir, args):
    from cuda.coop.numba_mlir._compiler._group_planner import (
        _GroupCallPlanner,
        has_group_markers,
    )
    from cuda.coop.numba_mlir._compiler._rewrite import CoopSinglePhaseRewrite

    state = SimpleNamespace(
        func_ir=func_ir,
        args=args,
        typingctx=_TypingContext(),
        typemap={},
        calltypes={},
        metadata={"targetoptions": {}},
    )
    planner = _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    )
    assert has_group_markers(func_ir)
    assert planner.run()
    assert not has_group_markers(func_ir)

    rewrite = CoopSinglePhaseRewrite(state)
    for label in sorted(func_ir.blocks):
        block = func_ir.blocks[label]
        while rewrite.match(func_ir, block, state.typemap, state.calltypes):
            block = rewrite.apply()
            func_ir.blocks[label] = block
    return planner, rewrite, state


def test_lazy_fake_family_proves_additive_registration_end_to_end():
    from numba_cuda_mlir import cuda, types

    import cuda.coop as portable_coop
    import cuda.coop.numba_mlir as qualified_coop
    from cuda.coop._core import (
        LaunchFacts,
        StorageOwnership,
        SynchronizationScope,
        make_group_primitive_call,
        plan_group_primitive,
        this_block,
        this_thread,
        this_warp,
    )
    from cuda.coop._core.api import _dispatch as portable_dispatch
    from cuda.coop._core.group import _dispatch as core_dispatch
    from cuda.coop.numba_mlir._compiler import _operations
    from cuda.coop.numba_mlir._compiler._group_planner import _GroupCallPlanner
    from cuda.coop.numba_mlir._compiler._group_planner_support import (
        _group_operation_name,
    )
    from cuda.coop.numba_mlir._compiler._group_planning import GroupPlanningContext

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
    missing = object()
    module_snapshot = sys.modules.get(_LAZY_FAKE_FAMILY_MODULE, missing)
    family = None
    operations = {
        "scalar": "_test_lazy_family_scalar",
        "array": "_test_lazy_family_array",
        "pair": "_test_lazy_family_pair",
    }
    markers = None
    try:
        assert module_snapshot is missing
        sys.modules.pop(_LAZY_FAKE_FAMILY_MODULE, None)
        operations, markers = _register_lazy_fake_frontends()
        assert _LAZY_FAKE_FAMILY_MODULE not in sys.modules
        assert all(
            operation not in _operations._GROUP_PRIMITIVES
            for operation in operations.values()
        )

        for shape, operation in operations.items():
            portable_marker = markers["portable"][shape]
            qualified_marker = markers["qualified"][shape]
            assert portable_marker is not qualified_marker
            assert (
                portable_dispatch._portable_group_operation_name(portable_marker)
                == operation
            )
            assert _operations.group_operation_name(qualified_marker) == operation
            assert _group_operation_name(portable_marker) == operation
            assert _group_operation_name(qualified_marker) == operation

            def impostor(group, value):
                del group, value

            impostor.__module__ = qualified_marker.__module__
            impostor.__name__ = qualified_marker.__name__
            impostor.__cuda_coop_backend_member__ = operation
            assert _group_operation_name(impostor) is None
            assert _LAZY_FAKE_FAMILY_MODULE not in sys.modules

        first_ir, first_args = _lazy_fake_family_frontend(
            portable_coop,
            markers["portable"]["scalar"],
            "scalar",
        )
        first_planner, first_rewrite, _ = _run_lazy_fake_family_pipeline(
            first_ir, first_args
        )
        assert isinstance(first_planner.context, GroupPlanningContext)
        assert first_planner.context.launch.exact_block_dim == (64, 1, 1)
        assert not hasattr(first_planner.context, "_group_cache")
        assert not hasattr(first_planner.context, "_planner")
        assert _LAZY_FAKE_FAMILY_MODULE in sys.modules
        family = sys.modules[_LAZY_FAKE_FAMILY_MODULE]
        assert tuple(family.OPERATIONS) == tuple(operations.values())
        assert all(
            _operations.group_primitive(operation) is not None
            and _operations.rewrite_operation(operation) is not None
            for operation in operations.values()
        )

        launch = LaunchFacts(
            exact_block_dim=(64, 1, 1),
            exact_grid_dim=(1, 1, 1),
        )
        core_cases = {
            "scalar": (
                this_thread,
                (types.int32,),
                SynchronizationScope.NONE,
                StorageOwnership.NONE,
                1,
            ),
            "array": (
                this_block,
                (types.float32,),
                SynchronizationScope.BLOCK,
                StorageOwnership.IMPLEMENTATION,
                1,
            ),
            "pair": (
                this_warp,
                (types.int32, types.float32),
                SynchronizationScope.WARP,
                StorageOwnership.IMPLEMENTATION,
                2,
            ),
        }
        for shape, (
            group_factory,
            result_dtypes,
            expected_scope,
            expected_ownership,
            result_count,
        ) in core_cases.items():
            semantics = family.LazyFamilySemantics(
                operation=operations[shape],
                result_dtypes=result_dtypes,
                array_extent=3,
            )
            call = make_group_primitive_call(group_factory(), semantics)
            plan = plan_group_primitive(call, launch).require_supported()
            assert plan.topology.execution_scope is expected_scope
            assert plan.synchronization.storage_reuse_barrier is expected_scope
            assert plan.temp_storage.ownership is expected_ownership
            assert len(plan.result.values) == result_count
            assert tuple(
                classification.name for classification in call.argument_classifications
            ) == (
                ("key", "values")
                if shape == "pair"
                else (("values",) if shape == "array" else ("value",))
            )

        provenance_cases = (
            ("scalar", None, types.int32, False, 1),
            ("array", None, types.float32, True, 3),
            ("pair", 0, types.int32, False, 1),
            ("pair", 1, types.float32, True, 3),
        )
        for frontend_name, module in (
            ("portable", portable_coop),
            ("qualified", qualified_coop),
        ):
            for (
                shape,
                result_index,
                expected_dtype,
                is_array,
                extent,
            ) in provenance_cases:
                func_ir, args = _lazy_fake_family_frontend(
                    module,
                    markers[frontend_name][shape],
                    shape,
                    result_index=result_index,
                )
                planner = _GroupCallPlanner(
                    SimpleNamespace(func_ir=func_ir, args=args),
                    {
                        "block": (64, 1, 1),
                        "grid": (1, 1, 1),
                        "cluster": None,
                    },
                )
                result = _return_value(func_ir)
                assert planner.context.dtype(result) == expected_dtype
                assert planner.context.is_array(operations[shape], result) is is_array
                assert planner.context.array_extent(result) == extent

        pipeline_cases = [
            (frontend_name, shape)
            for frontend_name in ("portable", "qualified")
            for shape in ("scalar", "array", "pair")
        ]
        pipeline_results = {("portable", "scalar"): (first_ir, first_rewrite)}
        for frontend_name, shape in pipeline_cases[1:]:
            module = portable_coop if frontend_name == "portable" else qualified_coop
            func_ir, args = _lazy_fake_family_frontend(
                module,
                markers[frontend_name][shape],
                shape,
            )
            _, rewrite, _ = _run_lazy_fake_family_pipeline(func_ir, args)
            pipeline_results[(frontend_name, shape)] = (func_ir, rewrite)

        expected_rewrite = {
            "scalar": (family.INVOCABLES[operations["scalar"]], 1, None),
            "array": (
                family.INVOCABLES[operations["array"]],
                2,
                cuda.syncthreads,
            ),
            "pair": (
                family.INVOCABLES[operations["pair"]],
                3,
                cuda.syncwarp,
            ),
        }
        for (_, shape), (func_ir, rewrite) in pipeline_results.items():
            invocable, arg_count, sync = expected_rewrite[shape]
            calls = _resolved_calls(func_ir)
            invocable_calls = [call for target, call in calls if target is invocable]
            assert len(invocable_calls) == 1
            assert len(invocable_calls[0].args) == arg_count
            sync_calls = [
                target
                for target, _ in calls
                if target in {cuda.syncthreads, cuda.syncwarp}
            ]
            assert sync_calls == ([] if sync is None else [sync])
            shared_allocations = sum(target is cuda.shared.array for target, _ in calls)
            assert shared_allocations == (0 if shape == "scalar" else 1)
            assert (rewrite._temp_storage_global_plan is None) is (shape == "scalar")

        assert {(operation, dtype) for operation, dtype in family.FACTORY_CALLS} == {
            (operations["scalar"], types.int32),
            (operations["array"], types.float32),
            (operations["pair"], types.float32),
        }
        assert {
            (operation, group_kind, is_common_root)
            for operation, group_kind, is_common_root, _ in family.PLANNING_EVENTS
        } == {
            (operations["scalar"], "thread", True),
            (operations["array"], "block", True),
            (operations["pair"], "warp", True),
            (operations["scalar"], "thread", False),
            (operations["array"], "block", False),
            (operations["pair"], "warp", False),
        }
        for operation, provider in family.PROVIDERS.items():
            metadata = _operations.factory_operation(provider)
            assert metadata.operation == operation
            assert metadata.namespace == "lazy_test_namespace"
            assert _operations.rewrite_operation(
                operation
            ).dtype_factory_kwargs == frozenset({"value_type"})
    finally:
        for registry, snapshot in zip(registries, snapshots):
            registry.clear()
            registry.update(snapshot)
        if module_snapshot is missing:
            sys.modules.pop(_LAZY_FAKE_FAMILY_MODULE, None)
        else:
            sys.modules[_LAZY_FAKE_FAMILY_MODULE] = module_snapshot

    assert _LAZY_FAKE_FAMILY_MODULE not in sys.modules
    assert markers is not None
    assert all(
        operation not in _operations._GROUP_PRIMITIVES
        and operation not in _operations._REWRITE_OPERATIONS
        and operation not in _operations._GROUP_FAMILY_MODULES
        and operation not in portable_dispatch._PORTABLE_GROUP_OPERATIONS_BY_NAME
        for operation in operations.values()
    )
    assert all(
        marker not in _operations._GROUP_OPERATIONS
        and marker not in portable_dispatch._PORTABLE_GROUP_OPERATIONS_BY_FUNCTION
        for frontend_markers in markers.values()
        for marker in frontend_markers.values()
    )
    assert family is not None
    assert family.LazyFamilySemantics not in core_dispatch._GROUP_OPERATION_FAMILIES
    assert all(
        provider not in _operations._FACTORY_OPERATIONS
        for provider in family.PROVIDERS.values()
    )


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
