# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.compiler import run_frontend
from numba_cuda_mlir.numbair_transforms import ir

import cuda.coop as common_coop
import cuda.coop.numba_mlir as coop
from cuda.coop._core import SynchronizationScope
from cuda.coop.numba_mlir._compiler import _operations
from cuda.coop.numba_mlir._compiler._rewrite import (
    CoopSinglePhaseRewrite,
    CoopSinglePhaseRewriteError,
    CoopWholeFunctionPlanner,
)
from cuda.coop.numba_mlir._compiler._rewrite_support import (
    _DEFAULT_STATIC_SHARED_MEMORY_BYTES,
    _TempStorageCtorSpec,
    _TempStoragePlan,
    _TempStorageRequirementSummary,
    _TempStorageUseRequirement,
)

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


class _StringSharing(str, Enum):
    SHARED = "shared"


class _TypingContext:
    def __init__(self):
        self.refresh_count = 0

    def refresh(self):
        self.refresh_count += 1


@pytest.fixture(autouse=True)
def _restore_rewrite_registries():
    operation_prefix = "_test_storage_rewrite_family_"
    try:
        yield
    finally:
        for factory, metadata in tuple(_operations._FACTORY_OPERATIONS.items()):
            if metadata.operation.startswith(operation_prefix):
                del _operations._FACTORY_OPERATIONS[factory]
        for operation in tuple(_operations._REWRITE_OPERATIONS):
            if operation.startswith(operation_prefix):
                del _operations._REWRITE_OPERATIONS[operation]


def _rewrite(function):
    func_ir = run_frontend(function)
    typingctx = _TypingContext()
    state = SimpleNamespace(
        func_ir=func_ir,
        typingctx=typingctx,
        typemap={},
        calltypes={},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    for label in sorted(func_ir.blocks):
        block = func_ir.blocks[label]
        while rewrite.match(func_ir, block, state.typemap, state.calltypes):
            block = rewrite.apply()
            func_ir.blocks[label] = block
    return func_ir, typingctx


def _call_targets(func_ir):
    rewrite = object.__new__(CoopSinglePhaseRewrite)
    rewrite._func_ir = func_ir
    targets = []
    for block in func_ir.blocks.values():
        rewrite._block_defs = {
            inst.target.name: inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
        }
        for inst in block.body:
            value = getattr(inst, "value", None)
            if isinstance(value, ir.Expr) and value.op == "call":
                targets.append(rewrite._resolve_python_value(value.func))
    return targets


def test_qualified_thread_data_lowers_to_a_compiler_array():
    def kernel():
        data = coop.ThreadData(2, types.int32, alignas=16)
        return data[0]

    func_ir, typingctx = _rewrite(kernel)
    targets = _call_targets(func_ir)

    assert cuda.local.array in targets
    assert typingctx.refresh_count == 1


@pytest.mark.parametrize(
    "alignment",
    [16, np.int64(16)],
    ids=["builtin-int", "index-integer"],
)
def test_thread_data_rewrite_uses_alignment(alignment):
    def kernel():
        data = coop.ThreadData(
            2,
            types.int32,
            alignas=alignment,
        )
        return data[0]

    func_ir, _ = _rewrite(kernel)

    assert cuda.local.array in _call_targets(func_ir)


def test_thread_data_rewrite_rejects_removed_alignment_alias():
    def kernel():
        return coop.ThreadData(2, types.int32, alignment=16)

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=r"ThreadData got unexpected keyword\(s\): alignment",
    ):
        _rewrite(kernel)


def test_thread_data_rewrite_accepts_explicit_default_alignment():
    def kernel():
        data = coop.ThreadData(2, types.int32, alignas=8)
        return data[0]

    func_ir, _ = _rewrite(kernel)

    assert cuda.local.array in _call_targets(func_ir)


def test_common_thread_data_uses_only_the_portable_signature():
    def kernel():
        data = common_coop.ThreadData(2, types.int32)
        return data[0]

    func_ir, _ = _rewrite(kernel)

    assert cuda.local.array in _call_targets(func_ir)


def test_common_thread_data_rejects_qualified_alignment_control():
    def kernel():
        return common_coop.ThreadData(2, types.int32, alignas=16)

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=r"cuda\.coop\.ThreadData got unexpected keyword.*alignas",
    ):
        _rewrite(kernel)


def test_whole_function_planner_reuses_the_foundation_rewrite():
    def kernel():
        return coop.ThreadData(2, types.int32)

    func_ir = run_frontend(kernel)
    typingctx = _TypingContext()
    state = SimpleNamespace(
        func_ir=func_ir,
        typingctx=typingctx,
        typemap={},
        calltypes={},
        metadata={"targetoptions": {}},
    )

    assert CoopWholeFunctionPlanner(state).run()
    assert cuda.local.array in _call_targets(func_ir)
    assert typingctx.refresh_count == 1


def _unused_temp_storage():
    _storage = coop.TempStorage(32)
    return 0


@pytest.mark.parametrize(
    "kernel",
    [
        pytest.param(_unused_temp_storage, id="no-consumer"),
        pytest.param(lambda: coop.TempStorage(32), id="return"),
        pytest.param(lambda: coop.TempStorage(32)[0], id="index"),
        pytest.param(lambda: len(coop.TempStorage(32)), id="other-call"),
    ],
)
def test_temp_storage_is_an_opaque_primitive_descriptor(kernel):
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="opaque compile-time descriptors",
    ):
        _rewrite(kernel)


def test_temp_storage_rewrite_rejects_string_enum_sharing():
    def kernel():
        return coop.TempStorage(sharing=_StringSharing.SHARED)

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="TempStorage sharing must be a string",
    ):
        _rewrite(kernel)


class _FakeInvocable:
    files = ("storage-rewrite-test.ltoir",)
    specialization = None
    storage_abi = "leading_pointer"
    execution_scope = "block"
    synchronization_scope = "block"

    def __init__(self, *, size_in_bytes=64, alignment=16):
        self.temp_storage_bytes = size_in_bytes
        self.temp_storage_alignment = alignment

    def __call__(self, *args):
        del args


def _register_leading_pointer_provider(
    invocable,
    *,
    execution_scope=SynchronizationScope.BLOCK,
    synchronization_scope=SynchronizationScope.BLOCK,
):
    operation = f"_test_storage_rewrite_family_{id(invocable)}"
    calls = []

    def provider(*runtime_args, **factory_kwargs):
        assert not runtime_args
        assert not factory_kwargs
        calls.append((runtime_args, factory_kwargs))
        return invocable

    provider.calls = calls

    _operations.register_factory(
        provider,
        operation=operation,
        namespace="storage_test",
        storage_abi=_operations.StorageABI.LEADING_POINTER,
        execution_scope=execution_scope,
        synchronization_scope=synchronization_scope,
    )
    _operations.register_rewrite_operation(
        operation,
        _operations.RewriteOperationSpec(
            factory_namespaces=frozenset({"storage_test"}),
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
    return provider


def _frontend(function, *, ssa=False):
    func_ir = run_frontend(function)
    if ssa:
        from numba_cuda_mlir.numba_cuda.core.ir_utils import build_definitions
        from numba_cuda_mlir.numba_cuda.core.ssa import reconstruct_ssa

        func_ir = reconstruct_ssa(func_ir)
        func_ir._definitions = build_definitions(func_ir.blocks)
    return func_ir


def _rewrite_registered_provider(function, *, ssa=False, lifo=False):
    func_ir = _frontend(function, ssa=ssa)
    typingctx = _TypingContext()
    state = SimpleNamespace(
        func_ir=func_ir,
        args=(),
        typingctx=typingctx,
        typemap={},
        calltypes={},
        metadata={"targetoptions": {}},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    items = list(func_ir.blocks.items())
    if not lifo:
        items.reverse()
    while items:
        label, block = items.pop()
        if rewrite.match(func_ir, block, state.typemap, state.calltypes):
            new_block = rewrite.apply()
            func_ir.blocks[label] = new_block
            items.append((label, new_block))
    return func_ir, rewrite, state


def _rewrite_preflight(function):
    func_ir = _frontend(function)
    state = SimpleNamespace(
        func_ir=func_ir,
        args=(),
        typingctx=_TypingContext(),
        typemap={},
        calltypes={},
        metadata={"targetoptions": {}},
    )
    return func_ir, state, CoopSinglePhaseRewrite(state)


@pytest.mark.parametrize(
    ("execution_scope", "synchronization_scope", "accepted"),
    [
        pytest.param(
            SynchronizationScope.BLOCK,
            SynchronizationScope.BLOCK,
            True,
            id="block-block",
        ),
        pytest.param(
            SynchronizationScope.BLOCK,
            SynchronizationScope.NONE,
            False,
            id="block-none",
        ),
        pytest.param(
            SynchronizationScope.WARP,
            SynchronizationScope.WARP,
            False,
            id="warp-warp",
        ),
        pytest.param(
            SynchronizationScope.NONE,
            SynchronizationScope.NONE,
            False,
            id="none-none",
        ),
        pytest.param(
            SynchronizationScope.GROUP,
            SynchronizationScope.GROUP,
            False,
            id="group-group",
        ),
    ],
)
def test_storage_provider_without_plan_requires_block_scope(
    execution_scope,
    synchronization_scope,
    accepted,
):
    invocable = _FakeInvocable()
    provider = _register_leading_pointer_provider(
        invocable,
        execution_scope=execution_scope,
        synchronization_scope=synchronization_scope,
    )

    def kernel(value):
        return provider(value)

    func_ir, state, rewrite = _rewrite_preflight(kernel)
    entry_block = func_ir.blocks[min(func_ir.blocks)]
    if accepted:
        prepared = []
        rewrite._prepare_ltoir_bundle_for_matches = lambda matches: prepared.append(
            tuple(matches)
        )

        assert rewrite.match(func_ir, entry_block, state.typemap, state.calltypes)
        assert len(prepared) == 1
        assert len(prepared[0]) == 1
        assert provider.calls == [((), {})]
        return

    rewrite._prepare_ltoir_bundle_for_matches = lambda _matches: pytest.fail(
        "invalid no-plan provider reached bundle preparation"
    )
    rewrite._materialize_invocable = lambda _match: pytest.fail(
        "invalid no-plan provider reached provider materialization"
    )
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="require block execution and block synchronization scopes",
    ):
        rewrite.match(func_ir, entry_block, state.typemap, state.calltypes)
    assert provider.calls == []


@pytest.mark.parametrize(
    ("case", "with_descriptor", "message"),
    [
        pytest.param(
            "group",
            False,
            "execution scope 'group' has no storage emitter",
            id="group",
        ),
        pytest.param(
            "address-space",
            False,
            "shared-address-space TempStorage",
            id="address-space",
        ),
        pytest.param(
            "implementation-with-descriptor",
            True,
            "TempStorage ownership disagrees with its runtime arguments",
            id="implementation-with-descriptor",
        ),
        pytest.param(
            "caller-without-descriptor",
            False,
            "TempStorage ownership disagrees with its runtime arguments",
            id="caller-without-descriptor",
        ),
        pytest.param(
            "caller-none",
            True,
            "caller-owned TempStorage is supported only for single-instance block",
            id="caller-none",
        ),
        pytest.param(
            "caller-group",
            True,
            "execution scope 'group' has no storage emitter",
            id="caller-group",
        ),
    ],
)
def test_planned_storage_guardrails_fail_before_materialization(
    case,
    with_descriptor,
    message,
):
    from dataclasses import replace

    from cuda.coop._core import (
        GroupLoadStoreAlgorithm,
        GroupLoweringTarget,
        GroupTopologyContract,
        LaunchFacts,
        ParticipationContract,
        StorageOwnership,
        SynchronizationContract,
        make_group_primitive_call,
        resolve_thread_group,
        this_block,
        this_thread,
    )
    from tests.support.group_planning import _load_store, _plan

    plan = _plan(
        this_block(),
        _load_store(algorithm=GroupLoadStoreAlgorithm.TRANSPOSE),
    )
    execution_scope = SynchronizationScope.BLOCK
    synchronization_scope = SynchronizationScope.BLOCK
    if case in {"group", "caller-group"}:
        execution_scope = SynchronizationScope.GROUP
        synchronization_scope = SynchronizationScope.GROUP
        plan = replace(
            plan,
            topology=replace(
                plan.topology,
                execution_scope=SynchronizationScope.GROUP,
            ),
            synchronization=replace(
                plan.synchronization,
                storage_reuse_barrier=SynchronizationScope.GROUP,
            ),
        )
    elif case == "address-space":
        plan = replace(
            plan,
            temp_storage=replace(plan.temp_storage, address_space="local"),
        )
    elif case == "caller-none":
        execution_scope = SynchronizationScope.NONE
        synchronization_scope = SynchronizationScope.NONE
        launch = LaunchFacts(exact_block_dim=(64, 1, 1))
        resolved_thread = resolve_thread_group(
            this_thread(), launch
        ).require_supported()
        operation = plan.call.operation
        plan = replace(
            plan,
            target=GroupLoweringTarget.CUDAX_GROUP,
            call=make_group_primitive_call(resolved_thread, operation),
            resolved_group=resolved_thread,
            topology=GroupTopologyContract(
                group_kind="thread",
                logical_width=1,
                instances=64,
                instance_index="linear_thread_rank",
                thread_rank="0",
                execution_scope=SynchronizationScope.NONE,
            ),
            participation=ParticipationContract(
                group_kind="thread",
                exact_group_size=1,
                exact_block_dim=(64, 1, 1),
                complete_membership=True,
                contiguous=True,
                aligned=True,
                converged_entry=True,
                complete_parent_partition=True,
            ),
            synchronization=SynchronizationContract(
                converged_entry=True,
                storage_reuse_barrier=SynchronizationScope.NONE,
            ),
            temp_storage=replace(
                plan.temp_storage,
                instances=64,
                instance_index="linear_thread_rank",
            ),
        )
    if case.startswith("caller-"):
        plan = replace(
            plan,
            temp_storage=replace(
                plan.temp_storage,
                ownership=StorageOwnership.CALLER,
                exact_layout_required=True,
                sharing="shared",
            ),
        )

    invocable = _FakeInvocable()
    provider = _register_leading_pointer_provider(
        invocable,
        execution_scope=execution_scope,
        synchronization_scope=synchronization_scope,
    )
    if with_descriptor:

        def kernel(value):
            storage = coop.TempStorage()
            return provider(
                value,
                temp_storage=storage,
                __cuda_coop_group_lowering_plan__=plan,
            )

    else:

        def kernel(value):
            return provider(
                value,
                __cuda_coop_group_lowering_plan__=plan,
            )

    func_ir, state, rewrite = _rewrite_preflight(kernel)
    rewrite._prepare_ltoir_bundle_for_matches = lambda _matches: pytest.fail(
        "invalid planned provider reached bundle preparation"
    )
    rewrite._materialize_invocable = lambda _match: pytest.fail(
        "invalid planned provider reached provider materialization"
    )

    with pytest.raises(CoopSinglePhaseRewriteError, match=message):
        rewrite.match(
            func_ir,
            func_ir.blocks[min(func_ir.blocks)],
            state.typemap,
            state.calltypes,
        )
    assert provider.calls == []


def _resolved_calls(func_ir):
    resolver = object.__new__(CoopSinglePhaseRewrite)
    resolver._func_ir = func_ir
    calls = []
    for label, block in func_ir.blocks.items():
        resolver._block_defs = {
            inst.target.name: inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
        }
        for inst in block.body:
            value = getattr(inst, "value", None)
            if isinstance(value, ir.Expr) and value.op == "call":
                calls.append((label, inst, resolver._resolve_python_value(value.func)))
    return calls


def _rewrite_with_fake_invocable(function, invocable, *, ssa=False):
    func_ir = _frontend(function, ssa=ssa)
    typingctx = _TypingContext()
    state = SimpleNamespace(
        func_ir=func_ir,
        args=(),
        typingctx=typingctx,
        typemap={},
        calltypes={},
        metadata={},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    rewrite._prepare_ltoir_bundle_for_matches = lambda _matches: None
    rewrite._materialize_invocable = lambda _match: (invocable, False)
    rewrite._record_invocable_specialization = lambda _invocable: None
    for label in sorted(func_ir.blocks):
        block = func_ir.blocks[label]
        while rewrite.match(func_ir, block, state.typemap, state.calltypes):
            block = rewrite.apply()
            func_ir.blocks[label] = block
    return func_ir, rewrite


def test_temp_storage_aliases_from_one_constructor_are_accepted():
    from cuda.coop.numba_mlir._lowering._load_store import load as provider_load

    def kernel(source, choose_first):
        storage = coop.TempStorage()
        if choose_first:
            selected = storage
        else:
            selected = storage
        payload = coop.ThreadData(2, types.int32)
        return provider_load(
            source,
            payload,
            dtype=types.int32,
            threads_per_block=32,
            items_per_thread=2,
            temp_storage=selected,
        )

    func_ir, _ = _rewrite_with_fake_invocable(
        kernel,
        _FakeInvocable(),
        ssa=True,
    )

    assert any(
        isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "phi"
        for block in func_ir.blocks.values()
        for inst in block.body
    )
    targets = _call_targets(func_ir)
    assert cuda.shared.array not in targets
    assert cuda.syncthreads not in targets


def test_temp_storage_phi_canonicalizes_equivalent_constructor_contracts():
    invocable = _FakeInvocable()
    provider = _register_leading_pointer_provider(invocable)

    def kernel(value, choose_first):
        if choose_first:
            selected = coop.TempStorage()
        else:
            selected = coop.TempStorage(auto_sync=True, sharing=" SHARED ")
        return provider(value, temp_storage=selected)

    func_ir, rewrite, _ = _rewrite_registered_provider(kernel, ssa=True)
    calls = _resolved_calls(func_ir)

    assert sum(target is cuda.shared.array for _, _, target in calls) == 1
    assert sum(target is invocable for _, _, target in calls) == 1
    assert rewrite._temp_storage_global_plan.total_size == 64
    assert len(set(rewrite._temp_storage_ctor_roots.values())) == 1


def test_leading_pointer_storage_can_disable_external_reuse_sync():
    invocable = _FakeInvocable()
    provider = _register_leading_pointer_provider(invocable)

    def kernel(value):
        storage = coop.TempStorage(auto_sync=False)
        return provider(value, temp_storage=storage)

    func_ir, _, _ = _rewrite_registered_provider(kernel)
    calls = _resolved_calls(func_ir)
    targets = [target for _, _, target in calls]
    invocable_calls = [inst.value for _, inst, target in calls if target is invocable]

    assert targets.count(cuda.shared.array) == 1
    assert len(invocable_calls) == 1
    assert len(invocable_calls[0].args) == 2
    assert cuda.syncthreads not in targets
    assert cuda.syncwarp not in targets


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((64, 16, True, "shared"), (96, 16, True, "shared")),
        ((64, 16, True, "shared"), (64, 32, True, "shared")),
        ((64, 16, True, "shared"), (64, 16, False, "shared")),
        ((64, 16, None, "shared"), (64, 16, None, "exclusive")),
    ],
    ids=["capacity", "alignment", "auto-sync", "sharing"],
)
def test_temp_storage_phi_rejects_incompatible_contracts_before_compile(left, right):
    invocable = _FakeInvocable()
    provider = _register_leading_pointer_provider(invocable)
    left_size, left_alignment, left_auto_sync, left_sharing = left
    right_size, right_alignment, right_auto_sync, right_sharing = right

    def kernel(value, choose_first):
        if choose_first:
            selected = coop.TempStorage(
                left_size,
                left_alignment,
                left_auto_sync,
                left_sharing,
            )
        else:
            selected = coop.TempStorage(
                right_size,
                right_alignment,
                right_auto_sync,
                right_sharing,
            )
        return provider(value, temp_storage=selected)

    func_ir = _frontend(kernel, ssa=True)
    state = SimpleNamespace(
        func_ir=func_ir,
        args=(),
        typingctx=_TypingContext(),
        typemap={},
        calltypes={},
        metadata={"targetoptions": {}},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    rewrite._prepare_ltoir_bundle_for_matches = lambda _matches: pytest.fail(
        "incompatible TempStorage reached provider preparation"
    )
    rewrite._materialize_invocable = lambda _match: pytest.fail(
        "incompatible TempStorage reached provider compilation"
    )

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="TempStorage aliases have inconsistent contracts",
    ):
        rewrite.match(
            func_ir,
            func_ir.blocks[min(func_ir.blocks)],
            state.typemap,
            state.calltypes,
        )


def test_group_planning_rejects_mixed_descriptor_phi():
    from cuda.coop.numba_mlir._compiler._group_planner import _GroupCallPlanner
    from cuda.coop.numba_mlir._compiler._group_planner_support import GroupRewriteError

    def consume(value):
        del value

    def kernel(choose_storage):
        if choose_storage:
            selected = coop.TempStorage()
        else:
            selected = None
        consume(selected)

    func_ir = _frontend(kernel, ssa=True)
    phi_assign = next(
        inst
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "phi"
    )
    planner = _GroupCallPlanner(
        SimpleNamespace(func_ir=func_ir, args=(types.boolean,)),
        {"block": (32, 1, 1), "grid": (1, 1, 1)},
    )

    with pytest.raises(GroupRewriteError, match="inconsistent contracts"):
        planner.context.temp_storage(phi_assign.target)


def test_temp_storage_backing_dominates_nonentry_call_under_lifo_rewrite():
    invocable = _FakeInvocable()
    provider = _register_leading_pointer_provider(invocable)

    def kernel(value, take_provider):
        if take_provider:
            result = provider(value)
        else:
            result = value
        return result

    func_ir, rewrite, _ = _rewrite_registered_provider(kernel, lifo=True)
    calls = _resolved_calls(func_ir)
    entry_label = min(func_ir.blocks)
    backing_calls = [
        (label, inst) for label, inst, target in calls if target is cuda.shared.array
    ]
    invocable_calls = [
        (label, inst) for label, inst, target in calls if target is invocable
    ]

    assert len(backing_calls) == 1
    assert len(invocable_calls) == 1
    backing_label, backing_assign = backing_calls[0]
    call_label, call_assign = invocable_calls[0]
    assert backing_label == entry_label
    assert call_label != entry_label
    storage_arg = call_assign.value.args[0]
    storage_definition = next(
        inst.value
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign) and inst.target.name == storage_arg.name
    )
    assert isinstance(storage_definition, ir.Expr)
    assert storage_definition.op == "getitem"
    assert storage_definition.value.name == backing_assign.target.name
    assert rewrite._temp_storage_backing_emitted


def test_equivalent_temp_storage_phi_escape_is_rejected_before_compile():
    invocable = _FakeInvocable()
    provider = _register_leading_pointer_provider(invocable)

    def kernel(value, choose_first):
        if choose_first:
            selected = coop.TempStorage()
        else:
            selected = coop.TempStorage(auto_sync=True)
        provider(value, temp_storage=selected)
        return selected

    func_ir = _frontend(kernel, ssa=True)
    state = SimpleNamespace(
        func_ir=func_ir,
        args=(),
        typingctx=_TypingContext(),
        typemap={},
        calltypes={},
        metadata={"targetoptions": {}},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    rewrite._prepare_ltoir_bundle_for_matches = lambda _matches: pytest.fail(
        "escaping TempStorage reached provider preparation"
    )
    rewrite._materialize_invocable = lambda _match: pytest.fail(
        "escaping TempStorage reached provider compilation"
    )

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="would escape to runtime",
    ):
        rewrite.match(
            func_ir,
            func_ir.blocks[min(func_ir.blocks)],
            state.typemap,
            state.calltypes,
        )


def test_leading_pointer_provider_stages_one_dynamic_backing(monkeypatch):
    from cuda.coop.numba_mlir._compiler import _rewrite_storage

    size_in_bytes = 64 * 1024
    invocable = _FakeInvocable(size_in_bytes=size_in_bytes)
    provider = _register_leading_pointer_provider(invocable)
    monkeypatch.setattr(
        _rewrite_storage,
        "_query_device_shared_memory_limits",
        lambda: {
            "max_default_shared_memory_per_block": 48 * 1024,
            "max_optin_shared_memory_per_block": 96 * 1024,
        },
    )

    def kernel(value):
        return provider(value)

    func_ir, rewrite, state = _rewrite_registered_provider(kernel, lifo=True)
    backing_calls = [
        inst.value
        for _, inst, target in _resolved_calls(func_ir)
        if target is cuda.shared.array
    ]

    assert len(backing_calls) == 1
    size_var = backing_calls[0].args[0]
    size_definition = next(
        inst.value
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign) and inst.target.name == size_var.name
    )
    assert isinstance(size_definition, ir.Const)
    assert size_definition.value == 0
    assert rewrite._temp_storage_global_plan.dynamic_shared_bytes == size_in_bytes
    assert state.metadata["required_dynamic_shared_memory"] == size_in_bytes


def test_mixed_temp_storage_primitive_and_escape_fails_before_compile():
    from cuda.coop.numba_mlir._lowering._load_store import load as provider_load

    def kernel(source):
        storage = coop.TempStorage()
        payload = coop.ThreadData(2, types.int32)
        provider_load(
            source,
            payload,
            dtype=types.int32,
            threads_per_block=32,
            items_per_thread=2,
            temp_storage=storage,
        )
        return storage

    func_ir = run_frontend(kernel)
    state = SimpleNamespace(
        func_ir=func_ir,
        args=(),
        typingctx=_TypingContext(),
        typemap={},
        calltypes={},
        metadata={},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    rewrite._prepare_ltoir_bundle_for_matches = lambda _matches: pytest.fail(
        "escaping TempStorage reached provider preparation"
    )
    rewrite._materialize_invocable = lambda _match: pytest.fail(
        "escaping TempStorage reached provider compilation"
    )

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="would escape to runtime",
    ):
        rewrite.match(
            func_ir,
            func_ir.blocks[min(func_ir.blocks)],
            state.typemap,
            state.calltypes,
        )


def test_direct_provider_uses_no_implicit_storage_or_automatic_sync():
    from cuda.coop.numba_mlir._lowering._load_store import load as provider_load

    def kernel(source):
        payload = coop.ThreadData(2, types.int32)
        return provider_load(
            source,
            payload,
            dtype=types.int32,
            threads_per_block=32,
            items_per_thread=2,
        )

    invocable = _FakeInvocable()
    func_ir, rewrite = _rewrite_with_fake_invocable(kernel, invocable)
    targets = _call_targets(func_ir)

    assert cuda.shared.array not in targets
    assert cuda.syncthreads not in targets
    assert rewrite._temp_storage_global_plan is None
    resolver = object.__new__(CoopSinglePhaseRewrite)
    resolver._func_ir = func_ir
    invocable_calls = []
    for block in func_ir.blocks.values():
        resolver._block_defs = {
            inst.target.name: inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
        }
        invocable_calls.extend(
            inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
            and isinstance(inst.value, ir.Expr)
            and inst.value.op == "call"
            and resolver._resolve_python_value(inst.value.func) is invocable
        )
    assert len(invocable_calls) == 1
    assert len(invocable_calls[0].args) == 2


def test_direct_provider_ignores_implicit_and_explicit_storage():
    from cuda.coop.numba_mlir._lowering._load_store import load as provider_load

    def kernel(source_a, source_b):
        storage = coop.TempStorage()
        payload_a = coop.ThreadData(2, types.int32)
        payload_b = coop.ThreadData(2, types.int32)
        provider_load(
            source_a,
            payload_a,
            dtype=types.int32,
            threads_per_block=32,
            items_per_thread=2,
            temp_storage=storage,
        )
        return provider_load(
            source_b,
            payload_b,
            dtype=types.int32,
            threads_per_block=32,
            items_per_thread=2,
        )

    invocable = _FakeInvocable()
    func_ir, rewrite = _rewrite_with_fake_invocable(kernel, invocable)
    targets = _call_targets(func_ir)

    assert cuda.shared.array not in targets
    assert cuda.syncthreads not in targets
    assert rewrite._temp_storage_global_plan is None


def test_getitem_temp_storage_syntax_is_not_an_accepted_descriptor_use():
    from cuda.coop.numba_mlir._lowering._load_store import load as provider_load

    def kernel(source):
        storage = coop.TempStorage()
        payload = coop.ThreadData(2, types.int32)
        return provider_load[storage](
            source,
            payload,
            dtype=types.int32,
            threads_per_block=32,
            items_per_thread=2,
        )

    func_ir = run_frontend(kernel)
    state = SimpleNamespace(
        func_ir=func_ir,
        args=(),
        typingctx=_TypingContext(),
        typemap={},
        calltypes={},
        metadata={},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    rewrite._prepare_ltoir_bundle_for_matches = lambda _matches: pytest.fail(
        "getitem TempStorage reached provider preparation"
    )
    rewrite._materialize_invocable = lambda _match: pytest.fail(
        "getitem TempStorage reached provider compilation"
    )

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="may only be passed as temp_storage=",
    ):
        rewrite.match(
            func_ir,
            func_ir.blocks[min(func_ir.blocks)],
            state.typemap,
            state.calltypes,
        )


def _planner_for_storage_policy(spec, use_specs):
    rewrite = object.__new__(CoopSinglePhaseRewrite)
    calls = [object() for _ in use_specs]
    rewrite._temp_storage_plans = {}
    rewrite._temp_storage_ctor_specs = {"storage": spec}
    rewrite._func_temp_storage_requirements = {
        "storage": _TempStorageRequirementSummary(
            uses=[
                _TempStorageUseRequirement(
                    call_assign=call,
                    order=index,
                    size_in_bytes=size,
                    alignment=alignment,
                )
                for index, (call, (size, alignment)) in enumerate(zip(calls, use_specs))
            ]
        )
    }
    return rewrite, calls


def test_shared_storage_reuses_one_aligned_slice_and_defaults_to_auto_sync():
    rewrite, calls = _planner_for_storage_policy(
        _TempStorageCtorSpec(None, None, None, "shared"),
        [(24, 8), (64, 16)],
    )

    plan = rewrite._finalize_temp_storage_plan_for_var("storage")

    assert (plan.size_in_bytes, plan.alignment, plan.auto_sync) == (64, 16, True)
    assert [plan.slices_by_call_id[id(call)].offset for call in calls] == [0, 0]


def test_shared_storage_can_delegate_synchronization_to_the_caller():
    rewrite, _ = _planner_for_storage_policy(
        _TempStorageCtorSpec(None, None, False, "shared"),
        [(24, 8), (64, 16)],
    )

    plan = rewrite._finalize_temp_storage_plan_for_var("storage")

    assert not plan.auto_sync


def test_exclusive_storage_assigns_distinct_aligned_slices_without_auto_sync():
    rewrite, calls = _planner_for_storage_policy(
        _TempStorageCtorSpec(None, None, None, "exclusive"),
        [(24, 8), (16, 16)],
    )

    plan = rewrite._finalize_temp_storage_plan_for_var("storage")

    assert (plan.size_in_bytes, plan.alignment, plan.auto_sync) == (48, 16, False)
    assert [plan.slices_by_call_id[id(call)].offset for call in calls] == [0, 32]


def test_exclusive_storage_rejects_automatic_synchronization():
    rewrite, _ = _planner_for_storage_policy(
        _TempStorageCtorSpec(None, None, True, "exclusive"),
        [(24, 8)],
    )

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="does not support auto_sync=True",
    ):
        rewrite._finalize_temp_storage_plan_for_var("storage")


def test_storage_capacity_and_alignment_are_validated_before_codegen():
    undersized, _ = _planner_for_storage_policy(
        _TempStorageCtorSpec(15, 16, None, "shared"),
        [(16, 16)],
    )
    underaligned, _ = _planner_for_storage_policy(
        _TempStorageCtorSpec(16, 8, None, "shared"),
        [(16, 16)],
    )

    with pytest.raises(CoopSinglePhaseRewriteError, match="smaller than required"):
        undersized._finalize_temp_storage_plan_for_var("storage")
    with pytest.raises(CoopSinglePhaseRewriteError, match="alignment is smaller"):
        underaligned._finalize_temp_storage_plan_for_var("storage")


def test_small_static_storage_does_not_require_a_device_query(monkeypatch):
    from cuda.coop.numba_mlir._compiler import _rewrite_storage

    rewrite = object.__new__(CoopSinglePhaseRewrite)
    monkeypatch.setattr(
        _rewrite_storage,
        "_query_device_shared_memory_limits",
        lambda: pytest.fail("small static storage queried a device"),
    )

    assert rewrite._get_device_shared_memory_limits(
        _DEFAULT_STATIC_SHARED_MEMORY_BYTES
    ) == (
        _DEFAULT_STATIC_SHARED_MEMORY_BYTES,
        _DEFAULT_STATIC_SHARED_MEMORY_BYTES,
    )


def test_large_storage_requires_an_exact_device_limit(monkeypatch):
    from cuda.coop.numba_mlir._compiler import _rewrite_storage

    rewrite = object.__new__(CoopSinglePhaseRewrite)
    monkeypatch.setattr(
        _rewrite_storage,
        "_query_device_shared_memory_limits",
        lambda: (_ for _ in ()).throw(RuntimeError("no current device")),
    )

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="require an exact current-device shared-memory query",
    ):
        rewrite._get_device_shared_memory_limits(
            _DEFAULT_STATIC_SHARED_MEMORY_BYTES + 1
        )


def _global_storage_planner(
    monkeypatch,
    *,
    size,
    default_limit,
    optin_limit,
    implicit_use_specs=(),
):
    from cuda.coop.numba_mlir._compiler import _rewrite_storage

    rewrite = object.__new__(CoopSinglePhaseRewrite)
    rewrite._state = object()
    rewrite._temp_storage_global_plan = None
    rewrite._temp_storage_ctor_specs = {"storage": object()}
    rewrite._temp_storage_ctor_order = {"storage": 0}
    rewrite._temp_storage_plans = {}
    rewrite._func_temp_storage_requirements = {
        "storage": _TempStorageRequirementSummary()
    }
    implicit_calls = [object() for _ in implicit_use_specs]
    rewrite._implicit_temp_storage_requirements = _TempStorageRequirementSummary(
        max_size_in_bytes=max(
            (use_size for use_size, _ in implicit_use_specs),
            default=0,
        ),
        max_alignment=max(
            (alignment for _, alignment in implicit_use_specs),
            default=1,
        ),
        uses=[
            _TempStorageUseRequirement(
                call_assign=call,
                order=index,
                size_in_bytes=use_size,
                alignment=alignment,
            )
            for index, (call, (use_size, alignment)) in enumerate(
                zip(implicit_calls, implicit_use_specs)
            )
        ],
    )
    rewrite._implicit_temp_storage_plan = None
    rewrite._finalize_temp_storage_plan_for_var = lambda _key: _TempStoragePlan(
        size_in_bytes=size,
        alignment=16,
        sharing="shared",
        auto_sync=True,
        slices_by_call_id={},
    )
    monkeypatch.setattr(
        _rewrite_storage,
        "_query_device_shared_memory_limits",
        lambda: {
            "max_default_shared_memory_per_block": default_limit,
            "max_optin_shared_memory_per_block": optin_limit,
        },
    )
    requested = []
    monkeypatch.setattr(
        _rewrite_storage,
        "set_required_dynamic_shared_memory",
        lambda state, value: requested.append((state, value)),
    )
    return rewrite, requested, implicit_calls


def test_storage_above_default_requests_exact_dynamic_shared_memory(monkeypatch):
    rewrite, requested, _ = _global_storage_planner(
        monkeypatch,
        size=64 * 1024,
        default_limit=48 * 1024,
        optin_limit=96 * 1024,
    )

    plan = rewrite._ensure_temp_storage_global_plan()

    assert plan.uses_dynamic_smem
    assert plan.dynamic_shared_bytes == 64 * 1024
    assert requested == [(rewrite._state, 64 * 1024)]


def test_storage_above_device_optin_limit_is_rejected(monkeypatch):
    rewrite, requested, _ = _global_storage_planner(
        monkeypatch,
        size=100 * 1024,
        default_limit=48 * 1024,
        optin_limit=96 * 1024,
    )

    with pytest.raises(CoopSinglePhaseRewriteError, match="device max opt-in"):
        rewrite._ensure_temp_storage_global_plan()
    assert requested == []


def test_implicit_and_explicit_storage_share_one_dynamic_backing(monkeypatch):
    rewrite, requested, implicit_calls = _global_storage_planner(
        monkeypatch,
        size=40 * 1024,
        implicit_use_specs=((16 * 1024, 16), (8 * 1024, 8)),
        default_limit=48 * 1024,
        optin_limit=96 * 1024,
    )

    plan = rewrite._ensure_temp_storage_global_plan()
    implicit = rewrite._implicit_temp_storage_plan

    assert plan.total_size == 56 * 1024
    assert plan.dynamic_shared_bytes == 56 * 1024
    assert requested == [(rewrite._state, 56 * 1024)]
    assert implicit is not None
    assert implicit.base_offset == 40 * 1024
    assert implicit.size_in_bytes == 16 * 1024
    assert set(implicit.slices_by_call_id) == {id(call) for call in implicit_calls}


def test_implicit_storage_counts_toward_the_optin_limit(monkeypatch):
    rewrite, requested, _ = _global_storage_planner(
        monkeypatch,
        size=80 * 1024,
        implicit_use_specs=((32 * 1024, 16),),
        default_limit=48 * 1024,
        optin_limit=96 * 1024,
    )

    with pytest.raises(CoopSinglePhaseRewriteError, match="device max opt-in"):
        rewrite._ensure_temp_storage_global_plan()
    assert requested == []
