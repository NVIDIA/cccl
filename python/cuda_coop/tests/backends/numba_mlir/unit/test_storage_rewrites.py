# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import numpy as np
import pytest
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.compiler import run_frontend
from numba_cuda_mlir.numbair_transforms import ir

import cuda.coop as common_coop
import cuda.coop.numba_mlir as coop
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


class _TypingContext:
    def __init__(self):
        self.refresh_count = 0

    def refresh(self):
        self.refresh_count += 1


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
def test_thread_data_rewrite_matches_runtime_alignment_aliases(alignment):
    def kernel():
        data = coop.ThreadData(
            2,
            types.int32,
            alignas=alignment,
            alignment=alignment,
        )
        return data[0]

    func_ir, _ = _rewrite(kernel)

    assert cuda.local.array in _call_targets(func_ir)


def test_thread_data_rewrite_rejects_conflicting_alignment_aliases():
    def kernel():
        return coop.ThreadData(2, types.int32, alignas=16, alignment=32)

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="alignas and alignment must match when both are set",
    ):
        _rewrite(kernel)


def test_thread_data_rewrite_accepts_explicit_default_alignment():
    def kernel():
        data = coop.ThreadData(2, types.int32, alignment=None)
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
        return common_coop.ThreadData(2, types.int32, alignment=16)

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=r"cuda\.coop\.ThreadData got unexpected keyword.*alignment",
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


class _FakeInvocable:
    files = ("storage-rewrite-test.ltoir",)
    storage_abi = "leading_pointer"
    execution_scope = "block"
    synchronization_scope = "block"

    def __init__(self, *, size_in_bytes=64, alignment=16):
        self.temp_storage_bytes = size_in_bytes
        self.temp_storage_alignment = alignment

    def __call__(self, *args):
        del args


def _frontend(function, *, ssa=False):
    func_ir = run_frontend(function)
    if ssa:
        from numba_cuda_mlir.numba_cuda.core.ir_utils import build_definitions
        from numba_cuda_mlir.numba_cuda.core.ssa import reconstruct_ssa

        func_ir = reconstruct_ssa(func_ir)
        func_ir._definitions = build_definitions(func_ir.blocks)
    return func_ir


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


def test_temp_storage_phi_rejects_distinct_constructors_before_compile():
    from cuda.coop.numba_mlir._lowering._load_store import load as provider_load

    def kernel(source, choose_first):
        storage_a = coop.TempStorage()
        storage_b = coop.TempStorage()
        if choose_first:
            selected = storage_a
        else:
            selected = storage_b
        payload = coop.ThreadData(2, types.int32)
        return provider_load(
            source,
            payload,
            dtype=types.int32,
            threads_per_block=32,
            items_per_thread=2,
            temp_storage=selected,
        )

    func_ir = _frontend(kernel, ssa=True)
    assert any(
        isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "phi"
        for block in func_ir.blocks.values()
        for inst in block.body
    )
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
        "ambiguous TempStorage reached provider preparation"
    )
    rewrite._materialize_invocable = lambda _match: pytest.fail(
        "ambiguous TempStorage reached provider compilation"
    )

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="merges distinct constructor instances",
    ):
        rewrite.match(
            func_ir,
            func_ir.blocks[min(func_ir.blocks)],
            state.typemap,
            state.calltypes,
        )


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
