# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import dataclasses
import inspect

import pytest

from ....support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT

cutlass_typing = pytest.importorskip("cutlass.base_dsl.typing")

import cuda.coop.cutlass as coop
from cuda.coop._core import (
    ArgumentBinding,
    GroupLoadStoreAlgorithm,
    GroupLoadStoreKind,
    LaunchFacts,
    StorageOwnership,
)
from cuda.coop.cutlass._dsl import _cub_load_store_provider as provider
from cuda.coop.cutlass._dsl import _provider as provider_support
from cuda.coop.cutlass._dsl import _single_phase as single_phase
from cuda.coop.cutlass._dsl._load_store import (
    ScopedLoadStoreRoute,
    ScopedLoadStoreRouteDecision,
)
from cuda.coop.cutlass._dsl.block import _load_store as scoped_load_store
from cuda.coop.cutlass._group_load_store import _make_group_load_store_plan

Int32 = cutlass_typing.Int32


def _plan(
    kind: GroupLoadStoreKind,
    *,
    group=None,
    algorithm: GroupLoadStoreAlgorithm = GroupLoadStoreAlgorithm.TRANSPOSE,
):
    if group is None:
        group = coop.this_block()
    return _make_group_load_store_plan(
        group=group,
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        kind=kind,
        dtype=Int32,
        items_per_thread=4,
        algorithm=algorithm,
        valid_items=ArgumentBinding.omitted(),
        oob_default=ArgumentBinding.omitted(),
        offset=ArgumentBinding.omitted(),
        source="cutlass_root",
    ).require_supported()


def _with_caller_owned_storage(plan):
    assert plan.temp_storage is not None
    return dataclasses.replace(
        plan,
        temp_storage=dataclasses.replace(
            plan.temp_storage,
            ownership=StorageOwnership.CALLER,
            address_space="shared",
            cpp_type="typename implementation_type::TempStorage",
            instances=1,
            instance_index="cta",
            exact_layout_required=True,
        ),
    )


def _request(kind: GroupLoadStoreKind, *, external_scratch: bool):
    plan = _plan(kind)
    if external_scratch:
        plan = _with_caller_owned_storage(plan)
    return provider._CubLoadStoreRequest(
        plan=plan,
        value_type=Int32,
        external_scratch=external_scratch,
    )


@pytest.mark.parametrize(
    ("kind", "cpp_class", "algorithm"),
    (
        (GroupLoadStoreKind.LOAD, "BlockLoad", "BLOCK_LOAD_TRANSPOSE"),
        (GroupLoadStoreKind.STORE, "BlockStore", "BLOCK_STORE_TRANSPOSE"),
    ),
)
def test_static_and_external_requests_have_distinct_abis_and_exact_probe(
    kind,
    cpp_class,
    algorithm,
):
    static_request = _request(kind, external_scratch=False)
    external_request = _request(kind, external_scratch=True)

    assert static_request != external_request
    assert hash(static_request) != hash(external_request)
    assert external_request.symbol_name == (
        f"{static_request.symbol_name}_external_scratch"
    )

    static_source = "\n".join(provider._render_cub_load_store(static_request))
    external_source = "\n".join(provider._render_cub_load_store(external_request))

    assert "__shared__ typename implementation_type::TempStorage storage;" in (
        static_source
    )
    assert "cuda_coop_cutlass_block_sync();" in static_source
    assert "unsigned int temp_storage_smem_addr" in external_source
    assert "int temp_storage_bytes" in external_source
    assert "int temp_storage_auto_sync" in external_source
    assert "cuda_coop_cutlass_shared_ptr(temp_storage_smem_addr)" in external_source
    assert "typename implementation_type::TempStorage*>(temp_storage_ptr)" in (
        external_source
    )
    assert "temp_storage_bytes < required_temp_bytes" in external_source
    assert "required_temp_alignment - 1ull" in external_source
    assert "__shared__ typename implementation_type::TempStorage" not in (
        external_source
    )
    assert "if (temp_storage_auto_sync != 0)" in external_source
    assert "cuda_coop_cutlass_block_sync();" in external_source

    assert provider._cub_load_store_scratch_layout_probe(static_request) is None
    probe = provider._cub_load_store_scratch_layout_probe(external_request)
    assert probe is not None
    assert probe.requirement_key == external_request.scratch_requirement_key
    assert probe.size_expression == f"sizeof({external_request.scratch_cpp_type})"
    assert probe.alignment_expression == (
        f"alignof({external_request.scratch_cpp_type})"
    )
    assert external_request.scratch_cpp_type == (
        f"typename ::cub::{cpp_class}<int, 64, 4, "
        f"::cub::{algorithm}, 1, 1>::TempStorage"
    )


def test_external_request_requires_exact_caller_owned_block_storage():
    with pytest.raises(ValueError, match="storage ownership"):
        provider._CubLoadStoreRequest(
            plan=_plan(GroupLoadStoreKind.LOAD),
            value_type=Int32,
            external_scratch=True,
        )

    warp_plan = _with_caller_owned_storage(
        _plan(
            GroupLoadStoreKind.LOAD,
            group=coop.this_warp(),
            algorithm=GroupLoadStoreAlgorithm.TRANSPOSE,
        )
    )
    with pytest.raises(ValueError, match="block-scoped only"):
        provider._CubLoadStoreRequest(
            plan=warp_plan,
            value_type=Int32,
            external_scratch=True,
        )


def test_root_and_scoped_frontends_forward_keyword_only_storage(
    monkeypatch, set_cutlass_launch_facts
):
    set_cutlass_launch_facts(64)
    storage = coop.TempStorage()
    calls = []

    def capture_load(**payload):
        calls.append(("load", payload, single_phase.get_active_single_phase_context()))
        return payload["output"]

    def capture_store(**payload):
        calls.append(("store", payload, single_phase.get_active_single_phase_context()))

    monkeypatch.setattr(provider, "provider_load", capture_load)
    monkeypatch.setattr(provider, "provider_store", capture_store)
    monkeypatch.setattr(
        scoped_load_store,
        "classify_scoped_load_store_route",
        lambda *args, **kwargs: ScopedLoadStoreRouteDecision(
            route=ScopedLoadStoreRoute.CANONICAL_CUB,
            reason="test canonical route",
            exact_block_dim=(64, 1, 1),
        ),
    )

    root_items = coop.ThreadData(4, dtype=Int32)
    assert (
        coop.load(
            coop.this_block(),
            object(),
            root_items,
            algorithm="transpose",
            temp_storage=storage,
        )
        is root_items
    )
    coop.store(
        coop.this_block(),
        object(),
        root_items,
        algorithm="transpose",
        temp_storage=storage,
    )

    scoped_items = coop.ThreadData(4, dtype=Int32)
    assert (
        coop._block.load(
            object(),
            scoped_items,
            temp_storage=storage,
            launch_metadata={"block": 64},
        )
        is scoped_items
    )
    coop._block.store(
        object(),
        scoped_items,
        temp_storage=storage,
        launch_metadata={"block": 64},
    )

    assert inspect.signature(coop.load).parameters["temp_storage"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )
    assert inspect.signature(coop.store).parameters["temp_storage"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )
    assert [kind for kind, _, _ in calls] == ["load", "store", "load", "store"]
    assert all(payload["temp_storage"] is storage for _, payload, _ in calls)


@pytest.mark.parametrize("primitive", ("load", "store"))
def test_scoped_indexing_route_rejects_deferred_storage(monkeypatch, primitive):
    storage = coop.TempStorage()
    monkeypatch.setattr(
        scoped_load_store,
        "classify_scoped_load_store_route",
        lambda *args, **kwargs: ScopedLoadStoreRouteDecision(
            route=ScopedLoadStoreRoute.CUTE_INDEXING_PAYLOAD_ADAPTER,
            reason="test fallback route",
            exact_block_dim=None,
        ),
    )

    if primitive == "load":

        def invoke():
            return coop._block.load(
                object(),
                coop.ThreadData(4, dtype=Int32),
                temp_storage=storage,
            )

    else:

        def invoke():
            return coop._block.store(
                object(),
                coop.ThreadData(4, dtype=Int32),
                temp_storage=storage,
            )

    with pytest.raises(
        NotImplementedError,
        match=r"deferred TempStorage.*canonical CUB|canonical CUB.*deferred TempStorage",
    ):
        invoke()


def test_provider_reconciles_explicit_and_active_context_storage(monkeypatch):
    explicit = coop.TempStorage()
    active = coop.TempStorage()
    context = single_phase.SinglePhaseContext(
        thread_data=None,
        temp_storage=active,
    )

    monkeypatch.setattr(provider, "_resolve_memory_type", lambda *args, **kwargs: Int32)
    monkeypatch.setattr(provider, "_memory_pointer", lambda *args, **kwargs: object())

    with single_phase.activate_single_phase_context(context):
        with pytest.raises(ValueError, match="two TempStorage objects"):
            provider.provider_load(
                group=coop.this_block(),
                launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
                source=object(),
                output=coop.ThreadData(4, dtype=Int32),
                algorithm=GroupLoadStoreAlgorithm.TRANSPOSE,
                valid_items=None,
                valid_items_binding=ArgumentBinding.omitted(),
                oob_default=None,
                oob_default_binding=ArgumentBinding.omitted(),
                offset=None,
                offset_binding=ArgumentBinding.omitted(),
                temp_storage=explicit,
            )


class _FakeTensor:
    class _Iterator:
        llvm_ptr = object()

    iterator = _Iterator()

    def __getitem__(self, index):
        return Int32(index)


@pytest.mark.parametrize("kind", (GroupLoadStoreKind.LOAD, GroupLoadStoreKind.STORE))
def test_materialization_restores_session_after_ffi_failure(monkeypatch, kind):
    storage = coop.TempStorage()
    snapshot = object()
    restored = []
    registrations = []
    ffi_calls = []

    monkeypatch.setattr(provider, "_resolve_memory_type", lambda *args, **kwargs: Int32)
    monkeypatch.setattr(provider, "_memory_pointer", lambda *args, **kwargs: object())
    monkeypatch.setattr(provider._cute, "make_rmem_tensor", lambda *args: _FakeTensor())
    monkeypatch.setattr(
        provider_support,
        "snapshot_active_session_state",
        lambda: snapshot,
    )
    monkeypatch.setattr(
        provider_support,
        "restore_active_session_state",
        restored.append,
    )
    monkeypatch.setattr(
        provider_support,
        "register_request",
        lambda request: registrations.append(("request", request)),
    )
    monkeypatch.setattr(
        provider_support,
        "register_deferred_temp_storage_event",
        lambda *args, **kwargs: (
            registrations.append(("event", args, kwargs))
            or (object(), object(), object())
        ),
    )

    def failing_ffi(**_kwargs):
        def invoke(*args):
            ffi_calls.append(args)
            raise RuntimeError("forced FFI failure")

        return invoke

    monkeypatch.setattr(provider, "ffi", failing_ffi)
    monkeypatch.setattr(provider.llvm.PointerType, "get", lambda *_args: object())

    common = {
        "group": coop.this_block(),
        "launch": LaunchFacts(exact_block_dim=(64, 1, 1)),
        "algorithm": GroupLoadStoreAlgorithm.TRANSPOSE,
        "valid_items": None,
        "valid_items_binding": ArgumentBinding.omitted(),
        "offset": None,
        "offset_binding": ArgumentBinding.omitted(),
        "temp_storage": storage,
    }
    if kind is GroupLoadStoreKind.LOAD:

        def invoke():
            return provider.provider_load(
                source=object(),
                output=coop.ThreadData(4, dtype=Int32),
                oob_default=None,
                oob_default_binding=ArgumentBinding.omitted(),
                **common,
            )

    else:

        def invoke():
            return provider.provider_store(
                destination=object(),
                value=coop.ThreadData.from_values(
                    Int32(1),
                    Int32(2),
                    Int32(3),
                    Int32(4),
                ),
                **common,
            )

    context = single_phase.SinglePhaseContext(
        thread_data=None,
        temp_storage=storage,
    )
    with single_phase.activate_single_phase_context(context):
        with pytest.raises(RuntimeError, match="forced FFI failure"):
            invoke()

    assert [entry[0] for entry in registrations] == ["request", "event"]
    event = registrations[1]
    assert event[2]["primitive_name"] == kind.value
    assert restored == [snapshot]
    if kind is GroupLoadStoreKind.LOAD:
        assert ffi_calls[0][-1] is _FakeTensor.iterator.llvm_ptr
