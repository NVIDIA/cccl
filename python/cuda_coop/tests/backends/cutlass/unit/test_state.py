# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gc
import weakref
from dataclasses import dataclass
from importlib import import_module
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass")
_state = import_module("cuda.coop.cutlass._compiler._state")
ir = import_module("cutlass._mlir.ir")

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


class _Options:
    pass


class _UnhashableOptions:
    __hash__ = None


@dataclass(frozen=True)
class _Request:
    symbol_name: str


@pytest.fixture(autouse=True)
def isolated_sessions(monkeypatch):
    monkeypatch.setattr(_state, "_SESSIONS", weakref.WeakKeyDictionary())
    monkeypatch.setattr(_state, "_ID_SESSIONS", {})


@pytest.mark.parametrize("options_type", [_Options, _UnhashableOptions])
def test_nested_modules_keep_independent_provider_sessions(options_type):
    options = options_type()
    outer_module, inner_module = object(), object()
    outer = _state.get_or_create_bundle_session(options, trace_module_op=outer_module)
    outer.add(_Request("outer"))
    inner = _state.get_or_create_bundle_session(options, trace_module_op=inner_module)
    inner.add(_Request("inner"))
    assert _state.lookup_bundle_session(options) is None
    assert _state.pop_bundle_session(options) is None
    assert _state.lookup_bundle_session(options, trace_module_op=object()) is None
    assert _state.pop_bundle_session(options, trace_module_op=object()) is None

    assert _state.pop_bundle_session(options, trace_module_op=inner_module) is inner
    assert inner.request_list() == [_Request("inner")]
    resumed = _state.get_or_create_bundle_session(options, trace_module_op=outer_module)
    assert resumed is outer
    resumed.add(_Request("outer_after_nested_trace"))
    assert resumed.request_list() == [
        _Request("outer"),
        _Request("outer_after_nested_trace"),
    ]
    assert _state.pop_bundle_session(options, trace_module_op=outer_module) is outer
    assert _state.lookup_bundle_session(options) is None


def test_nested_jit_in_the_same_mlir_module_reuses_one_bundle():
    options = _Options()
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        first = _state.get_or_create_bundle_session(options, trace_module_op=module)
        nested = _state.get_or_create_bundle_session(
            options, trace_module_op=module.operation
        )
        assert nested is first
        assert _state.lookup_bundle_session(options) is first
        assert (
            _state.pop_bundle_session(options, trace_module_op=module.operation)
            is first
        )


def test_unbound_session_binds_once_without_overwriting_other_modules():
    options = _Options()
    initial = _state.get_or_create_bundle_session(options)
    module = object()
    assert (
        _state.get_or_create_bundle_session(options, trace_module_op=module) is initial
    )
    assert initial.trace_module_op is module
    other = _state.get_or_create_bundle_session(options, trace_module_op=object())
    assert other is not initial
    assert _state.lookup_bundle_session(options, trace_module_op=module) is initial


@pytest.mark.parametrize("options_type", [_Options, _UnhashableOptions])
def test_rejected_unbound_request_does_not_leak_into_next_trace(
    monkeypatch, options_type
):
    options = options_type()
    monkeypatch.setattr(_state, "_ensure_trace_hook_registered", lambda: None)
    monkeypatch.setattr(
        _state, "_get_cute_dsl", lambda: SimpleNamespace(compile_options=options)
    )

    with ir.Context(), ir.Location.unknown():
        with pytest.raises(_state.DSLRuntimeError, match="active CuTe trace module"):
            _state.register_request(_Request("unbound"))
        assert _state.lookup_bundle_session(options) is None

        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            _state.register_request(_Request("current_trace"))
        session = _state.pop_bundle_session(options, trace_module_op=module)
        assert session.request_list() == [_Request("current_trace")]
        assert _state.lookup_bundle_session(options) is None


def test_rollback_of_new_inner_trace_preserves_outer_session(monkeypatch):
    options = _Options()
    outer_module, inner_module = object(), object()
    outer = _state.get_or_create_bundle_session(options, trace_module_op=outer_module)
    outer.add(_Request("outer"))
    monkeypatch.setattr(_state, "_active_trace_module_op", lambda: inner_module)

    def get_dsl():
        return SimpleNamespace(compile_options=options)

    snapshot = _state.snapshot_active_session_state_for(get_cute_dsl=get_dsl)
    inner = _state.get_or_create_bundle_session(options, trace_module_op=inner_module)
    inner.add(_Request("failed_inner"))
    _state.restore_active_session_state_for(snapshot, get_cute_dsl=get_dsl)
    assert _state.lookup_bundle_session(options, trace_module_op=inner_module) is None
    assert _state.lookup_bundle_session(options, trace_module_op=outer_module) is outer
    assert outer.request_list() == [_Request("outer")]


def test_rollback_restores_only_its_own_trace_requests(monkeypatch):
    options = _Options()
    outer_module, inner_module = object(), object()
    outer = _state.get_or_create_bundle_session(options, trace_module_op=outer_module)
    outer.add(_Request("before"))
    monkeypatch.setattr(_state, "_active_trace_module_op", lambda: outer_module)

    def get_dsl():
        return SimpleNamespace(compile_options=options)

    snapshot = _state.snapshot_active_session_state_for(get_cute_dsl=get_dsl)
    outer.add(_Request("failed"))
    inner = _state.get_or_create_bundle_session(options, trace_module_op=inner_module)
    inner.add(_Request("inner"))
    _state.restore_active_session_state_for(snapshot, get_cute_dsl=get_dsl)
    assert outer.request_list() == [_Request("before")]
    assert inner.request_list() == [_Request("inner")]
    assert _state.lookup_bundle_session(options, trace_module_op=inner_module) is inner


def test_retry_in_a_new_module_does_not_reuse_failed_trace(monkeypatch):
    options = _Options()
    failed_module, retry_module = object(), object()
    failed = _state.get_or_create_bundle_session(options, trace_module_op=failed_module)
    failed.add(_Request("failed_trace"))
    monkeypatch.setattr(_state, "_active_trace_module_op", lambda: retry_module)

    def get_dsl():
        return SimpleNamespace(compile_options=options)

    snapshot = _state.snapshot_active_session_state_for(get_cute_dsl=get_dsl)
    retry = _state.get_or_create_bundle_session(options, trace_module_op=retry_module)
    assert retry.is_empty()
    retry.add(_Request("retry"))
    assert retry.request_list() == [_Request("retry")]
    _state.restore_active_session_state_for(snapshot, get_cute_dsl=get_dsl)
    assert _state.lookup_bundle_session(options, trace_module_op=retry_module) is None


def test_rollback_after_options_switch_preserves_unrelated_sessions(monkeypatch):
    original_options, current_options = _Options(), _Options()
    original_module, current_module, unrelated_module = object(), object(), object()
    original = _state.get_or_create_bundle_session(
        original_options, trace_module_op=original_module
    )
    original.add(_Request("original"))
    monkeypatch.setattr(_state, "_active_trace_module_op", lambda: original_module)
    snapshot = _state.snapshot_active_session_state_for(
        get_cute_dsl=lambda: SimpleNamespace(compile_options=original_options)
    )
    original.add(_Request("failed"))
    _state.get_or_create_bundle_session(current_options, trace_module_op=current_module)
    unrelated = _state.get_or_create_bundle_session(
        current_options, trace_module_op=unrelated_module
    )
    unrelated.add(_Request("unrelated"))
    monkeypatch.setattr(_state, "_active_trace_module_op", lambda: current_module)
    _state.restore_active_session_state_for(
        snapshot,
        get_cute_dsl=lambda: SimpleNamespace(compile_options=current_options),
    )
    assert original.request_list() == [_Request("original")]
    assert (
        _state.lookup_bundle_session(current_options, trace_module_op=current_module)
        is None
    )
    assert (
        _state.lookup_bundle_session(current_options, trace_module_op=unrelated_module)
        is unrelated
    )


@pytest.mark.parametrize("options_type", [_Options, _UnhashableOptions])
def test_options_lifetime_releases_unfinalized_sessions(options_type):
    options = options_type()
    session = _state.get_or_create_bundle_session(options, trace_module_op=object())
    reference = weakref.ref(session)
    del session, options
    gc.collect()
    assert reference() is None
    assert not _state._SESSIONS
    assert not _state._ID_SESSIONS


def test_nonweak_options_report_dependency_error():
    with pytest.raises(_state.DSLRuntimeError, match="weak-referenceable"):
        _state.get_or_create_bundle_session(object(), trace_module_op=object())
