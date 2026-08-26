# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import dataclass

import pytest

pytest.importorskip("cutlass")


@dataclass(frozen=True)
class _Request:
    kind: str = "foundation_test"
    symbol_name: str = "cuda_coop_foundation_test"


def test_provider_requires_trace_finalize_capability():
    from cutlass.base_dsl.common import DSLRuntimeError

    from cuda.coop.cutlass._compiler import _state

    dsl_without_finalize = type("Dsl", (), {"compile_options": object()})()

    with pytest.raises(
        DSLRuntimeError,
        match=r"(?s)trace-finalize hook.*compatible.*CUTLASS DSL runtime",
    ):
        _state.ensure_trace_hook_registered(
            get_cute_dsl=lambda: dsl_without_finalize,
        )


def test_trace_hook_registration_holds_the_state_lock(monkeypatch):
    from cuda.coop.cutlass._compiler import _state

    class RecordingLock:
        depth = 0

        def __enter__(self):
            self.depth += 1
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            del exc_type, exc_value, traceback
            self.depth -= 1

    lock = RecordingLock()
    registered = []

    def register_hook(hook):
        assert lock.depth > 0
        registered.append(hook)

    dsl = type(
        "Dsl",
        (),
        {
            "compile_options": object(),
            "register_trace_finalize_hook": staticmethod(register_hook),
        },
    )()

    def finalizer(dsl, module, function_name):
        del dsl, module, function_name

    monkeypatch.setattr(_state, "_BUNDLE_FINALIZER", finalizer)
    monkeypatch.setattr(_state, "_SESSION_SCOPE", "cuda.coop.cutlass")
    monkeypatch.setattr(_state, "_STATE_LOCK", lock)

    _state.ensure_trace_hook_registered(
        finalizer=finalizer,
        scope="cuda.coop.cutlass",
        get_cute_dsl=lambda: dsl,
    )

    assert registered == [_state._trace_finalize_dispatcher]


def test_managed_link_cleanup_accepts_fresh_compile_options(monkeypatch, tmp_path):
    import cuda.coop.cutlass._compiler._cache as _provider_cache
    import cuda.coop.cutlass._compiler._finalize as _provider_finalizer

    managed_path = str(tmp_path / "managed.ltoir")
    monkeypatch.setattr(
        _provider_cache,
        "managed_bundle_paths",
        lambda: frozenset({managed_path}),
    )
    dsl = type(
        "Dsl",
        (),
        {"compile_options": type("Options", (), {"options": {}})()},
    )()

    _provider_finalizer._remove_managed_bundle_link_options(dsl)

    assert dsl.compile_options.options == {}


def test_trace_finalizer_compiles_and_links_registered_requests(monkeypatch):
    import cuda.coop.cutlass._compiler._bundle as _provider_bundle
    import cuda.coop.cutlass._compiler._finalize as finalizer
    import cuda.coop.cutlass._compiler._rendering as _rendering
    import cuda.coop.cutlass._compiler._state as _state
    import cuda.coop.cutlass._compiler._storage as _storage
    import cuda.coop.cutlass._compiler._types as _types

    renderer = _types.BundleRenderer(
        include_lines=(),
        cccl_headers=(),
        render=lambda request: [f'extern "C" void {request.symbol_name}() {{}}'],
    )
    monkeypatch.setitem(_rendering._BUNDLE_RENDERERS, "foundation_test", renderer)

    compile_options = type("CompileOptions", (), {"options": {}})()
    dsl = type("Dsl", (), {"compile_options": compile_options})()
    module = type("Module", (), {"operation": object()})()
    session = _state.BundleSession(module.operation)
    session.add(_Request())
    _state.set_bundle_session(compile_options, session)

    compiled_sources: list[str] = []
    materialized: list[tuple[object, ...]] = []
    linked: list[tuple[object, str]] = []

    def compile_source(source: str) -> str:
        compiled_sources.append(source)
        return "/tmp/cuda_coop_foundation_test.ltoir"

    monkeypatch.setattr(finalizer, "_compile_bundle_source", compile_source)
    monkeypatch.setattr(
        _storage,
        "materialize_deferred_temp_storage_plans",
        lambda plans: materialized.append(plans),
    )
    monkeypatch.setattr(
        _provider_bundle,
        "append_link_library_attr",
        lambda target, path: linked.append((target, path)),
    )

    finalizer._trace_finalize_hook(dsl, module, "kernel")

    assert len(compiled_sources) == 1
    assert "cuda_coop_foundation_test" in compiled_sources[0]
    assert materialized == [()]
    assert linked == [(module, "/tmp/cuda_coop_foundation_test.ltoir")]
    assert _state.lookup_bundle_session(compile_options) is None


def test_trace_finalizer_preserves_a_session_for_its_owning_module(monkeypatch):
    import cuda.coop.cutlass._compiler._bundle as _provider_bundle
    import cuda.coop.cutlass._compiler._finalize as finalizer
    import cuda.coop.cutlass._compiler._state as _state

    compile_options = type("CompileOptions", (), {"options": {}})()
    dsl = type("Dsl", (), {"compile_options": compile_options})()
    owning_module = type("Module", (), {"operation": object()})()
    unrelated_module = type("Module", (), {"operation": object()})()
    session = _state.BundleSession(owning_module.operation)
    session.add(_Request())
    _state.set_bundle_session(compile_options, session)

    compiled_sources: list[str] = []
    linked: list[tuple[object, str]] = []
    monkeypatch.setattr(
        finalizer,
        "_render_bundle_source",
        lambda requests: f"requests={len(requests)}",
    )
    monkeypatch.setattr(
        finalizer,
        "_compile_bundle_source",
        lambda source: compiled_sources.append(source) or "/tmp/provider.ltoir",
    )
    monkeypatch.setattr(
        _provider_bundle,
        "append_link_library_attr",
        lambda target, path: linked.append((target, path)),
    )

    finalizer._trace_finalize_hook(dsl, unrelated_module, "other")

    assert _state.lookup_bundle_session(compile_options) is session
    assert compiled_sources == []
    assert linked == []

    finalizer._trace_finalize_hook(dsl, owning_module, "owner")

    assert _state.lookup_bundle_session(compile_options) is None
    assert compiled_sources == ["requests=1"]
    assert linked == [(owning_module, "/tmp/provider.ltoir")]


def test_provider_session_snapshot_restores_trace_module_binding(monkeypatch):
    from cuda.coop.cutlass._compiler import _state

    compile_options = type("CompileOptions", (), {"options": {}})()
    dsl = type("Dsl", (), {"compile_options": compile_options})()
    trace_module = object()
    session = _state.BundleSession()
    _state.set_bundle_session(compile_options, session)
    monkeypatch.setattr(_state, "_active_trace_module_op", lambda: trace_module)

    snapshot = _state.snapshot_active_session_state_for(get_cute_dsl=lambda: dsl)
    assert session.trace_module_op is None

    assert session.bind_trace_module(trace_module)
    session.add(_Request())
    _state.restore_active_session_state_for(snapshot, get_cute_dsl=lambda: dsl)

    assert session.trace_module_op is None
    assert session.is_empty()


def test_provider_session_snapshot_restores_nonweak_scalar_types():
    from cutlass.base_dsl.typing import Int32

    from cuda.coop.cutlass._compiler import _state

    class Scalar:
        __slots__ = ()

    compile_options = type("CompileOptions", (), {"options": {}})()
    dsl = type("Dsl", (), {"compile_options": compile_options})()
    value = Scalar()
    session = _state.BundleSession()
    _state.set_bundle_session(compile_options, session)
    _state.remember_scalar_result_type(
        value,
        Int32,
        compile_options=compile_options,
    )
    snapshot = _state.snapshot_active_session_state_for(get_cute_dsl=lambda: dsl)

    assert _state.canonical_dsl_type(value) is Int32
    assert _state.pop_bundle_session(compile_options) is session
    assert _state.canonical_dsl_type(value) is Scalar

    try:
        _state.restore_active_session_state_for(snapshot, get_cute_dsl=lambda: dsl)
        assert _state.lookup_bundle_session(compile_options) is not session
        assert _state.canonical_dsl_type(value) is Int32
    finally:
        _state.pop_bundle_session(compile_options)


def test_provider_session_snapshot_rolls_back_weak_scalar_types():
    from cutlass.base_dsl.typing import Int32

    from cuda.coop._core import ResultVisibility
    from cuda.coop.cutlass import _value_metadata
    from cuda.coop.cutlass._compiler import _state

    class Scalar:
        pass

    compile_options = type("CompileOptions", (), {"options": {}})()
    dsl = type("Dsl", (), {"compile_options": compile_options})()
    value = Scalar()
    metadata = _value_metadata.ValueGroupMetadata(
        _value_metadata.DefinedThreadDomain.all_callers(),
        ResultVisibility.PER_MEMBER,
    )
    session = _state.BundleSession()
    _state.set_bundle_session(compile_options, session)
    snapshot = _state.snapshot_active_session_state_for(get_cute_dsl=lambda: dsl)

    try:
        _state.remember_scalar_result_type(
            value,
            Int32,
            compile_options=compile_options,
            group_metadata=metadata,
        )
        assert _state.canonical_dsl_type(value) is Int32
        assert _state.scalar_result_group_metadata(value) is metadata

        _state.restore_active_session_state_for(snapshot, get_cute_dsl=lambda: dsl)

        assert _state.canonical_dsl_type(value) is Scalar
        assert _state.scalar_result_group_metadata(value) is None
    finally:
        _state.pop_bundle_session(compile_options)


@pytest.mark.parametrize("snapshot_has_state", [False, True])
def test_provider_session_restore_closes_a_switched_compile_options_session(
    snapshot_has_state,
):
    from cuda.coop.cutlass._compiler import _state

    compile_options_a = type("CompileOptions", (), {"options": {}})()
    compile_options_b = type("CompileOptions", (), {"options": {}})()
    dsl = type("Dsl", (), {"compile_options": compile_options_a})()
    session_a = None
    if snapshot_has_state:
        session_a = _state.BundleSession()
        session_a.add(_Request())
        _state.set_bundle_session(compile_options_a, session_a)
    snapshot = _state.snapshot_active_session_state_for(get_cute_dsl=lambda: dsl)

    dsl.compile_options = compile_options_b
    session_b = _state.BundleSession()
    session_b.add(_Request(symbol_name="cuda_coop_switched_options"))
    _state.set_bundle_session(compile_options_b, session_b)

    try:
        _state.restore_active_session_state_for(snapshot, get_cute_dsl=lambda: dsl)

        assert _state.lookup_bundle_session(compile_options_b) is None
        restored_a = _state.lookup_bundle_session(compile_options_a)
        if snapshot_has_state:
            assert restored_a is session_a
            assert restored_a.request_list() == [_Request()]
        else:
            assert restored_a is None
    finally:
        _state.pop_bundle_session(compile_options_a)
        _state.pop_bundle_session(compile_options_b)
