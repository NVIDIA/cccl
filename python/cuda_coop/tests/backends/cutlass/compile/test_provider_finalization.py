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

    from cuda.coop.cutlass._dsl import _provider

    dsl_without_finalize = type("Dsl", (), {"compile_options": object()})()

    with pytest.raises(
        DSLRuntimeError,
        match=r"(?s)trace-finalize hook.*compatible.*CUTLASS DSL runtime",
    ):
        _provider.ensure_trace_hook_registered(
            get_cute_dsl=lambda: dsl_without_finalize,
        )


def test_managed_link_cleanup_accepts_fresh_compile_options(monkeypatch, tmp_path):
    from cuda.coop.cutlass._dsl import _provider_bundle, _provider_finalizer

    managed_path = str(tmp_path / "managed.ltoir")
    monkeypatch.setattr(
        _provider_bundle,
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
    from cuda.coop.cutlass._dsl import _provider, _provider_bundle
    from cuda.coop.cutlass._dsl import _provider_finalizer as finalizer

    renderer = _provider.BundleRenderer(
        include_lines=(),
        cccl_headers=(),
        render=lambda request: [f'extern "C" void {request.symbol_name}() {{}}'],
    )
    monkeypatch.setitem(_provider._BUNDLE_RENDERERS, "foundation_test", renderer)

    compile_options = type("CompileOptions", (), {"options": {}})()
    dsl = type("Dsl", (), {"compile_options": compile_options})()
    module = type("Module", (), {"operation": object()})()
    session = _provider.BundleSession(module.operation)
    session.add(_Request())
    _provider.set_bundle_session(compile_options, session)

    compiled_sources: list[tuple[str, tuple[str, ...]]] = []
    materialized: list[tuple[tuple[object, ...], object]] = []
    linked: list[tuple[object, str]] = []

    def compile_source(source: str, symbols: tuple[str, ...]) -> str:
        compiled_sources.append((source, symbols))
        return "/tmp/cuda_coop_foundation_test.ltoir"

    monkeypatch.setattr(finalizer, "_compile_bundle_source", compile_source)
    monkeypatch.setattr(
        _provider,
        "materialize_deferred_temp_storage_plans",
        lambda plans, target: materialized.append((plans, target)),
    )
    monkeypatch.setattr(
        _provider_bundle,
        "append_link_library_attr",
        lambda target, path: linked.append((target, path)),
    )

    finalizer._trace_finalize_hook(dsl, module, "kernel")

    assert len(compiled_sources) == 1
    assert "cuda_coop_foundation_test" in compiled_sources[0][0]
    assert compiled_sources[0][1] == ("cuda_coop_foundation_test",)
    assert materialized == [((), module)]
    assert linked == [(module, "/tmp/cuda_coop_foundation_test.ltoir")]
    assert _provider.lookup_bundle_session(compile_options) is None
