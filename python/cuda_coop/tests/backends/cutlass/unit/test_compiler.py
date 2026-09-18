# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provider artifact identity, corruption recovery, and trace finalization."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass")

from cutlass import Int32
from cutlass.base_dsl.compiler import LinkLibraries

from cuda.coop._core import ArgumentBinding, this_block
from cuda.coop.cutlass._compiler import (
    _bundle,
    _cache,
    _finalize,
    _nvrtc,
    _rendering,
    _state,
)
from cuda.coop.cutlass._lowering._load_store import _CubLoadStoreRequest
from tests.support.group_planning import _load_store, _plan

pytestmark = [pytest.mark.unit, pytest.mark.backend_cutlass]


@pytest.fixture
def compilation(monkeypatch, tmp_path):
    monkeypatch.setenv(_cache.CACHE_DIR_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(_cache, "_SOURCE_CACHE", {})
    monkeypatch.setattr(_cache, "_MANAGED_BUNDLE_PATHS", set())
    state = SimpleNamespace(
        calls=[],
        context=_nvrtc.CompileContext(
            ("/headers",),
            "headers-v1",
            "/toolkit",
            (13, 3),
            "/nvrtc",
            "/builtins",
            (13, 3),
            "/nvjitlink",
            (13, 3),
        ),
    )
    monkeypatch.setattr(
        _nvrtc, "resolve_compile_context", lambda headers: state.context
    )

    def compile_source(source, options):
        state.calls.append((source, options))
        return b"test-lto-ir:" + source.encode()

    monkeypatch.setattr(_nvrtc, "compile_ltoir", compile_source)
    return state


def _compile(source="provider", arch="compute_120"):
    return _bundle.compile_bundle_source(source, arch=arch, required_headers=())


def test_cache_reuses_only_intact_artifacts(compilation):
    path = Path(_compile())
    assert _compile() == str(path)
    assert len(compilation.calls) == 1
    # A fresh process can reuse disk artifacts; a damaged artifact is rebuilt.
    _cache._SOURCE_CACHE.clear()
    assert _compile() == str(path)
    assert len(compilation.calls) == 1
    path.write_bytes(b"damaged-lto")
    assert _compile() == str(path)
    assert len(compilation.calls) == 2
    assert path.read_bytes() == b"test-lto-ir:provider"
    assert str(path) in _cache.managed_bundle_paths()


def test_cache_identity_includes_source_target_headers_and_toolkit(compilation):
    paths = {_compile(), _compile("other"), _compile(arch="compute_90")}
    compilation.context = replace(compilation.context, header_identity="headers-v2")
    paths.add(_compile())
    compilation.context = replace(compilation.context, nvrtc_path="/other-nvrtc")
    paths.add(_compile())
    assert len(paths) == len(compilation.calls) == 5


def test_concurrent_compilation_publishes_one_artifact(compilation):
    with ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(lambda _: _compile(), range(12)))
    assert len(set(paths)) == 1
    assert len(compilation.calls) == 1


def test_compilation_failure_does_not_poison_retry(compilation, monkeypatch):
    compile_source = _nvrtc.compile_ltoir

    def fail(*args):
        raise RuntimeError("deliberate compiler failure")

    monkeypatch.setattr(_nvrtc, "compile_ltoir", fail)
    with pytest.raises(RuntimeError, match="deliberate compiler failure"):
        _compile()
    assert not list(Path(_cache.configured_cache_dir()).glob("*.ltoir"))
    monkeypatch.setattr(_nvrtc, "compile_ltoir", compile_source)
    assert Path(_compile()).is_file()


def _request(kind="load", **kwargs):
    return _CubLoadStoreRequest(
        _plan(this_block(), _load_store(kind, dtype=Int32, **kwargs)), Int32
    )


def test_direct_provider_has_no_storage_abi_or_barriers():
    load = _request(valid_items=ArgumentBinding.runtime())
    store = _request("store")
    source = _rendering.render_bundle_source([store, load, load])
    assert source.count(f"void {load.symbol_name}(") == 1
    assert source.count(f"void {store.symbol_name}(") == 1
    assert "TempStorage" not in source
    assert "__shared__" not in source
    assert "__syncthreads" not in source
    # Guarded loads seed invalid items from the existing output payload.
    assert "result_items[0], result_items[1]" in source
    assert "valid_items < 0 || valid_items > 128" in source


def test_storage_free_request_identity_ignores_storage_controls():
    from cuda.coop._core import StorageOwnership

    default = _request()
    explicit = _request(
        storage_ownership=StorageOwnership.CALLER,
        storage_sharing="exclusive",
        storage_size_in_bytes=256,
        storage_alignment=64,
        storage_auto_sync=False,
    )
    assert default == explicit
    assert default.symbol_name == explicit.symbol_name


def test_finalize_preserves_unrelated_session_and_user_link_libraries(
    compilation, monkeypatch
):
    class Options:
        def __init__(self):
            self.options = {}

    options = Options()
    owned_module, unrelated_module = object(), object()
    session = _state.get_or_create_bundle_session(options, trace_module_op=owned_module)
    session.add(_request())
    path = _compile()
    options.options[LinkLibraries] = LinkLibraries(f"/user/provider.ltoir,{path}")
    dsl = SimpleNamespace(compile_options=options)
    _finalize._trace_finalize_hook(dsl, unrelated_module, "unrelated")
    assert _state.lookup_bundle_session(options) is session
    assert options.options[LinkLibraries].value == "/user/provider.ltoir"
    linked = []
    monkeypatch.setattr(
        _finalize._target, "resolve_nvrtc_arch", lambda *a: "compute_120"
    )
    monkeypatch.setattr(
        _bundle, "append_link_library_attr", lambda m, p: linked.append((m, p))
    )
    _finalize._trace_finalize_hook(dsl, owned_module, "owned")
    assert _state.lookup_bundle_session(options) is None
    assert len(linked) == 1
    assert linked[0][0] is owned_module
    assert Path(linked[0][1]).is_file()
    _finalize._trace_finalize_hook(dsl, owned_module, "owned")
    assert len(linked) == 1
