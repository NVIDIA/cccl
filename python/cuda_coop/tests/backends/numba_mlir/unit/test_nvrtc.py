# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import inspect
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cuda.coop import _headers
from cuda.coop._headers._identity import include_dirs_identity
from cuda.coop._headers._toolkit import ToolkitCompilerLibraries
from cuda.coop.numba_mlir._compiler import _nvrtc


def _fake_nvrtc(
    actual_version: tuple[int, int],
    *,
    result: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        nvrtcResult=SimpleNamespace(NVRTC_SUCCESS=0),
        nvrtcVersion=lambda: (result, *actual_version),
    )


def _libraries(**changes) -> ToolkitCompilerLibraries:
    libraries = ToolkitCompilerLibraries(
        toolkit_root="/cuda/toolkit",
        toolkit_version=(13, 3),
        nvrtc_path="/cuda/toolkit/lib/libnvrtc.so.13",
        nvrtc_builtins_path="/cuda/toolkit/lib/libnvrtc-builtins.so.13.3",
        nvjitlink_path="/cuda/toolkit/lib/libnvJitLink.so.13",
        nvrtc_version=(13, 3),
        nvjitlink_version=(13, 4),
    )
    return replace(libraries, **changes)


def _context(**changes) -> _nvrtc.CompileContext:
    context = _nvrtc.CompileContext(
        toolkit_root="/cuda/toolkit",
        toolkit_version=(13, 3),
        nvrtc_path="/cuda/toolkit/lib/libnvrtc.so.13",
        nvrtc_builtins_path="/cuda/toolkit/lib/libnvrtc-builtins.so.13.3",
        nvjitlink_path="/cuda/toolkit/lib/libnvJitLink.so.13",
        nvrtc_version=_nvrtc.version(13, 3),
        nvjitlink_version=(13, 4),
        include_dirs=("/cccl/include", "/cuda/toolkit/include"),
        header_identity="headers-a",
    )
    return replace(context, **changes)


def test_include_options_use_filesystem_encoding() -> None:
    include_dir = "/tmp/cuda-coop-π/include"

    assert _nvrtc._include_options((include_dir,)) == [
        os.fsencode(f"--include-path={include_dir}")
    ]


def test_required_headers_cover_executable_provider_includes() -> None:
    assert {
        "cub/block/block_load.cuh",
        "cub/block/block_store.cuh",
    } <= set(_nvrtc._REQUIRED_HEADERS)


def test_required_headers_reject_a_partial_configured_bundle(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "partial-bundle"
    missing = "cuda/experimental/group.cuh"
    present_headers = (set(_nvrtc._REQUIRED_HEADERS) - {missing}) | {"cub/version.cuh"}
    for header in present_headers:
        destination = bundle / header
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.touch()

    with pytest.raises(_headers.HeaderResolutionError, match=missing):
        _headers.resolve_include_paths(
            start=Path(__file__),
            configured_roots=(bundle,),
            required_headers=_nvrtc._REQUIRED_HEADERS,
        )


def test_resolve_context_captures_exact_toolchain_after_lazy_binding_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cccl_include = tmp_path / "cccl" / "include"
    cuda_include = tmp_path / "cuda" / "include"
    cccl_include.mkdir(parents=True)
    cuda_include.mkdir(parents=True)
    (cccl_include / "primitive.cuh").write_bytes(b"cccl header")
    (cuda_include / "cuda_runtime.h").write_bytes(b"cuda header")
    include_dirs = (cccl_include, cuda_include)
    libraries = _libraries()
    events: list[object] = []
    real_identity = include_dirs_identity

    def resolve_paths(**kwargs):
        events.append("headers")
        assert kwargs["required_headers"] == _nvrtc._REQUIRED_HEADERS
        return SimpleNamespace(
            as_tuple=lambda: include_dirs,
            cuda=(cuda_include,),
        )

    def preload(paths):
        events.append(("preload", paths))
        return libraries

    def load_nvrtc():
        events.append("binding")
        return _fake_nvrtc((13, 3))

    def identify(paths):
        events.append(("identity", paths))
        return real_identity(paths)

    monkeypatch.setattr(_nvrtc, "resolve_include_paths", resolve_paths)
    monkeypatch.setattr(_nvrtc, "preload_toolkit_compiler_libraries", preload)
    monkeypatch.setattr(_nvrtc, "_load_nvrtc", load_nvrtc)
    monkeypatch.setattr(_nvrtc, "include_dirs_identity", identify)

    context = _nvrtc.resolve_compile_context()
    expected_identity = real_identity(tuple(str(path) for path in include_dirs))

    assert events == [
        "headers",
        ("preload", (cuda_include,)),
        "binding",
        ("identity", tuple(str(path) for path in include_dirs)),
    ]
    assert expected_identity.recursive_walks == 2
    assert context == _nvrtc.CompileContext(
        toolkit_root=libraries.toolkit_root,
        toolkit_version=libraries.toolkit_version,
        nvrtc_path=libraries.nvrtc_path,
        nvrtc_builtins_path=libraries.nvrtc_builtins_path,
        nvjitlink_path=libraries.nvjitlink_path,
        nvrtc_version=_nvrtc.version(13, 3),
        nvjitlink_version=libraries.nvjitlink_version,
        include_dirs=tuple(str(path) for path in include_dirs),
        header_identity=expected_identity.digest,
    )


def test_resolve_context_preserves_order_and_recursive_header_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    cuda = tmp_path / "cuda"
    for include_dir in (first, second, cuda):
        include_dir.mkdir()
        (include_dir / "header.cuh").write_bytes(b"same content")
    ordered = [first, second, cuda]
    monkeypatch.setattr(
        _nvrtc,
        "resolve_include_paths",
        lambda **kwargs: SimpleNamespace(
            as_tuple=lambda: tuple(ordered),
            cuda=(cuda,),
        ),
    )
    monkeypatch.setattr(
        _nvrtc,
        "preload_toolkit_compiler_libraries",
        lambda paths: _libraries(),
    )
    monkeypatch.setattr(
        _nvrtc,
        "_load_nvrtc",
        lambda: _fake_nvrtc((13, 3)),
    )

    forward = _nvrtc.resolve_compile_context()
    ordered[:] = [second, first, cuda]
    reverse = _nvrtc.resolve_compile_context()
    (first / "header.cuh").write_bytes(b"mutated content")
    mutated = _nvrtc.resolve_compile_context()

    assert forward.include_dirs == tuple(str(path) for path in (first, second, cuda))
    assert reverse.include_dirs == tuple(str(path) for path in (second, first, cuda))
    assert forward.header_identity != reverse.header_identity
    assert reverse.header_identity != mutated.header_identity


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("toolkit_root", "/other/toolkit"),
        ("toolkit_version", (13, 4)),
        ("nvrtc_path", "/other/libnvrtc.so.13"),
        ("nvrtc_builtins_path", "/other/libnvrtc-builtins.so.13.3"),
        ("nvjitlink_path", "/other/libnvJitLink.so.13"),
        ("nvrtc_version", _nvrtc.version(13, 4)),
        ("nvjitlink_version", (13, 5)),
        ("include_dirs", ("/cuda/toolkit/include", "/cccl/include")),
        ("header_identity", "headers-b"),
    ),
)
def test_compile_context_symbol_suffix_covers_every_identity_field(
    field: str,
    replacement: object,
) -> None:
    context = _context()
    changed = replace(context, **{field: replacement})

    assert context.symbol_suffix == _context().symbol_suffix
    assert context.symbol_suffix != changed.symbol_suffix


def test_compile_forwards_complete_context_into_cache_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    context = _context()
    monkeypatch.setattr(
        _nvrtc,
        "resolve_compile_context",
        lambda: pytest.fail("compile must not resolve a supplied context again"),
    )
    monkeypatch.setattr(_nvrtc, "_dump_source", lambda *args: None)
    monkeypatch.setattr(
        _nvrtc,
        "compile_impl",
        lambda **kwargs: calls.append(kwargs) or b"ltoir",
    )

    runtime_version, result = _nvrtc.compile(
        cpp="int provider;",
        cc=90,
        rdc=True,
        code="lto",
        context=context,
    )

    assert runtime_version == context.nvrtc_version
    assert result == b"ltoir"
    assert calls == [
        {
            "cpp": "int provider;",
            "cc": 90,
            "rdc": True,
            "code": "lto",
            "toolkit_root": context.toolkit_root,
            "toolkit_version": context.toolkit_version,
            "nvrtc_path": context.nvrtc_path,
            "nvrtc_builtins_path": context.nvrtc_builtins_path,
            "nvjitlink_path": context.nvjitlink_path,
            "nvrtc_version": context.nvrtc_version,
            "nvjitlink_version": context.nvjitlink_version,
            "include_dirs": context.include_dirs,
            "header_identity": context.header_identity,
            "compiler_options": _nvrtc._compiler_options(
                cc=90,
                rdc=True,
                code="lto",
                include_dirs=context.include_dirs,
            ),
        }
    ]


def test_resolve_context_rejects_loaded_nvrtc_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    include_dir = Path("/cuda/toolkit/include")
    events: list[str] = []
    monkeypatch.setattr(
        _nvrtc,
        "resolve_include_paths",
        lambda **kwargs: SimpleNamespace(
            as_tuple=lambda: (include_dir,),
            cuda=(include_dir,),
        ),
    )
    monkeypatch.setattr(
        _nvrtc,
        "preload_toolkit_compiler_libraries",
        lambda paths: events.append("preload") or _libraries(),
    )
    monkeypatch.setattr(
        _nvrtc,
        "_load_nvrtc",
        lambda: events.append("binding") or _fake_nvrtc((12, 8)),
    )
    monkeypatch.setattr(
        _nvrtc,
        "include_dirs_identity",
        lambda paths: pytest.fail("mismatch must fail before header identity"),
    )

    with pytest.raises(
        RuntimeError,
        match=r"headers report Toolkit 13\.3, but loaded NVRTC .* reports 12\.8",
    ):
        _nvrtc.resolve_compile_context()

    assert events == ["preload", "binding"]


def test_compile_impl_rejects_nvrtc_version_change_before_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation = inspect.unwrap(_nvrtc.compile_impl)
    monkeypatch.setattr(
        _nvrtc,
        "_load_nvrtc",
        lambda: _fake_nvrtc((13, 2)),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"loaded NVRTC version changed after compile-context resolution: "
            r"expected version\(major=13, minor=3\), got "
            r"version\(major=13, minor=2\)"
        ),
    ):
        implementation(
            cpp="int provider;",
            cc=90,
            rdc=True,
            code="lto",
            toolkit_root="/cuda/toolkit",
            toolkit_version=(13, 3),
            nvrtc_path="/cuda/toolkit/lib/libnvrtc.so.13",
            nvrtc_builtins_path="/cuda/toolkit/lib/libnvrtc-builtins.so.13.3",
            nvjitlink_path="/cuda/toolkit/lib/libnvJitLink.so.13",
            nvrtc_version=_nvrtc.version(13, 3),
            nvjitlink_version=(13, 4),
            include_dirs=("/cccl/include", "/cuda/toolkit/include"),
            header_identity="headers-a",
            compiler_options=_nvrtc._compiler_options(
                cc=90,
                rdc=True,
                code="lto",
                include_dirs=("/cccl/include", "/cuda/toolkit/include"),
            ),
        )
