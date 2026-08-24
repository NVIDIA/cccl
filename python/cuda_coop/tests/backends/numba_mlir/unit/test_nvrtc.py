# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from cuda.bindings import nvrtc
from cuda.coop._headers._toolkit import ToolkitCompilerLibraries
from cuda.coop.numba_mlir._compiler import _nvrtc


def test_include_options_use_filesystem_encoding():
    include_dir = "/tmp/cuda-coop-π/include"

    assert _nvrtc._include_options((include_dir,)) == [
        os.fsencode(f"--include-path={include_dir}")
    ]


def test_compile_cache_inputs_include_resolved_header_identity(monkeypatch):
    calls = []
    events = []
    include_dirs = (Path("/cccl/include"), Path("/cuda/include"))

    monkeypatch.setattr(
        _nvrtc,
        "resolve_include_paths",
        lambda **kwargs: (
            events.append("headers")
            or SimpleNamespace(
                as_tuple=lambda: include_dirs,
                cuda=(include_dirs[-1],),
            )
        ),
    )
    monkeypatch.setattr(
        _nvrtc,
        "preload_toolkit_compiler_libraries",
        lambda paths: (
            events.append(("preload", paths))
            or ToolkitCompilerLibraries(
                nvrtc_path="/cuda/lib/libnvrtc.so.13",
                nvjitlink_path="/cuda/lib/libnvJitLink.so.13",
                toolkit_version=(13, 3),
            )
        ),
    )
    monkeypatch.setattr(
        _nvrtc,
        "include_dirs_identity",
        lambda paths: SimpleNamespace(digest="header-digest"),
    )
    monkeypatch.setattr(
        _nvrtc.nvrtc,
        "nvrtcVersion",
        lambda: events.append("version") or (nvrtc.nvrtcResult.NVRTC_SUCCESS, 13, 3),
    )

    def compile_impl(**kwargs):
        calls.append(kwargs)
        return b"ltoir"

    monkeypatch.setattr(_nvrtc, "compile_impl", compile_impl)

    runtime_version, result = _nvrtc.compile(
        cpp="int provider;",
        cc=90,
        rdc=True,
        code="lto",
    )

    assert runtime_version == _nvrtc.version(13, 3)
    assert result == b"ltoir"
    assert calls == [
        {
            "cpp": "int provider;",
            "cc": 90,
            "rdc": True,
            "code": "lto",
            "nvrtc_path": "/cuda/lib/libnvrtc.so.13",
            "nvrtc_version": _nvrtc.version(13, 3),
            "include_dirs": ("/cccl/include", "/cuda/include"),
            "header_identity": "header-digest",
        }
    ]
    assert events == ["headers", ("preload", (include_dirs[-1],)), "version"]


def test_compile_context_rejects_header_nvrtc_version_mismatch(monkeypatch):
    include_dir = Path("/cuda/include")
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
        lambda paths: ToolkitCompilerLibraries(
            nvrtc_path="/cuda/lib/libnvrtc.so.12",
            nvjitlink_path="/cuda/lib/libnvJitLink.so.12",
            toolkit_version=(13, 2),
        ),
    )
    monkeypatch.setattr(
        _nvrtc.nvrtc,
        "nvrtcVersion",
        lambda: (nvrtc.nvrtcResult.NVRTC_SUCCESS, 12, 8),
    )

    with pytest.raises(RuntimeError, match="headers report Toolkit 13.2"):
        _nvrtc.resolve_compile_context()


def test_compile_uses_one_pre_resolved_context(monkeypatch):
    calls = []
    context = _nvrtc.CompileContext(
        nvrtc_path="/toolkit/lib/libnvrtc.so.13",
        nvrtc_version=_nvrtc.version(13, 2),
        include_dirs=("/toolkit/include",),
        header_identity="headers-a",
    )

    monkeypatch.setattr(
        _nvrtc,
        "resolve_compile_context",
        lambda: pytest.fail("compile must not resolve a supplied context again"),
    )
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
    assert calls[0]["nvrtc_path"] == context.nvrtc_path
    assert calls[0]["include_dirs"] == context.include_dirs
    assert calls[0]["header_identity"] == context.header_identity
