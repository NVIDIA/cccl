# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from types import SimpleNamespace

from cuda.bindings import nvrtc
from cuda.coop.numba_mlir import _nvrtc


def test_compile_cache_inputs_include_resolved_header_identity(monkeypatch):
    calls = []
    include_dirs = (Path("/cccl/include"), Path("/cuda/include"))

    monkeypatch.setattr(
        _nvrtc,
        "resolve_include_paths",
        lambda **kwargs: SimpleNamespace(as_tuple=lambda: include_dirs),
    )
    monkeypatch.setattr(
        _nvrtc,
        "include_dirs_identity",
        lambda paths: SimpleNamespace(digest="header-digest"),
    )
    monkeypatch.setattr(
        _nvrtc.nvrtc,
        "nvrtcVersion",
        lambda: (nvrtc.nvrtcResult.NVRTC_SUCCESS, 13, 3),
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
            "nvrtc_path": _nvrtc.nvrtc.__file__,
            "nvrtc_version": _nvrtc.version(13, 3),
            "include_dirs": ("/cccl/include", "/cuda/include"),
            "header_identity": "header-digest",
        }
    ]
