# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

from cuda.coop.numba_mlir import _nvrtc


def test_nvrtc_source_dump_is_content_addressed(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_COOP_NUMBA_MLIR_NVRTC_DUMP_DIR", str(tmp_path))

    first = _nvrtc._dump_source("// first\n", 120, "lto")
    repeated = _nvrtc._dump_source("// first\n", 120, "lto")
    second = _nvrtc._dump_source("// second\n", 120, "lto")

    assert isinstance(first, Path)
    assert first == repeated
    assert first != second
    assert first.name.startswith("cuda_coop_numba_mlir_")
    assert first.name.endswith("_cc120_lto.cu")
    assert first.read_text(encoding="utf-8") == "// first\n"
    assert second.read_text(encoding="utf-8") == "// second\n"
    assert sorted(tmp_path.glob("*.cu")) == sorted((first, second))


def test_nvrtc_source_dump_is_disabled_without_env(monkeypatch):
    monkeypatch.delenv("CUDA_COOP_NUMBA_MLIR_NVRTC_DUMP_DIR", raising=False)

    assert _nvrtc._dump_source("// source\n", 120, "ptx") is None


def test_compile_dumps_source_before_compile_cache_lookup(monkeypatch):
    events = []

    monkeypatch.setattr(
        _nvrtc,
        "_dump_source",
        lambda cpp, cc, code: events.append(("dump", cpp, cc, code)),
    )
    monkeypatch.setattr(
        _nvrtc.nvrtc,
        "nvrtcVersion",
        lambda: (_nvrtc.nvrtc.nvrtcResult.NVRTC_SUCCESS, 13, 1),
    )
    monkeypatch.setattr(
        _nvrtc,
        "compile_impl",
        lambda **kwargs: events.append(("compile", kwargs)) or b"cached",
    )

    version, result = _nvrtc.compile(cpp="// source\n", cc=120, rdc=True, code="lto")

    assert (version.major, version.minor) == (13, 1)
    assert result == b"cached"
    assert events[0] == ("dump", "// source\n", 120, "lto")
    assert events[1][0] == "compile"
