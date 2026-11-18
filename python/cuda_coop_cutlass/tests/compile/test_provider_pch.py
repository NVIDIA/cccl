# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check process-local PCH reuse and layout queries on distinct JIT misses."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cuda.coop.cutlass._compiler import _bundle, _cache, _nvrtc
from cuda.coop.cutlass._compiler._types import ScratchLayout, ScratchLayoutProbe
from cuda.coop.cutlass._headers._identity import include_dirs_identity

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def test_pch_gating_and_unavailable_storage(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("temporary directory unavailable")

    context = SimpleNamespace(nvrtc_version=(12, 8), header_identity="test")
    monkeypatch.setattr(_nvrtc.tempfile, "mkdtemp", unavailable)
    monkeypatch.setenv(_nvrtc._PCH_ENV, "auto")
    assert _nvrtc.pch_options(context) == ()
    monkeypatch.setattr(_nvrtc.tempfile, "mkdtemp", pytest.fail)
    monkeypatch.setenv(_nvrtc._PCH_ENV, "off")
    assert _nvrtc.pch_options(context) == ()
    monkeypatch.setenv(_nvrtc._PCH_ENV, "auto")
    context.nvrtc_version = (12, 7)
    assert _nvrtc.pch_options(context) == ()


def test_header_reuse_and_edit_preserve_compiled_layouts(tmp_path, monkeypatch):
    nvrtc_context = _nvrtc.resolve_compile_context(())
    if nvrtc_context.nvrtc_version < (12, 8):
        pytest.skip("NVRTC automatic PCH requires CUDA 12.8 or later")
    include_dir = tmp_path / "include"
    include_dir.mkdir()
    header = include_dir / "layout.cuh"
    header.write_text("struct alignas(16) S { int value; };")
    include_dirs = (str(include_dir),)

    def context(required_headers):
        return replace(
            nvrtc_context,
            include_dirs=include_dirs,
            header_identity=include_dirs_identity(include_dirs).digest,
        )

    monkeypatch.setattr(_nvrtc, "resolve_compile_context", context)
    monkeypatch.setenv(_nvrtc._PCH_ENV, "auto")
    monkeypatch.setenv(_cache.CACHE_DIR_ENV, str(tmp_path / "cache"))
    nvrtc = _nvrtc._load_nvrtc()
    compile_program = nvrtc.nvrtcCompileProgram
    logs = []

    def record_compile(program, count, options):
        result = compile_program(program, count, options)
        logs.append(_nvrtc._program_log(nvrtc, program))
        return result

    monkeypatch.setattr(nvrtc, "nvrtcCompileProgram", record_compile)

    def compile_layout(value):
        source = (
            '#include "layout.cuh"\n'
            f'extern "C" __device__ int get_value() {{ return {value}; }}'
        )
        result = _bundle.compile_bundle_source_with_layouts(
            source,
            arch="compute_80",
            required_headers=("layout.cuh",),
            layout_probes=[ScratchLayoutProbe("s", "sizeof(S)", "alignof(S)")],
        )
        assert Path(result.path).stat().st_size > 0
        return result.layouts["s"]

    assert compile_layout(1) == ScratchLayout(16, 16)
    assert compile_layout(2) == ScratchLayout(16, 16)
    assert "creating precompiled header" in logs[0].lower()
    assert "using precompiled header" in logs[1].lower()

    header.write_text("struct alignas(32) S { int value; };")
    assert compile_layout(1) == ScratchLayout(32, 32)
    assert "creating precompiled header" in logs[2].lower()


def test_pch_compile_failure_retries_with_registered_names(monkeypatch):
    from cuda.coop.cutlass._compiler import _layout

    context = _nvrtc.resolve_compile_context(())
    nvrtc = _nvrtc._load_nvrtc()
    compile_program = nvrtc.nvrtcCompileProgram
    attempts = []

    def reject_pch(program, count, options):
        attempts.append(tuple(options))
        if b"--pch" in options:
            options = [*options, b"--invalid-option-for-pch-test"]
        return compile_program(program, len(options), options)

    monkeypatch.setattr(nvrtc, "nvrtcCompileProgram", reject_pch)
    options = _nvrtc.compiler_options(context, "compute_80")
    prepared = _layout._prepare_layout_probes(
        'extern "C" __device__ int value() { return 42; }',
        [ScratchLayoutProbe("value", "sizeof(int)", "alignof(int)")],
    )
    blob, layouts = _nvrtc.compile_ltoir_with_layouts(
        prepared, options + (b"--pch", b"--pch-dir=unused")
    )
    assert blob and layouts[prepared.expressions[0]] == ScratchLayout(4, 4)
    assert attempts == [options + (b"--pch", b"--pch-dir=unused"), options]
