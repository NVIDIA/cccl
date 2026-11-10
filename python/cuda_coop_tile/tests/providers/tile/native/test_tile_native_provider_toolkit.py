# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from pathlib import Path

import pytest

from .harness import (
    TILEIR_LP_ERROR,
    _assert_malformed,
    _create_provider,
    _load_provider_or_skip,
    _provide_impls,
)
from .reduce_scan import _resolved_scan_call, _scan_callsite
from .test_tile_native_provider_diagnostics import (
    _instantiate_result,
    _provide_request,
)

_TOOLKIT_ROOT_VARIABLES = (
    "CUDA_COOP_TILE_CUDA_ROOT",
    "CUDA_HOME",
    "CUDA_ROOT",
    "CUDA_PATH",
    "CUDA_TOOLKIT_PATH",
)


@pytest.mark.parametrize(
    "configuration", ["conflicting-roots", "different-nvrtc", "missing-headers"]
)
def test_native_provider_reports_toolkit_mismatch_before_scan_compilation(
    monkeypatch, tmp_path, configuration
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    for name in _TOOLKIT_ROOT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    selected = tmp_path / "selected"
    (selected / "include").mkdir(parents=True)
    if configuration != "missing-headers":
        (selected / "include/cuda_runtime.h").touch()
    monkeypatch.setenv("CUDA_ROOT", str(selected))
    if configuration == "conflicting-roots":
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv("CUDA_HOME", str(other))
    elif configuration == "different-nvrtc":
        (selected / "lib64").mkdir()
        (selected / "lib64/libnvrtc.so").touch()
    trace_file = tmp_path / "provider-trace.txt"
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE_FILE", str(trace_file))

    instance = _create_provider(lib)
    try:
        callsite, discovery_keepalive = _scan_callsite(1, "f32")
        request, request_keepalive = _provide_request(callsite)
        error = _assert_malformed(instance, request, "[LP-TOOLKIT-001]")
        expected = {
            "conflicting-roots": "conflicting CUDA toolkit roots",
            "different-nvrtc": "loaded NVRTC",
            "missing-headers": "has no cuda_runtime.h",
        }[configuration]
        assert expected in error.details
        assert str(selected) in error.details

        call, keepalive = _resolved_scan_call(1, "f32")
        result, error = _instantiate_result(instance, [call])
        assert result == TILEIR_LP_ERROR
        assert error is not None
        assert "[LP-TOOLKIT-001]" in error.details
        assert expected in error.details
        trace = trace_file.read_text() if trace_file.exists() else ""
        assert "resource_query" not in trace
        assert "nvrtc_" not in trace
        assert discovery_keepalive and request_keepalive and keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_scan_accepts_symlinked_toolkit(monkeypatch, tmp_path):
    roots = [
        Path(value)
        for name in _TOOLKIT_ROOT_VARIABLES
        if (value := os.environ.get(name))
    ]
    if not roots:
        pytest.skip("symlinked toolkit test requires a configured CUDA toolkit")
    source = roots[0]
    include = next(
        (
            source / suffix
            for suffix in (
                "include",
                "bin/include",
                "bin/nvrtc/include",
                "nvrtc/include",
            )
            if (source / suffix / "cuda_runtime.h").is_file()
        ),
        None,
    )
    library_directory = next(
        (
            source / suffix
            for suffix in (
                "bin/nvrtc/lib64",
                "nvrtc/lib64",
                "lib64",
                "lib",
                "bin/x64",
                "bin/x86_64",
                "bin/arm64",
                "bin",
            )
            if any((source / suffix).glob("*nvrtc*"))
        ),
        None,
    )
    assert include is not None, source
    assert library_directory is not None, source
    lib, _ = _load_provider_or_skip(monkeypatch)
    for name in _TOOLKIT_ROOT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    selected = tmp_path / "selected"
    selected.mkdir()
    alias = tmp_path / "alias"
    try:
        (selected / "include").symlink_to(
            include.resolve(), target_is_directory=True
        )
        (selected / "lib64").symlink_to(
            library_directory.resolve(), target_is_directory=True
        )
        alias.symlink_to(selected, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"toolkit directory symlinks are unavailable: {exc}")
    monkeypatch.setenv("CUDA_ROOT", str(selected))
    monkeypatch.setenv("CUDA_HOME", str(alias))
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    (fallback / "cuda_runtime.h").write_text(
        "#error wrong CUDA toolkit headers\n"
    )
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_INCLUDE_PATHS", str(fallback))
    trace_file = tmp_path / "provider-trace.txt"
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE_FILE", str(trace_file))

    instance = _create_provider(lib)
    try:
        callsite, keepalive = _scan_callsite(1, "f32")
        request, request_keepalive = _provide_request(callsite)
        assert _provide_impls(instance, request).startswith(b"\x7fTileIR\x00")
        trace = trace_file.read_text()
        assert f"toolkit={selected}" in trace
        assert " nvrtc=" in trace
        assert "resource_query_nvrtc_compile" in trace
        assert keepalive and request_keepalive
    finally:
        instance.vtable.destroy(instance.handle)
