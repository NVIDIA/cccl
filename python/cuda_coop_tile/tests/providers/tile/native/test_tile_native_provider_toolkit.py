# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import subprocess
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


def _configured_toolkit_paths():
    roots = [
        Path(value)
        for name in _TOOLKIT_ROOT_VARIABLES
        if (value := os.environ.get(name))
    ]
    if not roots:
        pytest.skip("toolkit mirror test requires a configured CUDA toolkit")
    source = roots[0]
    includes = [
        source / suffix
        for suffix in (
            "bin/nvrtc/include",
            "nvrtc/include",
            "include",
            "bin/include",
        )
        if any(
            (source / suffix / name).is_file()
            for name in ("nvrtc.h", "cuda_runtime.h")
        )
    ]
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
    assert any((path / "cuda_runtime.h").is_file() for path in includes), source
    assert library_directory is not None, source
    return source, includes, library_directory


def _link_toolkit_directory(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            # Junction creation needs no symlink privilege, but Windows
            # process policy can still prevent traversing the junction.
            subprocess.run(
                [
                    "cmd",
                    "/d",
                    "/c",
                    "mklink",
                    "/J",
                    str(link),
                    str(target.resolve()),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            link.symlink_to(target.resolve(), target_is_directory=True)
        assert link.samefile(target), (link, target)
    except subprocess.CalledProcessError as exc:
        pytest.skip(
            f"toolkit mirror requires directory junctions: {exc.stderr.strip()}"
        )
    except OSError as exc:
        if os.name == "nt" and exc.winerror not in (5, 50, 448, 1314):
            raise
        pytest.skip(
            f"toolkit mirror requires traversable directory links: {exc}"
        )


def test_native_provider_scan_accepts_symlinked_toolkit(monkeypatch, tmp_path):
    source, includes, library_directory = _configured_toolkit_paths()
    lib, _ = _load_provider_or_skip(monkeypatch)
    for name in _TOOLKIT_ROOT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    selected = tmp_path / "selected"
    selected.mkdir()
    alias = tmp_path / "alias"
    for include in includes:
        _link_toolkit_directory(selected / include.relative_to(source), include)
    _link_toolkit_directory(selected / "lib64", library_directory)
    _link_toolkit_directory(alias, selected)
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


@pytest.mark.parametrize(
    "configuration", ["same-inputs", "different-headers", "different-nvrtc"]
)
def test_native_provider_checks_distinct_toolkit_mirrors(
    monkeypatch, tmp_path, configuration
):
    source, includes, library_directory = _configured_toolkit_paths()
    lib, _ = _load_provider_or_skip(monkeypatch)
    for name in _TOOLKIT_ROOT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    mirror = tmp_path / "toolkit mirror"
    mirror.mkdir()
    for include in includes:
        destination = mirror / include.relative_to(source)
        if configuration == "different-headers":
            destination.mkdir(parents=True)
            # Matching top-level header contents do not make a different CUDA
            # header tree interchangeable with the selected toolkit.
            for name in ("cuda_runtime.h", "nvrtc.h"):
                if (include / name).is_file():
                    shutil.copyfile(include / name, destination / name)
            (destination / "vector_types.h").write_text(
                "#error mismatched CUDA header tree\n"
            )
        else:
            _link_toolkit_directory(destination, include)
    if configuration == "different-nvrtc":
        (mirror / "lib64").mkdir()
        for library in library_directory.glob("*nvrtc*"):
            if library.is_file():
                (mirror / "lib64" / library.name).touch()
    else:
        _link_toolkit_directory(mirror / "lib64", library_directory)
    assert not mirror.samefile(source)
    monkeypatch.setenv("CUDA_ROOT", str(source))
    monkeypatch.setenv("CUDA_HOME", str(mirror))
    trace_file = tmp_path / "provider-trace.txt"
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE_FILE", str(trace_file))

    instance = _create_provider(lib)
    try:
        callsite, keepalive = _scan_callsite(1, "f32")
        request, request_keepalive = _provide_request(callsite)
        if configuration == "same-inputs":
            assert _provide_impls(instance, request).startswith(
                b"\x7fTileIR\x00"
            )
            trace = trace_file.read_text()
            assert " nvrtc=" in trace
            assert "resource_query_nvrtc_compile" in trace
        else:
            error = _assert_malformed(instance, request, "[LP-TOOLKIT-001]")
            assert "conflicting CUDA toolkit roots" in error.details
            assert str(mirror) in error.details
            assert str(source.resolve()) in error.details
            trace = trace_file.read_text() if trace_file.exists() else ""
            assert "resource_query" not in trace
            assert "nvrtc_" not in trace
        assert keepalive and request_keepalive
    finally:
        instance.vtable.destroy(instance.handle)
