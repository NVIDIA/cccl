# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

import pytest

from ..support import frontend as frontend_support


@pytest.mark.parametrize(
    ("machine", "dll_arch"), [("AMD64", "x86_64"), ("ARM64", "arm64")]
)
def test_cpp_frontend_windows_paths_include_versioned_component_dlls(
    monkeypatch, tmp_path, machine, dll_arch
):
    monkeypatch.setattr(frontend_support.platform, "machine", lambda: machine)
    cuda_path = (
        tmp_path / "installed CUDA components with spaces" / "nvidia" / "cu13"
    )
    tileiras = cuda_path / "bin" / "tileiras.exe"
    dll_directory = cuda_path / "bin" / dll_arch
    tileiras.parent.mkdir(parents=True)
    tileiras.touch()
    dll_directory.mkdir()

    assert (
        frontend_support._tileiras_lookup_name(tileiras, windows=True)
        == "tileiras.exe"
    )
    assert frontend_support._frontend_path_entries(
        tileiras,
        cuda_path,
        windows=True,
    ) == (tileiras.parent, dll_directory)


@pytest.mark.parametrize(
    ("machine", "dll_arch"), [("AMD64", "x86_64"), ("ARM64", "arm64")]
)
def test_cpp_frontend_windows_paths_require_versioned_component_dlls(
    monkeypatch, tmp_path, machine, dll_arch
):
    monkeypatch.setattr(frontend_support.platform, "machine", lambda: machine)
    cuda_path = tmp_path / "nvidia" / "cu13"
    tileiras = cuda_path / "bin" / "tileiras.exe"
    tileiras.parent.mkdir(parents=True)
    tileiras.touch()
    other_arch = "x86_64" if dll_arch == "arm64" else "arm64"
    (cuda_path / "bin" / other_arch).mkdir()

    with pytest.raises(
        pytest.fail.Exception,
        match=rf"installed CUDA component DLL directory: .*bin.{dll_arch}",
    ):
        frontend_support._frontend_path_entries(
            tileiras,
            cuda_path,
            windows=True,
        )


@pytest.mark.parametrize(
    ("machine", "dll_arch"), [("AMD64", "x86_64"), ("ARM64", "arm64")]
)
def test_cpp_frontend_environment_keeps_path_changes_child_scoped(
    monkeypatch,
    tmp_path,
    machine,
    dll_arch,
):
    monkeypatch.setattr(frontend_support.platform, "machine", lambda: machine)
    cuda_path = tmp_path / "installed components" / "nvidia" / "cu13"
    nvcc = cuda_path / "bin" / "nvcc.exe"
    tileiras = (
        cuda_path / "bin" / ("tileiras.exe" if os.name == "nt" else "tileiras")
    )
    tile_library = tmp_path / "tile-library" / "cuda_coop_tile.tilelib"
    for executable in (nvcc, tileiras, tile_library):
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.touch()
    dll_directory = cuda_path / "bin" / dll_arch
    dll_directory.mkdir()
    ambient_path = str(tmp_path / "ambient tools")
    monkeypatch.setenv("CUDA_COOP_TILE_CPP_TEST_NVCC", str(nvcc))
    monkeypatch.setenv("CUDA_COOP_TILE_CPP_TEST_TILEIRAS", str(tileiras))
    monkeypatch.setenv("CUDA_COOP_TILE_LIBRARY_PATH", str(tile_library))
    monkeypatch.setenv("CUDA_COOP_TILE_CPP_TEST_CUDA_PATH", str(cuda_path))
    monkeypatch.setenv("CUDA_COOP_TILE_CPP_TEST_ARCH", "sm_120")
    monkeypatch.setenv("PATH", ambient_path)
    for name in ("CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_ROOT", "CUDA_PATH"):
        monkeypatch.setenv(name, f"ambient {name}")
    monkeypatch.setattr(
        frontend_support.shutil,
        "which",
        lambda name, *, path: str(tileiras),
    )

    frontend = frontend_support._real_frontend_config("child environment")

    expected_path_entries = [str(tileiras.parent)]
    if os.name == "nt":
        expected_path_entries.append(str(dll_directory))
    expected_path_entries.append(ambient_path)
    assert (
        frontend.environment["PATH"].split(os.pathsep) == expected_path_entries
    )
    assert os.environ["PATH"] == ambient_path
    for name in ("CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_ROOT", "CUDA_PATH"):
        assert frontend.environment[name] == str(cuda_path)
        assert os.environ[name] == f"ambient {name}"


def test_cpp_frontend_runtime_timeout_has_actionable_diagnostics(
    monkeypatch, tmp_path
):
    frontend = SimpleNamespace(
        nvcc=tmp_path / "nvcc",
        cuda_path=tmp_path / "cuda",
        arch="sm_120",
        config=SimpleNamespace(nvcc_options=lambda: ()),
        environment={},
    )
    monkeypatch.setattr(
        frontend_support, "_real_frontend_config", lambda _description: frontend
    )
    calls = 0

    def run(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(command, 0, "compiled", "")
        raise subprocess.TimeoutExpired(
            command,
            kwargs["timeout"],
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(frontend_support.subprocess, "run", run)

    with pytest.raises(pytest.fail.Exception) as error:
        frontend_support.compile_and_run_frontend(
            tmp_path,
            source=tmp_path / "source.cu",
            executable_name="stress",
            description="stress",
            run_timeout=17,
        )

    diagnostic = str(error.value)
    assert "runtime exceeded 17 seconds" in diagnostic
    assert "partial stdout" in diagnostic
    assert "partial stderr" in diagnostic


def test_provider_trace_target_assertion_accepts_same_code_family_refinement():
    frontend_support.assert_provider_trace_uses_requested_architecture(
        frontend_support.FrontendResult(
            compile_trace="discover sm=sm_120a\ninstantiate sm=sm_120a",
            stdout="",
            requested_arch="sm_120a",
        )
    )
    frontend_support.assert_provider_trace_uses_requested_architecture(
        frontend_support.FrontendResult(
            compile_trace="discover sm=sm_120\ninstantiate sm=sm_120a",
            stdout="",
            requested_arch="sm_120",
        )
    )
    frontend_support.assert_provider_trace_uses_requested_architecture(
        frontend_support.FrontendResult(
            compile_trace=(
                "[coop_linkage_provider][trace] resource_query_nvrtc_compile "
                "sm=sm_120 primitives=4\ninstantiate sm=sm_120a"
            ),
            stdout="",
            requested_arch="sm_120",
        )
    )
    frontend_support.assert_provider_trace_uses_requested_architecture(
        frontend_support.FrontendResult(
            compile_trace=(
                "[coop_linkage_provider][trace] resource_query_nvrtc_compile "
                "sm=sm_120 primitives=4\n"
                "discover sm=sm_120a\ninstantiate sm=sm_120a"
            ),
            stdout="",
            requested_arch="sm_120a",
        )
    )


@pytest.mark.parametrize(
    ("requested_arch", "compile_trace"),
    [
        pytest.param(
            "sm_120a",
            "discover sm=sm_120\ninstantiate sm=sm_120a",
            id="reverse-refinement",
        ),
        pytest.param(
            "sm_120",
            "discover sm=sm_120\ninstantiate sm=sm_100a",
            id="cross-code",
        ),
        pytest.param(
            "sm_120",
            "discover sm=sm_120a\ninstantiate sm=sm_120f",
            id="mixed-refinements",
        ),
        pytest.param(
            "sm_120a",
            "[coop_linkage_provider][trace] resource_query_nvrtc_compile "
            "sm=sm_100 primitives=4\ninstantiate sm=sm_120a",
            id="cross-code-resource-query",
        ),
    ],
)
def test_provider_trace_target_assertion_rejects_callback_disagreement(
    requested_arch, compile_trace
):
    with pytest.raises(
        AssertionError,
        match=r"provider (?:trace|resource-query) target mismatch",
    ):
        frontend_support.assert_provider_trace_uses_requested_architecture(
            frontend_support.FrontendResult(
                compile_trace=compile_trace,
                stdout="",
                requested_arch=requested_arch,
            )
        )


def test_provider_trace_accepts_discovery_scores_without_late_callback():
    frontend_support.assert_provider_trace_scored_selected_implementations(
        frontend_support.FrontendResult(
            compile_trace=(
                "[coop_linkage_provider][trace] discovery candidate=0 "
                "semantic_key=scan128 provider_score=1049572\n"
                "[coop_linkage_provider][trace] discovery candidate=1 "
                "semantic_key=scan256 provider_score=918372\n"
                "[coop_linkage_provider][trace] resource_estimate "
                "semantic_key=scan128 threads=128"
            ),
            stdout="",
            requested_arch="sm_120",
        )
    )


@pytest.mark.parametrize(
    ("compile_trace", "message"),
    [
        ("", "did not instantiate"),
        (
            (
                "[coop_linkage_provider][trace] resource_estimate "
                "semantic_key=scan128"
            ),
            "without scored discovery rows",
        ),
        (
            (
                "[coop_linkage_provider][trace] discovery candidate=0 "
                "semantic_key=scan256 provider_score=918372\n"
                "[coop_linkage_provider][trace] resource_estimate "
                "semantic_key=scan128"
            ),
            "without scored discovery rows",
        ),
        (
            (
                "[coop_linkage_provider][trace] discovery candidate=0 "
                "semantic_key=scan128 provider_score=100\n"
                "[coop_linkage_provider][trace] resource_estimate "
                "semantic_key=scan128"
            ),
            "lack their discovery preference score",
        ),
    ],
)
def test_provider_trace_selection_requires_scored_selected_rows(
    compile_trace, message
):
    with pytest.raises(AssertionError, match=message):
        frontend_support.assert_provider_trace_scored_selected_implementations(
            frontend_support.FrontendResult(
                compile_trace=compile_trace,
                stdout="",
                requested_arch="sm_120",
            )
        )


@pytest.mark.parametrize("arch", ("sm_90", "sm_120a", "sm_100f"))
def test_cpp_frontend_accepts_only_canonical_test_architectures(arch):
    assert frontend_support._validate_test_arch(arch) == arch


@pytest.mark.parametrize(
    "arch", ("120", "sm120", "compute_120", "sm_0120", "sm_120A", "sm_0")
)
def test_cpp_frontend_rejects_noncanonical_test_architectures(arch):
    with pytest.raises(pytest.fail.Exception, match="must use canonical"):
        frontend_support._validate_test_arch(arch)
