# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cutlass")

from cuda.coop.cutlass._compiler import _bundle as provider_bundle
from cuda.coop.cutlass._compiler import _bundle_contract as provider_contract
from cuda.coop.cutlass._compiler import _cache as provider_cache
from cuda.coop.cutlass._compiler import _nvrtc as provider_nvrtc


class _NvrtcResult:
    NVRTC_SUCCESS = 0


class _FakeNvrtc:
    nvrtcResult = _NvrtcResult

    def __init__(self):
        self.calls = []
        self.blob = b"fake-ltoir"
        self.version = (13, 0)

    def nvrtcVersion(self):
        return 0, *self.version

    def nvrtcCreateProgram(self, source, name, num_headers, headers, include_names):
        self.calls.append(("create", source))
        return 0, object()

    def nvrtcAddNameExpression(self, program, expression):
        self.calls.append(("add", expression))
        return (0,)

    def nvrtcCompileProgram(self, program, num_options, options):
        self.calls.append(("compile", tuple(options)))
        return (0,)

    def nvrtcGetLoweredName(self, program, expression):
        self.calls.append(("get_lowered", expression))
        decoded = expression.decode("utf-8")
        symbol_match = re.match(r"&([^<]+)<", decoded)
        assert symbol_match is not None
        symbol = symbol_match.group(1)
        return 0, f"_Z{len(symbol)}{symbol}ILy40ELy8EE".encode()

    def nvrtcGetLTOIRSize(self, program):
        self.calls.append(("get_ltoir_size",))
        return 0, len(self.blob)

    def nvrtcGetLTOIR(self, program, blob):
        self.calls.append(("get_ltoir",))
        blob[:] = self.blob
        return (0,)

    def nvrtcDestroyProgram(self, program):
        self.calls.append(("destroy",))
        return (0,)


def _compile_kwargs():
    return {
        "scope": "cuda.coop.cutlass",
        "provider_dir": provider_bundle.__file__,
        "registered_headers": dict,
        "select_bundle_format": lambda: "ltoir",
        "resolve_nvrtc_sm_arch": lambda: "sm_80",
        "resolve_nvrtc_arch": lambda: "compute_80",
    }


def _probe():
    return provider_contract.LayoutProbe(
        key="storage",
        size_expression="sizeof(Storage)",
        alignment_expression="alignof(Storage)",
    )


def _precompiled_resolution(request, path):
    return provider_contract.BundleResolution(
        request=request,
        path=str(path),
        layouts_by_expression={
            expression: provider_contract.StorageLayout(40, 8)
            for expression in request.identity.layout_expressions
        },
        route=provider_contract.RESOLUTION_ROUTE_PRECOMPILED,
        producer_compiler="nvcc",
        producer_compiler_version="13.0",
        producer_toolkit_version="13.0",
        phase_timings_ns={},
    )


@pytest.fixture(autouse=True)
def _isolated_bundle_cache(monkeypatch, tmp_path):
    monkeypatch.setenv(
        provider_cache.CACHE_DIR_ENV,
        str(tmp_path / "provider-cache"),
    )
    provider_bundle.reset_compile_state()
    yield
    provider_bundle.reset_compile_state()


def test_resolution_routes_preserve_nvrtc_memory_and_disk_behavior(monkeypatch):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_nvrtc, "cuda_nvrtc", fake_nvrtc)
    observed = []

    with provider_bundle.activate_bundle_resolution_observer(observed.append):
        first = provider_bundle.compile_bundle_source(
            'extern "C" {}',
            **_compile_kwargs(),
        )
        second = provider_bundle.compile_bundle_source(
            'extern "C" {}',
            **_compile_kwargs(),
        )
        provider_bundle.reset_compile_state()
        third = provider_bundle.compile_bundle_source(
            'extern "C" {}',
            **_compile_kwargs(),
        )

    assert first == second == third
    assert [result.route for result in observed] == [
        provider_contract.RESOLUTION_ROUTE_NVRTC,
        provider_contract.RESOLUTION_ROUTE_MEMORY,
        provider_contract.RESOLUTION_ROUTE_DISK,
    ]
    assert [call[0] for call in fake_nvrtc.calls].count("compile") == 1
    assert observed[0].producer_compiler == "nvrtc"
    assert observed[0].producer_compiler_version == "13.0"
    assert observed[1].producer_compiler_version == "13.0"
    assert observed[2].producer_compiler_version == "13.0"
    compile_options = next(call[1] for call in fake_nvrtc.calls if call[0] == "compile")
    identity_options = tuple(
        option.encode("ascii")
        for option in observed[0].request.identity.compiler_options
    )
    assert compile_options[: len(identity_options)] == identity_options

    telemetry = provider_bundle.get_bundle_telemetry()
    assert telemetry.route_counts == {provider_contract.RESOLUTION_ROUTE_DISK: 1}
    assert telemetry.phase_counts["total"] == 1
    assert "compiler" not in telemetry.phase_counts


def test_precompile_hit_skips_mutable_jit_io_and_is_context_local(
    monkeypatch,
    tmp_path,
):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_nvrtc, "cuda_nvrtc", fake_nvrtc)
    precompiled_path = tmp_path / "captured.ltoir"
    precompiled_path.write_bytes(b"captured")
    resolver_requests = []
    observed = []

    def resolver(request):
        resolver_requests.append(request)
        return _precompiled_resolution(request, precompiled_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("exact precompile hits must bypass mutable JIT I/O")

    with monkeypatch.context() as hit_patch:
        hit_patch.setattr(provider_bundle, "cccl_include_dirs", forbidden)
        hit_patch.setattr(provider_bundle, "include_dirs_identity", forbidden)
        hit_patch.setattr(provider_bundle, "maybe_dump_source", forbidden)
        with (
            provider_bundle.activate_bundle_precompile_resolver(resolver),
            provider_bundle.activate_bundle_resolution_observer(observed.append),
        ):
            compilation = provider_bundle.compile_bundle_source_with_layouts(
                "struct Storage {};",
                layout_probes=(_probe(),),
                **_compile_kwargs(),
            )

    assert compilation.path == str(precompiled_path)
    assert compilation.layouts == {"storage": provider_contract.StorageLayout(40, 8)}
    assert len(resolver_requests) == 1
    request = resolver_requests[0]
    assert request.identity.provider_abi_version == 1
    assert request.identity.bundle_arch == "compute_80"
    assert request.identity.bundle_sm_arch == "sm_80"
    assert request.identity.layout_expressions
    assert observed[0].route == provider_contract.RESOLUTION_ROUTE_PRECOMPILED
    assert observed[0].producer_toolkit_version == "13.0"
    assert "header_resolution" not in observed[0].phase_timings_ns
    assert "header_fingerprint" not in observed[0].phase_timings_ns
    assert "source_dump" not in observed[0].phase_timings_ns
    assert [call[0] for call in fake_nvrtc.calls].count("compile") == 0

    provider_bundle.compile_bundle_source(
        'extern "C" {}',
        **_compile_kwargs(),
    )
    assert [call[0] for call in fake_nvrtc.calls].count("compile") == 1
    assert len(observed) == 1


def test_nested_precompile_resolvers_are_tried_newest_first(tmp_path):
    precompiled_path = tmp_path / "captured.ltoir"
    precompiled_path.write_bytes(b"captured")
    calls = []

    def outer(request):
        calls.append("outer")
        return _precompiled_resolution(request, precompiled_path)

    def inner(request):
        calls.append("inner")

    with (
        provider_bundle.activate_bundle_precompile_resolver(outer),
        provider_bundle.activate_bundle_precompile_resolver(inner),
    ):
        path = provider_bundle.compile_bundle_source(
            'extern "C" {}',
            **_compile_kwargs(),
        )

    assert path == str(precompiled_path)
    assert calls == ["inner", "outer"]


def test_nested_resolution_observers_restore_outer_context(tmp_path):
    precompiled_path = tmp_path / "captured.ltoir"
    precompiled_path.write_bytes(b"captured")
    outer_observed = []
    inner_observed = []

    def resolver(request):
        return _precompiled_resolution(request, precompiled_path)

    with provider_bundle.activate_bundle_precompile_resolver(resolver):
        with provider_bundle.activate_bundle_resolution_observer(outer_observed.append):
            provider_bundle.compile_bundle_source(
                'extern "C" {}',
                **_compile_kwargs(),
            )
            with provider_bundle.activate_bundle_resolution_observer(
                inner_observed.append
            ):
                provider_bundle.compile_bundle_source(
                    'extern "C" {}',
                    **_compile_kwargs(),
                )
            provider_bundle.compile_bundle_source(
                'extern "C" {}',
                **_compile_kwargs(),
            )

        provider_bundle.compile_bundle_source(
            'extern "C" {}',
            **_compile_kwargs(),
        )

    assert [result.route for result in outer_observed] == [
        provider_contract.RESOLUTION_ROUTE_PRECOMPILED,
        provider_contract.RESOLUTION_ROUTE_PRECOMPILED,
        provider_contract.RESOLUTION_ROUTE_PRECOMPILED,
    ]
    assert [result.route for result in inner_observed] == [
        provider_contract.RESOLUTION_ROUTE_PRECOMPILED
    ]


def test_precompile_resolver_rejects_incompatible_layout_metadata(tmp_path):
    precompiled_path = tmp_path / "captured.ltoir"
    precompiled_path.write_bytes(b"captured")

    def resolver(request):
        return provider_contract.BundleResolution(
            request=request,
            path=str(precompiled_path),
            layouts_by_expression={},
            route=provider_contract.RESOLUTION_ROUTE_PRECOMPILED,
            producer_compiler=None,
            producer_compiler_version=None,
            producer_toolkit_version=None,
            phase_timings_ns={},
        )

    with (
        provider_bundle.activate_bundle_precompile_resolver(resolver),
        pytest.raises(ValueError, match="incompatible layout metadata"),
    ):
        provider_bundle.compile_bundle_source_with_layouts(
            "struct Storage {};",
            layout_probes=(_probe(),),
            **_compile_kwargs(),
        )


def test_resolution_telemetry_counts_routes_and_exact_layouts(monkeypatch):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_nvrtc, "cuda_nvrtc", fake_nvrtc)
    observed = []

    with provider_bundle.activate_bundle_resolution_observer(observed.append):
        first = provider_bundle.compile_bundle_source_with_layouts(
            "struct Storage {};",
            layout_probes=(_probe(),),
            **_compile_kwargs(),
        )
        second = provider_bundle.compile_bundle_source_with_layouts(
            "struct Storage {};",
            layout_probes=(_probe(),),
            **_compile_kwargs(),
        )

    assert first == second
    assert [result.route for result in observed] == [
        provider_contract.RESOLUTION_ROUTE_NVRTC,
        provider_contract.RESOLUTION_ROUTE_MEMORY,
    ]
    for result in observed:
        assert set(result.layouts_by_expression) == set(
            result.request.identity.layout_expressions
        )
        assert all(value >= 0 for value in result.phase_timings_ns.values())

    telemetry = provider_bundle.get_bundle_telemetry()
    assert telemetry.route_counts == {
        provider_contract.RESOLUTION_ROUTE_NVRTC: 1,
        provider_contract.RESOLUTION_ROUTE_MEMORY: 1,
    }
    assert telemetry.phase_counts["total"] == 2
    assert telemetry.phase_counts["compiler"] == 1
    assert all(value >= 0 for value in telemetry.phase_timings_ns.values())


def test_nvrtc_version_changes_mutable_cache_identity(monkeypatch):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_nvrtc, "cuda_nvrtc", fake_nvrtc)
    observed = []

    with provider_bundle.activate_bundle_resolution_observer(observed.append):
        first = provider_bundle.compile_bundle_source(
            'extern "C" {}',
            **_compile_kwargs(),
        )
        fake_nvrtc.version = (13, 1)
        second = provider_bundle.compile_bundle_source(
            'extern "C" {}',
            **_compile_kwargs(),
        )

    assert first != second
    assert [result.route for result in observed] == [
        provider_contract.RESOLUTION_ROUTE_NVRTC,
        provider_contract.RESOLUTION_ROUTE_NVRTC,
    ]
    assert [result.producer_compiler_version for result in observed] == [
        "13.0",
        "13.1",
    ]
    assert [call[0] for call in fake_nvrtc.calls].count("compile") == 2


@pytest.mark.skipif(os.name == "nt", reason="test exercises POSIX file locking")
def test_artifact_lock_serializes_across_processes(tmp_path):
    artifact_path = str(tmp_path / "bundle.ltoir")
    acquired_path = tmp_path / "acquired"
    source_path = str(Path(provider_bundle.__file__).resolve().parents[2])
    script = """
import sys
from pathlib import Path
import cuda.coop

cuda.coop.__path__.insert(0, sys.argv[1])
from cuda.coop.cutlass._compiler import _cache as provider_cache

with provider_cache.artifact_lock(sys.argv[2], scope="test"):
    Path(sys.argv[3]).write_text("acquired", encoding="utf-8")
"""

    with provider_cache.artifact_lock(artifact_path, scope="test"):
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                source_path,
                artifact_path,
                str(acquired_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.2)
        assert not acquired_path.exists()

    _, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, stderr
    assert acquired_path.read_text(encoding="utf-8") == "acquired"
