# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import multiprocessing
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass")

from cuda.coop.cutlass._dsl import _provider_bundle as provider_bundle

_SUBPROCESS_READY_TIMEOUT_SECONDS = 120.0


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
        "registered_headers": lambda: {},
        "select_bundle_format": lambda: "ltoir",
        "resolve_nvrtc_sm_arch": lambda: "sm_80",
        "resolve_nvrtc_arch": lambda: "compute_80",
    }


def _probe():
    return provider_bundle.LayoutProbe(
        key="storage",
        size_expression="sizeof(Storage)",
        alignment_expression="alignof(Storage)",
    )


def _precompiled_resolution(request, path):
    return provider_bundle.BundleResolution(
        request=request,
        path=str(path),
        layouts_by_expression={
            expression: provider_bundle.StorageLayout(40, 8)
            for expression in request.identity.layout_expressions
        },
        route=provider_bundle.RESOLUTION_ROUTE_PRECOMPILED,
        producer_compiler="nvcc",
        producer_compiler_version="13.0",
        producer_toolkit_version="13.0",
        phase_timings_ns={},
    )


@pytest.fixture(autouse=True)
def _isolated_bundle_cache(monkeypatch, tmp_path):
    monkeypatch.setenv(
        provider_bundle.CACHE_DIR_ENV,
        str(tmp_path / "provider-cache"),
    )
    provider_bundle.reset_compile_state()
    yield
    provider_bundle.reset_compile_state()


def test_resolution_routes_preserve_nvrtc_memory_and_disk_behavior(monkeypatch):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)
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
        provider_bundle.RESOLUTION_ROUTE_NVRTC,
        provider_bundle.RESOLUTION_ROUTE_MEMORY,
        provider_bundle.RESOLUTION_ROUTE_DISK,
    ]
    assert [call[0] for call in fake_nvrtc.calls].count("compile") == 1
    assert observed[0].producer_compiler == "nvrtc"
    assert observed[0].producer_compiler_version == "13.0"
    assert observed[1].producer_compiler_version == "13.0"
    assert observed[2].producer_compiler_version == "13.0"
    assert {
        "nvrtc_compile",
        "lto_retrieval",
        "artifact_io",
        "metadata_io",
        "compiler",
        "total",
    } <= set(observed[0].phase_timings_ns)
    assert "render" not in observed[0].phase_timings_ns
    assert observed[0].phase_timings_ns["total"] >= max(
        duration_ns
        for phase, duration_ns in observed[0].phase_timings_ns.items()
        if phase != "total"
    )
    for cache_hit in observed[1:]:
        assert "nvrtc_compile" not in cache_hit.phase_timings_ns
        assert "lto_retrieval" not in cache_hit.phase_timings_ns
        assert "artifact_io" not in cache_hit.phase_timings_ns
        assert "metadata_io" not in cache_hit.phase_timings_ns
    compile_options = next(call[1] for call in fake_nvrtc.calls if call[0] == "compile")
    identity_options = tuple(
        option.encode("ascii")
        for option in observed[0].request.identity.compiler_options
    )
    assert compile_options[: len(identity_options)] == identity_options

    telemetry = provider_bundle.get_bundle_telemetry()
    assert telemetry.route_counts == {provider_bundle.RESOLUTION_ROUTE_DISK: 1}
    assert telemetry.phase_counts["total"] == 1
    assert "compiler" not in telemetry.phase_counts


def test_block_finalizer_includes_render_in_resolution_total(
    monkeypatch,
    capsys,
):
    from cuda.coop.cutlass._dsl.block import _provider as block_provider

    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)

    request = SimpleNamespace(symbol_name="provider_a")

    class _Session:
        def belongs_to_trace_module(self, _operation):
            return True

        def is_empty(self):
            return False

        def request_list(self):
            return [request]

        def deferred_temp_storage_event_list(self):
            return []

    monkeypatch.setattr(
        block_provider._provider_support,
        "pop_bundle_session",
        lambda _compile_options: _Session(),
    )
    monkeypatch.setattr(
        block_provider._provider_support,
        "bundle_scratch_layout_probes",
        lambda _requests: {},
    )
    monkeypatch.setattr(
        block_provider._provider_support,
        "plan_deferred_temp_storage_events",
        lambda _events, _layouts: (),
    )
    monkeypatch.setattr(
        block_provider._provider_support,
        "materialize_deferred_temp_storage_plans",
        lambda _plans, _module: None,
    )
    monkeypatch.setattr(
        block_provider,
        "_remove_managed_bundle_link_options",
        lambda _dsl: None,
    )
    monkeypatch.setattr(
        block_provider,
        "_render_bundle_source",
        lambda _requests: 'extern "C" __device__ void provider_a() {}\n',
    )
    monkeypatch.setattr(
        block_provider,
        "_append_link_library_attr",
        lambda _module, _path: None,
    )

    def compile_rendered_source(source, symbols, **timing):
        return provider_bundle.compile_bundle_source(
            source,
            symbols=symbols,
            **_compile_kwargs(),
            **timing,
        )

    monkeypatch.setattr(
        block_provider,
        "_compile_bundle_source",
        compile_rendered_source,
    )
    observed = []
    dsl = SimpleNamespace(compile_options=object())
    module = SimpleNamespace(operation=object())
    with provider_bundle.activate_bundle_resolution_observer(observed.append):
        block_provider._trace_finalize_hook(dsl, module, "kernel")

    assert len(observed) == 1
    timings = observed[0].phase_timings_ns
    assert "render" in timings
    assert timings["total"] >= timings["render"]
    assert {
        "nvrtc_compile",
        "lto_retrieval",
        "artifact_io",
        "metadata_io",
    } <= set(timings)
    assert capsys.readouterr() == ("", "")


def test_precompile_hit_skips_mutable_jit_io_and_is_context_local(
    monkeypatch,
    tmp_path,
):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)
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
        hit_patch.setattr(provider_bundle, "include_dirs_cache_key", forbidden)
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
    assert compilation.layouts == {"storage": provider_bundle.StorageLayout(40, 8)}
    assert len(resolver_requests) == 1
    request = resolver_requests[0]
    assert request.identity.provider_abi_version == 1
    assert request.identity.bundle_arch == "compute_80"
    assert request.identity.bundle_sm_arch == "sm_80"
    assert request.identity.layout_expressions
    assert observed[0].route == provider_bundle.RESOLUTION_ROUTE_PRECOMPILED
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
        return None

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
        provider_bundle.RESOLUTION_ROUTE_PRECOMPILED,
        provider_bundle.RESOLUTION_ROUTE_PRECOMPILED,
        provider_bundle.RESOLUTION_ROUTE_PRECOMPILED,
    ]
    assert [result.route for result in inner_observed] == [
        provider_bundle.RESOLUTION_ROUTE_PRECOMPILED
    ]


def test_precompile_resolver_rejects_incompatible_layout_metadata(tmp_path):
    precompiled_path = tmp_path / "captured.ltoir"
    precompiled_path.write_bytes(b"captured")

    def resolver(request):
        return provider_bundle.BundleResolution(
            request=request,
            path=str(precompiled_path),
            layouts_by_expression={},
            route=provider_bundle.RESOLUTION_ROUTE_PRECOMPILED,
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
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)
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
        provider_bundle.RESOLUTION_ROUTE_NVRTC,
        provider_bundle.RESOLUTION_ROUTE_MEMORY,
    ]
    for result in observed:
        assert set(result.layouts_by_expression) == set(
            result.request.identity.layout_expressions
        )
        assert all(value >= 0 for value in result.phase_timings_ns.values())

    telemetry = provider_bundle.get_bundle_telemetry()
    assert telemetry.route_counts == {
        provider_bundle.RESOLUTION_ROUTE_NVRTC: 1,
        provider_bundle.RESOLUTION_ROUTE_MEMORY: 1,
    }
    assert telemetry.phase_counts["total"] == 2
    assert telemetry.phase_counts["compiler"] == 1
    assert all(value >= 0 for value in telemetry.phase_timings_ns.values())


def test_nvrtc_version_changes_mutable_cache_identity(monkeypatch):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)
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
        provider_bundle.RESOLUTION_ROUTE_NVRTC,
        provider_bundle.RESOLUTION_ROUTE_NVRTC,
    ]
    assert [result.producer_compiler_version for result in observed] == [
        "13.0",
        "13.1",
    ]
    assert [call[0] for call in fake_nvrtc.calls].count("compile") == 2


@pytest.mark.skipif(os.name == "nt", reason="test exercises POSIX file locking")
def test_artifact_lock_serializes_across_processes(tmp_path):
    artifact_path = str(tmp_path / "bundle.ltoir")
    ready_path = tmp_path / "ready"
    acquired_path = tmp_path / "acquired"
    source_path = str(Path(provider_bundle.__file__).resolve().parents[2])
    script = """
import sys
from pathlib import Path
import cuda.coop

cuda.coop.__path__.insert(0, sys.argv[1])
from cuda.coop.cutlass._dsl import _provider_bundle as provider_bundle

Path(sys.argv[3]).write_text("ready", encoding="utf-8")
with provider_bundle.artifact_lock(sys.argv[2], scope="test"):
    Path(sys.argv[4]).write_text("acquired", encoding="utf-8")
"""

    process = None
    try:
        with provider_bundle.artifact_lock(artifact_path, scope="test"):
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    script,
                    source_path,
                    artifact_path,
                    str(ready_path),
                    str(acquired_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Importing the installed CuTe/CUDA stack can exceed 30 seconds on
            # a cold or contended network filesystem. The ready marker keeps
            # that startup time separate from the lock assertion below.
            deadline = time.monotonic() + _SUBPROCESS_READY_TIMEOUT_SECONDS
            while (
                not ready_path.exists()
                and process.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert ready_path.exists()
            with pytest.raises(subprocess.TimeoutExpired):
                process.wait(timeout=0.2)
            assert not acquired_path.exists()

        _, stderr = process.communicate(timeout=10)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.communicate(timeout=5)
    assert process is not None
    assert process.returncode == 0, stderr
    assert acquired_path.read_text(encoding="utf-8") == "acquired"


def _forked_artifact_lock_worker(
    artifact_path,
    inherited_descriptor,
    inherited_identity,
    connection,
):
    try:
        descriptor_stat = os.fstat(inherited_descriptor)
    except OSError:
        inherited_lock_descriptor = False
    else:
        inherited_lock_descriptor = (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        ) == inherited_identity
    connection.send(("inherited_lock_descriptor", inherited_lock_descriptor))
    with provider_bundle.artifact_lock(artifact_path, scope="test"):
        connection.send(("acquired", True))
    connection.close()


def _fork_inside_artifact_lock_worker(artifact_path, connection):
    report_read, report_write = os.pipe()
    child_pid = None
    descriptor = None
    with provider_bundle.artifact_lock(artifact_path, scope="test"):
        with provider_bundle._STATE_LOCK:
            active_descriptors = tuple(provider_bundle._ACTIVE_ARTIFACT_LOCK_FDS)
        assert len(active_descriptors) == 1
        descriptor = active_descriptors[0]
        child_pid = os.fork()
        if child_pid == 0:
            os.close(report_read)
            reused_read, reused_write = os.pipe()
            if descriptor not in (reused_read, reused_write):
                os.dup2(reused_read, descriptor)

    assert child_pid is not None
    assert descriptor is not None
    if child_pid == 0:
        try:
            os.fstat(descriptor)
        except OSError:
            os.write(report_write, b"0")
        else:
            os.write(report_write, b"1")
        os._exit(0)

    os.close(report_write)
    descriptor_survived = os.read(report_read, 1) == b"1"
    _, child_status = os.waitpid(child_pid, 0)
    connection.send(
        {
            "child_exitcode": os.waitstatus_to_exitcode(child_status),
            "descriptor_survived": descriptor_survived,
        }
    )
    connection.close()


@pytest.mark.skipif(
    not hasattr(os, "fork"),
    reason="test requires POSIX fork state reset",
)
@pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded, use of fork.*:DeprecationWarning"
)
def test_artifact_lock_state_resets_after_fork(tmp_path):
    artifact_path = str(tmp_path / "bundle.ltoir")
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_artifact_lock():
        with provider_bundle.artifact_lock(artifact_path, scope="test"):
            lock_held.set()
            release_lock.wait()

    holder = threading.Thread(target=hold_artifact_lock)
    holder.start()
    process = None
    parent_connection = None
    child_connection = None
    try:
        assert lock_held.wait(timeout=5)
        with provider_bundle._STATE_LOCK:
            inherited_descriptors = tuple(provider_bundle._ACTIVE_ARTIFACT_LOCK_FDS)
        assert len(inherited_descriptors) == 1
        inherited_descriptor = inherited_descriptors[0]
        descriptor_stat = os.fstat(inherited_descriptor)
        inherited_identity = (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        )
        context = multiprocessing.get_context("fork")
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_forked_artifact_lock_worker,
            args=(
                artifact_path,
                inherited_descriptor,
                inherited_identity,
                child_connection,
            ),
        )
        process.start()
        assert parent_connection.poll(5)
        assert parent_connection.recv() == (
            "inherited_lock_descriptor",
            False,
        )
        assert not parent_connection.poll(0.2)
        release_lock.set()
        assert parent_connection.poll(5)
        assert parent_connection.recv() == ("acquired", True)
        process.join(timeout=5)
    finally:
        release_lock.set()
        holder.join(timeout=5)
        if process is not None and process.is_alive():
            process.terminate()
        if process is not None:
            process.join(timeout=5)
        if parent_connection is not None:
            parent_connection.close()
        if child_connection is not None:
            child_connection.close()

    assert process is not None
    assert not holder.is_alive()
    assert not process.is_alive()
    assert process.exitcode == 0


@pytest.mark.skipif(
    not hasattr(os, "fork"),
    reason="test requires POSIX fork state reset",
)
@pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded, use of fork.*:DeprecationWarning"
)
def test_artifact_lock_context_ignores_stale_child_descriptor(tmp_path):
    artifact_path = str(tmp_path / "bundle.ltoir")
    context = multiprocessing.get_context("fork")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_fork_inside_artifact_lock_worker,
        args=(artifact_path, child_connection),
    )
    process.start()
    try:
        assert parent_connection.poll(10)
        result = parent_connection.recv()
        process.join(timeout=5)
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        parent_connection.close()
        child_connection.close()

    assert not process.is_alive()
    assert process.exitcode == 0
    assert result == {
        "child_exitcode": 0,
        "descriptor_survived": True,
    }
