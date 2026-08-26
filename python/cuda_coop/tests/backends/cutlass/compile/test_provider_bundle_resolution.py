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
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass")

from cutlass.base_dsl.common import DSLRuntimeError

import cuda.coop.cutlass._compiler._bundle as provider_bundle
import cuda.coop.cutlass._compiler._bundle_contract as provider_contract
import cuda.coop.cutlass._compiler._cache as provider_cache
import cuda.coop.cutlass._compiler._nvrtc as provider_nvrtc
from cuda.coop._headers import _toolkit


def test_include_options_use_filesystem_encoding():
    include_dir = "/tmp/cuda-coop-\udcff/include"

    assert provider_nvrtc.include_options([include_dir]) == [
        os.fsencode(f"-I{include_dir}")
    ]


def test_preload_toolkit_nvrtc_wraps_malformed_header_encoding(tmp_path):
    include_dir = tmp_path / "toolkit" / "include"
    include_dir.mkdir(parents=True)
    (include_dir / "cuda_runtime_api.h").write_bytes(b"\xff")

    with pytest.raises(
        DSLRuntimeError,
        match="Failed aligning provider NVRTC",
    ) as error:
        provider_nvrtc.preload_toolkit_nvrtc([str(include_dir)])

    assert isinstance(error.value.__cause__, RuntimeError)
    assert isinstance(error.value.__cause__.__cause__, UnicodeDecodeError)


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


def test_bitcode_resolution_tracks_the_selected_sm_arch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        provider_bundle._clang_support,
        "resolve_clang_compiler",
        lambda which: ("clang++", "clang-test", "clang-test"),
    )

    def compile_bundle(source, **kwargs):
        del source
        calls.append(kwargs)
        return provider_cache._CachedBundle(
            path=kwargs["output_path"], layouts_by_expression={}
        )

    monkeypatch.setattr(
        provider_bundle._clang_support,
        "compile_bundle",
        compile_bundle,
    )
    kwargs = _compile_kwargs()
    kwargs.update(
        select_bundle_format=lambda: "bc",
        resolve_nvrtc_sm_arch=lambda: "sm_90",
        resolve_nvrtc_arch=lambda: (_ for _ in ()).throw(
            AssertionError("bitcode must not resolve an NVRTC compute target")
        ),
    )

    path = provider_bundle.compile_bundle_source('extern "C" {}', **kwargs)

    assert path.endswith(".bc")
    assert len(calls) == 1
    assert "-march=sm_90" in calls[0]["compiler_options"]
    assert calls[0]["cache_identity"].bundle.bundle_sm_arch == "sm_90"


@pytest.fixture(autouse=True)
def _isolated_bundle_cache(monkeypatch, tmp_path):
    monkeypatch.setenv(
        provider_cache.CACHE_DIR_ENV,
        str(tmp_path / "provider-cache"),
    )
    provider_bundle.reset_compile_state()
    with _toolkit._PRELOAD_LOCK:
        _toolkit._EXACT_LIBRARY_HANDLES.clear()
    yield
    provider_bundle.reset_compile_state()
    with _toolkit._PRELOAD_LOCK:
        _toolkit._EXACT_LIBRARY_HANDLES.clear()


def test_configured_cache_dir_creates_missing_parents(monkeypatch, tmp_path):
    cache_dir = tmp_path / "nested" / "provider" / "cache"
    monkeypatch.setenv(provider_cache.CACHE_DIR_ENV, str(cache_dir))

    assert provider_cache.ensure_cache_dir("cuda.coop.cutlass") == str(cache_dir)
    assert cache_dir.is_dir()
    assert cache_dir.stat().st_mode & 0o777 == 0o700


def test_preload_toolkit_nvrtc_checks_lib64_and_ignores_unversioned_files(
    monkeypatch,
    tmp_path,
):
    include_dir = tmp_path / "toolkit" / "include"
    lib64_dir = tmp_path / "toolkit" / "lib64"
    include_dir.mkdir(parents=True)
    lib64_dir.mkdir()
    nvrtc = lib64_dir / "libnvrtc.so.13"
    nvjitlink = lib64_dir / "libnvJitLink.so.13"
    unversioned = lib64_dir / "libnvrtc.so"
    for library in (nvrtc, nvjitlink, unversioned):
        library.touch()
    (include_dir / "cuda_runtime_api.h").write_text(
        "#define CUDART_VERSION 13000\n",
        encoding="utf-8",
    )

    loaded = []
    monkeypatch.setattr(
        "ctypes.CDLL",
        lambda path, *, mode: loaded.append((path, mode)),
    )
    monkeypatch.setattr(
        "cuda.pathfinder.load_nvidia_dynamic_lib",
        lambda name: SimpleNamespace(
            abs_path=str(nvrtc if name == "nvrtc" else nvjitlink)
        ),
    )
    monkeypatch.setattr(_toolkit, "_nvjitlink_version", lambda path: (13, 0))
    monkeypatch.setattr(provider_nvrtc, "get_version_tuple", lambda: (13, 0))

    provider_nvrtc.preload_toolkit_nvrtc([str(include_dir)])

    assert [Path(path).name for path, _ in loaded] == [
        "libnvrtc.so.13",
        "libnvJitLink.so.13",
    ]
    with _toolkit._PRELOAD_LOCK:
        assert len(_toolkit._EXACT_LIBRARY_HANDLES) == 2


def test_preload_toolkit_nvrtc_loads_split_library_directories(
    monkeypatch,
    tmp_path,
):
    first_include = tmp_path / "first" / "include"
    second_include = tmp_path / "second" / "include"
    first_lib = tmp_path / "first" / "lib"
    second_lib = tmp_path / "second" / "lib"
    for directory in (first_include, second_include, first_lib, second_lib):
        directory.mkdir(parents=True)
    for include_dir in (first_include, second_include):
        (include_dir / "cuda_runtime_api.h").write_text(
            "#define CUDART_VERSION 13000\n",
            encoding="utf-8",
        )
    nvrtc = first_lib / "libnvrtc.so.13"
    nvjitlink = second_lib / "libnvJitLink.so.13"
    nvrtc.touch()
    nvjitlink.touch()

    loaded = []
    monkeypatch.setattr(
        "ctypes.CDLL",
        lambda path, *, mode: loaded.append((path, mode)),
    )
    monkeypatch.setattr(
        "cuda.pathfinder.load_nvidia_dynamic_lib",
        lambda name: SimpleNamespace(
            abs_path=str(nvrtc if name == "nvrtc" else nvjitlink)
        ),
    )
    monkeypatch.setattr(_toolkit, "_nvjitlink_version", lambda path: (13, 0))
    monkeypatch.setattr(provider_nvrtc, "get_version_tuple", lambda: (13, 0))

    provider_nvrtc.preload_toolkit_nvrtc([str(first_include), str(second_include)])

    assert [Path(path) for path, _ in loaded] == [nvrtc, nvjitlink]


def test_preload_toolkit_nvrtc_requires_a_reported_version(monkeypatch):
    monkeypatch.setattr(
        provider_nvrtc,
        "preload_toolkit_compiler_libraries",
        lambda _include_dirs: SimpleNamespace(
            nvrtc_path="/toolkit/libnvrtc.so",
            toolkit_version=(13, 0),
        ),
    )
    monkeypatch.setattr(provider_nvrtc, "get_version_tuple", lambda: None)

    with pytest.raises(
        DSLRuntimeError,
        match="Failed aligning provider NVRTC",
    ) as exc_info:
        provider_nvrtc.preload_toolkit_nvrtc(("/toolkit/include",))

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "loaded NVRTC did not report its version"


def test_resolution_routes_preserve_nvrtc_memory_and_disk_behavior(monkeypatch):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_nvrtc, "cuda_nvrtc", fake_nvrtc)

    first = provider_bundle.compile_bundle_source(
        'extern "C" {}',
        **_compile_kwargs(),
    )
    assert provider_bundle.get_bundle_telemetry().route_counts == {
        provider_contract.RESOLUTION_ROUTE_NVRTC: 1
    }
    second = provider_bundle.compile_bundle_source(
        'extern "C" {}',
        **_compile_kwargs(),
    )
    assert provider_bundle.get_bundle_telemetry().route_counts == {
        provider_contract.RESOLUTION_ROUTE_NVRTC: 1,
        provider_contract.RESOLUTION_ROUTE_MEMORY: 1,
    }
    provider_bundle.reset_compile_state()
    third = provider_bundle.compile_bundle_source(
        'extern "C" {}',
        **_compile_kwargs(),
    )

    assert first == second == third
    assert [call[0] for call in fake_nvrtc.calls].count("compile") == 1
    compile_options = next(call[1] for call in fake_nvrtc.calls if call[0] == "compile")
    identity_options = tuple(
        option.encode("ascii")
        for option in provider_contract.bundle_compiler_options(
            "ltoir",
            "compute_80",
        )
    )
    assert compile_options[: len(identity_options)] == identity_options

    telemetry = provider_bundle.get_bundle_telemetry()
    assert telemetry.route_counts == {provider_contract.RESOLUTION_ROUTE_DISK: 1}
    assert telemetry.phase_counts["total"] == 1
    assert "compiler" not in telemetry.phase_counts


def test_resolution_telemetry_counts_routes_and_exact_layouts(monkeypatch):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_nvrtc, "cuda_nvrtc", fake_nvrtc)

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
    assert first.layouts == {"storage": provider_contract.StorageLayout(40, 8)}

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
    assert provider_bundle.get_bundle_telemetry().route_counts == {
        provider_contract.RESOLUTION_ROUTE_NVRTC: 2
    }
    assert [call[0] for call in fake_nvrtc.calls].count("compile") == 2


@pytest.mark.skipif(os.name == "nt", reason="test exercises POSIX file locking")
def test_artifact_lock_serializes_across_processes(tmp_path):
    artifact_path = str(tmp_path / "bundle.ltoir")
    acquired_path = tmp_path / "acquired"
    source_path = str(Path(provider_bundle.__file__).resolve().parents[4])
    script = """
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
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
    with provider_cache.artifact_lock(artifact_path, scope="test"):
        connection.send(("acquired", True))
    connection.close()


def _forked_nvrtc_counter_worker(connection):
    outcome = {}

    def access_counter():
        try:
            provider_nvrtc.reset_compile_state()
            outcome["counter"] = provider_nvrtc.get_compile_program_counter()
        except Exception as exc:
            outcome["error"] = repr(exc)

    thread = threading.Thread(target=access_counter, daemon=True)
    thread.start()
    thread.join(timeout=2)
    connection.send(
        {
            "completed": not thread.is_alive(),
            "counter": outcome.get("counter"),
            "error": outcome.get("error"),
        }
    )
    connection.close()


def _forked_unknown_compiler_token_worker(connection):
    connection.send(provider_bundle._unknown_compiler_process_token())
    connection.close()


def _fork_inside_artifact_lock_worker(artifact_path, connection):
    report_read, report_write = os.pipe()
    child_pid = None
    descriptor = None
    with provider_cache.artifact_lock(artifact_path, scope="test"):
        with provider_cache._STATE_LOCK:
            active_descriptors = tuple(provider_cache._ACTIVE_ARTIFACT_LOCK_FDS)
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
def test_nvrtc_counter_lock_state_resets_after_fork():
    context = multiprocessing.get_context("fork")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_forked_nvrtc_counter_worker,
        args=(child_connection,),
    )
    process.start()
    child_connection.close()
    result = None
    try:
        assert parent_connection.poll(5)
        result = parent_connection.recv()
        process.join(timeout=5)
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        parent_connection.close()

    assert not process.is_alive()
    assert process.exitcode == 0
    assert result == {"completed": True, "counter": 0, "error": None}


@pytest.mark.skipif(
    not hasattr(os, "fork"),
    reason="test requires POSIX fork state reset",
)
@pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded, use of fork.*:DeprecationWarning"
)
def test_unknown_compiler_token_partitions_forked_processes():
    parent_token = provider_bundle._unknown_compiler_process_token()
    assert provider_bundle._unknown_compiler_process_token() == parent_token

    context = multiprocessing.get_context("fork")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_forked_unknown_compiler_token_worker,
        args=(child_connection,),
    )
    process.start()
    child_connection.close()
    child_token = None
    try:
        assert parent_connection.poll(5)
        child_token = parent_connection.recv()
        process.join(timeout=5)
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        parent_connection.close()

    assert process.exitcode == 0
    assert child_token != parent_token


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
        with provider_cache.artifact_lock(artifact_path, scope="test"):
            lock_held.set()
            release_lock.wait()

    holder = threading.Thread(target=hold_artifact_lock)
    holder.start()
    process = None
    parent_connection = None
    child_connection = None
    try:
        assert lock_held.wait(timeout=5)
        with provider_cache._STATE_LOCK:
            inherited_descriptors = tuple(provider_cache._ACTIVE_ARTIFACT_LOCK_FDS)
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

    assert not holder.is_alive()
    assert process is not None
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
