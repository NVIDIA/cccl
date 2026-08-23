# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import multiprocessing
import os
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass")

from cutlass.base_dsl.common import DSLRuntimeError

from cuda.coop._headers._identity import IncludeDirsIdentity
from cuda.coop.cutlass._dsl import _provider_bundle as provider_bundle
from cuda.coop.cutlass._dsl import _provider_pch as provider_pch


class _NvrtcResult:
    NVRTC_SUCCESS = 0
    NVRTC_ERROR_COMPILATION = 6
    NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED = 13
    NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED = 14
    NVRTC_ERROR_PCH_CREATE = 15


class _Program:
    def __init__(self, identifier, source):
        self.identifier = identifier
        self.source = source
        self.log = ""
        self.pch_status = _NvrtcResult.NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED
        self.destroyed = False


class _FakeNvrtc:
    nvrtcResult = _NvrtcResult

    def __init__(
        self,
        *,
        version=(13, 0),
        status_shape="binding",
        create_status=_NvrtcResult.NVRTC_SUCCESS,
        fail_first_pch_compile=False,
        compile_delay=0.0,
        compile_barrier=None,
    ):
        self.version = version
        self.status_shape = status_shape
        self.create_status = create_status
        self.fail_first_pch_compile = fail_first_pch_compile
        self.compile_delay = compile_delay
        self.compile_barrier = compile_barrier
        self.calls = []
        self.blob = b"fake-ltoir"
        self._next_program = 0
        self._failed_pch = False
        self._pch_directories = set()
        self._lock = threading.Lock()
        self._active_compiles = 0
        self.max_active_compiles = 0

    def nvrtcVersion(self):
        return self.nvrtcResult.NVRTC_SUCCESS, *self.version

    def nvrtcCreateProgram(self, source, name, num_headers, headers, include_names):
        with self._lock:
            self._next_program += 1
            program = _Program(self._next_program, source)
            self.calls.append(("create", program.identifier, source))
        return self.nvrtcResult.NVRTC_SUCCESS, program

    def nvrtcAddNameExpression(self, program, expression):
        self.calls.append(("add", program.identifier, expression))
        return (self.nvrtcResult.NVRTC_SUCCESS,)

    def nvrtcCompileProgram(self, program, num_options, options):
        options = tuple(options)
        with self._lock:
            self.calls.append(("compile", program.identifier, options))
            self._active_compiles += 1
            self.max_active_compiles = max(
                self.max_active_compiles,
                self._active_compiles,
            )
        try:
            if self.compile_barrier is not None:
                self.compile_barrier.wait(timeout=2)
            if self.compile_delay:
                time.sleep(self.compile_delay)
            pch_directory = next(
                (
                    option.decode("utf-8").removeprefix("--pch-dir=")
                    for option in options
                    if option.startswith(b"--pch-dir=")
                ),
                None,
            )
            if (
                pch_directory is not None
                and self.fail_first_pch_compile
                and not self._failed_pch
            ):
                self._failed_pch = True
                program.log = "simulated PCH compile failure"
                return (self.nvrtcResult.NVRTC_ERROR_COMPILATION,)
            if pch_directory is None:
                program.log = ""
                program.pch_status = (
                    self.nvrtcResult.NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED
                )
            elif pch_directory in self._pch_directories:
                program.log = (
                    f'using precompiled header file "{pch_directory}/provider.pch"'
                )
                program.pch_status = (
                    self.nvrtcResult.NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED
                )
            else:
                program.pch_status = self.create_status
                if self.create_status == self.nvrtcResult.NVRTC_SUCCESS:
                    self._pch_directories.add(pch_directory)
                    program.log = (
                        "creating precompiled header file "
                        f'"{pch_directory}/provider.pch"'
                    )
                else:
                    program.log = "PCH creation warning"
            return (self.nvrtcResult.NVRTC_SUCCESS,)
        finally:
            with self._lock:
                self._active_compiles -= 1

    def nvrtcGetPCHCreateStatus(self, program):
        if self.status_shape == "api_and_status":
            return self.nvrtcResult.NVRTC_SUCCESS, program.pch_status
        return (program.pch_status,)

    def nvrtcGetProgramLogSize(self, program):
        return self.nvrtcResult.NVRTC_SUCCESS, len(program.log.encode()) + 1

    def nvrtcGetProgramLog(self, program, log):
        log[:] = program.log.encode() + b"\0"
        return (self.nvrtcResult.NVRTC_SUCCESS,)

    def nvrtcGetLTOIRSize(self, program):
        return self.nvrtcResult.NVRTC_SUCCESS, len(self.blob)

    def nvrtcGetLTOIR(self, program, blob):
        blob[:] = self.blob
        return (self.nvrtcResult.NVRTC_SUCCESS,)

    def nvrtcDestroyProgram(self, program):
        program.destroyed = True
        self.calls.append(("destroy", program.identifier))
        return (self.nvrtcResult.NVRTC_SUCCESS,)

    def nvrtcSetPCHHeapSize(self, size):
        self.calls.append(("set_pch_heap_size", size))
        raise AssertionError("provider PCH support must not resize the NVRTC heap")


def _compile_kwargs(arch="80"):
    return {
        "scope": "cuda.coop.cutlass",
        "provider_dir": provider_bundle.__file__,
        "registered_headers": lambda: {},
        "select_bundle_format": lambda: "ltoir",
        "resolve_nvrtc_sm_arch": lambda: f"sm_{arch}",
        "resolve_nvrtc_arch": lambda: f"compute_{arch}",
    }


def _pch_directories(fake_nvrtc):
    return [
        next(
            (
                option.decode("utf-8").removeprefix("--pch-dir=")
                for option in call[2]
                if option.startswith(b"--pch-dir=")
            ),
            None,
        )
        for call in fake_nvrtc.calls
        if call[0] == "compile"
    ]


def _reset_pch_state():
    provider_pch._cleanup_pool()
    provider_pch._PCH_STATE_LOCK = threading.RLock()
    provider_pch._PCH_PID = os.getpid()
    provider_pch._PCH_POOL_PATH = None
    provider_pch._PCH_DISABLED_DOMAINS = set()
    provider_pch._PCH_DOMAIN_LOCKS = {}


@pytest.fixture(autouse=True)
def _isolated_provider_state(monkeypatch, tmp_path):
    header_digest = {"value": "headers-a"}
    monkeypatch.setenv(
        provider_bundle.CACHE_DIR_ENV,
        str(tmp_path / "provider-cache"),
    )
    monkeypatch.setenv(provider_pch.PCH_ENV, provider_pch.PCH_MODE_AUTO)
    monkeypatch.setattr(
        provider_bundle,
        "cccl_include_dirs",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        provider_bundle,
        "include_dirs_identity",
        lambda _include_dirs: IncludeDirsIdentity(
            roots=(),
            digest=header_digest["value"],
            recursive_walks=0,
            duration_ns=0,
        ),
    )
    provider_bundle.reset_compile_state()
    _reset_pch_state()
    yield header_digest
    provider_bundle.reset_compile_state()
    _reset_pch_state()


@pytest.mark.parametrize("status_shape", ["binding", "api_and_status"])
def test_distinct_uncached_bundles_share_one_pch_domain(
    monkeypatch,
    status_shape,
):
    fake_nvrtc = _FakeNvrtc(status_shape=status_shape)
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)
    preamble = "#define CUDA_COOP_FEATURE 1\n#include <stdint.h>\n"

    first = provider_bundle.compile_bundle_source(
        preamble + 'extern "C" __device__ void provider_a() {}\n',
        **_compile_kwargs(),
    )
    assert provider_bundle.get_nvrtc_compile_program_counter() == 1
    second = provider_bundle.compile_bundle_source(
        preamble + 'extern "C" __device__ void provider_b() {}\n',
        **_compile_kwargs(),
    )

    assert first != second
    assert _pch_directories(fake_nvrtc)[0] == _pch_directories(fake_nvrtc)[1]
    assert all(_pch_directories(fake_nvrtc))
    assert len([call for call in fake_nvrtc.calls if call[0] == "compile"]) == 2
    telemetry = provider_bundle.get_bundle_telemetry()
    assert telemetry.phase_counts["pch_lookup"] == 2
    assert telemetry.phase_counts["pch_create"] == 1
    assert telemetry.phase_counts["pch_hit"] == 1


def test_preamble_arch_and_header_identity_partition_domains(
    monkeypatch,
    _isolated_provider_state,
):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)
    preamble = "#define CUDA_COOP_FEATURE 1\n#include <stdint.h>\n"

    provider_bundle.compile_bundle_source(
        preamble + 'extern "C" __device__ void provider_a() {}\n',
        **_compile_kwargs(),
    )
    provider_bundle.compile_bundle_source(
        "#define CUDA_COOP_FEATURE 2\n#include <stdint.h>\n"
        'extern "C" __device__ void provider_b() {}\n',
        **_compile_kwargs(),
    )
    provider_bundle.compile_bundle_source(
        preamble + 'extern "C" __device__ void provider_c() {}\n',
        **_compile_kwargs("90"),
    )
    _isolated_provider_state["value"] = "headers-b"
    provider_bundle.compile_bundle_source(
        preamble + 'extern "C" __device__ void provider_d() {}\n',
        **_compile_kwargs(),
    )

    directories = _pch_directories(fake_nvrtc)
    assert len(directories) == 4
    assert len(set(directories)) == 4


def test_preamble_identity_excludes_request_body():
    preamble = (
        "// generated provider\n#define CUDA_COOP_FEATURE 1\n#include <stdint.h>\n"
    )
    first = provider_pch.provider_preamble_identity(
        preamble + 'extern "C" void provider_a();\n'
    )
    second = provider_pch.provider_preamble_identity(
        preamble + 'extern "C" void provider_b();\n'
    )
    different = provider_pch.provider_preamble_identity(
        "#define CUDA_COOP_FEATURE 2\n"
        "#include <stdint.h>\n"
        'extern "C" void provider_a();\n'
    )

    assert first == second
    assert different != first


def test_unsupported_nvrtc_and_explicit_off_skip_pch(monkeypatch):
    fake_nvrtc = _FakeNvrtc(version=(12, 7))
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)

    provider_bundle.compile_bundle_source(
        'extern "C" __device__ void provider_a() {}\n',
        **_compile_kwargs(),
    )
    assert _pch_directories(fake_nvrtc) == [None]
    assert provider_pch._PCH_POOL_PATH is None
    assert provider_bundle.get_bundle_telemetry().phase_counts["pch_unsupported"] == 1

    provider_bundle.reset_compile_state()
    monkeypatch.setenv(provider_pch.PCH_ENV, provider_pch.PCH_MODE_OFF)
    provider_bundle.compile_bundle_source(
        'extern "C" __device__ void provider_b() {}\n',
        **_compile_kwargs(),
    )
    assert _pch_directories(fake_nvrtc) == [None, None]
    assert provider_bundle.get_bundle_telemetry().phase_counts["pch_off"] == 1


def test_invalid_mode_fails_before_creating_an_nvrtc_program(monkeypatch):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)
    monkeypatch.setenv(provider_pch.PCH_ENV, "sometimes")

    with pytest.raises(DSLRuntimeError, match="Invalid .* PCH configuration"):
        provider_bundle.compile_bundle_source(
            'extern "C" __device__ void provider_a() {}\n',
            **_compile_kwargs(),
        )

    assert not any(call[0] == "create" for call in fake_nvrtc.calls)


@pytest.mark.parametrize(
    "failure_site",
    [
        "pool_mkdtemp",
        "pool_chmod",
        "pool_lstat",
        "pool_security",
        "domain_mkdir",
        "domain_lstat",
        "domain_security",
    ],
)
def test_operational_setup_failure_compiles_once_and_disables_domain(
    monkeypatch,
    failure_site,
):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)
    original_chmod = provider_pch.os.chmod
    original_lstat = provider_pch.os.lstat
    original_mkdir = provider_pch.os.mkdir

    def is_pool(path):
        return Path(os.fspath(path)).name.startswith("cuda-coop-cutlass-pch-")

    def is_domain(path):
        return Path(os.fspath(path)).name.startswith("domain-")

    def unavailable(*_args, **_kwargs):
        raise OSError("simulated PCH setup failure")

    if failure_site == "pool_mkdtemp":
        monkeypatch.setattr(provider_pch.tempfile, "mkdtemp", unavailable)
    elif failure_site == "pool_chmod":

        def chmod(path, mode):
            if is_pool(path):
                unavailable()
            return original_chmod(path, mode)

        monkeypatch.setattr(provider_pch.os, "chmod", chmod)
    elif failure_site == "domain_mkdir":

        def mkdir(path, mode=0o777):
            if is_domain(path):
                unavailable()
            return original_mkdir(path, mode)

        monkeypatch.setattr(provider_pch.os, "mkdir", mkdir)
    else:
        expected_path = is_pool if failure_site.startswith("pool_") else is_domain

        def lstat(path):
            result = original_lstat(path)
            if not expected_path(path):
                return result
            if failure_site.endswith("_lstat"):
                unavailable()
            return SimpleNamespace(
                st_mode=(result.st_mode & ~0o777) | 0o755,
            )

        monkeypatch.setattr(provider_pch.os, "lstat", lstat)

    preamble = "#include <stdint.h>\n"
    first = provider_bundle.compile_bundle_source(
        preamble + 'extern "C" __device__ void provider_a() {}\n',
        **_compile_kwargs(),
    )
    second = provider_bundle.compile_bundle_source(
        preamble + 'extern "C" __device__ void provider_b() {}\n',
        **_compile_kwargs(),
    )
    if "lstat" in failure_site or "security" in failure_site:
        monkeypatch.setattr(provider_pch.os, "lstat", original_lstat)

    assert Path(first).is_file()
    assert Path(second).is_file()
    assert _pch_directories(fake_nvrtc) == [None, None]
    assert provider_bundle.get_nvrtc_compile_program_counter() == 2
    assert len([call for call in fake_nvrtc.calls if call[0] == "create"]) == 2
    assert len([call for call in fake_nvrtc.calls if call[0] == "destroy"]) == 2
    telemetry = provider_bundle.get_bundle_telemetry()
    assert telemetry.phase_counts["pch_unavailable"] == 1
    assert telemetry.phase_counts["pch_disabled"] == 1
    assert "pch_fallback" not in telemetry.phase_counts


def test_failed_pch_compile_retries_once_and_disables_domain(monkeypatch):
    fake_nvrtc = _FakeNvrtc(fail_first_pch_compile=True)
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)
    preamble = "#include <stdint.h>\n"

    first = provider_bundle.compile_bundle_source(
        preamble + 'extern "C" __device__ void provider_a() {}\n',
        **_compile_kwargs(),
    )
    assert provider_bundle.get_nvrtc_compile_program_counter() == 2
    second = provider_bundle.compile_bundle_source(
        preamble + 'extern "C" __device__ void provider_b() {}\n',
        **_compile_kwargs(),
    )

    assert Path(first).is_file()
    assert Path(second).is_file()
    assert _pch_directories(fake_nvrtc) == [
        _pch_directories(fake_nvrtc)[0],
        None,
        None,
    ]
    assert _pch_directories(fake_nvrtc)[0] is not None
    assert len([call for call in fake_nvrtc.calls if call[0] == "create"]) == 3
    assert len([call for call in fake_nvrtc.calls if call[0] == "destroy"]) == 3
    assert provider_bundle.get_nvrtc_compile_program_counter() == 3
    telemetry = provider_bundle.get_bundle_telemetry()
    assert telemetry.phase_counts["pch_fallback"] == 1
    assert telemetry.phase_counts["pch_disabled"] == 1


@pytest.mark.parametrize(
    "create_status",
    [
        _NvrtcResult.NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED,
        _NvrtcResult.NVRTC_ERROR_PCH_CREATE,
    ],
)
def test_pch_create_warning_preserves_artifact_without_heap_resize(
    monkeypatch,
    create_status,
):
    fake_nvrtc = _FakeNvrtc(create_status=create_status)
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)

    artifact = provider_bundle.compile_bundle_source(
        '#include <stdint.h>\nextern "C" __device__ void provider_a() {}\n',
        **_compile_kwargs(),
    )
    provider_bundle.compile_bundle_source(
        '#include <stdint.h>\nextern "C" __device__ void provider_b() {}\n',
        **_compile_kwargs(),
    )

    assert Path(artifact).read_bytes() == fake_nvrtc.blob
    assert not any(call[0] == "set_pch_heap_size" for call in fake_nvrtc.calls)
    assert _pch_directories(fake_nvrtc)[0] is not None
    assert _pch_directories(fake_nvrtc)[1] is None
    telemetry = provider_bundle.get_bundle_telemetry()
    assert telemetry.phase_counts["pch_create"] == 1
    assert telemetry.phase_counts["pch_create_warning"] == 1
    assert telemetry.phase_counts["pch_disabled"] == 1


def test_pch_lifecycle_is_serialized_across_distinct_artifacts(monkeypatch):
    fake_nvrtc = _FakeNvrtc(compile_delay=0.08)
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)
    barrier = threading.Barrier(2)
    preamble = "#include <stdint.h>\n"

    def compile_one(name):
        barrier.wait(timeout=2)
        return provider_bundle.compile_bundle_source(
            preamble + f'extern "C" __device__ void {name}() {{}}\n',
            **_compile_kwargs(),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        artifacts = tuple(executor.map(compile_one, ("provider_a", "provider_b")))

    assert len(set(artifacts)) == 2
    assert fake_nvrtc.max_active_compiles == 1
    telemetry = provider_bundle.get_bundle_telemetry()
    assert telemetry.phase_counts["pch_create"] == 1
    assert telemetry.phase_counts["pch_hit"] == 1


def test_unrelated_pch_domains_compile_concurrently(monkeypatch):
    compile_barrier = threading.Barrier(2)
    fake_nvrtc = _FakeNvrtc(compile_barrier=compile_barrier)
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)
    start_barrier = threading.Barrier(2)

    def compile_one(feature):
        start_barrier.wait(timeout=2)
        return provider_bundle.compile_bundle_source(
            f"#define CUDA_COOP_FEATURE {feature}\n"
            f'extern "C" __device__ void provider_{feature}() {{}}\n',
            **_compile_kwargs(),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        artifacts = tuple(executor.map(compile_one, (1, 2)))

    assert len(set(artifacts)) == 2
    assert fake_nvrtc.max_active_compiles == 2


def _forked_pool_worker(connection):
    with provider_pch.pch_session(
        nvrtc_version=(13, 0),
        bundle_arch="compute_80",
        bundle_sm_arch="sm_80",
        compiler_options=("--std=c++17",),
        include_dirs=("/include",),
        header_identity="headers",
        preamble_identity="preamble",
    ) as session:
        connection.send((os.getpid(), session.directory))
    _reset_pch_state()
    connection.close()


@pytest.mark.skipif(
    not hasattr(os, "fork"),
    reason="test requires POSIX fork state reset",
)
@pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded, use of fork.*:DeprecationWarning"
)
def test_fork_uses_a_new_process_private_pool():
    with provider_pch.pch_session(
        nvrtc_version=(13, 0),
        bundle_arch="compute_80",
        bundle_sm_arch="sm_80",
        compiler_options=("--std=c++17",),
        include_dirs=("/include",),
        header_identity="headers",
        preamble_identity="preamble",
    ) as parent_session:
        parent_directory = Path(parent_session.directory)
    parent_pool = parent_directory.parent

    context = multiprocessing.get_context("fork")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_forked_pool_worker,
        args=(child_connection,),
    )
    process.start()
    assert parent_connection.poll(5)
    child_pid, child_directory_text = parent_connection.recv()
    process.join(timeout=5)

    assert not process.is_alive()
    assert process.exitcode == 0
    child_pool = Path(child_directory_text).parent
    assert child_pool != parent_pool
    assert str(child_pid) in child_pool.name
    assert str(os.getpid()) in parent_pool.name
    assert parent_pool.is_dir()
    assert stat.S_IMODE(parent_pool.stat().st_mode) == 0o700
