# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check process-local PCH reuse and layout queries on distinct JIT misses."""

import multiprocessing
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cutlass
import pytest
from cutlass import Int32, cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

import cuda.coop.cutlass as coop
from cuda.coop.cutlass._compiler import (
    _bundle,
    _cache,
    _layout,
    _nvrtc,
    _rendering,
)
from cuda.coop.cutlass._compiler._types import ScratchLayout, ScratchLayoutProbe
from cuda.coop.cutlass._headers._identity import include_dirs_identity
from cuda.coop.cutlass._lowering._load_store import (
    _CubLoadStoreRequest,
    _scratch_layout_probe,
)
from cuda.coop.cutlass._planning import this_block

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def _record_compile_logs(monkeypatch):
    nvrtc = _nvrtc._load_nvrtc()
    compile_program = nvrtc.nvrtcCompileProgram
    logs = []

    def record(program, count, flags):
        result = compile_program(program, count, flags)
        logs.append(_nvrtc._program_log(nvrtc, program).lower())
        return result

    monkeypatch.setattr(nvrtc, "nvrtcCompileProgram", record)
    return logs


def test_pch_gating_and_unavailable_storage(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("temporary directory unavailable")

    context = SimpleNamespace(nvrtc_version=(12, 8), header_identity="test")
    monkeypatch.setattr(_nvrtc.tempfile, "mkdtemp", unavailable)
    monkeypatch.setenv(_nvrtc._PCH_ENV, "")
    assert _nvrtc.pch_options(context) == ()
    monkeypatch.setattr(_nvrtc.tempfile, "mkdtemp", pytest.fail)
    monkeypatch.setenv(_nvrtc._PCH_ENV, "off")
    assert _nvrtc.pch_options(context) == ()
    context.nvrtc_version = (12, 7)
    for mode in ("auto", "common"):
        monkeypatch.setenv(_nvrtc._PCH_ENV, mode)
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
    logs = _record_compile_logs(monkeypatch)

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
    assert "creating precompiled header" in logs[0]
    assert "using precompiled header" in logs[1]

    header.write_text("struct alignas(32) S { int value; };")
    assert compile_layout(1) == ScratchLayout(32, 32)
    assert "creating precompiled header" in logs[2]


def _prepared_provider(kind, items):
    from tests._group_planning import _load_store, _plan

    operation = _load_store(
        kind, dtype=Int32, items_per_thread=items, algorithm="transpose"
    )
    request = _CubLoadStoreRequest(_plan(this_block(), operation), Int32)
    return _layout._prepare_layout_probes(
        _rendering.render_bundle_source([request]),
        [_scratch_layout_probe(request)],
    )


def test_common_pch_reuses_across_cub_operations(tmp_path, monkeypatch):
    if _nvrtc.resolve_compile_context(()).nvrtc_version < (12, 8):
        pytest.skip("NVRTC automatic PCH requires CUDA 12.8 or later")

    @cute.kernel
    def kernel(memory: cute.Pointer, scan: cutlass.Constexpr):
        block = coop.this_block()
        data = cute.make_tensor(memory, cute.make_layout(64))
        value = data[block.rank()]
        if cutlass.const_expr(scan):
            data[block.rank()] = coop.inclusive_scan(block, value)
        else:
            total = coop.sum(block, value)
            if block.rank() == 0:
                data[0] = total

    @cute.jit
    def launch(memory: cute.Pointer, scan: cutlass.Constexpr):
        kernel(memory, scan).launch(grid=1, block=64)

    logs = _record_compile_logs(monkeypatch)
    monkeypatch.setattr(_nvrtc, "_PCH_DIRECTORIES", {})
    monkeypatch.setenv(_nvrtc._PCH_ENV, "common")
    monkeypatch.setenv(_cache.CACHE_DIR_ENV, str(tmp_path / "cache"))
    memory = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    compile_kernel = cute.compile[(GPUArch("sm_80"),)]
    assert compile_kernel(launch, memory, True) is not None
    assert compile_kernel(launch, memory, False) is not None
    assert len(logs) == 2
    assert "creating precompiled header" in logs[0]
    assert "using precompiled header" in logs[1]


def test_interleaved_cub_prefixes_and_simultaneous_compiles(monkeypatch):
    context = _nvrtc.resolve_compile_context(())
    options = _nvrtc.compiler_options(context, "compute_80")
    inputs = [
        _prepared_provider(kind, n)
        for n in (2, 4)
        for kind in ("load", "store")
    ]
    reference = [
        _nvrtc.compile_ltoir_with_layouts(provider, options)[1]
        for provider in inputs
    ]

    monkeypatch.setattr(_nvrtc, "_PCH_DIRECTORIES", {})
    monkeypatch.setenv(_nvrtc._PCH_ENV, "")
    pch = _nvrtc.pch_options(context)
    assert pch
    logs = _record_compile_logs(monkeypatch)

    # Load, Store, then a distinct Load specialization with the first prefix.
    def compile_provider(index):
        blob, layouts = _nvrtc.compile_ltoir_with_layouts(
            inputs[index], options + pch
        )
        assert blob and layouts == reference[index]

    for index in (0, 1, 2):
        compile_provider(index)
    assert "using precompiled header" in logs[2]

    # Start simultaneous misses with a fresh directory, as on a cold worker.
    monkeypatch.setattr(_nvrtc, "_PCH_DIRECTORIES", {})
    pch = _nvrtc.pch_options(context)
    start = threading.Barrier(len(inputs))

    def worker(index):
        start.wait(timeout=30)
        compile_provider(index)

    with ThreadPoolExecutor(len(inputs)) as workers:
        list(workers.map(worker, range(len(inputs))))


def test_pch_failure_retries_with_probes_and_disables_affected_directory(
    monkeypatch, caplog
):
    context = _nvrtc.resolve_compile_context(())
    monkeypatch.setattr(_nvrtc, "_PCH_DIRECTORIES", {})
    monkeypatch.setenv(_nvrtc._PCH_ENV, "auto")
    pch = _nvrtc.pch_options(context)
    directory = Path(os.fsdecode(pch[1].split(b"=", 1)[1]))
    directory.rmdir()  # NVRTC diagnoses an invalid PCH directory.
    prepared = _prepared_provider("load", 2)
    blob, layouts = _nvrtc.compile_ltoir_with_layouts(
        prepared, _nvrtc.compiler_options(context, "compute_80") + pch
    )
    assert blob and layouts
    assert _nvrtc.pch_options(context) == ()
    assert "Disabling NVRTC PCH" in caplog.text


def test_source_error_after_checkpoint_is_reported_once(tmp_path, monkeypatch):
    header = tmp_path / "source_error.cuh"
    header.write_text("struct S { int value; };")
    context = _nvrtc.resolve_compile_context(())
    monkeypatch.setattr(_nvrtc, "_PCH_DIRECTORIES", {})
    monkeypatch.setenv(_nvrtc._PCH_ENV, "auto")
    options = (
        *_nvrtc.compiler_options(context, "compute_80"),
        os.fsencode(f"--include-path={tmp_path}"),
        *_nvrtc.pch_options(context),
    )
    nvrtc = _nvrtc._load_nvrtc()
    calls = []
    compile_program = nvrtc.nvrtcCompileProgram

    def record(program, count, flags):
        calls.append(tuple(flags))
        return compile_program(program, count, flags)

    monkeypatch.setattr(nvrtc, "nvrtcCompileProgram", record)
    with pytest.raises(RuntimeError, match="deliberate_source_error"):
        _nvrtc.compile_ltoir(
            '#include "source_error.cuh"\n#error deliberate_source_error\n',
            options,
        )
    assert calls == [options]
    assert _nvrtc.pch_options(context)


def test_exhausted_pch_heap_keeps_ltoir_and_retires_directory(monkeypatch):
    # Changing the NVRTC heap invalidates PCH from other users in the process.
    # Inject just the status so this test cannot resize that shared heap.
    context = _nvrtc.resolve_compile_context(())
    monkeypatch.setattr(_nvrtc, "_PCH_DIRECTORIES", {})
    monkeypatch.setenv(_nvrtc._PCH_ENV, "auto")
    nvrtc = _nvrtc._load_nvrtc()
    monkeypatch.setattr(
        nvrtc,
        "nvrtcGetPCHCreateStatus",
        lambda program: (
            nvrtc.nvrtcResult.NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED,
        ),
    )
    data = _nvrtc.compile_ltoir(
        'extern "C" __device__ int value() { return 42; }',
        _nvrtc.compiler_options(context, "compute_80")
        + _nvrtc.pch_options(context),
    )
    assert data
    assert _nvrtc.pch_options(context) == ()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="fork-specific PCH directory ownership",
)
def test_fork_worker_releases_only_its_pch_directory(tmp_path):
    # The pytest process may have compiler or BLAS worker threads. Fork
    # from a fresh interpreter with NumPy's worker pool limited to one.
    subprocess.run(
        [sys.executable, __file__, str(tmp_path)],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(sys.path),
            "OPENBLAS_NUM_THREADS": "1",
            _nvrtc._PCH_ENV: "auto",
        },
        timeout=30,
        check=True,
    )


def _check_fork_cleanup(tmp_path):
    context = SimpleNamespace(
        nvrtc_version=(13, 3), header_identity=str(tmp_path)
    )

    def pch_directory():
        return Path(
            os.fsdecode(_nvrtc.pch_options(context)[1].split(b"=", 1)[1])
        )

    parent = pch_directory()
    record = tmp_path / "child-directory"

    def worker():
        record.write_text(str(pch_directory()))

    process = multiprocessing.get_context("fork").Process(target=worker)
    process.start()
    process.join(timeout=10)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    assert process.exitcode == 0
    child = Path(record.read_text())
    assert child != parent
    assert parent.is_dir()
    assert not child.exists()


if __name__ == "__main__":
    _check_fork_cleanup(Path(sys.argv[1]))
