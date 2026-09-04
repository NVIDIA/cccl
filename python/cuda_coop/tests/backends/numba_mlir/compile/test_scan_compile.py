# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GPU-hidden, real-toolchain compilation contracts for Scan providers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("numba_cuda_mlir")

import numba_cuda_mlir.tools as numba_mlir_tools
from numba_cuda_mlir import cuda, types

from cuda.coop._core import ArgumentBinding, SynchronizationScope
from cuda.coop.numba_mlir import StatefulFunction, _types
from cuda.coop.numba_mlir._compiler import _nvrtc
from cuda.coop.numba_mlir._compiler._operations import StorageABI
from cuda.coop.numba_mlir._lowering import _scan

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]

_FIXED_COMPUTE_CAPABILITY = (9, 0)
_FIXED_CC = 90
_BLOCK_THREADS = 64


class _RunningPrefixFunctor:
    def __call__(self_ptr, block_aggregate):
        previous = self_ptr[0]
        self_ptr[0] = previous + block_aggregate
        return previous


@pytest.fixture(autouse=True)
def _fixed_compiler_target(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", (
        "the Numba-CUDA-MLIR compile stage must hide all CUDA devices"
    )
    queries: list[tuple[int, int]] = []

    def current_device() -> SimpleNamespace:
        queries.append(_FIXED_COMPUTE_CAPABILITY)
        return SimpleNamespace(compute_capability=_FIXED_COMPUTE_CAPABILITY)

    def compute_capability(as_type=str):
        assert as_type in (str, tuple)
        return _FIXED_COMPUTE_CAPABILITY if as_type is tuple else "sm_90"

    monkeypatch.setattr(_types.cuda, "get_current_device", current_device)
    monkeypatch.setattr(
        numba_mlir_tools,
        "get_gpu_compute_capability",
        compute_capability,
    )
    _types._DEVICE_LTOIR_CACHE.clear()
    return queries


@pytest.fixture(scope="module")
def compile_context() -> _nvrtc.CompileContext:
    return _nvrtc.resolve_compile_context()


def _collect(factory, compile_context: _nvrtc.CompileContext, /, **kwargs):
    with _types.collect_specializations() as collected:
        result = factory(**kwargs)
    assert len(collected) == 1
    algorithm, threads, block_threads = collected[0]
    assert result is algorithm
    algorithm._compile_context = compile_context
    return algorithm, threads, block_threads


def _source(collected) -> str:
    algorithm, threads, block_threads = collected
    return algorithm._source_code(
        threads=threads,
        block_threads=block_threads,
    )[0]


def _compile_bundle(collected, *, name: str) -> bytes:
    ltoir = _types.prepare_ltoir_bundle(
        [item[0] for item in collected],
        bundle_name=name,
        allow_single=True,
        threads_by_algo={id(item[0]): item[1] for item in collected},
        block_threads_by_algo={id(item[0]): item[2] for item in collected},
    )
    assert isinstance(ltoir, bytes)
    assert ltoir
    return ltoir


def test_block_scalar_array_algorithms_methods_and_aggregates_compile(
    compile_context: _nvrtc.CompileContext,
    _fixed_compiler_target: list[tuple[int, int]],
) -> None:
    cases = (
        (
            _scan.block_scan_scalar,
            {
                "dtype": types.int32,
                "threads_per_block": _BLOCK_THREADS,
                "mode": "exclusive",
                "algorithm": "raking",
            },
            ".ExclusiveSum(",
            "::cub::BLOCK_SCAN_RAKING",
        ),
        (
            _scan.block_scan_array,
            {
                "dtype": types.float32,
                "threads_per_block": _BLOCK_THREADS,
                "items_per_thread": 2,
                "value_kind": "array",
                "mode": "inclusive",
                "algorithm": "raking_memoize",
            },
            ".InclusiveSum(",
            "::cub::BLOCK_SCAN_RAKING_MEMOIZE",
        ),
        (
            _scan.block_scan_array,
            {
                "dtype": types.int16,
                "threads_per_block": _BLOCK_THREADS,
                "items_per_thread": 3,
                "value_kind": "array",
                "mode": "exclusive",
                "scan_op": "maximum",
                "initial_value": ArgumentBinding.static(-7),
                "block_aggregate": True,
                "algorithm": "warp_scans",
            },
            ".ExclusiveScan(",
            "::cub::BLOCK_SCAN_WARP_SCANS",
        ),
        (
            _scan.block_scan_scalar,
            {
                "dtype": types.uint32,
                "threads_per_block": _BLOCK_THREADS,
                "mode": "inclusive",
                "scan_op": np.bitwise_or,
                "block_aggregate": True,
                "algorithm": "raking",
            },
            ".InclusiveScan(",
            "::cub::BLOCK_SCAN_RAKING",
        ),
    )
    collected = [
        _collect(factory, compile_context, **kwargs)
        for factory, kwargs, _method, _algorithm in cases
    ]

    for item, (_factory, kwargs, method, algorithm_name) in zip(collected, cases):
        algorithm = item[0]
        source = _source(item)
        assert method in source
        assert algorithm_name in source
        assert "cub::BlockScan<" in source
        assert "TempStorage" in source
        assert "__shared__" in source
        assert "__syncthreads();" in source
        assert "__syncwarp" not in source
        assert algorithm.storage_abi is StorageABI.LEADING_POINTER
        assert algorithm.execution_scope is SynchronizationScope.BLOCK
        assert algorithm.synchronization_scope is SynchronizationScope.BLOCK
        if kwargs.get("value_kind") == "array":
            cpp_dtype = _types.numba_type_to_cpp(kwargs["dtype"])
            items_per_thread = kwargs["items_per_thread"]
            array_cast = f"reinterpret_cast<{cpp_dtype} (*)[{items_per_thread}]>(param_"
            # Both the compiler-owned and caller-owned storage entry points
            # accept distinct input/output array pointers and reinterpret them
            # as CUB's fixed-size array references.
            assert source.count(array_cast) >= 4
            assert "void *abi_param_0, void *abi_param_1" in source
        if kwargs.get("block_aggregate"):
            call_lines = [
                line.strip()
                for line in source.splitlines()
                if method in line and "algorithm_t_" in line
            ]
            assert call_lines
            assert all(re.search(r", \*param_\d+\);$", line) for line in call_lines)

    bundle = _compile_bundle(
        collected,
        name="cuda_coop_numba_mlir_block_scan_variants",
    )
    ptx = _types._ltoir_to_ptx(
        bundle,
        name="cuda_coop_numba_mlir_block_scan_variants",
        cc=_FIXED_CC,
    )
    assert ".version" in ptx
    assert all(item[0].temp_storage_bytes > 0 for item in collected)
    assert all(item[0].temp_storage_alignment > 0 for item in collected)
    assert _fixed_compiler_target


def test_remaining_supported_numeric_dtypes_compile(
    compile_context: _nvrtc.CompileContext,
) -> None:
    collected = [
        _collect(
            _scan.block_scan_scalar,
            compile_context,
            dtype=dtype,
            threads_per_block=_BLOCK_THREADS,
            mode="inclusive",
        )
        for dtype in (types.int8, types.uint8, types.uint64, types.float64)
    ]

    for item in collected:
        source = _source(item)
        assert ".InclusiveSum(" in source
        assert "cub::BlockScan<" in source

    bundle = _compile_bundle(
        collected,
        name="cuda_coop_numba_mlir_scan_remaining_dtypes",
    )
    assert ".version" in _types._ltoir_to_ptx(
        bundle,
        name="cuda_coop_numba_mlir_scan_remaining_dtypes",
        cc=_FIXED_CC,
    )


def test_physical_and_logical_warp_methods_prefixes_and_aggregates_compile(
    compile_context: _nvrtc.CompileContext,
) -> None:
    cases = (
        (
            {
                "dtype": types.int32,
                "threads_in_warp": 32,
                "threads_per_block": _BLOCK_THREADS,
                "mode": "exclusive",
            },
            ".ExclusiveSum(",
        ),
        (
            {
                "dtype": types.float32,
                "threads_in_warp": 8,
                "threads_per_block": _BLOCK_THREADS,
                "mode": "inclusive",
                "scan_op": np.maximum,
                "warp_aggregate": True,
            },
            ".InclusiveScan(",
        ),
        (
            {
                "dtype": types.int64,
                "threads_in_warp": 8,
                "threads_per_block": _BLOCK_THREADS,
                "mode": "exclusive",
                "scan_op": "multiplies",
                "initial_value": ArgumentBinding.static(1),
                "valid_items": ArgumentBinding.static(5),
                "warp_aggregate": True,
            },
            ".ExclusiveScanPartial(",
        ),
        (
            {
                "dtype": types.int32,
                "threads_in_warp": 8,
                "threads_per_block": _BLOCK_THREADS,
                "mode": "exclusive",
                "valid_items": ArgumentBinding.static(5),
            },
            ".ExclusiveScanPartial(",
        ),
        (
            {
                "dtype": types.uint16,
                "threads_in_warp": 8,
                "threads_per_block": _BLOCK_THREADS,
                "mode": "inclusive",
                "scan_op": "bit_xor",
                "valid_items": ArgumentBinding.runtime(),
            },
            ".InclusiveScanPartial(",
        ),
    )
    collected = [
        _collect(_scan.warp_scan, compile_context, **kwargs)
        for kwargs, _method in cases
    ]

    for item, (kwargs, method) in zip(collected, cases):
        algorithm = item[0]
        source = _source(item)
        width = kwargs["threads_in_warp"]
        assert method in source
        assert (
            f"cub::WarpScan<{_types.numba_type_to_cpp(kwargs['dtype'])}, {width}>"
            in source
        )
        assert "TempStorage" in source
        assert "__shared__" in source
        assert "__syncthreads" not in source
        assert "__syncwarp" in source
        assert algorithm.storage_abi is StorageABI.LEADING_POINTER
        assert algorithm.execution_scope is SynchronizationScope.WARP
        assert algorithm.synchronization_scope is SynchronizationScope.WARP
        if kwargs.get("warp_aggregate"):
            call_lines = [
                line.strip()
                for line in source.splitlines()
                if method in line and "algorithm_t_" in line
            ]
            assert call_lines
            assert all(re.search(r", \*param_\d+\);$", line) for line in call_lines)

    runtime_prefix = _source(collected[-1])
    assert "::cuda::std::int64_t" in runtime_prefix
    assert "static_cast<::cuda::std::int32_t>" in runtime_prefix
    assert 'asm volatile("trap;" : : :);' in runtime_prefix
    assert re.search(
        r"if \((param_\d+) < 1 \|\| \1 > 8\)",
        runtime_prefix,
    )
    assert ", 5," in _source(collected[-3])

    partial_exclusive_sum = _source(collected[-2])
    assert ".ExclusiveScanPartial(" in partial_exclusive_sum
    assert "::cuda::std::int32_t{0}" in partial_exclusive_sum
    assert ", 0," not in partial_exclusive_sum

    bundle = _compile_bundle(
        collected,
        name="cuda_coop_numba_mlir_warp_scan_variants",
    )
    ptx = _types._ltoir_to_ptx(
        bundle,
        name="cuda_coop_numba_mlir_warp_scan_variants",
        cc=_FIXED_CC,
    )
    assert ".version" in ptx


def _link_ltoir_files(paths: list[str], *, name: str) -> str:
    from cuda.core import Linker, LinkerOptions, ObjectCode

    objects = [
        ObjectCode.from_ltoir(Path(path).read_bytes(), name=f"{name}_{index}")
        for index, path in enumerate(paths)
    ]
    options = LinkerOptions(
        arch=f"sm_{_FIXED_CC}",
        link_time_optimization=True,
        ptx=True,
    )
    return Linker(*objects, options=options).link("ptx").code.decode("utf-8")


def test_stateless_block_and_warp_scan_callbacks_link_with_provider_lto(
    compile_context: _nvrtc.CompileContext,
) -> None:
    def maximum(lhs, rhs):
        return lhs if lhs > rhs else rhs  # noqa: FURB136

    device_maximum = cuda.jit(device=True)(maximum)
    cases = (
        _collect(
            _scan.block_scan_array,
            compile_context,
            dtype=types.int32,
            threads_per_block=_BLOCK_THREADS,
            items_per_thread=2,
            value_kind="array",
            mode="inclusive",
            scan_op=device_maximum,
            algorithm="warp_scans",
        ),
        _collect(
            _scan.warp_scan,
            compile_context,
            dtype=types.int32,
            threads_in_warp=8,
            threads_per_block=_BLOCK_THREADS,
            mode="exclusive",
            scan_op=device_maximum,
            initial_value=ArgumentBinding.static(-(2**31)),
        ),
    )

    for index, collected in enumerate(cases):
        algorithm = collected[0]
        source, support_ltoirs, _, declarations = algorithm._source_code(
            threads=collected[1],
            block_threads=collected[2],
        )
        assert len(support_ltoirs) == 1
        assert len(declarations) == 1
        callback_name = next(iter(declarations))
        assert declarations[callback_name] in source
        assert f"return {callback_name}(wp_0, wp_1);" in source

        name = f"cuda_coop_numba_mlir_scan_callback_{index}"
        invocable = _types.make_invocable_from_specialization(
            algorithm,
            threads=collected[1],
            block_threads=collected[2],
        )
        assert len(invocable.files) == 2
        assert all(Path(path).suffix == ".ltoir" for path in invocable.files)
        assert ".version" in _link_ltoir_files(invocable.files, name=name)
        assert invocable.temp_storage_bytes > 0
        assert invocable.temp_storage_alignment > 0


def test_block_prefix_callbacks_compile_for_scalar_array_and_algorithms(
    compile_context: _nvrtc.CompileContext,
) -> None:
    def prefix_from_aggregate(block_aggregate):
        return block_aggregate + 7

    def carry_prefix(state, block_aggregate):
        previous = state[0]
        state[0] = previous + block_aggregate
        return previous

    device_prefix = cuda.jit(device=True)(prefix_from_aggregate)
    running = StatefulFunction(
        cuda.jit(device=True)(carry_prefix),
        types.int64,
        name="running_prefix",
    )
    running_int32 = StatefulFunction(
        _RunningPrefixFunctor,
        types.int32,
        name="running_prefix",
    )
    cases = (
        (
            _scan.block_scan_scalar,
            {
                "dtype": types.int32,
                "threads_per_block": _BLOCK_THREADS,
                "mode": "exclusive",
                "prefix_op": device_prefix,
                "algorithm": "raking",
            },
            ".ExclusiveSum(",
            False,
        ),
        (
            _scan.block_scan_array,
            {
                "dtype": types.int32,
                "threads_per_block": _BLOCK_THREADS,
                "items_per_thread": 2,
                "value_kind": "array",
                "mode": "exclusive",
                "scan_op": "max",
                "prefix_op": device_prefix,
                "algorithm": "raking_memoize",
            },
            ".ExclusiveScan(",
            False,
        ),
        (
            _scan.block_scan_scalar,
            {
                "dtype": types.int32,
                "threads_per_block": _BLOCK_THREADS,
                "mode": "inclusive",
                "prefix_op": running,
                "prefix_state": True,
                "algorithm": "warp_scans",
            },
            ".InclusiveSum(",
            True,
        ),
        (
            _scan.block_scan_array,
            {
                "dtype": types.int32,
                "threads_per_block": _BLOCK_THREADS,
                "items_per_thread": 2,
                "value_kind": "array",
                "mode": "inclusive",
                "prefix_op": running_int32,
                "prefix_state": True,
                "algorithm": "raking",
            },
            ".InclusiveSum(",
            True,
        ),
    )
    collected = [
        _collect(factory, compile_context, **kwargs)
        for factory, kwargs, _method, _stateful in cases
    ]

    callback_names = set()
    for item, (_factory, _kwargs, method, stateful) in zip(collected, cases):
        algorithm = item[0]
        source, support_ltoirs, _, declarations = algorithm._source_code(
            threads=item[1],
            block_threads=item[2],
        )
        assert method in source
        assert len(support_ltoirs) == 1
        assert len(declarations) == 1
        callback_name = next(iter(declarations))
        callback_names.add(callback_name)
        assert declarations[callback_name] in source
        assert f"return {callback_name}(" in source
        if stateful:
            assert "char *param_" in source
            assert re.search(
                rf"return {re.escape(callback_name)}\(param_\d+_state, wp_0\);",
                source,
            )
        else:
            assert f"return {callback_name}(wp_0);" in source

    assert len(callback_names) == 3
    bundle = _compile_bundle(
        collected,
        name="cuda_coop_numba_mlir_scan_prefix_callbacks",
    )
    assert all(item[0].temp_storage_bytes > 0 for item in collected)
    assert all(item[0].temp_storage_alignment > 0 for item in collected)
    assert ".version" in _types._ltoir_to_ptx(
        bundle,
        name="cuda_coop_numba_mlir_scan_prefix_callbacks",
        cc=_FIXED_CC,
    )


def test_production_kernel_compile_consumes_prefix_descriptors() -> None:
    import cuda.coop.numba_mlir as coop

    @cuda.jit(device=True)
    def stateless_prefix(block_aggregate):
        return block_aggregate + 7

    @cuda.jit(device=True)
    def running_prefix(state, block_aggregate):
        previous = state[0]
        state[0] = previous + block_aggregate
        return previous

    running = StatefulFunction(
        running_prefix,
        types.int64,
        name="compile_only_running_prefix",
    )
    running_int32 = StatefulFunction(
        running_prefix,
        types.int32,
        name="compile_only_running_prefix",
    )

    @cuda.jit(chip="sm_90")
    def kernel(source, destination, final_state):
        thread = cuda.threadIdx.x
        state = coop.ThreadData(1, dtype=types.int64)
        state[0] = 11
        destination[thread] = coop.exclusive_sum(
            coop.this_block(),
            source[thread],
            state,
            prefix_op=running,
            algorithm=coop.BlockScanAlgorithm.RAKING,
        )
        state_int32 = cuda.local.array(1, dtype=types.int32)
        state_int32[0] = 5
        destination[_BLOCK_THREADS + thread] = coop.exclusive_sum(
            coop.this_block(),
            source[thread],
            state_int32,
            prefix_op=running_int32,
            algorithm=coop.BlockScanAlgorithm.RAKING_MEMOIZE,
        )
        values = cuda.local.array(2, dtype=types.int32)
        values[0] = source[thread * 2]
        values[1] = source[thread * 2 + 1]
        scanned = coop.inclusive_sum(
            coop.this_block(),
            values,
            prefix_op=stateless_prefix,
            algorithm=coop.BlockScanAlgorithm.WARP_SCANS,
        )
        destination[2 * _BLOCK_THREADS + thread * 2] = scanned[0]
        destination[2 * _BLOCK_THREADS + thread * 2 + 1] = scanned[1]
        if thread == 0:
            final_state[0] = state[0]

    signature = types.void(
        types.int32[::1],
        types.int32[::1],
        types.int64[::1],
    )
    launch_config_key = (
        ("grid", (1, 1, 1)),
        ("block", (_BLOCK_THREADS, 1, 1)),
        ("sharedmem", 0),
        ("cluster", None),
    )
    result = kernel._compile_launch_config_signature(
        signature,
        launch_config_key,
    )

    assert isinstance(result.metadata["ltoir"], bytes)
    assert result.metadata["ltoir"]
    assert isinstance(result.metadata["cubin"], bytes)
    assert result.metadata["cubin"]
    assert result.metadata["linked_external_link_items"]
    ptx = next(iter(kernel.inspect_lto_ptx().values()))
    assert ".visible .entry" in ptx
    assert ".shared" in ptx
    assert "bar.sync" in ptx
