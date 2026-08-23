# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

from numba_cuda_mlir import types

from cuda.coop.numba_mlir._compiler._rewrite import CoopSinglePhaseRewriteError
from cuda.coop.numba_mlir._lowering._reduce import sum as block_sum

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]

_THREADS_1D = 64
_THREADS_2D = (8, 4)


@cuda.jit
def _implicit_1d_block_sum(source, output):
    tid = cuda.threadIdx.x
    total = block_sum(source[tid])
    if tid == 0:
        output[0] = total


@cuda.jit
def _implicit_2d_block_sum(source, output):
    tid = cuda.threadIdx.x + cuda.blockDim.x * cuda.threadIdx.y
    total = block_sum(source[tid])
    if tid == 0:
        output[0] = total


@cuda.jit
def _dim_alias_block_sum(source, output):
    tid = cuda.threadIdx.x
    total = block_sum(source[tid], dtype=types.int32, dim=_THREADS_1D)
    if tid == 0:
        output[0] = total


@cuda.jit
def _mismatched_explicit_block_sum(source, output):
    tid = cuda.threadIdx.x
    total = block_sum(source[tid], dtype=types.int32, threads_per_block=32)
    if tid == 0:
        output[0] = total


@cuda.jit(device=True, inline=True)
def _implicit_device_block_sum(value):
    return block_sum(value)


@cuda.jit
def _device_helper_block_sum(source, output):
    tid = cuda.threadIdx.x
    total = _implicit_device_block_sum(source[tid])
    if tid == 0:
        output[0] = total


@cuda.jit(launch_bounds=128)
def _bounded_block_sum(source, output):
    tid = cuda.threadIdx.x
    total = block_sum(source[tid])
    if tid == 0:
        output[0] = total


def _check_sum(kernel, block):
    threads = int(np.prod(block)) if isinstance(block, tuple) else block
    source = np.arange(1, threads + 1, dtype=np.int32)
    output = np.full(1, -1, dtype=np.int32)

    kernel[1, block](source, output)

    assert output[0] == source.sum(dtype=np.int32)


def test_block_dimension_is_inferred_from_1d_launch():
    _check_sum(_implicit_1d_block_sum, _THREADS_1D)


def test_block_dimension_preserves_a_2d_launch_shape():
    _check_sum(_implicit_2d_block_sum, _THREADS_2D)


def test_dim_alias_selects_the_block_dimension():
    _check_sum(_dim_alias_block_sum, _THREADS_1D)


def test_explicit_dimension_rejects_an_exact_launch_mismatch():
    source = np.arange(1, _THREADS_1D + 1, dtype=np.int32)
    output = np.full(1, -1, dtype=np.int32)

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=(
            r"threads_per_block=32, but the exact kernel launch block is "
            r"\(64, 1, 1\)"
        ),
    ):
        _mismatched_explicit_block_sum[1, _THREADS_1D](source, output)


def test_device_function_rewrite_is_deferred_to_its_kernel_caller():
    _check_sum(_device_helper_block_sum, _THREADS_1D)


def test_launch_bounds_do_not_replace_the_exact_launch_dimension():
    _check_sum(_bounded_block_sum, _THREADS_1D)
