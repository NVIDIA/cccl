# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

import cuda.coop.numba_mlir as coop

_PHYSICAL_WARP_THREADS = 32
_MODULE_WARP = coop.this_warp()


@cuda.jit(device=True)
def _linear_thread_rank():
    return cuda.threadIdx.x + cuda.blockDim.x * (
        cuda.threadIdx.y + cuda.blockDim.y * cuda.threadIdx.z
    )


@cuda.jit
def _warp_sum(source, output):
    thread = _linear_thread_rank()
    result = coop.sum(coop.this_warp(), source[thread])
    if thread % _PHYSICAL_WARP_THREADS == 0:
        output[thread // _PHYSICAL_WARP_THREADS] = result


@pytest.mark.parametrize("block_dim", ((64,), (8, 8), (8, 4, 2), (4, 4, 4)))
def test_physical_warp_sum_uses_x_major_multidimensional_rank(block_dim):
    source = np.arange(64, dtype=np.int32)
    output = np.zeros(2, dtype=np.int32)

    _warp_sum[1, block_dim](source, output)

    np.testing.assert_array_equal(
        output,
        np.array(
            [
                source[:32].sum(dtype=np.int32),
                source[32:].sum(dtype=np.int32),
            ],
            dtype=np.int32,
        ),
    )


@cuda.jit
def _warp_max(source, output):
    thread = _linear_thread_rank()
    result = coop.reduce(
        coop.this_warp(),
        source[thread],
        binary_op="max",
    )
    if thread % _PHYSICAL_WARP_THREADS == 0:
        output[thread // _PHYSICAL_WARP_THREADS] = result


def test_physical_warp_reduce_supports_non_sum_operator():
    source = np.arange(64, dtype=np.int32) - 37
    output = np.zeros(2, dtype=np.int32)

    _warp_max[1, (8, 4, 2)](source, output)

    np.testing.assert_array_equal(
        output,
        np.array([source[:32].max(), source[32:].max()], dtype=np.int32),
    )


@cuda.jit
def _warp_valid_prefix(source, output, valid_items):
    thread = _linear_thread_rank()
    result = coop.sum(
        coop.this_warp(),
        source[thread],
        valid_items=valid_items,
    )
    if thread % _PHYSICAL_WARP_THREADS == 0:
        output[thread // _PHYSICAL_WARP_THREADS] = result


@pytest.mark.parametrize("valid_items", (1, 17, 32))
def test_runtime_valid_prefix_is_applied_per_physical_warp(valid_items):
    source = np.arange(64, dtype=np.uint64)
    output = np.zeros(2, dtype=np.uint64)

    _warp_valid_prefix[1, (4, 4, 4)](
        source,
        output,
        np.int64(valid_items),
    )

    np.testing.assert_array_equal(
        output,
        np.array(
            [
                source[:valid_items].sum(dtype=np.uint64),
                source[32 : 32 + valid_items].sum(dtype=np.uint64),
            ],
            dtype=np.uint64,
        ),
    )


@cuda.jit
def _warp_max_valid_prefix(source, output, valid_items):
    thread = _linear_thread_rank()
    result = coop.reduce(
        coop.this_warp(),
        source[thread],
        binary_op="max",
        valid_items=valid_items,
    )
    if thread % _PHYSICAL_WARP_THREADS == 0:
        output[thread // _PHYSICAL_WARP_THREADS] = result


def test_non_sum_runtime_valid_prefix_uses_cub_reduce_overload():
    source = np.arange(64, dtype=np.int32) - 19
    output = np.zeros(2, dtype=np.int32)
    valid_items = 17

    _warp_max_valid_prefix[1, (8, 8)](
        source,
        output,
        np.int32(valid_items),
    )

    np.testing.assert_array_equal(
        output,
        np.array(
            [
                source[:valid_items].max(),
                source[32 : 32 + valid_items].max(),
            ],
            dtype=np.int32,
        ),
    )


@cuda.jit
def _warp_static_valid_prefix(source, output):
    thread = _linear_thread_rank()
    result = coop.sum(coop.this_warp(), source[thread], valid_items=17)
    if thread % _PHYSICAL_WARP_THREADS == 0:
        output[thread // _PHYSICAL_WARP_THREADS] = result


def test_static_valid_prefix_is_lowered_per_physical_warp():
    source = np.arange(64, dtype=np.int32)
    output = np.zeros(2, dtype=np.int32)

    _warp_static_valid_prefix[1, (4, 4, 4)](source, output)

    np.testing.assert_array_equal(
        output,
        np.array(
            [
                source[:17].sum(dtype=np.int32),
                source[32:49].sum(dtype=np.int32),
            ],
            dtype=np.int32,
        ),
    )


@cuda.jit
def _warp_sum_with_module_descriptor(source, output):
    thread = _linear_thread_rank()
    result = coop.sum(_MODULE_WARP, source[thread])
    if thread % _PHYSICAL_WARP_THREADS == 0:
        output[thread // _PHYSICAL_WARP_THREADS] = result


def test_module_level_warp_descriptor_is_a_compile_time_constant():
    source = np.arange(64, dtype=np.int32)
    output = np.zeros(2, dtype=np.int32)

    _warp_sum_with_module_descriptor[1, (8, 8)](source, output)

    np.testing.assert_array_equal(
        output,
        np.array(
            [
                source[:32].sum(dtype=np.int32),
                source[32:].sum(dtype=np.int32),
            ],
            dtype=np.int32,
        ),
    )


@cuda.jit
def _back_to_back_warp_reductions(source, output):
    thread = _linear_thread_rank()
    first = coop.sum(coop.this_warp(), source[thread])
    second = coop.sum(coop.this_warp(), source[thread] + 1)
    if thread % _PHYSICAL_WARP_THREADS == 0:
        warp_id = thread // _PHYSICAL_WARP_THREADS
        output[2 * warp_id] = first
        output[2 * warp_id + 1] = second


def test_back_to_back_warp_reductions_reuse_storage_safely():
    source = np.arange(64, dtype=np.int32)
    output = np.zeros(4, dtype=np.int32)

    _back_to_back_warp_reductions[1, (8, 4, 2)](source, output)

    np.testing.assert_array_equal(
        output,
        np.array(
            [
                source[:32].sum(dtype=np.int32),
                source[:32].sum(dtype=np.int32) + 32,
                source[32:].sum(dtype=np.int32),
                source[32:].sum(dtype=np.int32) + 32,
            ],
            dtype=np.int32,
        ),
    )


@cuda.jit
def _looped_warp_reduction(source, output):
    thread = _linear_thread_rank()
    value = source[thread]
    for iteration in range(3):
        result = coop.sum(coop.this_warp(), value + iteration)
        if thread % _PHYSICAL_WARP_THREADS == 0:
            warp_id = thread // _PHYSICAL_WARP_THREADS
            output[3 * warp_id + iteration] = result


def test_looped_warp_reduction_reuses_storage_safely():
    source = np.arange(64, dtype=np.int32)
    output = np.zeros(6, dtype=np.int32)

    _looped_warp_reduction[1, (8, 4, 2)](source, output)

    first = source[:32].sum(dtype=np.int32)
    second = source[32:].sum(dtype=np.int32)
    np.testing.assert_array_equal(
        output,
        np.array(
            [first, first + 32, first + 64, second, second + 32, second + 64],
            dtype=np.int32,
        ),
    )
