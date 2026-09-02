# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

from cuda import coop as root_coop

_GLOBAL_SCALAR = np.int32(3)


@cuda.jit(device=True)
def _inlined_root_sum(value, valid_items=None):
    return root_coop.sum(
        root_coop.this_block(),
        value,
        valid_items=valid_items,
    )


@cuda.jit
def _root_sum_through_device_function(source, output):
    thread = cuda.threadIdx.x
    result = _inlined_root_sum(source[thread])
    if thread == 0:
        output[0] = result


def test_root_calls_compile_after_device_function_inlining():
    source = np.arange(64, dtype=np.int32)
    output = np.zeros(1, dtype=np.int32)

    _root_sum_through_device_function[1, 64](source, output)

    assert output[0] == source.sum(dtype=np.int32)


@cuda.jit(device=True)
def _inlined_root_warp_sum(value, valid_items=None):
    return root_coop.sum(
        root_coop.this_warp(),
        value,
        valid_items=valid_items,
    )


@cuda.jit
def _root_warp_sum_through_device_function(source, output):
    thread = cuda.threadIdx.x + cuda.blockDim.x * (
        cuda.threadIdx.y + cuda.blockDim.y * cuda.threadIdx.z
    )
    result = _inlined_root_warp_sum(source[thread])
    if thread % 32 == 0:
        output[thread // 32] = result


def test_root_warp_calls_compile_after_device_function_inlining():
    source = np.arange(64, dtype=np.int32)
    output = np.zeros(2, dtype=np.int32)

    _root_warp_sum_through_device_function[1, (8, 4, 2)](source, output)

    np.testing.assert_array_equal(
        output,
        np.array(
            [source[:32].sum(dtype=np.int32), source[32:].sum(dtype=np.int32)],
            dtype=np.int32,
        ),
    )


@cuda.jit
def _root_sum_payload_origins(source, output):
    thread = cuda.threadIdx.x
    global_thread = cuda.grid(1)
    thread_result = root_coop.sum(root_coop.this_block(), thread)
    grid_result = root_coop.sum(root_coop.this_block(), global_thread)
    global_result = root_coop.sum(root_coop.this_block(), _GLOBAL_SCALAR)
    promoted_result = root_coop.sum(
        root_coop.this_block(),
        source[thread] + 1,
    )
    if thread == 0:
        output[0] = thread_result
        output[1] = grid_result
        output[2] = global_result
        output[3] = promoted_result


def test_root_calls_compile_with_types_resolved_by_overload_typing():
    source = np.arange(32, dtype=np.int32)
    output = np.zeros(4, dtype=np.int64)

    _root_sum_payload_origins[1, 32](source, output)

    np.testing.assert_array_equal(
        output,
        np.array(
            [
                sum(range(32)),
                sum(range(32)),
                32 * int(_GLOBAL_SCALAR),
                source.sum(dtype=np.int64) + 32,
            ],
            dtype=np.int64,
        ),
    )


@cuda.jit
def _root_max(source, output):
    thread = cuda.threadIdx.x
    result = root_coop.reduce(
        root_coop.this_block(),
        source[thread],
        binary_op="max",
    )
    if thread == 0:
        output[0] = result


def test_root_reduce_compiles_with_a_non_sum_operator():
    source = np.arange(32, dtype=np.int32) - 11
    output = np.zeros(1, dtype=np.int32)

    _root_max[1, 32](source, output)

    assert output[0] == source.max()
