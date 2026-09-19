# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Independent numerical oracles for batched warp reductions."""

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("width", [1, 8, 32])
@pytest.mark.parametrize("batches", [3, 33])
@pytest.mark.parametrize("layout", ["striped", "blocked"])
@pytest.mark.parametrize("dtype", [np.int32, np.float64])
def test_batch_reduction_layouts_preserve_input(width, batches, layout, dtype):
    output_count = (batches + width - 1) // width

    @cuda.jit
    def kernel(source, output, preserved):
        block = coop.this_block()
        warp = coop.this_warp().group_by(width)
        values = coop.ThreadData(batches)
        coop.load(block, source, values)
        result = coop.reduce_batched(warp, values, output_layout=layout)
        thread = cuda.threadIdx.x
        lane = thread % width
        base = (thread // width) * batches
        for i in range(output_count):
            if layout == "striped":
                batch = lane + i * width
            else:
                batch = lane * output_count + i
            if batch < batches:
                output[base + batch] = result[i]
        coop.store(block, preserved, values)

    source = ((np.arange(64 * batches) * 7) % 29 - 14).astype(dtype)
    d_source = cuda.to_device(source)
    output = cuda.device_array((64 // width) * batches, dtype=dtype)
    preserved = cuda.device_array_like(source)
    kernel[1, 64](d_source, output, preserved)
    expected = source.reshape(64 // width, width, batches).sum(axis=1, dtype=dtype)
    np.testing.assert_array_equal(output.copy_to_host(), expected.ravel())
    np.testing.assert_array_equal(d_source.copy_to_host(), source)
    np.testing.assert_array_equal(preserved.copy_to_host(), source)


@pytest.mark.parametrize("operator", ["min", "max", "bit_xor"])
def test_batch_builtin_operators(operator):
    @cuda.jit
    def kernel(source, output):
        values = coop.ThreadData(3)
        coop.load(coop.this_block(), source, values)
        result = coop.reduce_batched(coop.this_warp(), values, binary_op=operator)
        if cuda.threadIdx.x < 3:
            output[cuda.threadIdx.x] = result[0]

    source = (np.arange(96) * 13 % 43 - 21).astype(np.int32)
    output = cuda.device_array(3, dtype=np.int32)
    kernel[1, 32](cuda.to_device(source), output)
    reducer = {"min": np.minimum, "max": np.maximum, "bit_xor": np.bitwise_xor}[
        operator
    ]
    np.testing.assert_array_equal(
        output.copy_to_host(), reducer.reduce(source.reshape(32, 3), axis=0)
    )


def test_custom_operator_and_chained_result_extent():
    @cuda.jit(device=True)
    def maximum(left, right):
        return left if left > right else right

    @cuda.jit
    def kernel(source, output):
        values = coop.ThreadData(64)
        coop.load(coop.this_block(), source, values)
        result = numba_coop.reduce_batched(
            coop.this_warp(), values, binary_op=maximum, output_layout="blocked"
        )
        # The result has two slots, each a distinct batch for the next call.
        combined = coop.reduce_batched(coop.this_warp(), result)
        if cuda.threadIdx.x < 2:
            output[cuda.threadIdx.x] = combined[0]

    source = (np.arange(32 * 64) * 7 % 37).astype(np.int32)
    output = cuda.device_array(2, dtype=np.int32)
    kernel[1, 32](cuda.to_device(source), output)
    maxima = source.reshape(32, 64).max(axis=0)
    np.testing.assert_array_equal(
        output.copy_to_host(), maxima.reshape(32, 2).sum(axis=0)
    )


def test_only_one_logical_warp_participates():
    @cuda.jit
    def kernel(source, output):
        values = coop.ThreadData(3)
        coop.load(coop.this_block(), source, values)
        if cuda.threadIdx.x < 8:
            result = coop.reduce_batched(coop.this_warp().group_by(8), values)
            if cuda.threadIdx.x < 3:
                output[cuda.threadIdx.x] = result[0]

    source = np.arange(96, dtype=np.int32)
    output = cuda.device_array(3, dtype=np.int32)
    kernel[1, 32](cuda.to_device(source), output)
    np.testing.assert_array_equal(
        output.copy_to_host(), source[:24].reshape(8, 3).sum(axis=0)
    )


def test_warp_feature_sums_example():
    # example-begin reduce-batched-features
    @cuda.jit
    def feature_sums(samples, totals):
        warp = coop.this_warp()
        features = coop.ThreadData(3)
        coop.load(warp, samples, features)
        sums = coop.reduce_batched(warp, features)
        # Lane 0 owns feature 0's sum, lane 1 feature 1's, and so on.
        if warp.rank() < 3:
            totals[(cuda.threadIdx.x // 32) * 3 + warp.rank()] = sums[0]

    # example-end reduce-batched-features

    samples = np.arange(64 * 3, dtype=np.float32)
    totals = cuda.device_array(6, dtype=np.float32)
    feature_sums[1, 64](cuda.to_device(samples), totals)
    np.testing.assert_array_equal(
        totals.copy_to_host(), samples.reshape(2, 32, 3).sum(axis=1).ravel()
    )
