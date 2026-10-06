# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check independent per-slot reductions and distributed result ownership.

Host references arrange input as logical groups, lanes, and batch slots,
then reduce only the lane axis. Store results only where layout maps a slot
to an existing batch. Other cases cover built-in operators, a qualified
callback, chained result shapes, and one independently participating group.
"""

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
]


@pytest.mark.parametrize(
    "width,batches,layout,dtype",
    [
        (width, batches, layout, dtype)
        for width in (1, 8, 32)
        for batches in (1, 4, 33)
        for layout in ("striped", "blocked")
        for dtype in (np.int32, np.float64)
    ]
    + [
        (8, 3, "striped", dtype)
        for dtype in (
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.uint32,
            np.int64,
            np.uint64,
            np.float32,
        )
    ],
)
def test_batch_reduction_layouts_preserve_input(width, batches, layout, dtype):
    """Check each output layout against a reduction over the lane axis.

    Map every defined result back to its batch index before comparison. Batch
    counts above the width require several slots per lane; rounded capacity
    leaves holes that the kernel must not read. Retain input dtype in the host
    sum to match the provider's type rules, and check input preservation.
    """

    output_count = (batches + width - 1) // width

    @cuda.jit
    def kernel(source, output, preserved, items_per_thread):
        block = coop.this_block()
        warp = coop.this_warp().group_by(width)
        values = coop.ThreadData(items_per_thread)
        coop.load(block, source, values)
        result = coop.reduce_batched(warp, values, output_layout=layout)
        thread = cuda.threadIdx.x
        lane = thread % width
        base = (thread // width) * items_per_thread
        for i in range(output_count):
            if layout == "striped":
                batch = lane + i * width
            else:
                batch = lane * output_count + i
            if batch < items_per_thread:
                output[base + batch] = result[i]
        coop.store(block, preserved, values)

    source = ((np.arange(64 * batches) * 5) % 7).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        source = source / dtype(4) - dtype(0.625)
    elif np.issubdtype(dtype, np.signedinteger):
        source -= dtype(3)
    if dtype == np.float64:
        source += dtype(2**-30)
    elif dtype == np.int64:
        source *= dtype(1 << 33)
    elif dtype == np.uint64:
        source += dtype(1 << 33)
    d_source = cuda.to_device(source)
    output = cuda.device_array((64 // width) * batches, dtype=dtype)
    preserved = cuda.device_array_like(source)
    kernel[1, 64](d_source, output, preserved, batches)
    expected = source.reshape(64 // width, width, batches).sum(
        axis=1, dtype=dtype
    )
    np.testing.assert_array_equal(output.copy_to_host(), expected.ravel())
    np.testing.assert_array_equal(d_source.copy_to_host(), source)
    np.testing.assert_array_equal(preserved.copy_to_host(), source)


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("operator", ["min", "max", "bit_xor"])
def test_batch_builtin_operators(operator, items_per_thread):
    @cuda.jit
    def kernel(source, output, items_per_thread):
        values = coop.ThreadData(items_per_thread)
        coop.load(coop.this_block(), source, values)
        result = coop.reduce_batched(
            coop.this_warp(), values, binary_op=operator
        )
        if cuda.threadIdx.x < items_per_thread:
            output[cuda.threadIdx.x] = result[0]

    source = (np.arange(32 * items_per_thread) * 13 % 43 - 21).astype(np.int32)
    output = cuda.device_array(items_per_thread, dtype=np.int32)
    kernel[1, 32](cuda.to_device(source), output, items_per_thread)
    reducer = {"min": np.minimum, "max": np.maximum, "bit_xor": np.bitwise_xor}[
        operator
    ]
    np.testing.assert_array_equal(
        output.copy_to_host(),
        reducer.reduce(source.reshape(32, items_per_thread), axis=0),
    )


def test_custom_operator_and_chained_result_extent():
    """Use callback results as two new batches in a second reduction.

    Reduce 64 batches to two blocked output slots in each physical-warp lane.
    Every slot is defined, so a second call can safely sum those slots as two
    independent batches. The host first computes maxima, then regroups them
    by lane to check the new extent and meaning of the chained result.
    """

    @cuda.jit(device=True)
    def maximum(left, right):
        return max(right, left)

    @cuda.jit
    def kernel(source, output, items_per_thread):
        values = coop.ThreadData(items_per_thread)
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
    kernel[1, 32](cuda.to_device(source), output, 64)
    maxima = source.reshape(32, 64).max(axis=0)
    np.testing.assert_array_equal(
        output.copy_to_host(), maxima.reshape(32, 2).sum(axis=0)
    )


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_only_one_logical_warp_participates(items_per_thread):
    """Let one complete logical warp reduce while neighboring groups skip it.

    All eight selected lanes enter the call, but other lanes in the physical
    warp take no part. The result checks that CUB synchronization is limited
    to the selected group and does not require unrelated logical warps.
    """

    @cuda.jit
    def kernel(source, output, items_per_thread):
        values = coop.ThreadData(items_per_thread)
        coop.load(coop.this_block(), source, values)
        if cuda.threadIdx.x < 8:
            result = coop.reduce_batched(coop.this_warp().group_by(8), values)
            if cuda.threadIdx.x < items_per_thread:
                output[cuda.threadIdx.x] = result[0]

    source = np.arange(32 * items_per_thread, dtype=np.int32)
    output = cuda.device_array(items_per_thread, dtype=np.int32)
    kernel[1, 32](cuda.to_device(source), output, items_per_thread)
    np.testing.assert_array_equal(
        output.copy_to_host(),
        source[: 8 * items_per_thread].reshape(8, items_per_thread).sum(axis=0),
    )


def test_warp_feature_sums_example():
    """Check the feature-sums example independently for two physical warps."""

    # example-begin reduce-batched-features
    import numpy as np
    from numba_cuda_mlir import cuda

    from cuda import coop

    @cuda.jit
    def feature_sums(samples, totals, items_per_thread):
        warp = coop.this_warp()
        features = coop.ThreadData(items_per_thread)
        coop.load(warp, samples, features)
        sums = coop.reduce_batched(warp, features)
        # Lane 0 owns feature 0's sum, lane 1 feature 1's, and so on.
        if warp.rank() < items_per_thread:
            totals[
                (cuda.threadIdx.x // 32) * items_per_thread + warp.rank()
            ] = sums[0]

    for items_per_thread in (1, 4):
        samples = np.arange(64 * items_per_thread, dtype=np.float32)
        totals = cuda.device_array(2 * items_per_thread, dtype=np.float32)
        feature_sums[1, 64](cuda.to_device(samples), totals, items_per_thread)
        np.testing.assert_array_equal(
            totals.copy_to_host(),
            samples.reshape(2, 32, items_per_thread).sum(axis=1).ravel(),
        )
    # example-end reduce-batched-features
