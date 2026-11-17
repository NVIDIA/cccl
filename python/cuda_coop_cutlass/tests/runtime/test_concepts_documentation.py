# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run the kernels shown for the core concepts in the documentation.

Each marked region is a snippet that the Programming concepts page or an API
section includes. The tests launch every kernel and check its output on the
host with NumPy.
"""

import cutlass
import numpy as np
import pytest
from cutlass import cute

import cuda.coop.cutlass as coop
from tests.runtime_support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


# TempStorage. Each kernel is the prefix sum from examples/prefix_sum.py.
# Load and Store ask for the transpose algorithm because their default
# algorithm uses no scratch. The kernels are defined at module level so the
# marked lines keep that example's names and need no dedent.


# cutlass-concept-temp-storage-begin
@cute.kernel
def prefix_sum_shared(
    source: cute.Pointer,
    destination: cute.Pointer,
    items_per_thread: cutlass.Constexpr,
):
    block = coop.this_block()
    scratch = coop.TempStorage(auto_sync=True)
    values = coop.load(
        block,
        source,
        items_per_thread=items_per_thread,
        algorithm="transpose",
        temp_storage=scratch,
    )
    sums = coop.inclusive_sum(block, values, temp_storage=scratch)
    coop.store(
        block, destination, sums, algorithm="transpose", temp_storage=scratch
    )


# cutlass-concept-temp-storage-end


@cute.kernel
def prefix_sum_exclusive(
    source: cute.Pointer,
    destination: cute.Pointer,
    items_per_thread: cutlass.Constexpr,
):
    block = coop.this_block()
    scratch = coop.TempStorage(sharing="exclusive", auto_sync=True)
    values = coop.load(
        block,
        source,
        items_per_thread=items_per_thread,
        algorithm="transpose",
        temp_storage=scratch,
    )
    sums = coop.inclusive_sum(block, values, temp_storage=scratch)
    coop.store(
        block, destination, sums, algorithm="transpose", temp_storage=scratch
    )


@cute.kernel
def prefix_sum_omitted(
    source: cute.Pointer,
    destination: cute.Pointer,
    items_per_thread: cutlass.Constexpr,
):
    block = coop.this_block()
    values = coop.load(
        block, source, items_per_thread=items_per_thread, algorithm="transpose"
    )
    sums = coop.inclusive_sum(block, values)
    coop.store(block, destination, sums, algorithm="transpose")


@cute.kernel
def prefix_sum_manual_sync(
    source: cute.Pointer,
    destination: cute.Pointer,
    items_per_thread: cutlass.Constexpr,
):
    block = coop.this_block()
    # cutlass-concept-temp-storage-manual-sync-begin
    scratch = coop.TempStorage()
    values = coop.load(
        block,
        source,
        items_per_thread=items_per_thread,
        algorithm="transpose",
        temp_storage=scratch,
    )
    # Scan reuses the bytes Load used. Wait until every thread finishes Load.
    scratch.sync()
    sums = coop.inclusive_sum(block, values, temp_storage=scratch)
    # Store reuses the same bytes. Wait until every thread finishes Scan.
    scratch.sync()
    coop.store(
        block, destination, sums, algorithm="transpose", temp_storage=scratch
    )
    # Store is the last call that uses the scratch, so no barrier follows.
    # cutlass-concept-temp-storage-manual-sync-end


# cutlass-concept-temp-storage-release-begin
@cute.kernel
def sum_then_mirror(source: cute.Pointer, destination: cute.Pointer):
    block = coop.this_block()
    rank = block.rank()
    # Ask for 512 bytes: room for one Int32 per thread once the sum is done.
    scratch = coop.TempStorage(512, auto_sync=True)
    values = coop.load(block, source, items_per_thread=1)
    total = coop.sum(block, values, temp_storage=scratch)

    # The sum is finished. Take the region back as ordinary shared memory.
    shared = scratch.release(cutlass.Int32)
    shared[rank] = values[0]
    block.sync()
    output = cute.make_tensor(destination, cute.make_layout(129))
    output[rank] = shared[127 - rank]
    if rank == 0:
        output[128] = total


# cutlass-concept-temp-storage-release-end


def _check_prefix_sum(kernel):
    """Launch ``kernel`` on one block of 128 threads and check its sums.

    Each thread owns four Int32 items. The input mixes signs and the output
    starts from a sentinel, so a dropped or misplaced item changes the result.
    """

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, destination, items_per_thread).launch(grid=1, block=128)

    values = (np.arange(128 * 4, dtype=np.int32) * 13) % 31 - 15
    observed = np.full_like(values, -101)
    with device_array(values) as src, device_array(observed) as dst:
        launch(src, dst, 4)
    np.testing.assert_array_equal(observed, values.cumsum(dtype=np.int32))


@pytest.mark.parametrize(
    "kernel",
    (prefix_sum_shared, prefix_sum_exclusive, prefix_sum_omitted),
    ids=("shared", "exclusive", "omitted"),
)
def test_temp_storage_concept(kernel):
    """Check the prefix sum with each way of giving the calls scratch."""

    _check_prefix_sum(kernel)


def test_temp_storage_manual_sync_concept():
    """Check the prefix sum when the kernel writes its own barriers."""

    _check_prefix_sum(prefix_sum_manual_sync)


def test_temp_storage_release_concept():
    """Check the block sum and the values mirrored through released scratch."""

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        sum_then_mirror(source, destination).launch(grid=1, block=128)

    values = (np.arange(128, dtype=np.int32) * 13) % 31 - 15
    observed = np.full(129, -101, dtype=np.int32)
    with device_array(values) as src, device_array(observed) as dst:
        launch(src, dst)
    np.testing.assert_array_equal(observed[:128], values[::-1])
    assert observed[128] == values.sum()
