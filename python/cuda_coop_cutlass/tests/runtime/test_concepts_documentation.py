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


# TempStorage. The prefix_sum_* kernels are the prefix sum from
# examples/prefix_sum.py. Load and Store ask for the transpose algorithm
# because their default algorithm uses no scratch. Kernels that the
# documentation includes whole are defined at module level, so those snippets
# keep that example's names and need no dedent.


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
def prefix_sum_exclusive_no_sync(
    source: cute.Pointer,
    destination: cute.Pointer,
    items_per_thread: cutlass.Constexpr,
):
    block = coop.this_block()
    # Each call has its own slice, so no call reuses another call's bytes
    # and the kernel needs no barrier between them.
    scratch = coop.TempStorage(sharing="exclusive")
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


# Thread groups. The same call works on a block, on each warp, and on each
# group of eight lanes. The block has 128 threads, so it holds four warps and
# sixteen groups of eight.


# cutlass-concept-groups-begin
@cute.kernel
def group_sums(source: cute.Pointer, destination: cute.Pointer):
    block = coop.this_block()
    warp = coop.this_warp()
    lanes = warp.group_by(8)
    values = coop.load(block, source, items_per_thread=1)

    block_sum = coop.sum(block, values)  # one sum for all 128 threads
    warp_sum = coop.sum(warp, values)  # one sum for each warp of 32
    lane_sum = coop.sum(lanes, values)  # one sum for each group of 8

    # The thread with rank 0 in a group holds that group's sum.
    thread = block.rank()
    output = cute.make_tensor(destination, cute.make_layout(21))
    if block.rank() == 0:
        output[0] = block_sum
    if warp.rank() == 0:
        output[1 + thread // 32] = warp_sum
    if lanes.rank() == 0:
        output[5 + thread // 8] = lane_sum
    # cutlass-concept-groups-end


@cute.kernel
def group_queries(destination: cute.Pointer):
    # cutlass-concept-group-queries-begin
    block = coop.this_block()
    warp = coop.this_warp()
    lanes = warp.group_by(8)

    thread = block.rank()  # 0 to 127: this thread in its block
    threads = block.count()  # 128: threads in the block
    warp_index = block.rank("warp")  # 0 to 3: this thread's warp in the block
    warps = block.count("warp")  # 4: warps in the block
    lane = warp.rank()  # 0 to 31: this thread in its warp
    slot = lanes.rank()  # 0 to 7: this thread in its group of eight
    lanes_index = lanes.rank("warp")  # 0 to 3: this group in its warp
    block_index = block.rank("grid")  # 0 or 1: this block in a grid of two
    # cutlass-concept-group-queries-end
    index = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
    output = cute.make_tensor(destination, cute.make_layout(8 * 256))
    output[0 * 256 + index] = thread
    output[1 * 256 + index] = threads
    output[2 * 256 + index] = warp_index
    output[3 * 256 + index] = warps
    output[4 * 256 + index] = lane
    output[5 * 256 + index] = slot
    output[6 * 256 + index] = lanes_index
    output[7 * 256 + index] = block_index


@cute.kernel
def partial_group_sums(source: cute.Pointer, destination: cute.Pointer):
    block = coop.this_block()
    thread = block.rank()
    values = coop.load(block, source, items_per_thread=1)
    output = cute.make_tensor(destination, cute.make_layout(4))
    # cutlass-concept-group-by-partial-begin
    # One group of 24 lanes in each warp. Lanes 24 to 31 are in no group.
    lanes = coop.this_warp().group_by(24, exhaustive=False)
    if lanes.is_member():
        total = coop.sum(lanes, values)
        if lanes.rank() == 0:
            output[thread // 32] = total
    # cutlass-concept-group-by-partial-end


# ThreadData. Load returns one, a kernel can build another from computed
# values, and a primitive reads either.


# cutlass-concept-thread-data-begin
@cute.kernel
def sum_of_squares(
    source: cute.Pointer,
    destination: cute.Pointer,
    items_per_thread: cutlass.Constexpr,
):
    block = coop.this_block()
    values = coop.load(block, source, items_per_thread=items_per_thread)

    squares = coop.ThreadData(items_per_thread)
    for i in cutlass.range_constexpr(items_per_thread):
        squares[i] = values[i] * values[i]

    total = coop.sum(block, squares)
    if block.rank() == 0:
        output = cute.make_tensor(destination, cute.make_layout(1))
        output[0] = total
    # cutlass-concept-thread-data-end


@cute.kernel
def sum_of_squares_from_fn(
    source: cute.Pointer,
    destination: cute.Pointer,
    items_per_thread: cutlass.Constexpr,
):
    block = coop.this_block()
    values = coop.load(block, source, items_per_thread=items_per_thread)
    # cutlass-concept-thread-data-from-fn-begin
    squares = coop.ThreadData.from_fn(
        items_per_thread, lambda i: values[i] * values[i]
    )
    # cutlass-concept-thread-data-from-fn-end
    total = coop.sum(block, squares)
    if block.rank() == 0:
        output = cute.make_tensor(destination, cute.make_layout(1))
        output[0] = total


# More than one block. Each block is a group of its own and works on its
# own tile, so the kernel gives Load and Store a per-block offset.


# cutlass-concept-blocks-begin
@cute.kernel
def prefix_sum_per_block(
    source: cute.Pointer,
    destination: cute.Pointer,
    items_per_thread: cutlass.Constexpr,
):
    block = coop.this_block()
    # A block of 128 threads owns 128 * items_per_thread elements.
    offset = cute.arch.block_idx()[0] * 128 * items_per_thread
    values = coop.load(
        block, source, items_per_thread=items_per_thread, offset=offset
    )
    sums = coop.inclusive_sum(block, values)
    coop.store(block, destination, sums, offset=offset)
    # cutlass-concept-blocks-end


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
    (
        prefix_sum_shared,
        prefix_sum_exclusive,
        prefix_sum_exclusive_no_sync,
        prefix_sum_omitted,
    ),
    ids=("shared", "exclusive", "exclusive-no-sync", "omitted"),
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


def test_groups_concept():
    """Check one sum per block, per warp and per group of eight lanes."""

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        group_sums(source, destination).launch(grid=1, block=128)

    values = (np.arange(128, dtype=np.int32) * 13) % 31 - 15
    observed = np.full(21, -101, dtype=np.int32)
    with device_array(values) as src, device_array(observed) as dst:
        launch(src, dst)
    expected = np.concatenate(
        (
            [values.sum()],
            values.reshape(4, 32).sum(axis=1),
            values.reshape(16, 8).sum(axis=1),
        )
    )
    np.testing.assert_array_equal(observed, expected)


def test_group_queries_concept():
    """Check each query in two blocks against the host's arithmetic."""

    @cute.jit
    def launch(destination: cute.Pointer):
        group_queries(destination).launch(grid=2, block=128)

    observed = np.full(8 * 256, -1, dtype=np.int64)
    with device_array(observed) as dst:
        launch(dst)
    thread = np.tile(np.arange(128), 2)
    expected = np.stack(
        (
            thread,
            np.full(256, 128),
            thread // 32,
            np.full(256, 4),
            thread % 32,
            thread % 8,
            (thread % 32) // 8,
            np.arange(256) // 128,
        )
    )
    np.testing.assert_array_equal(observed.reshape(8, 256), expected)


def test_group_by_partial_concept():
    """Check that each warp sums its first 24 lanes and skips the rest."""

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        partial_group_sums(source, destination).launch(grid=1, block=128)

    values = (np.arange(128, dtype=np.int32) * 13) % 31 - 15
    observed = np.full(4, -101, dtype=np.int32)
    with device_array(values) as src, device_array(observed) as dst:
        launch(src, dst)
    np.testing.assert_array_equal(
        observed, values.reshape(4, 32)[:, :24].sum(axis=1)
    )


@pytest.mark.parametrize(
    "kernel",
    (sum_of_squares, sum_of_squares_from_fn),
    ids=("loop", "from-fn"),
)
def test_thread_data_concept(kernel):
    """Check a sum over a payload the kernel filled from loaded values."""

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, destination, items_per_thread).launch(grid=1, block=128)

    values = (np.arange(128 * 4, dtype=np.int32) * 13) % 31 - 15
    observed = np.full(1, -101, dtype=np.int32)
    with device_array(values) as src, device_array(observed) as dst:
        launch(src, dst, 4)
    assert observed[0] == (values * values).sum(dtype=np.int32)


def test_blocks_concept():
    """Check that each of three blocks scans its own tile and no other."""

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel = prefix_sum_per_block(source, destination, items_per_thread)
        kernel.launch(grid=3, block=128)

    values = (np.arange(3 * 128 * 4, dtype=np.int32) * 13) % 31 - 15
    observed = np.full_like(values, -101)
    with device_array(values) as src, device_array(observed) as dst:
        launch(src, dst, 4)
    expected = values.reshape(3, 512).cumsum(axis=1, dtype=np.int32)
    np.testing.assert_array_equal(observed, expected.reshape(-1))
