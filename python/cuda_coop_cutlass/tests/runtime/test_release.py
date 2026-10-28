# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Use released scratch as native shared memory without another allocation."""

import cutlass
import numpy as np
import pytest
from cutlass import cute

import cuda.coop.cutlass as coop
import cuda.coop.cutlass as cutlass_coop
from tests.runtime_support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("api", (cutlass_coop,))
@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("capacity", (None, 1024))
@pytest.mark.parametrize("dtype", (None, np.int32, cutlass.Float64))
def test_released_view_covers_existing_allocation(
    api, sharing, capacity, dtype
):
    @cute.kernel
    def kernel(destination: cute.Pointer):
        thread, _, _ = cute.arch.thread_idx()
        storage = api.TempStorage(capacity, sharing=sharing)
        first = storage.reserve(65, cutlass.Int32)
        second = storage.reserve(64, cutlass.Float64, alignment=128)
        released = storage.release(dtype)
        released[thread] = released.element_type(3 * thread + 7)
        cute.arch.sync_threads()
        output = cute.make_tensor(destination, cute.make_layout(69))
        output[thread] = cutlass.Int64(released[63 - thread])
        if thread == 0:
            output[64] = cutlass.Int64(cute.size(released))
            output[65] = cutlass.Int64(released.iterator.toint())
            output[66] = cutlass.Int64(first.iterator.toint())
            output[67] = cutlass.Int64(second.iterator.toint())
            output[68] = cutlass.Int64(cute.arch.dynamic_smem_size())

    @cute.jit
    def launch(destination: cute.Pointer):
        kernel(destination).launch(grid=1, block=64)

    result = np.zeros(69, dtype=np.int64)
    with device_array(result) as pointer:
        launch(pointer)
    size = capacity or (512 if sharing == "shared" else 896)
    itemsize = 1 if dtype is None else 4 if dtype is np.int32 else 8
    np.testing.assert_array_equal(result[:64], 3 * np.arange(63, -1, -1) + 7)
    assert result[64] == size // itemsize
    assert result[65] == result[66]
    assert result[65] % 128 == 0
    assert result[67] - result[65] == (0 if sharing == "shared" else 384)
    assert result[68] == size


@pytest.mark.parametrize("items_per_thread", (1, 4))
@pytest.mark.parametrize("auto_sync", (False, True))
def test_release_after_primitive_and_fresh_loop_instance(
    items_per_thread, auto_sync
):
    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer):
        thread, _, _ = cute.arch.thread_idx()
        output = cute.make_tensor(destination, cute.make_layout(128))
        for iteration in range(2):
            storage = coop.TempStorage(auto_sync=auto_sync)
            values = coop.load(
                coop.this_block(),
                source,
                items_per_thread=items_per_thread,
                algorithm="transpose",
                temp_storage=storage,
            )
            if cutlass.const_expr(not auto_sync):
                cute.arch.sync_threads()
            released = storage.release(cutlass.Int32)
            released[thread] = cutlass.Int32(values[0] + iteration)
            cute.arch.sync_threads()
            output[64 * iteration + thread] = released[63 - thread]
            cute.arch.sync_threads()

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        kernel(source, destination).launch(grid=1, block=64)

    source = np.arange(64 * items_per_thread, dtype=np.int32)
    output = np.zeros(128, dtype=np.int32)
    with (
        device_array(source) as source_pointer,
        device_array(output) as destination,
    ):
        launch(source_pointer, destination)
    expected = np.arange(63, -1, -1) * items_per_thread
    np.testing.assert_array_equal(output[:64], expected)
    np.testing.assert_array_equal(output[64:], expected + 1)
