# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise typed shared reservations with native CuTe and cooperative calls."""

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.memory import SmemAllocator

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
def test_shared_reservations_alias_across_native_and_cooperative_phases(api):
    """Reuse one region while another descriptor and native arrays stay live."""

    @cute.jit
    def fill(tensor):
        thread, _, _ = cute.arch.thread_idx()
        tensor[thread] = 3 * thread + 7

    @cute.kernel
    def kernel(destination: cute.Pointer):
        group = api.this_block()
        thread = group.rank()
        scratch = api.TempStorage(512, sharing="shared")
        alias = scratch
        first = scratch.reserve(128, cutlass.Int32, alignment=64)
        second = alias.reserve(64, cutlass.Float64, alignment=128)
        persistent = api.TempStorage().reserve(64, cutlass.Int32, alignment=32)
        native = SmemAllocator().allocate_tensor(
            cutlass.Int32, cute.make_layout(64), byte_alignment=64
        )
        output = cute.make_tensor(destination, cute.make_layout(260))
        fill(first)
        persistent[thread] = 5 * thread + 19
        native[thread] = 11 * thread + 3
        scratch.sync()
        output[thread] = cutlass.Int64(first[63 - thread])
        scratch.sync()
        total = api.sum(group, cutlass.Int32(thread + 1), temp_storage=alias)
        scratch.sync()
        second[thread] = cutlass.Float64(2 * thread + 17)
        scratch.sync()
        output[64 + thread] = cutlass.Int64(second[63 - thread])
        output[128 + thread] = cutlass.Int64(persistent[63 - thread])
        output[192 + thread] = cutlass.Int64(native[63 - thread])
        if thread == 0:
            output[256] = cutlass.Int64(total)
            output[257] = cutlass.Int64(first.iterator.toint())
            output[258] = cutlass.Int64(second.iterator.toint())
            output[259] = cutlass.Int64(cute.arch.dynamic_smem_size())

    @cute.jit
    def launch(destination: cute.Pointer):
        kernel(destination).launch(grid=1, block=64)

    output = np.zeros(260, dtype=np.int64)
    with device_array(output) as pointer:
        launch(pointer)
    reverse = np.arange(63, -1, -1, dtype=np.int64)
    np.testing.assert_array_equal(output[:64], 3 * reverse + 7)
    np.testing.assert_array_equal(output[64:128], 2 * reverse + 17)
    np.testing.assert_array_equal(output[128:192], 5 * reverse + 19)
    np.testing.assert_array_equal(output[192:256], 11 * reverse + 3)
    assert output[256] == 2080
    assert output[257] == output[258]
    assert output[257] % 128 == 0
    assert output[259] == 1024


@pytest.mark.parametrize("count", (64, 13312))
def test_exclusive_reservations_preserve_values_and_account_for_launch(count):
    """Keep every slice live across a collective, including above 48 KiB."""

    @cute.kernel
    def kernel(destination: cute.Pointer, count: cutlass.Constexpr):
        group = coop.this_block()
        thread = group.rank()
        scratch = coop.TempStorage(sharing="exclusive")
        first = scratch.reserve(count, np.dtype(np.int32), alignment=64)
        second = scratch.reserve(64, cutlass.Int32, alignment=128)
        for i in range(thread, count, 64):
            first[i] = 3 * i + 17
        second[thread] = 7 * thread + 11
        scratch.sync()
        total = coop.sum(group, cutlass.Int32(thread + 1), temp_storage=scratch)
        scratch.sync()
        output = cute.make_tensor(destination, cute.make_layout(count + 68))
        for i in range(thread, count, 64):
            output[i] = cutlass.Int64(first[count - 1 - i])
        output[count + thread] = cutlass.Int64(second[63 - thread])
        if thread == 0:
            output[count + 64] = cutlass.Int64(total)
            output[count + 65] = cutlass.Int64(first.iterator.toint())
            output[count + 66] = cutlass.Int64(second.iterator.toint())
            output[count + 67] = cutlass.Int64(cute.arch.dynamic_smem_size())

    @cute.jit
    def launch(destination: cute.Pointer, count: cutlass.Constexpr):
        kernel(destination, count).launch(grid=1, block=64)

    output = np.zeros(count + 68, dtype=np.int64)
    with device_array(output) as pointer:
        launch(pointer, count)
    np.testing.assert_array_equal(
        output[:count], 3 * np.arange(count - 1, -1, -1) + 17
    )
    np.testing.assert_array_equal(
        output[count : count + 64], 7 * np.arange(63, -1, -1) + 11
    )
    assert output[-4] == 2080
    assert output[-3] % 64 == 0
    assert output[-2] % 128 == 0
    assert output[-2] - output[-3] >= count * 4
    assert output[-1] >= count * 4 + 256


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("storage_free_call", (False, True))
def test_reservation_only_helper_and_runtime_loop(sharing, storage_free_call):
    """Resolve typed views without provider layouts, including in loops."""

    @cute.jit
    def reserve(storage):
        return storage.reserve(64, cutlass.Int32, alignment=64)

    @cute.kernel
    def kernel(destination: cute.Pointer, iterations: cutlass.Int32):
        thread, _, _ = cute.arch.thread_idx()
        scratch = coop.TempStorage(sharing=sharing)
        output = cute.make_tensor(destination, cute.make_layout(131))
        for iteration in range(iterations):
            values = reserve(scratch)
            values[thread] = cutlass.Int32(thread + 3 * iteration)
            scratch.sync()
            if cutlass.const_expr(storage_free_call):
                payload = coop.ThreadData(items_per_thread=1)
                payload[0] = cutlass.Int64(values[63 - thread])
                coop.store(
                    coop.this_block(), destination, payload, algorithm="direct"
                )
            else:
                output[thread] = cutlass.Int64(values[63 - thread])
            if thread == 0:
                output[128 + iteration] = cutlass.Int64(values.iterator.toint())
            scratch.sync()
        other = reserve(scratch)
        other[thread] = cutlass.Int32(thread + 101)
        scratch.sync()
        output[64 + thread] = cutlass.Int64(other[63 - thread])
        if thread == 0:
            output[130] = cutlass.Int64(other.iterator.toint())

    @cute.jit
    def launch(destination: cute.Pointer, iterations: cutlass.Int32):
        kernel(destination, iterations).launch(grid=1, block=64)

    output = np.zeros(131, dtype=np.int64)
    with device_array(output) as pointer:
        launch(pointer, cutlass.Int32(2))
    np.testing.assert_array_equal(output[:64], np.arange(63, -1, -1) + 3)
    np.testing.assert_array_equal(output[64:128], np.arange(63, -1, -1) + 101)
    assert output[128] == output[129]
    assert (output[129] == output[130]) == (sharing == "shared")
