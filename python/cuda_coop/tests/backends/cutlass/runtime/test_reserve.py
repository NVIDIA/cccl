# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise typed shared reservations with native CuTe and cooperative calls."""

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute

from cuda import coop
from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


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
