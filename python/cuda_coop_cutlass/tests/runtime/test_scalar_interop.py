# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise host scalar selectors and floating prefix semantics on device."""

import cutlass
import numpy as np
import pytest
from cutlass import cute

import cuda.coop.cutlass as coop
from tests.runtime_support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize(
    "value", (np.int16(-3), np.uint8(3), np.int32(3), np.float32(0.5))
)
def test_numpy_scalar_reduction(value):
    @cute.kernel
    def kernel(destination: cute.Pointer):
        total = coop.sum(coop.this_warp(), value)
        if cute.arch.thread_idx()[0] == 0:
            cute.make_tensor(destination, cute.make_layout(1))[0] = (
                cutlass.Float64(total)
            )

    @cute.jit
    def launch(destination: cute.Pointer):
        kernel(destination).launch(grid=1, block=32)

    observed = np.zeros(1, dtype=np.float64)
    with device_array(observed) as destination:
        launch(destination)
    assert observed[0] == float(value) * 32
