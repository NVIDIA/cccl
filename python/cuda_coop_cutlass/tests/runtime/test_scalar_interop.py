# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise host scalar selectors and floating prefix semantics on device."""

import cutlass
import numpy as np
import pytest
from cutlass import cute

import cuda.coop.cutlass as coop
from tests.runtime_support import cutlass_dtype, device_array

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


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("operation", ("min", "max"))
@pytest.mark.parametrize("group_kind", ("warp", "logical", "block"))
@pytest.mark.parametrize("runtime", (False, True))
def test_nan_only_valid_prefix(dtype, operation, group_kind, runtime):
    value_type = cutlass_dtype(dtype)

    @cute.kernel
    def kernel(destination: cute.Pointer, count: cutlass.Int32):
        thread = cute.arch.thread_idx()[0]
        if cutlass.const_expr(group_kind == "warp"):
            group = coop.this_warp()
        elif cutlass.const_expr(group_kind == "logical"):
            group = coop.this_warp().group_by(8)
        else:
            group = coop.this_block()
        value = value_type(float("nan"))
        if cutlass.const_expr(runtime):
            result = coop.reduce(
                group, value, binary_op=operation, valid_items=count
            )
        else:
            result = coop.reduce(
                group, value, binary_op=operation, valid_items=4
            )
        if thread == 0:
            cute.make_tensor(destination, cute.make_layout(1))[0] = result

    @cute.jit
    def launch(destination: cute.Pointer, count: cutlass.Int32):
        kernel(destination, count).launch(grid=1, block=64)

    observed = np.zeros(1, dtype=dtype)
    with device_array(observed) as destination:
        launch(destination, cutlass.Int32(4))
    assert np.isnan(observed[0])


@pytest.mark.parametrize("region", ("branch", "loop"))
def test_scan_aggregate_explicit_copy(region):
    @cute.kernel
    def kernel(destination: cute.Pointer, count: cutlass.Int32):
        thread = cute.arch.thread_idx()[0]
        aggregate = coop.ThreadData.from_values(cutlass.Int32(0))
        total = cutlass.Int32(0)
        if cutlass.const_expr(region == "loop"):
            for _ in cutlass.range(count):
                total = total + aggregate[0]
                temporary = coop.ThreadData(1)
                coop.inclusive_sum(
                    coop.this_block(),
                    cutlass.Int32(thread + 1),
                    aggregate_output=temporary,
                )
                aggregate[0] = temporary[0]
        else:
            if count > 0:
                temporary = coop.ThreadData(1)
                coop.inclusive_sum(
                    coop.this_block(),
                    cutlass.Int32(thread + 1),
                    aggregate_output=temporary,
                )
                aggregate[0] = temporary[0]
            total = aggregate[0]
        cute.make_tensor(destination, cute.make_layout(32))[thread] = total

    @cute.jit
    def launch(destination: cute.Pointer, count: cutlass.Int32):
        kernel(destination, count).launch(grid=1, block=32)

    observed = np.zeros(32, dtype=np.int32)
    with device_array(observed) as destination:
        launch(destination, cutlass.Int32(3))
    np.testing.assert_array_equal(observed, 1056 if region == "loop" else 528)
