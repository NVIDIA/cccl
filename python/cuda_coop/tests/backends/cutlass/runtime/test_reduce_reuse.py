# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Repeated CUDAX primitives with independent group results."""

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute

from cuda import coop
from cuda.bindings import driver
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import check_cuda, device_array, values_for

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("kind", ("block", "mapped", "cluster"))
def test_cudax_loop(api, kind):
    block_threads = 128
    grid = 1
    width = block_threads if kind == "block" else 64
    if kind == "cluster":
        cutlass.cuda.initialize_cuda_context()
        device = check_cuda(driver.cuCtxGetDevice())
        supported = check_cuda(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH, device
            )
        )
        if not supported:
            pytest.skip("device does not support thread-block cluster launch")
        grid = 4
        block_threads = 32
    total_threads = block_threads * grid

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer, iterations: cutlass.Int32):
        thread = cute.arch.block_idx()[0] * block_threads + cute.arch.thread_idx()[0]
        inputs = cute.make_tensor(source, cute.make_layout(total_threads))
        outputs = cute.make_tensor(observed, cute.make_layout(total_threads))
        if cutlass.const_expr(kind == "block"):
            group = api.this_block()
        elif cutlass.const_expr(kind == "mapped"):
            group = api.this_block().group_by(2)
        else:
            group = api.this_cluster()
        accumulated = cutlass.Int32(0)
        for iteration in range(iterations):
            total = api.sum(group, inputs[thread] + iteration)
            largest = api.reduce(group, inputs[thread], binary_op="max")
            accumulated += total + largest
        outputs[thread] = accumulated

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer, iterations: cutlass.Int32):
        if cutlass.const_expr(kind == "cluster"):
            kernel(source, observed, iterations).launch(
                grid=grid, block=block_threads, cluster=(2, 1, 1)
            )
        else:
            kernel(source, observed, iterations).launch(grid=grid, block=block_threads)

    iterations = 5
    source = values_for(np.int32, total_threads, shift=79)
    observed = np.zeros_like(source)
    rows = source.reshape(-1, width)
    expected = np.repeat(
        iterations * (rows.sum(axis=1) + rows.max(axis=1))
        + width * sum(range(iterations)),
        width,
    )
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, iterations)
    np.testing.assert_array_equal(observed, expected)
