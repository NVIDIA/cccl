# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executable examples included in the common API docstrings."""

import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]


def test_exchange_example():
    # exchange-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
    from cuda import coop

    @cuda.jit
    def exchange_tile(source, destination):
        block = coop.this_block()
        striped = coop.ThreadData(2, dtype=np.int32)
        coop.load(block, source, striped, algorithm="striped")
        blocked = coop.exchange(block, striped, mode="striped_to_blocked")
        coop.store(block, destination, blocked)

    values = np.arange(256, dtype=np.int32) * 3 - 200
    source = cuda.to_device(values)
    destination = cuda.device_array_like(source)
    exchange_tile[1, 128](source, destination)
    np.testing.assert_array_equal(destination.copy_to_host(), values)
    # exchange-example-end


def test_exchange_striped_output():
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
    from cuda import coop

    @cuda.jit
    def exchange_tile(source, destination, preserved):
        block = coop.this_block()
        blocked = coop.ThreadData(2, dtype=np.int32)
        coop.load(block, source, blocked)
        striped = coop.exchange(block, blocked, mode="blocked_to_striped")
        coop.store(block, destination, striped)
        coop.store(block, preserved, blocked)

    values = np.arange(256, dtype=np.int32) * 3 - 200
    source = cuda.to_device(values)
    destination = cuda.device_array_like(source)
    preserved = cuda.device_array_like(source)
    exchange_tile[1, 128](source, destination, preserved)
    expected = values.reshape(2, 128).T.flatten()
    np.testing.assert_array_equal(destination.copy_to_host(), expected)
    np.testing.assert_array_equal(preserved.copy_to_host(), values)


def test_shuffle_example():
    # shuffle-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
    from cuda import coop

    @cuda.jit
    def shift_tile(source, following, preceding):
        block = coop.this_block()
        items = coop.ThreadData(2, dtype=np.int32)
        coop.load(block, source, items)
        down = coop.shuffle(block, items, mode="down")
        up = coop.shuffle(block, items, mode="up")
        # Define each exposed boundary before it is read by store.
        if cuda.threadIdx.x == cuda.blockDim.x - 1:
            down[1] = 0
        if cuda.threadIdx.x == 0:
            up[0] = 0
        coop.store(block, following, down)
        coop.store(block, preceding, up)

    values = np.arange(256, dtype=np.int32) * 3 - 200
    source = cuda.to_device(values)
    following = cuda.device_array_like(source)
    preceding = cuda.device_array_like(source)
    shift_tile[1, 128](source, following, preceding)
    np.testing.assert_array_equal(following.copy_to_host(), np.append(values[1:], 0))
    np.testing.assert_array_equal(
        preceding.copy_to_host(), np.insert(values[:-1], 0, 0)
    )
    # shuffle-example-end
