# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import numba
import numpy as np
import pytest

torch = pytest.importorskip("torch")

numba.config.CUDA_ENABLE_PYNVJITLINK = 1
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

from cuda.cccl.experimental.stf._stf_bindings import (  # noqa: E402
    context,
    rw,
)


def test_pytorch():
    n = 1024 * 1024
    X = np.ones(n, dtype=np.float32)
    Y = np.ones(n, dtype=np.float32)
    Z = np.ones(n, dtype=np.float32)

    ctx = context()
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    lZ = ctx.logical_data(Z)

    with ctx.task(rw(lX)) as t:
        torch_stream = torch.cuda.ExternalStream(t.stream_ptr())
        with torch.cuda.stream(torch_stream):
            tX = t.tensor_arguments()
            tX[:] = tX * 2  # In-place multiplication

    with ctx.task(lX.read(), lY.write()) as t:
        torch_stream = torch.cuda.ExternalStream(t.stream_ptr())
        with torch.cuda.stream(torch_stream):
            tX = t.get_arg_as_tensor(0)
            tY = t.get_arg_as_tensor(1)
            tY[:] = tX * 2  # Copy result into tY tensor

    with (
        ctx.task(lX.read(), lZ.write()) as t,
        torch.cuda.stream(torch.cuda.ExternalStream(t.stream_ptr())),
    ):
        tX, tZ = t.tensor_arguments()  # Get tX and tZ tensors
        tZ[:] = tX * 4 + 1  # Copy result into tZ tensor

    with (
        ctx.task(lY.read(), lZ.rw()) as t,
        torch.cuda.stream(torch.cuda.ExternalStream(t.stream_ptr())),
    ):
        tY, tZ = t.tensor_arguments()  # Get tY and tZ tensors
        tZ[:] = tY * 2 - 3  # Copy result into tZ tensor

    ctx.finalize()

    # Verify results on host after finalize
    print("Verifying results...")

    # Expected values:
    # X: 1.0 -> 2.0 (multiplied by 2)
    # Y: 1.0 -> 4.0 (X * 2 = 2.0 * 2 = 4.0)
    # Z: 1.0 -> 9.0 (X * 4 + 1 = 2.0 * 4 + 1 = 9.0) -> 5.0 (Y * 2 - 3 = 4.0 * 2 - 3 = 5.0)

    expected_X = 2.0
    expected_Y = 4.0
    expected_Z = 5.0

    # Check a few values to verify correctness
    assert np.allclose(X[:10], expected_X), (
        f"X mismatch: got {X[:10]}, expected {expected_X}"
    )
    assert np.allclose(Y[:10], expected_Y), (
        f"Y mismatch: got {Y[:10]}, expected {expected_Y}"
    )
    assert np.allclose(Z[:10], expected_Z), (
        f"Z mismatch: got {Z[:10]}, expected {expected_Z}"
    )

    # Check entire arrays
    assert np.all(X == expected_X), (
        f"X array not uniform: min={X.min()}, max={X.max()}, expected={expected_X}"
    )
    assert np.all(Y == expected_Y), (
        f"Y array not uniform: min={Y.min()}, max={Y.max()}, expected={expected_Y}"
    )
    assert np.all(Z == expected_Z), (
        f"Z array not uniform: min={Z.min()}, max={Z.max()}, expected={expected_Z}"
    )

    print(f"✅ All checks passed! X={X[0]}, Y={Y[0]}, Z={Z[0]}")


if __name__ == "__main__":
    print("Running CUDASTF examples...")
    test_pytorch()
