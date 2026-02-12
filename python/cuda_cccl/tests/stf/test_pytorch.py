# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import numpy as np
import pytest

torch = pytest.importorskip("torch")

import cuda.stf as stf  # noqa: E402


def test_pytorch():
    n = 1024 * 1024
    X = np.ones(n, dtype=np.float32)
    Y = np.ones(n, dtype=np.float32)
    Z = np.ones(n, dtype=np.float32)

    ctx = stf.context()
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    lZ = ctx.logical_data(Z)

    with ctx.task(lX.rw()) as t:
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
    # Expected values:
    # X: 1.0 -> 2.0 (multiplied by 2)
    # Y: 1.0 -> 4.0 (X * 2 = 2.0 * 2 = 4.0)
    # Z: 1.0 -> 9.0 (X * 4 + 1 = 2.0 * 4 + 1 = 9.0) -> 5.0 (Y * 2 - 3 = 4.0 * 2 - 3 = 5.0)
    assert np.allclose(X, 2.0)
    assert np.allclose(Y, 4.0)
    assert np.allclose(Z, 5.0)


def test_pytorch_task():
    """Test the pytorch_task functionality with simplified syntax"""
    n = 1024 * 1024
    X = np.ones(n, dtype=np.float32)
    Y = np.ones(n, dtype=np.float32)
    Z = np.ones(n, dtype=np.float32)

    ctx = stf.context()

    # Note: We could use ctx.logical_data_full instead of creating NumPy arrays first
    # For example: lX = ctx.logical_data_full((n,), 1.0, dtype=np.float32)
    # However, this would create logical data without underlying NumPy arrays,
    # so we wouldn't be able to check results after ctx.finalize() in this test
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    lZ = ctx.logical_data(Z)

    # Equivalent operations to test_pytorch() but using pytorch_task syntax

    # In-place multiplication using pytorch_task (single tensor)
    with ctx.pytorch_task(lX.rw()) as (tX,):
        tX[:] = tX * 2

    # Copy and multiply using pytorch_task (multiple tensors)
    with ctx.pytorch_task(lX.read(), lY.write()) as (tX, tY):
        tY[:] = tX * 2

    # Another operation combining tensors
    with ctx.pytorch_task(lX.read(), lZ.write()) as (tX, tZ):
        tZ[:] = tX * 4 + 1

    # Final operation with read-write access
    with ctx.pytorch_task(lY.read(), lZ.rw()) as (tY, tZ):
        tZ[:] = tY * 2 - 3

    ctx.finalize()

    # Verify results on host after finalize (same as original test)
    # Expected values:
    # X: 1.0 -> 2.0 (multiplied by 2)
    # Y: 1.0 -> 4.0 (X * 2 = 2.0 * 2 = 4.0)
    # Z: 1.0 -> 9.0 (X * 4 + 1 = 2.0 * 4 + 1 = 9.0) -> 5.0 (Y * 2 - 3 = 4.0 * 2 - 3 = 5.0)
    assert np.allclose(X, 2.0)
    assert np.allclose(Y, 4.0)
    assert np.allclose(Z, 5.0)


if __name__ == "__main__":
    test_pytorch()
