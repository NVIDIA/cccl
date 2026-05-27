# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Examples combining STF task graphs with cuda.compute (CUB/Thrust) algorithms.

These demonstrate the Python equivalent of C++ STF examples that call CUB
device-wide algorithms (reduce, scan) and Thrust transforms inside STF tasks.
The key integration point is the CUDA stream: STF provides a per-task stream
via task.stream_ptr(), and cuda.compute algorithms accept a stream= parameter
implementing the __cuda_stream__ protocol.

Ported from:
  - cudax/examples/stf/08-cub-reduce.cu  (reduce)
  - cudax/examples/stf/scan.cu           (inclusive scan)
  - cudax/examples/stf/thrust_zip_iterator.cu (binary transform)
"""

import numpy as np
import pytest

import cuda.stf._experimental as stf

try:
    import cuda.compute
    from cuda.compute import OpKind

    _HAS_CUDA_COMPUTE = True
except ImportError:
    _HAS_CUDA_COMPUTE = False

try:
    import numba.cuda

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

from cuda.stf._experimental.interop.numba import numba_task

pytestmark = pytest.mark.skipif(
    not (_HAS_CUDA_COMPUTE and _HAS_NUMBA),
    reason="cuda.compute and numba-cuda required",
)


# ---------------------------------------------------------------------------
# Example 1: Device-wide reduction (cf. 08-cub-reduce.cu)
#
# C++ version uses cub::BlockReduce in custom kernels launched from an STF
# task.  In Python we simply call cuda.compute.reduce_into() on the task
# stream, which dispatches to CUB DeviceReduce internally.
# ---------------------------------------------------------------------------


def test_stf_reduce():
    """Reduce an array inside an STF task using cuda.compute.reduce_into."""
    N = 1024
    ctx = stf.context()

    h_values = np.arange(N, dtype=np.int32)
    lValues = ctx.logical_data(h_values, name="values")

    h_result = np.zeros(1, dtype=np.int32)
    lResult = ctx.logical_data(h_result, name="result")

    with ctx.task(lValues.read(), lResult.rw()) as t:
        d_in = numba.cuda.from_cuda_array_interface(t.get_arg_cai(0), sync=False)
        d_out = numba.cuda.from_cuda_array_interface(t.get_arg_cai(1), sync=False)
        h_init = np.array([0], dtype=np.int32)
        stream = t.stream_ptr()
        cuda.compute.reduce_into(d_in, d_out, OpKind.PLUS, N, h_init, stream=stream)

    ctx.finalize()

    expected = int(h_values.sum())
    assert h_result[0] == expected, f"got {h_result[0]}, expected {expected}"


@pytest.mark.skip(
    reason=(
        "cuda.compute uses cudaMallocAsync for temp storage, creating mem-alloc "
        "graph nodes that require ownership transfer in cuGraphAddChildGraphNode"
    )
)
def test_stf_reduce_graph():
    """Same reduction but with a graph-mode context."""
    N = 1024
    ctx = stf.context(use_graph=True)

    h_values = np.arange(N, dtype=np.int32)
    lValues = ctx.logical_data(h_values, name="values")

    h_result = np.zeros(1, dtype=np.int32)
    lResult = ctx.logical_data(h_result, name="result")

    with ctx.task(lValues.read(), lResult.rw()) as t:
        # sync=False is required: numba.cuda.from_cuda_array_interface defaults to
        # synchronizing the CAI stream, which is illegal during graph capture.
        d_in = numba.cuda.from_cuda_array_interface(t.get_arg_cai(0), sync=False)
        d_out = numba.cuda.from_cuda_array_interface(t.get_arg_cai(1), sync=False)
        h_init = np.array([0], dtype=np.int32)
        stream = t.stream_ptr()
        cuda.compute.reduce_into(d_in, d_out, OpKind.PLUS, N, h_init, stream=stream)

    ctx.finalize()

    expected = int(h_values.sum())
    assert h_result[0] == expected


# ---------------------------------------------------------------------------
# Example 2: Device-wide inclusive scan (cf. scan.cu)
#
# C++ version queries CUB temp storage, creates a logical_data<slice<char>>
# for it, then calls cub::DeviceScan::InclusiveSum.  In Python,
# cuda.compute.inclusive_scan handles temp storage automatically.
# ---------------------------------------------------------------------------


def test_stf_inclusive_scan():
    """In-place inclusive prefix sum using cuda.compute inside an STF task."""
    N = 1024
    ctx = stf.context()

    h_data = np.ones(N, dtype=np.float64)
    lData = ctx.logical_data(h_data, name="scan_data")

    lOut = ctx.logical_data(np.zeros(N, dtype=np.float64), name="scan_out")

    with ctx.task(lData.read(), lOut.rw()) as t:
        d_in = numba.cuda.from_cuda_array_interface(t.get_arg_cai(0), sync=False)
        d_out = numba.cuda.from_cuda_array_interface(t.get_arg_cai(1), sync=False)
        stream = t.stream_ptr()
        cuda.compute.inclusive_scan(d_in, d_out, OpKind.PLUS, None, N, stream=stream)

    # Verify via host_launch that reads the result
    results = []
    ctx.host_launch(lOut.read(), fn=lambda out: results.append(out.copy()))
    ctx.finalize()

    expected = np.cumsum(h_data)
    np.testing.assert_allclose(results[0], expected)


def test_stf_exclusive_scan():
    """Exclusive prefix sum using cuda.compute inside an STF task."""
    N = 512
    ctx = stf.context()

    h_data = np.arange(1, N + 1, dtype=np.int32)
    lData = ctx.logical_data(h_data, name="data")

    h_out = np.zeros(N, dtype=np.int32)
    lOut = ctx.logical_data(h_out, name="scan_out")

    with ctx.task(lData.read(), lOut.rw()) as t:
        d_in = numba.cuda.from_cuda_array_interface(t.get_arg_cai(0), sync=False)
        d_out = numba.cuda.from_cuda_array_interface(t.get_arg_cai(1), sync=False)
        h_init = np.array([0], dtype=np.int32)
        stream = t.stream_ptr()
        cuda.compute.exclusive_scan(d_in, d_out, OpKind.PLUS, h_init, N, stream=stream)

    ctx.finalize()

    expected = np.concatenate(([0], np.cumsum(h_data)[:-1]))
    np.testing.assert_array_equal(h_out, expected)


# ---------------------------------------------------------------------------
# Example 3: Binary transform (cf. thrust_zip_iterator.cu)
#
# C++ version creates a zip iterator from two Thrust vectors and calls
# thrust::transform with a custom functor.  In Python we use
# cuda.compute.binary_transform with a user-defined operator.
# ---------------------------------------------------------------------------


def test_stf_binary_transform():
    """Element-wise addition of two arrays using cuda.compute.binary_transform."""
    N = 256
    ctx = stf.context()

    h_a = np.arange(N, dtype=np.float32)
    h_b = np.arange(N, dtype=np.float32) * 2.0
    h_c = np.zeros(N, dtype=np.float32)

    lA = ctx.logical_data(h_a, name="A")
    lB = ctx.logical_data(h_b, name="B")
    lC = ctx.logical_data(h_c, name="C")

    with ctx.task(lA.read(), lB.read(), lC.rw()) as t:
        dA = numba.cuda.from_cuda_array_interface(t.get_arg_cai(0), sync=False)
        dB = numba.cuda.from_cuda_array_interface(t.get_arg_cai(1), sync=False)
        dC = numba.cuda.from_cuda_array_interface(t.get_arg_cai(2), sync=False)
        stream = t.stream_ptr()
        cuda.compute.binary_transform(dA, dB, dC, OpKind.PLUS, N, stream=stream)

    ctx.finalize()

    expected = h_a + h_b
    np.testing.assert_allclose(h_c, expected)


def test_stf_unary_transform():
    """Negate each element using cuda.compute.unary_transform with a custom op."""
    N = 128
    ctx = stf.context()

    h_in = np.arange(N, dtype=np.float64) + 1.0
    h_out = np.zeros(N, dtype=np.float64)

    lIn = ctx.logical_data(h_in, name="input")
    lOut = ctx.logical_data(h_out, name="output")

    def negate(x):
        return -x

    with ctx.task(lIn.read(), lOut.rw()) as t:
        d_in = numba.cuda.from_cuda_array_interface(t.get_arg_cai(0), sync=False)
        d_out = numba.cuda.from_cuda_array_interface(t.get_arg_cai(1), sync=False)
        stream = t.stream_ptr()
        cuda.compute.unary_transform(d_in, d_out, negate, N, stream=stream)

    ctx.finalize()

    np.testing.assert_allclose(h_out, -h_in)


# ---------------------------------------------------------------------------
# Example 4: Multi-task pipeline — reduce after transform
#
# Shows how STF dependency tracking automatically sequences tasks that use
# cuda.compute algorithms on different streams.
# ---------------------------------------------------------------------------


def test_stf_pipeline_transform_then_reduce():
    """Pipeline: binary_transform in one task, then reduce_into in the next."""
    N = 512
    ctx = stf.context()

    h_a = np.ones(N, dtype=np.float32) * 3.0
    h_b = np.ones(N, dtype=np.float32) * 7.0
    h_c = np.zeros(N, dtype=np.float32)
    h_sum = np.zeros(1, dtype=np.float32)

    lA = ctx.logical_data(h_a, name="A")
    lB = ctx.logical_data(h_b, name="B")
    lC = ctx.logical_data(h_c, name="C")
    lSum = ctx.logical_data(h_sum, name="sum")

    # Task 1: C = A + B
    with ctx.task(lA.read(), lB.read(), lC.rw(), symbol="add") as t:
        dA = numba.cuda.from_cuda_array_interface(t.get_arg_cai(0), sync=False)
        dB = numba.cuda.from_cuda_array_interface(t.get_arg_cai(1), sync=False)
        dC = numba.cuda.from_cuda_array_interface(t.get_arg_cai(2), sync=False)
        stream = t.stream_ptr()
        cuda.compute.binary_transform(dA, dB, dC, OpKind.PLUS, N, stream=stream)

    # Task 2: sum = reduce(C)  — automatically waits for task 1
    with ctx.task(lC.read(), lSum.rw(), symbol="reduce") as t:
        dC = numba.cuda.from_cuda_array_interface(t.get_arg_cai(0), sync=False)
        dSum = numba.cuda.from_cuda_array_interface(t.get_arg_cai(1), sync=False)
        h_init = np.array([0.0], dtype=np.float32)
        stream = t.stream_ptr()
        cuda.compute.reduce_into(dC, dSum, OpKind.PLUS, N, h_init, stream=stream)

    ctx.finalize()

    expected = float(N) * (3.0 + 7.0)
    assert abs(h_sum[0] - expected) < 1e-3, f"got {h_sum[0]}, expected {expected}"


# ===========================================================================
# Simplified versions using numba_task
#
# numba_task(ctx, ...) yields (args, stream) where args are numba.cuda
# device arrays and stream implements __cuda_stream__.  Mirrors
# pytorch_task which yields torch.Tensor objects.
# ===========================================================================


def test_simple_reduce():
    """Simplified reduce using numba_task + ctx.wait()."""
    N = 2048
    ctx = stf.context()

    h_vals = np.arange(N, dtype=np.int64)
    lVals = ctx.logical_data(h_vals, name="vals")
    lOut = ctx.logical_data_empty((1,), dtype=np.int64, name="out")

    with numba_task(ctx, lVals.read(), lOut.write()) as (args, stream):
        cuda.compute.reduce_into(
            args[0],
            args[1],
            OpKind.PLUS,
            N,
            np.array([0], dtype=np.int64),
            stream=stream,
        )

    result = ctx.wait(lOut)
    assert result[0] == h_vals.sum()
    ctx.finalize()


def test_simple_scan():
    """Simplified inclusive scan using numba_task + ctx.wait()."""
    N = 256
    ctx = stf.context()

    lIn = ctx.logical_data(np.ones(N, dtype=np.float32), name="in")
    lOut = ctx.logical_data_empty((N,), dtype=np.float32, name="out")

    with numba_task(ctx, lIn.read(), lOut.write()) as (args, stream):
        cuda.compute.inclusive_scan(
            args[0], args[1], OpKind.PLUS, None, N, stream=stream
        )

    result = ctx.wait(lOut)
    np.testing.assert_allclose(result, np.arange(1, N + 1, dtype=np.float32))
    ctx.finalize()


def test_simple_transform():
    """Simplified binary transform using numba_task + ctx.wait()."""
    N = 64
    ctx = stf.context()

    lA = ctx.logical_data(np.full(N, 3.0, dtype=np.float32), name="A")
    lB = ctx.logical_data(np.full(N, 7.0, dtype=np.float32), name="B")
    lC = ctx.logical_data_empty((N,), dtype=np.float32, name="C")

    with numba_task(ctx, lA.read(), lB.read(), lC.write()) as (args, stream):
        cuda.compute.binary_transform(
            args[0], args[1], args[2], OpKind.PLUS, N, stream=stream
        )

    result = ctx.wait(lC)
    np.testing.assert_allclose(result, np.full(N, 10.0, dtype=np.float32))
    ctx.finalize()


def test_simple_pipeline():
    """Pipeline: transform then reduce, using numba_task + ctx.wait()."""
    N = 100
    ctx = stf.context()

    lX = ctx.logical_data(np.arange(N, dtype=np.float64), name="X")
    lY = ctx.logical_data(np.arange(N, dtype=np.float64) * 2, name="Y")
    lZ = ctx.logical_data_empty((N,), dtype=np.float64, name="Z")
    lSum = ctx.logical_data_empty((1,), dtype=np.float64, name="sum")

    # Z = X + Y
    with numba_task(ctx, lX.read(), lY.read(), lZ.write(), symbol="add") as (
        args,
        stream,
    ):
        cuda.compute.binary_transform(
            args[0], args[1], args[2], OpKind.PLUS, N, stream=stream
        )

    # sum = reduce(Z)
    with numba_task(ctx, lZ.read(), lSum.write(), symbol="reduce") as (args, stream):
        cuda.compute.reduce_into(
            args[0],
            args[1],
            OpKind.PLUS,
            N,
            np.array([0.0], dtype=np.float64),
            stream=stream,
        )

    result = ctx.wait(lSum)
    ctx.finalize()

    expected = sum(i + i * 2 for i in range(N))
    assert abs(result[0] - expected) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
