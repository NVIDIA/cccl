# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for host_launch on regular contexts and stackable contexts.

host_launch schedules a Python callable as a host-side task graph node.
Dependencies are auto-unpacked as numpy arrays and passed as the first
positional arguments to the callback.  Extra user data can be supplied
via ``args`` (evaluated eagerly at submission time).
"""

import numba
import numpy as np
from numba import cuda
from numba_helpers import numba_arguments

import cuda.stf as stf

numba.cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@cuda.jit
def fill_kernel(x, val):
    i = cuda.grid(1)
    if i < x.size:
        x[i] = val


@cuda.jit
def scale_kernel(x, alpha):
    i = cuda.grid(1)
    if i < x.size:
        x[i] = x[i] * alpha


# ===========================================================================
# Regular context (stream backend)
# ===========================================================================


def test_context_basic():
    """host_launch on a stream context; dep auto-unpacked as numpy array."""
    n = 1024
    X_host = np.zeros(n, dtype=np.float64)

    ctx = stf.context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 256
    bpg = (n + tpb - 1) // tpb

    with ctx.task(lX.write()) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        dX = numba_arguments(t)
        fill_kernel[bpg, tpb, nb_stream](dX, 42.0)

    result = {}

    def verify(x_arr, res):
        res["ok"] = bool(np.allclose(x_arr, 42.0))

    ctx.host_launch(lX.read(), fn=verify, args=[result])
    ctx.finalize()

    assert result.get("ok", False), "host_launch callback did not verify"


def test_context_multiple_deps():
    """host_launch with two read deps on a stream context."""
    n = 512
    X_host = np.ones(n, dtype=np.float64) * 3.0
    Y_host = np.ones(n, dtype=np.float64) * 7.0

    ctx = stf.context()
    lX = ctx.logical_data(X_host, name="X")
    lY = ctx.logical_data(Y_host, name="Y")

    result = {}

    def check_sum(x_arr, y_arr, res):
        res["dot"] = float(np.dot(x_arr, y_arr))

    ctx.host_launch(lX.read(), lY.read(), fn=check_sum, args=[result])
    ctx.finalize()

    expected = 3.0 * 7.0 * n
    assert abs(result["dot"] - expected) < 1e-6, (
        f"Expected {expected}, got {result['dot']}"
    )


def test_context_write_back():
    """host_launch with rw dep writes back through numpy array."""
    n = 64
    X_host = np.ones(n, dtype=np.float64) * 5.0

    ctx = stf.context()
    lX = ctx.logical_data(X_host, name="X")

    def zero_out(x_arr):
        x_arr[:] = 0.0

    ctx.host_launch(lX.rw(), fn=zero_out)
    ctx.finalize()

    assert np.allclose(X_host, 0.0), f"Expected zeros, got {X_host[:5]}"


def test_context_chained():
    """Two host_launch calls with proper dependency ordering."""
    n = 128
    X_host = np.ones(n, dtype=np.float64)

    ctx = stf.context()
    lX = ctx.logical_data(X_host, name="X")

    log = []

    def step_one(x_arr, log_list):
        x_arr[:] = x_arr * 2
        log_list.append("step1")

    def step_two(x_arr, log_list):
        x_arr[:] = x_arr + 10
        log_list.append("step2")

    ctx.host_launch(lX.rw(), fn=step_one, args=[log])
    ctx.host_launch(lX.rw(), fn=step_two, args=[log])
    ctx.finalize()

    assert log == ["step1", "step2"], f"Unexpected ordering: {log}"
    assert np.allclose(X_host, 12.0), f"Expected 12.0, got {X_host[0]}"


def test_context_no_deps():
    """host_launch with no deps (just ordering / side-effect)."""
    called = [False]

    ctx = stf.context()

    def mark(flag):
        flag[0] = True

    ctx.host_launch(fn=mark, args=[called])
    ctx.finalize()

    assert called[0], "Callback was not invoked"


def test_context_loop_value_capture():
    """args capture values eagerly in a loop (like C++ capture-by-value)."""
    n = 32
    X_host = np.ones(n, dtype=np.float64)

    ctx = stf.context()
    lX = ctx.logical_data(X_host, name="X")

    results = {}

    def record(x_arr, step, res):
        res[step] = float(x_arr.sum())

    for i in range(5):
        ctx.host_launch(lX.read(), fn=record, args=[i, results])

    ctx.finalize()

    for i in range(5):
        assert i in results, f"Step {i} was not recorded"
        assert abs(results[i] - n) < 1e-10, f"Step {i}: expected {n}, got {results[i]}"


def test_context_with_symbol():
    """host_launch with a debug symbol (smoke test)."""
    n = 64
    X_host = np.ones(n, dtype=np.float64)

    ctx = stf.context()
    lX = ctx.logical_data(X_host, name="X")

    called = [False]

    def noop(x_arr, flag):
        flag[0] = True

    ctx.host_launch(lX.read(), fn=noop, args=[called], symbol="my_noop")
    ctx.finalize()

    assert called[0], "Callback was not invoked"


# ===========================================================================
# Regular context (graph backend)
# ===========================================================================


def test_graph_ctx_basic():
    """host_launch on a graph context."""
    n = 128
    X_host = np.zeros(n, dtype=np.float64)

    ctx = stf.context(use_graph=True)
    lX = ctx.logical_data(X_host, name="X")

    tpb = 128
    bpg = (n + tpb - 1) // tpb

    with ctx.task(lX.write()) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        dX = numba_arguments(t)
        fill_kernel[bpg, tpb, nb_stream](dX, 3.14)

    result = {}

    def verify(x_arr, res):
        res["ok"] = bool(np.allclose(x_arr, 3.14))

    ctx.host_launch(lX.read(), fn=verify, args=[result])
    ctx.finalize()

    assert result.get("ok", False), "graph context host_launch did not verify"


def test_graph_ctx_write_back():
    """host_launch with rw dep on a graph context writes back."""
    n = 64
    X_host = np.ones(n, dtype=np.float64) * 5.0

    ctx = stf.context(use_graph=True)
    lX = ctx.logical_data(X_host, name="X")

    def double_it(x_arr):
        x_arr[:] = x_arr * 2

    ctx.host_launch(lX.rw(), fn=double_it)
    ctx.finalize()

    assert np.allclose(X_host, 10.0), f"Expected 10.0, got {X_host[:5]}"


# ===========================================================================
# Stackable context
# ===========================================================================


def test_stackable_basic():
    """host_launch on a stackable context; dep auto-unpacked as numpy array."""
    n = 1024
    X_host = np.zeros(n, dtype=np.float64)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 256
    bpg = (n + tpb - 1) // tpb

    with ctx.task(lX.write()) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        dX = numba_arguments(t)
        fill_kernel[bpg, tpb, nb_stream](dX, 42.0)

    result = {}

    def verify(x_arr, res):
        res["ok"] = bool(np.allclose(x_arr, 42.0))

    ctx.host_launch(lX.read(), fn=verify, args=[result])
    ctx.finalize()

    assert result.get("ok", False), "host_launch callback did not verify"


def test_stackable_multiple_deps():
    """host_launch with two read deps on a stackable context."""
    n = 512
    X_host = np.ones(n, dtype=np.float64) * 3.0
    Y_host = np.ones(n, dtype=np.float64) * 7.0

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")
    lY = ctx.logical_data(Y_host, name="Y")

    result = {}

    def check_sum(x_arr, y_arr, res):
        res["dot"] = float(np.dot(x_arr, y_arr))

    ctx.host_launch(lX.read(), lY.read(), fn=check_sum, args=[result])
    ctx.finalize()

    expected = 3.0 * 7.0 * n
    assert abs(result["dot"] - expected) < 1e-6, (
        f"Expected {expected}, got {result['dot']}"
    )


def test_stackable_in_graph_scope():
    """host_launch after a graph scope on a stackable context."""
    n = 256
    X_host = np.zeros(n, dtype=np.float64)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 256
    bpg = (n + tpb - 1) // tpb

    with ctx.graph_scope():
        with ctx.task(lX.write()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            fill_kernel[bpg, tpb, nb_stream](dX, 99.0)

    result = {}

    def verify(x_arr, res):
        res["max"] = float(np.max(x_arr))

    ctx.host_launch(lX.read(), fn=verify, args=[result])
    ctx.finalize()

    assert abs(result["max"] - 99.0) < 1e-10, f"Expected 99.0, got {result['max']}"


def test_stackable_write_back():
    """host_launch with rw dep writes back through numpy array."""
    n = 64
    X_host = np.ones(n, dtype=np.float64) * 5.0

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    def zero_out(x_arr):
        x_arr[:] = 0.0

    ctx.host_launch(lX.rw(), fn=zero_out)
    ctx.finalize()

    assert np.allclose(X_host, 0.0), f"Expected zeros, got {X_host[:5]}"


def test_stackable_chained():
    """Two host_launch calls with proper dependency ordering."""
    n = 128
    X_host = np.ones(n, dtype=np.float64)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    log = []

    def step_one(x_arr, log_list):
        x_arr[:] = x_arr * 2
        log_list.append("step1")

    def step_two(x_arr, log_list):
        x_arr[:] = x_arr + 10
        log_list.append("step2")

    ctx.host_launch(lX.rw(), fn=step_one, args=[log])
    ctx.host_launch(lX.rw(), fn=step_two, args=[log])
    ctx.finalize()

    assert log == ["step1", "step2"], f"Unexpected ordering: {log}"
    assert np.allclose(X_host, 12.0), f"Expected 12.0, got {X_host[0]}"


def test_stackable_loop_value_capture():
    """args capture values eagerly in a loop (like C++ capture-by-value)."""
    n = 32
    X_host = np.ones(n, dtype=np.float64)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    results = {}

    def record(x_arr, step, res):
        res[step] = float(x_arr.sum())

    for i in range(5):
        ctx.host_launch(lX.read(), fn=record, args=[i, results])

    ctx.finalize()

    for i in range(5):
        assert i in results, f"Step {i} was not recorded"
        assert abs(results[i] - n) < 1e-10, f"Step {i}: expected {n}, got {results[i]}"


def test_stackable_no_deps():
    """host_launch with no deps (just ordering / side-effect)."""
    called = [False]

    ctx = stf.stackable_context()

    def mark(flag):
        flag[0] = True

    ctx.host_launch(fn=mark, args=[called])
    ctx.finalize()

    assert called[0], "Callback was not invoked"


def test_stackable_with_symbol():
    """host_launch with a debug symbol (smoke test)."""
    n = 64
    X_host = np.ones(n, dtype=np.float64)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    called = [False]

    def noop(x_arr, flag):
        flag[0] = True

    ctx.host_launch(lX.read(), fn=noop, args=[called], symbol="my_noop")
    ctx.finalize()

    assert called[0], "Callback was not invoked"


if __name__ == "__main__":
    # Regular context (stream)
    test_context_basic()
    test_context_multiple_deps()
    test_context_write_back()
    test_context_chained()
    test_context_no_deps()
    test_context_loop_value_capture()
    test_context_with_symbol()

    # Regular context (graph)
    test_graph_ctx_basic()
    test_graph_ctx_write_back()

    # Stackable context
    test_stackable_basic()
    test_stackable_multiple_deps()
    test_stackable_in_graph_scope()
    test_stackable_write_back()
    test_stackable_chained()
    test_stackable_loop_value_capture()
    test_stackable_no_deps()
    test_stackable_with_symbol()

    print("All host_launch tests passed!")
