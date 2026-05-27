# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for the re-launchable ``stackable_context.launchable_graph_scope()``
context manager. Mirrors the C unit tests in
``c/experimental/stf/test/test_stackable.cu``.
"""

import numpy as np
import pytest

numba = pytest.importorskip("numba")
pytest.importorskip("numba.cuda")
from numba import cuda  # noqa: E402
from cuda.stf._experimental.interop.numba import numba_arguments  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402

numba.cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@cuda.jit
def add_kernel(x, val):
    i = cuda.grid(1)
    if i < x.size:
        x[i] = x[i] + val


@cuda.jit
def scale_kernel(x, alpha):
    i = cuda.grid(1)
    if i < x.size:
        x[i] = x[i] * alpha


def test_launchable_graph_scope_relaunch():
    """Re-launching the same graph N times accumulates N increments."""
    n = 1024
    N = 16
    X_host = np.zeros(n, dtype=np.float32)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 256
    bpg = (n + tpb - 1) // tpb

    with ctx.launchable_graph_scope() as scope:
        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            add_kernel[bpg, tpb, nb_stream](dX, 1.0)

        # prologue instantiates the graph but does not launch it; we launch
        # it exactly N times via scope.launch().
        for _ in range(N):
            scope.launch()

    ctx.finalize()

    assert np.allclose(X_host, float(N)), f"Expected {N}, got {X_host[0]}"


def test_launchable_graph_scope_zero_launches():
    """Exiting the scope without launching unfreezes the data cleanly."""
    n = 512
    X_host = np.full(n, 7.0, dtype=np.float32)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 256
    bpg = (n + tpb - 1) // tpb

    # Enter the scope, submit work, never call launch(). The __exit__ path
    # must still run the prologue+epilogue so that lX is reusable below.
    with ctx.launchable_graph_scope():
        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            add_kernel[bpg, tpb, nb_stream](dX, 1.0)

    # Subsequent regular graph_scope() must still work and mutate lX.
    with ctx.graph_scope():
        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            scale_kernel[bpg, tpb, nb_stream](dX, 2.0)

    ctx.finalize()

    # The launchable scope never launched => 7.0 unchanged.
    # The following graph_scope doubled it to 14.0.
    assert np.allclose(X_host, 14.0), f"Expected 14.0, got {X_host[0]}"


def test_launchable_graph_scope_exec_and_stream():
    """exec_graph and stream accessors return non-null pointers."""
    n = 256
    X_host = np.zeros(n, dtype=np.float32)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 256
    bpg = (n + tpb - 1) // tpb

    with ctx.launchable_graph_scope() as scope:
        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            add_kernel[bpg, tpb, nb_stream](dX, 1.0)

        # Trigger the lazy prologue and run the graph once.
        scope.launch()

        # Accessors are observable as raw integer pointers; they must all
        # be non-null while the scope is still active. ``graph`` does not
        # force instantiation, but should still return a live cudaGraph_t.
        assert scope.exec_graph != 0
        assert scope.stream != 0
        assert scope.graph != 0

    ctx.finalize()

    assert np.allclose(X_host, 1.0), f"Expected 1.0, got {X_host[0]}"


def test_launchable_graph_scope_graph_only():
    """``scope.graph`` returns a non-null cudaGraph_t without requiring a
    launch; ``exec_graph`` is never touched so no ``cudaGraphInstantiate``
    is triggered through the scope."""
    n = 128
    X_host = np.zeros(n, dtype=np.float32)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 128
    bpg = (n + tpb - 1) // tpb

    with ctx.launchable_graph_scope() as scope:
        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            add_kernel[bpg, tpb, nb_stream](dX, 1.0)

        # Only touch graph / stream - never exec_graph / launch - to prove
        # the topology accessor works standalone.
        assert scope.graph != 0
        assert scope.stream != 0

    ctx.finalize()

    # No launch happened through the scope, so the data stays at 0.
    assert np.allclose(X_host, 0.0), f"Expected 0.0, got {X_host[0]}"


def test_pop_prologue_shared_basic():
    """Shared launchable graph: launch N times, drop the last ref, verify."""
    n = 256
    N = 5
    X_host = np.zeros(n, dtype=np.float32)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 128
    bpg = (n + tpb - 1) // tpb

    ctx.push()
    with ctx.task(lX.rw()) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        dX = numba_arguments(t)
        add_kernel[bpg, tpb, nb_stream](dX, 1.0)
    g = ctx.pop_prologue_shared()
    assert g.valid
    assert g.stream != 0

    for _ in range(N):
        g.launch()

    # Dropping the sole Python reference triggers the C _free which fires
    # pop_epilogue; the ctx is usable for a fresh push afterwards.
    del g

    with ctx.graph_scope():
        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            scale_kernel[bpg, tpb, nb_stream](dX, 2.0)

    ctx.finalize()

    expected = float(N) * 2.0
    assert np.allclose(X_host, expected), f"Expected {expected}, got {X_host[0]}"


def test_pop_prologue_shared_stored_in_list():
    """Handle kept in a Python list outside the creating function."""
    n = 128
    X_host = np.zeros(n, dtype=np.float32)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 128
    bpg = (n + tpb - 1) // tpb

    cache = []

    def build_step():
        ctx.push()
        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            add_kernel[bpg, tpb, nb_stream](dX, 1.0)
        cache.append(ctx.pop_prologue_shared())

    build_step()
    assert cache[0].valid

    for _ in range(4):
        cache[0].launch()

    # Clearing the list drops the last reference, which fires pop_epilogue.
    cache.clear()

    # Context must be reusable after the shared release.
    with ctx.graph_scope():
        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            add_kernel[bpg, tpb, nb_stream](dX, 10.0)

    ctx.finalize()

    assert np.allclose(X_host, 14.0), f"Expected 14.0, got {X_host[0]}"


def test_pop_prologue_shared_reset_is_idempotent():
    """Double ``reset()`` is safe; accessors raise after reset."""
    n = 64
    X_host = np.zeros(n, dtype=np.float32)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 64
    bpg = (n + tpb - 1) // tpb

    ctx.push()
    with ctx.task(lX.rw()) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        dX = numba_arguments(t)
        add_kernel[bpg, tpb, nb_stream](dX, 1.0)
    g = ctx.pop_prologue_shared()
    g.launch()

    g.reset()
    assert not g.valid
    g.reset()  # idempotent; no-op
    assert not g.valid

    # launch() / accessors must refuse a reset handle.
    import pytest

    with pytest.raises(RuntimeError):
        g.launch()
    with pytest.raises(RuntimeError):
        _ = g.exec_graph
    with pytest.raises(RuntimeError):
        _ = g.stream
    with pytest.raises(RuntimeError):
        _ = g.graph

    ctx.finalize()

    assert np.allclose(X_host, 1.0), f"Expected 1.0, got {X_host[0]}"


def test_pop_prologue_shared_context_manager():
    """``with`` shorthand: __exit__ resets the handle."""
    n = 64
    X_host = np.zeros(n, dtype=np.float32)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 64
    bpg = (n + tpb - 1) // tpb

    ctx.push()
    with ctx.task(lX.rw()) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        dX = numba_arguments(t)
        add_kernel[bpg, tpb, nb_stream](dX, 1.0)
    with ctx.pop_prologue_shared() as g:
        for _ in range(3):
            g.launch()
        assert g.valid
    # After the with-block the handle is reset.
    assert not g.valid

    ctx.finalize()

    assert np.allclose(X_host, 3.0), f"Expected 3.0, got {X_host[0]}"


if __name__ == "__main__":
    test_launchable_graph_scope_relaunch()
    test_launchable_graph_scope_zero_launches()
    test_launchable_graph_scope_exec_and_stream()
    test_launchable_graph_scope_graph_only()
    test_pop_prologue_shared_basic()
    test_pop_prologue_shared_stored_in_list()
    test_pop_prologue_shared_reset_is_idempotent()
    test_pop_prologue_shared_context_manager()
    print("All launchable_graph_scope tests passed!")
