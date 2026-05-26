# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for a single ``cuda.stf._experimental`` DAG that mixes Warp tasks (via
``warp.stf_experimental.task``) and PyTorch tasks (via ``pytorch_task``).

Three properties are validated:

1. ``test_warp_pytorch_pipeline``
   Buffer round-trip Warp -> PyTorch -> Warp on the same logical_data.
   Verifies that ``__cuda_array_interface__``-backed views work in both
   directions and STF orders the steps correctly via the access modes.

2. ``test_warp_pytorch_concurrent_siblings``
   Two siblings on disjoint write sets (one Warp, one PyTorch) plus a
   joining Warp task that reads both. Verifies STF allows the Warp and
   PyTorch siblings to execute on independent streams and the joining
   task sees both outputs after STF's join.

3. ``test_warp_pytorch_mixed_in_captured_dag``
   The mixed pipeline of (1) wrapped in a ``stackable_context``: the
   whole Warp+PyTorch DAG is captured once into a single
   ``cudaGraph_t`` and replayed several times. Checks that the mixed
   DAG is capture-pure.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import warp as wp  # noqa: E402
from pytorch_task import pytorch_task  # noqa: E402
from warp import stf_experimental as wp_stf  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402

N = 1 << 14
FRAMES = 3


# ---------------------------------------------------------------------------
# Warp kernels.
# ---------------------------------------------------------------------------


@wp.kernel
def scale_kernel(arr: wp.array(dtype=wp.float32), factor: wp.float32):
    """In-place: arr[i] *= factor."""
    i = wp.tid()
    if i >= arr.shape[0]:
        return
    arr[i] = arr[i] * factor


@wp.kernel
def fill_kernel(arr: wp.array(dtype=wp.float32), value: wp.float32):
    """arr[i] = value."""
    i = wp.tid()
    if i >= arr.shape[0]:
        return
    arr[i] = value


@wp.kernel
def add_kernel(
    out: wp.array(dtype=wp.float32),
    a: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
):
    """out[i] = a[i] + b[i]."""
    i = wp.tid()
    if i >= out.shape[0]:
        return
    out[i] = a[i] + b[i]


# ---------------------------------------------------------------------------
# Test 1: round-trip on a single shared buffer.
# ---------------------------------------------------------------------------


def test_warp_pytorch_pipeline():
    """Pipeline Warp -> PyTorch -> Warp on the same logical_data.

    Starting from X = 1.0 (host-initialised numpy array):

      step 1  (Warp)    : X *= 3            -> X = 3
      step 2  (PyTorch) : X += 7            -> X = 10
      step 3  (Warp)    : X = X * X         -> X = 100

    All three steps share the same ``lX`` logical_data; STF serialises
    them by access mode (every step is ``rw()``). Final value is read
    back to the host via ``ctx.finalize()`` and verified.
    """
    wp.init()

    X = np.ones(N, dtype=np.float32)

    ctx = stf.context()
    lX = ctx.logical_data(X)

    # Step 1: Warp scales X by 3.
    with wp_stf.task(ctx, lX.rw()) as (s, wX):
        wp.launch(scale_kernel, dim=N, inputs=[wX, 3.0], stream=s)

    # Step 2: PyTorch adds 7 in place.
    with pytorch_task(ctx, lX.rw()) as (tX,):
        tX.add_(7.0)

    # Step 3: Warp squares X (X = X * X via two writes through scratch).
    # Implemented as in-place: read X, multiply by X (i.e. x*x = x^2).
    with wp_stf.task(ctx, lX.rw()) as (s, wX):
        # arr *= arr -> implement as a custom kernel inline via a
        # second kernel that does arr[i] = arr[i] * arr[i].
        wp.launch(square_kernel, dim=N, inputs=[wX], stream=s)

    ctx.finalize()

    assert np.allclose(X, 100.0), (
        f"pipeline result mismatch: expected 100.0 everywhere, got "
        f"unique values {np.unique(X).tolist()}"
    )


@wp.kernel
def square_kernel(arr: wp.array(dtype=wp.float32)):
    """In-place: arr[i] = arr[i] * arr[i]."""
    i = wp.tid()
    if i >= arr.shape[0]:
        return
    v = arr[i]
    arr[i] = v * v


# ---------------------------------------------------------------------------
# Test 2: concurrent siblings with disjoint write sets.
# ---------------------------------------------------------------------------


def test_warp_pytorch_concurrent_siblings():
    """Two siblings on independent buffers + one joining task.

    DAG::

                +-- Warp     fill A = 3.0  --+
                |                            |
        ctx ----+                            +---> Warp  C = A + B
                |                            |
                +-- PyTorch  fill B = 5.0  --+

    The two filler tasks share no logical data, so STF schedules them
    on independent streams. The joining task reads both and writes C;
    its body sees both fills already applied (validates the join).
    """
    wp.init()

    A = np.zeros(N, dtype=np.float32)
    B = np.zeros(N, dtype=np.float32)
    C = np.zeros(N, dtype=np.float32)

    ctx = stf.context()
    lA = ctx.logical_data(A)
    lB = ctx.logical_data(B)
    lC = ctx.logical_data(C)

    # Sibling 1 (Warp): A := 3.0.
    with wp_stf.task(ctx, lA.write()) as (s, wA):
        wp.launch(fill_kernel, dim=N, inputs=[wA, 3.0], stream=s)

    # Sibling 2 (PyTorch): B := 5.0. No shared dep with Sibling 1, so
    # STF is free to overlap the two.
    with pytorch_task(ctx, lB.write()) as (tB,):
        tB.fill_(5.0)

    # Join (Warp): C := A + B. Reads both, writes C.
    with wp_stf.task(ctx, lA.read(), lB.read(), lC.write()) as (s, wA, wB, wC):
        wp.launch(add_kernel, dim=N, inputs=[wC, wA, wB], stream=s)

    ctx.finalize()

    assert np.allclose(A, 3.0), f"A mismatch: unique={np.unique(A).tolist()}"
    assert np.allclose(B, 5.0), f"B mismatch: unique={np.unique(B).tolist()}"
    assert np.allclose(C, 8.0), f"C mismatch: unique={np.unique(C).tolist()}"


# ---------------------------------------------------------------------------
# Test 3: mixed DAG captured into one cudaGraph_t and replayed.
# ---------------------------------------------------------------------------


def test_warp_pytorch_mixed_in_captured_dag():
    """The mixed pipeline of test 1, but wrapped in a stackable_context
    so the whole Warp+PyTorch DAG is captured into a single
    ``cudaGraph_t`` and replayed FRAMES times.

    Per-frame transformation, starting from X = 1.0:

        X *= 3      (Warp)         X = 3
        X += 7      (PyTorch)      X = 10
        X = X * X   (Warp)         X = 100

    Then the next replay starts from X = 100 and applies the same
    sequence again. Reference computed below in pure numpy.
    """
    wp.init()

    X = np.ones(N, dtype=np.float32)

    graph = stf.task_graph()
    outer_ctx = graph.context
    lX = outer_ctx.logical_data(X)

    with graph:
        with wp_stf.task(outer_ctx, lX.rw()) as (s, wX):
            wp.launch(scale_kernel, dim=N, inputs=[wX, 3.0], stream=s)

        with pytorch_task(outer_ctx, lX.rw()) as (tX,):
            tX.add_(7.0)

        with wp_stf.task(outer_ctx, lX.rw()) as (s, wX):
            wp.launch(square_kernel, dim=N, inputs=[wX], stream=s)

    for _ in range(FRAMES):
        graph.launch()

    graph.reset()
    graph.finalize()

    # Reference: apply the same closed-form per-frame map FRAMES times.
    ref = np.ones(N, dtype=np.float32)
    for _ in range(FRAMES):
        ref *= 3.0
        ref += 7.0
        ref = ref * ref

    assert np.allclose(X, ref), (
        f"captured-replay result mismatch: "
        f"unique X = {np.unique(X).tolist()}, "
        f"unique ref = {np.unique(ref).tolist()}"
    )


# ---------------------------------------------------------------------------
# CLI runner.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    test_warp_pytorch_pipeline()
    test_warp_pytorch_concurrent_siblings()
    test_warp_pytorch_mixed_in_captured_dag()
