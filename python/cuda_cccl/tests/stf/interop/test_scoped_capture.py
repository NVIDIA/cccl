# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Smoke test: STF local context inside a Warp ``ScopedCapture``.

Exercises the exact configuration the C++ test ``legacy_to_stf_in_capture.cu``
validates, but through the Python/Warp surface:

    1. Create a CUDA stream, wrap it as a ``wp.Stream``.
    2. Open a ``wp.ScopedCapture(capture_mode=wp.CaptureMode.Relaxed)`` on
       that stream. ``Relaxed`` is needed because STF's first-context init
       (``cudaFree(0)`` in ``backend_ctx::impl`` and the ``machine::instance()``
       Meyers singleton) performs capture-unsafe CUDA runtime calls that the
       default ``ThreadLocal`` mode would reject.
    3. Create a local ``stf.context(stream=s)`` *without* an explicit
       ``async_resources_handle`` -- this is the supported in-capture config.
    4. Submit a small fork-join + tail DAG via tokens --
       ``fill_a`` and ``fill_b`` run in parallel (no input deps), ``add``
       joins them by reading both, and ``scale`` chains a single linear
       step on the result. Each task launches a Warp kernel on the stream
       that STF hands out.
    5. ``ctx.finalize()`` while still inside the capture.
    6. Close the capture, launch the instantiated graph, check the host-side
       result.

The STF-side fix (``acquire_release.cuh`` merging ``start_events`` for
input-less tasks + ``stream_ctx``'s capture-safety assertion) is what lets
step 5 complete without a ``cudaErrorStreamCaptureIsolation``.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
from cuda.bindings import runtime as cudart  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402

wp = pytest.importorskip("warp")

N = 1 << 12


@wp.kernel
def fill_kernel(arr: wp.array(dtype=wp.int32), value: wp.int32):
    i = wp.tid()
    if i >= arr.shape[0]:
        return
    arr[i] = value


@wp.kernel
def add_kernel(
    out: wp.array(dtype=wp.int32),
    a: wp.array(dtype=wp.int32),
    b: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if i >= out.shape[0]:
        return
    out[i] = a[i] + b[i]


@wp.kernel
def scale_kernel(arr: wp.array(dtype=wp.int32), factor: wp.int32):
    i = wp.tid()
    if i >= arr.shape[0]:
        return
    arr[i] = arr[i] * factor


def _check_cuda(err) -> None:
    if isinstance(err, tuple):
        err = err[0]
    if int(err) != 0:
        raise RuntimeError(f"cudart error {int(err)}")


def run_fork_join_in_capture_pure_warp() -> np.ndarray:
    """Reference: same fork-join + tail DAG, pure Warp, no STF inside the
    capture.

    Kernels are launched sequentially on the caller stream (no parallel
    branches, no tokens, no ``stf.context``). Used as a numerical baseline
    for :func:`run_fork_join_in_capture_relaxed`.
    """
    wp.init()
    device = wp.get_device("cuda:0")

    err, s_raw = cudart.cudaStreamCreate()
    _check_cuda(err)
    caller_wp = wp.Stream(device, cuda_stream=int(s_raw))

    a = wp.zeros(N, dtype=wp.int32, device=device)
    b = wp.zeros(N, dtype=wp.int32, device=device)
    c = wp.zeros(N, dtype=wp.int32, device=device)

    with wp.ScopedCapture(device=device, stream=caller_wp) as capture:
        wp.launch(fill_kernel, dim=N, inputs=[a, 3], device=device, stream=caller_wp)
        wp.launch(fill_kernel, dim=N, inputs=[b, 4], device=device, stream=caller_wp)
        wp.launch(add_kernel, dim=N, inputs=[c, a, b], device=device, stream=caller_wp)
        wp.launch(scale_kernel, dim=N, inputs=[c, 2], device=device, stream=caller_wp)

    wp.capture_launch(capture.graph, stream=caller_wp)
    wp.synchronize_stream(caller_wp)

    h_c = c.numpy()
    _check_cuda(cudart.cudaStreamDestroy(s_raw))
    return h_c


def run_fork_join_in_capture_relaxed() -> np.ndarray:
    """Same fork-join + tail DAG as
    :func:`run_fork_join_in_capture_pure_warp`, but the body of the capture
    is expressed as an STF token-DAG: ``fill_a`` || ``fill_b`` (independent
    writers) -> ``add`` (reads both) -> ``scale`` (rw on the join result).

    ``wp.ScopedCapture`` is opened in ``cudaStreamCaptureModeRelaxed`` via
    the ``capture_mode`` kwarg so that STF's first-context init (``cudaFree(0)``
    in ``backend_ctx::impl`` and device / peer-access enumeration inside
    ``machine::instance()``) is tolerated by the driver and does not poison
    the capture.
    """
    wp.init()
    device = wp.get_device("cuda:0")

    err, s_raw = cudart.cudaStreamCreate()
    _check_cuda(err)
    caller_wp = wp.Stream(device, cuda_stream=int(s_raw))

    a = wp.zeros(N, dtype=wp.int32, device=device)
    b = wp.zeros(N, dtype=wp.int32, device=device)
    c = wp.zeros(N, dtype=wp.int32, device=device)

    with wp.ScopedCapture(
        device=device, stream=caller_wp, capture_mode=wp.CaptureMode.Relaxed
    ) as capture:
        ctx = stf.context(stream=int(s_raw))

        tok_a = ctx.token()
        tok_b = ctx.token()
        tok_c = ctx.token()

        with ctx.task(tok_a.write()) as t:
            s = wp.Stream(device, cuda_stream=int(t.stream_ptr()))
            wp.launch(fill_kernel, dim=N, inputs=[a, 3], device=device, stream=s)

        with ctx.task(tok_b.write()) as t:
            s = wp.Stream(device, cuda_stream=int(t.stream_ptr()))
            wp.launch(fill_kernel, dim=N, inputs=[b, 4], device=device, stream=s)

        with ctx.task(tok_a.read(), tok_b.read(), tok_c.write()) as t:
            s = wp.Stream(device, cuda_stream=int(t.stream_ptr()))
            wp.launch(add_kernel, dim=N, inputs=[c, a, b], device=device, stream=s)

        with ctx.task(tok_c.rw()) as t:
            s = wp.Stream(device, cuda_stream=int(t.stream_ptr()))
            wp.launch(scale_kernel, dim=N, inputs=[c, 2], device=device, stream=s)

        ctx.finalize()

    wp.capture_launch(capture.graph, stream=caller_wp)
    wp.synchronize_stream(caller_wp)

    h_c = c.numpy()
    _check_cuda(cudart.cudaStreamDestroy(s_raw))
    return h_c


def test_fork_join_inside_warp_scoped_capture_pure_warp() -> None:
    h_c = run_fork_join_in_capture_pure_warp()
    expected = (3 + 4) * 2
    assert np.all(h_c == expected), (
        f"pure-Warp fork-join+tail mismatch: got unique values "
        f"{np.unique(h_c)}, expected all == {expected}"
    )


def test_stf_local_ctx_inside_relaxed_scoped_capture() -> None:
    h_c = run_fork_join_in_capture_relaxed()
    expected = (3 + 4) * 2
    assert np.all(h_c == expected), (
        f"relaxed-capture fork-join+tail mismatch: got unique values "
        f"{np.unique(h_c)}, expected all == {expected}"
    )


if __name__ == "__main__":
    test_fork_join_inside_warp_scoped_capture_pure_warp()
    print("pure-warp                                  : OK")
    test_stf_local_ctx_inside_relaxed_scoped_capture()
    print("stf-in-capture (Relaxed)                   : OK")
