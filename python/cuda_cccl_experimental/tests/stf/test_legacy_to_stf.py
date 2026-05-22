# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python port of ``cudax/test/stf/local_stf/legacy_to_stf.cu``.

Walks through the same gradual-adoption story as the C++ example:

1. ``ref_lib_call``   - the "legacy" single-stream version with no expressed
                        concurrency: ``init_a``, ``init_b``, ``axpy``,
                        ``empty_kernel`` launched back-to-back on one CUDA
                        stream, so ``init_a`` and ``init_b`` serialize even
                        though they touch disjoint buffers.

2. ``lib_call``       - same kernels, but the two device buffers are wrapped
                        as STF ``logical_data`` on ``data_place.device(0)``
                        (zero-copy over the caller-provided Numba device
                        arrays). STF discovers the DAG from the access modes
                        (``init_a`` writes A, ``init_b`` writes B,
                        ``axpy`` reads A / writes B, ``empty_kernel`` has no
                        dependencies) and can overlap the two ``init_*``
                        kernels on different streams of its pool.

3. ``lib_call_token`` - the slide-5 token form. The buffers are *not* handed
                        to STF; each logical_data is a ``ctx.token()`` used
                        purely to express ordering. Kernels still receive
                        the raw device arrays by closure, exactly like the
                        C++ ``lib_call_token`` variant.

4. ``lib_call_token(..., use_graph=True)``
                      - same token-based DAG as variant 3, but built on the
                        CUDA-graph backend (``stf.context(use_graph=True)``,
                        equivalent to ``graph_ctx ctx`` in C++). STF captures
                        the four tasks into a CUDA graph instead of emitting
                        them onto streams, which removes per-task launch
                        overhead at the cost of one graph instantiation.

5. ``lib_call_token(..., stream=..., handle=...)``
                      - same token-based DAG as variants 3/4, but the
                        context is created with a caller-owned
                        ``cudaStream_t`` *and* a shared
                        ``stf.async_resources`` handle. This mirrors the
                        C++ ``stream_ctx ctx(stream, handle)`` /
                        ``graph_ctx ctx(stream, handle)`` idiom used by
                        ``lib_call_with_handle`` in legacy_to_stf.cu. The
                        stream keeps STF non-blocking w.r.t. the rest of
                        the caller's pipeline; the handle caches the
                        instantiated graph (or stream pools) across calls
                        so the graph backend finally amortizes its
                        per-context construction cost.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

numba = pytest.importorskip("numba")
pytest.importorskip("numba.cuda")
from numba import cuda

import cuda.stf as stf

numba.cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

N = 128 * 1024
NITER = 128


# ---------------------------------------------------------------------------
# Kernels (mirror legacy_to_stf.cu)
# ---------------------------------------------------------------------------


@cuda.jit
def init_a(d_a):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(tid, d_a.size, stride):
        d_a[i] = np.float64(np.sin(np.float64(i)))


@cuda.jit
def init_b(d_b):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(tid, d_b.size, stride):
        d_b[i] = np.float64(np.cos(np.float64(i)))


@cuda.jit
def axpy(alpha, d_a, d_b):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(tid, d_a.size, stride):
        d_b[i] += alpha * d_a[i]


@cuda.jit
def empty_kernel():
    pass


BLOCKS = 128
THREADS = 32
EMPTY_BLOCKS = 16
EMPTY_THREADS = 8


# ---------------------------------------------------------------------------
# Reference closed-form check
# ---------------------------------------------------------------------------


def expected(n: int, alpha: float = 3.0) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    return np.cos(i) + alpha * np.sin(i)


# ---------------------------------------------------------------------------
# Variant 1: reference single-stream version (no STF)
# ---------------------------------------------------------------------------


def ref_lib_call(stream, d_a, d_b):
    """Legacy code: four launches on a single caller-provided stream.

    Nothing tells the runtime that ``init_a`` and ``init_b`` are independent,
    so they serialize on ``stream``.
    """
    init_a[BLOCKS, THREADS, stream](d_a)
    init_b[BLOCKS, THREADS, stream](d_b)
    axpy[BLOCKS, THREADS, stream](3.0, d_a, d_b)
    empty_kernel[EMPTY_BLOCKS, EMPTY_THREADS, stream]()


# ---------------------------------------------------------------------------
# Variant 2: STF with real logical_data over the existing device pointers
# ---------------------------------------------------------------------------


def lib_call(d_a, d_b):
    """STF wrapping the existing device arrays as logical_data.

    Equivalent to ``lib_call`` in legacy_to_stf.cu. The two ``init_*`` tasks
    write *different* logical_data and can therefore run concurrently;
    ``axpy`` reads A and rw()s B, so it serializes after both inits.
    """
    ctx = stf.context()
    device = stf.data_place.device(0)

    l_a = ctx.logical_data(d_a, device, name="A")
    l_b = ctx.logical_data(d_b, device, name="B")

    with ctx.task(l_a.write()) as t:
        s = cuda.external_stream(t.stream_ptr())
        da = cuda.from_cuda_array_interface(t.get_arg_cai(0), owner=None, sync=False)
        init_a[BLOCKS, THREADS, s](da)

    with ctx.task(l_b.write()) as t:
        s = cuda.external_stream(t.stream_ptr())
        db = cuda.from_cuda_array_interface(t.get_arg_cai(0), owner=None, sync=False)
        init_b[BLOCKS, THREADS, s](db)

    with ctx.task(l_a.read(), l_b.rw()) as t:
        s = cuda.external_stream(t.stream_ptr())
        da = cuda.from_cuda_array_interface(t.get_arg_cai(0), owner=None, sync=False)
        db = cuda.from_cuda_array_interface(t.get_arg_cai(1), owner=None, sync=False)
        axpy[BLOCKS, THREADS, s](3.0, da, db)

    with ctx.task() as t:
        s = cuda.external_stream(t.stream_ptr())
        empty_kernel[EMPTY_BLOCKS, EMPTY_THREADS, s]()

    ctx.finalize()


# ---------------------------------------------------------------------------
# Variant 3: STF with tokens (slide 5)
# ---------------------------------------------------------------------------


def lib_call_token(d_a, d_b, use_graph: bool = False, stream=None, handle=None):
    """STF using tokens only: buffers stay under caller ownership.

    Mirrors ``lib_call_token`` in legacy_to_stf.cu. Tokens ``l_a`` / ``l_b``
    carry no data; they just declare the dependency DAG. Kernels receive the
    user-owned ``d_a`` / ``d_b`` directly via closure, so STF never touches
    the buffers -- yet ``init_a`` and ``init_b`` still overlap because
    the tokens live on different logical_data.

    Parameters
    ----------
    use_graph : bool, default False
        Selects the STF backend. ``False`` uses the stream backend (slide-5
        default); ``True`` uses the CUDA-graph backend, equivalent to
        ``graph_ctx ctx;`` in C++ -- the four tasks are captured into a
        CUDA graph and launched together, collapsing per-task launch
        overhead at the cost of one-time graph construction.
    stream : optional
        Caller-owned CUDA stream (any object implementing
        ``__cuda_stream__``, e.g. a ``numba.cuda.stream()``). When provided,
        STF inherits it instead of picking a stream from its internal pool
        -- equivalent to the C++ ``stream_ctx ctx(stream)`` /
        ``graph_ctx ctx(stream)`` constructors.
    handle : stf.async_resources, optional
        Shared resources handle reused across calls. Reusing one handle
        lets the graph backend cache instantiated graphs, and lets the
        stream backend reuse its stream pools -- equivalent to the C++
        ``stream_ctx ctx(stream, handle)`` / ``graph_ctx ctx(stream, handle)``
        overloads used in ``lib_call_with_handle``.
    """
    ctx = stf.context(use_graph=use_graph, stream=stream, handle=handle)

    l_a = ctx.token()
    l_b = ctx.token()

    with ctx.task(l_a.write()) as t:
        s = cuda.external_stream(t.stream_ptr())
        init_a[BLOCKS, THREADS, s](d_a)

    with ctx.task(l_b.write()) as t:
        s = cuda.external_stream(t.stream_ptr())
        init_b[BLOCKS, THREADS, s](d_b)

    with ctx.task(l_a.read(), l_b.rw()) as t:
        s = cuda.external_stream(t.stream_ptr())
        axpy[BLOCKS, THREADS, s](3.0, d_a, d_b)

    with ctx.task() as t:
        s = cuda.external_stream(t.stream_ptr())
        empty_kernel[EMPTY_BLOCKS, EMPTY_THREADS, s]()

    ctx.finalize()


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


@pytest.fixture
def device_buffers():
    d_a = cuda.device_array(N, dtype=np.float64)
    d_b = cuda.device_array(N, dtype=np.float64)
    return d_a, d_b


def _check(d_a, d_b):
    cuda.synchronize()
    np.testing.assert_allclose(d_b.copy_to_host(), expected(N), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        d_a.copy_to_host(),
        np.sin(np.arange(N, dtype=np.float64)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_ref_lib_call(device_buffers):
    d_a, d_b = device_buffers
    stream = cuda.stream()
    ref_lib_call(stream, d_a, d_b)
    stream.synchronize()
    _check(d_a, d_b)


def test_lib_call(device_buffers):
    d_a, d_b = device_buffers
    lib_call(d_a, d_b)
    _check(d_a, d_b)


def test_lib_call_token(device_buffers):
    d_a, d_b = device_buffers
    lib_call_token(d_a, d_b)
    _check(d_a, d_b)


def test_lib_call_token_graph(device_buffers):
    d_a, d_b = device_buffers
    lib_call_token(d_a, d_b, use_graph=True)
    _check(d_a, d_b)


def test_lib_call_token_shared_handle(device_buffers):
    """Reuse a single async_resources + caller stream across two contexts.

    Exercises the ``stream=`` / ``handle=`` kwargs end-to-end: two back-to-back
    calls on the graph backend share a resources handle so the second call
    hits the cached graph, and both calls land on the caller's stream.
    """
    d_a, d_b = device_buffers
    stream = cuda.stream()
    h = stf.async_resources()
    lib_call_token(d_a, d_b, use_graph=True, stream=stream, handle=h)
    lib_call_token(d_a, d_b, use_graph=True, stream=stream, handle=h)
    stream.synchronize()
    _check(d_a, d_b)


# ---------------------------------------------------------------------------
# Benchmark entry point (``python test_legacy_to_stf.py``)
#
# Mirrors the ``nvtx_range`` timed blocks in legacy_to_stf.cu's main().
# Use ``nsys profile -c cudaProfilerApi --capture-range=cudaProfilerApi
# python test_legacy_to_stf.py`` to see the streams laid out on a timeline
# and confirm that ``init_a`` / ``init_b`` overlap in the two STF variants.
# ---------------------------------------------------------------------------


def _time(label: str, fn, niter: int, warmup: int = 4) -> float:
    for _ in range(warmup):
        fn()
    cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(niter):
        fn()
    cuda.synchronize()
    us = (time.perf_counter() - t0) / niter * 1e6
    print(f"  {label:<26s} {us:10.1f} us/iter")
    return us


def _benchmark(sizes=None):
    """Sweep problem size to make the overhead / concurrency crossover visible.

    At small N, Python/Cython overhead per STF task dominates and the STF
    variants look slower than the single-stream baseline. At larger N, each
    kernel is big enough that overlapping ``init_a`` with ``init_b`` on
    separate streams wins back the overhead and yields the slide-5 speedup.
    Expected asymptote is ~1.17x (two of four kernels overlap).
    """
    if sizes is None:
        sizes = [
            128 * 1024,
            1024 * 1024,
            8 * 1024 * 1024,
            32 * 1024 * 1024,
            128 * 1024 * 1024,
        ]
    for n in sizes:
        niter = 128 if n <= 8 * 1024 * 1024 else 32
        d_a = cuda.device_array(n, dtype=np.float64)
        d_b = cuda.device_array(n, dtype=np.float64)
        stream = cuda.stream()
        # One shared handle per size: reused across every _time() iteration
        # so the graph backend caches its instantiated graph across calls,
        # mirroring `lib_call_with_handle` in legacy_to_stf.cu.
        handle = stf.async_resources()
        print(f"\n=== N = {n:>12,}  ({n * 8 / 1e6:.1f} MB/array)  niter={niter} ===")
        ref = _time("ref_lib_call", lambda: ref_lib_call(stream, d_a, d_b), niter)
        ld = _time("lib_call (logical_data)", lambda: lib_call(d_a, d_b), niter)
        tok = _time("lib_call_token", lambda: lib_call_token(d_a, d_b), niter)
        tokg = _time(
            "lib_call_token (graph)",
            lambda: lib_call_token(d_a, d_b, use_graph=True),
            niter,
        )
        tokh = _time(
            "lib_call_token (+stream,+handle)",
            lambda: lib_call_token(d_a, d_b, stream=stream, handle=handle),
            niter,
        )
        tokgh = _time(
            "lib_call_token (graph,+stream,+handle)",
            lambda: lib_call_token(
                d_a, d_b, use_graph=True, stream=stream, handle=handle
            ),
            niter,
        )
        print(f"  token/ref                         {tok / ref:10.2f}x")
        print(f"  token(graph)/ref                  {tokg / ref:10.2f}x")
        print(f"  token(+stream,+handle)/ref        {tokh / ref:10.2f}x")
        print(f"  token(graph,+stream,+handle)/ref  {tokgh / ref:10.2f}x")
        print(f"  logical_data/ref                  {ld / ref:10.2f}x")

        ref_lib_call(stream, d_a, d_b)
        stream.synchronize()
        np.testing.assert_allclose(
            d_b.copy_to_host(), expected(n), rtol=1e-12, atol=1e-12
        )
        # Drop the shared handle *after* we're done using it for this size,
        # and only once all contexts built on top of it have finalized.
        handle = None
    print("\ncorrectness: OK for all sizes")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--n",
        type=int,
        nargs="*",
        default=None,
        help="problem size(s) in elements (default sweeps 128K..128M)",
    )
    args = p.parse_args()
    _benchmark(sizes=args.n)
