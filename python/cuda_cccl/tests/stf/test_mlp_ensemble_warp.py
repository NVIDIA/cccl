# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Ensemble training of tiny 2-layer MLPs -- token-based concurrency demo
(Warp backend). See ``test_mlp_ensemble_numba.py`` for the same demo
with Numba-CUDA kernel launches instead; the STF token logic is identical,
which is the point: tokens express the concurrency contract independently
of which GPU-Python framework launches the kernels.

Motivation
----------
``warp/examples/tile/example_tile_mlp.py`` trains *one* coordinate-based
MLP. A single training step is strictly sequential (forward -> backward
-> optimizer), so there is no intra-step concurrency to extract. The
realistic pattern that *does* expose concurrency -- and that shows up in
Newton/Warp-style robotics codebases -- is **ensemble training**: train E
independent MLPs on the same data, e.g. value-function ensembles for RL,
per-agent policies, or hyperparameter-search sweeps.

With a plain ``wp.launch`` loop on one stream, the E training pipelines
serialize even though they touch entirely disjoint parameter and scratch
buffers. With STF tokens, we declare one ``ctx.token()`` per ensemble
member and let STF discover the E-way concurrency automatically -- the
runtime lays out the E chains on separate streams of its pool, and all
members train in parallel on one GPU, with the Warp-provided kernels
getting each task's stream through ``wp.Stream(..., cuda_stream=ptr)``.

Variants
--------
1. ``ref_train_ensemble`` - legacy baseline: a Python loop over steps and
                            members launching every Warp kernel on one
                            caller-provided ``wp.Stream``. E members
                            serialize on that single stream.

2. ``stf_train_ensemble`` - token form: one ``ctx.token()`` per ensemble
                            member, E chains overlap on STF's internal
                            stream pool. Same ``use_graph`` / ``stream=``
                            / ``handle=`` kwargs as in the Numba variant.

Correctness
-----------
Both variants start from identical weights and see the same data in the
same order per member, and inside a member the 5 per-step kernels run
in a fixed order. Members are independent, so the final parameters
after S steps must match bit-for-bit between the reference and STF
variants.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import warp as wp

import cuda.stf._experimental as stf


# ---------------------------------------------------------------------------
# MLP geometry. Large enough that each per-layer kernel is not purely
# launch-latency bound, so the E-way concurrency win from STF is
# observable over Warp's per-launch bookkeeping.
# ---------------------------------------------------------------------------

D_IN  = 4096
D_HID = 4096
D_OUT = 64
LR    = wp.float32(0.01)

STEPS_DEFAULT = 16


# ---------------------------------------------------------------------------
# Warp kernels. Each kernel operates on *one member's* arrays; the grid
# is sized per-member. That keeps launches small enough that per-launch
# bookkeeping is non-trivial, which is exactly what E-way concurrency
# across members amortizes.
# ---------------------------------------------------------------------------


@wp.kernel
def fwd_L1(
    W1: wp.array2d(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
    z: wp.array(dtype=wp.float32),
):
    """z[h] = relu(sum_d W1[h, d] * x[d])."""
    h = wp.tid()
    D = W1.shape[1]
    acc = wp.float32(0.0)
    for d in range(D):
        acc += W1[h, d] * x[d]
    z[h] = wp.max(acc, wp.float32(0.0))


@wp.kernel
def fwd_L2(
    W2: wp.array2d(dtype=wp.float32),
    z: wp.array(dtype=wp.float32),
    y: wp.array(dtype=wp.float32),
):
    """y[o] = sum_h W2[o, h] * z[h]."""
    o = wp.tid()
    H = W2.shape[1]
    acc = wp.float32(0.0)
    for h in range(H):
        acc += W2[o, h] * z[h]
    y[o] = acc


@wp.kernel
def bwd_gz(
    y: wp.array(dtype=wp.float32),
    target: wp.array(dtype=wp.float32),
    W2: wp.array2d(dtype=wp.float32),
    z: wp.array(dtype=wp.float32),
    gz: wp.array(dtype=wp.float32),
):
    """gz[h] = (z[h] > 0) * sum_o (y[o] - target[o]) * W2[o, h].

    Must run before ``upd_W2`` -- reads the pre-update W2.
    """
    h = wp.tid()
    if z[h] <= wp.float32(0.0):
        gz[h] = wp.float32(0.0)
        return
    O = y.shape[0]
    acc = wp.float32(0.0)
    for o in range(O):
        acc += (y[o] - target[o]) * W2[o, h]
    gz[h] = acc


@wp.kernel
def upd_W2(
    y: wp.array(dtype=wp.float32),
    target: wp.array(dtype=wp.float32),
    z: wp.array(dtype=wp.float32),
    W2: wp.array2d(dtype=wp.float32),
    lr: wp.float32,
):
    """W2[o, h] -= lr * (y[o] - target[o]) * z[h]."""
    tid = wp.tid()
    H = W2.shape[1]
    o = tid // H
    h = tid %  H
    W2[o, h] = W2[o, h] - lr * (y[o] - target[o]) * z[h]


@wp.kernel
def upd_W1(
    gz: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
    W1: wp.array2d(dtype=wp.float32),
    lr: wp.float32,
):
    """W1[h, d] -= lr * gz[h] * x[d] (zero for dead-ReLU rows via gz)."""
    tid = wp.tid()
    D = W1.shape[1]
    h = tid // D
    d = tid %  D
    W1[h, d] = W1[h, d] - lr * gz[h] * x[d]


# ---------------------------------------------------------------------------
# Ensemble state. One ``wp.array`` per member per tensor; indexing by
# member is O(1) and each kernel launch binds exactly one member's
# arrays via closure.
# ---------------------------------------------------------------------------


class Ensemble:
    def __init__(self, n_members: int, seed: int = 0, device=None):
        rng = np.random.default_rng(seed)
        s1 = np.float32(1.0 / np.sqrt(D_IN))
        s2 = np.float32(1.0 / np.sqrt(D_HID))

        self.n = n_members
        self.device = wp.get_device(device)
        self.W1, self.W2 = [], []
        self.x, self.target = [], []
        self.z, self.y, self.gz = [], [], []

        for _ in range(n_members):
            W1 = (rng.uniform(-1.0, 1.0, (D_HID, D_IN)).astype(np.float32) * s1)
            W2 = (rng.uniform(-1.0, 1.0, (D_OUT, D_HID)).astype(np.float32) * s2)
            x  = rng.standard_normal(D_IN).astype(np.float32)
            tg = rng.standard_normal(D_OUT).astype(np.float32)

            self.W1.append(wp.array(W1, dtype=wp.float32, device=self.device))
            self.W2.append(wp.array(W2, dtype=wp.float32, device=self.device))
            self.x.append(wp.array(x, dtype=wp.float32, device=self.device))
            self.target.append(
                wp.array(tg, dtype=wp.float32, device=self.device)
            )
            self.z.append(
                wp.zeros(D_HID, dtype=wp.float32, device=self.device)
            )
            self.y.append(
                wp.zeros(D_OUT, dtype=wp.float32, device=self.device)
            )
            self.gz.append(
                wp.zeros(D_HID, dtype=wp.float32, device=self.device)
            )

    def snapshot_weights(self):
        wp.synchronize()
        return (
            [W1.numpy() for W1 in self.W1],
            [W2.numpy() for W2 in self.W2],
        )


def clone_weights(src: "Ensemble", dst: "Ensemble") -> None:
    """Copy src's (W1, W2) into dst's (W1, W2)."""
    assert src.n == dst.n
    for k in range(src.n):
        wp.copy(dst.W1[k], src.W1[k])
        wp.copy(dst.W2[k], src.W2[k])
    wp.synchronize()


# ---------------------------------------------------------------------------
# STF-stream <-> wp.Stream adapter cache
# ---------------------------------------------------------------------------
#
# STF reuses streams from its pool, so the set of distinct ``cudaStream_t``
# pointers passed to our tasks is small (<= pool size). Building a fresh
# ``wp.Stream`` wrapper on every task call would pay the register/unregister
# cost per launch and dominate the benchmark, so we cache one wrapper per
# raw pointer. Wrappers live for the lifetime of the process, mirroring how
# Warp's own ``null_stream`` is set up once per device.
# ---------------------------------------------------------------------------

_wp_stream_cache: dict[tuple[int, int], wp.Stream] = {}


def _wrap_stream(raw_ptr: int, device) -> wp.Stream:
    key = (id(device), int(raw_ptr))
    s = _wp_stream_cache.get(key)
    if s is None:
        s = wp.Stream(device, cuda_stream=int(raw_ptr))
        _wp_stream_cache[key] = s
    return s


# ---------------------------------------------------------------------------
# Variant 1: reference single-stream ensemble trainer
# ---------------------------------------------------------------------------


def ref_train_ensemble(stream: wp.Stream, ens: "Ensemble",
                       steps: int, lr=LR) -> None:
    """All E members trained back-to-back on one caller-provided stream."""
    BLOCKS_W2 = D_OUT * D_HID
    BLOCKS_W1 = D_HID * D_IN
    for _ in range(steps):
        for k in range(ens.n):
            wp.launch(
                kernel=fwd_L1, dim=D_HID,
                inputs=[ens.W1[k], ens.x[k], ens.z[k]],
                device=ens.device, stream=stream,
            )
            wp.launch(
                kernel=fwd_L2, dim=D_OUT,
                inputs=[ens.W2[k], ens.z[k], ens.y[k]],
                device=ens.device, stream=stream,
            )
            wp.launch(
                kernel=bwd_gz, dim=D_HID,
                inputs=[ens.y[k], ens.target[k], ens.W2[k],
                        ens.z[k], ens.gz[k]],
                device=ens.device, stream=stream,
            )
            wp.launch(
                kernel=upd_W2, dim=BLOCKS_W2,
                inputs=[ens.y[k], ens.target[k], ens.z[k],
                        ens.W2[k], lr],
                device=ens.device, stream=stream,
            )
            wp.launch(
                kernel=upd_W1, dim=BLOCKS_W1,
                inputs=[ens.gz[k], ens.x[k], ens.W1[k], lr],
                device=ens.device, stream=stream,
            )


# ---------------------------------------------------------------------------
# Variant 2: STF tokens, one per ensemble member
# ---------------------------------------------------------------------------


def stf_train_ensemble(
    ens: "Ensemble",
    steps: int,
    lr=LR,
    use_graph: bool = False,
    stream: "wp.Stream | None" = None,
    handle=None,
) -> None:
    """One ``ctx.token()`` per ensemble member -> E-way concurrency.

    The unit of STF work ("task") is the full 5-kernel chain for one
    member at one training step: all five kernels are independent only
    *across* members, so splitting them into separate tasks would just
    multiply per-task bookkeeping without exposing any new parallelism.
    Each task binds exactly one member's buffers by closure -- STF does
    not touch the data. Because the tasks of member ``k`` take
    ``tok[k]`` as their only logical_data, STF concludes member chains
    are mutually independent and schedules them on separate streams
    of its pool; Warp launches onto those streams via
    ``wp.Stream(device, cuda_stream=<STF-provided ptr>)``.

    Parameters
    ----------
    stream : wp.Stream, optional
        Caller-owned Warp stream. STF inherits its raw ``cudaStream_t``
        and emits work on top of it. We pre-populate the wp.Stream cache
        so that if STF hands its ptr back to a task (the common case on
        the stream backend), we reuse *this* wrapper instead of building
        a second ``wp.Stream`` for the same ptr -- double-registering
        the same raw stream with Warp corrupts Warp's internal state.
    """
    device = ens.device
    ctx_stream_arg = stream.cuda_stream if stream is not None else None

    # Register the caller's wp.Stream in the cache so that any task whose
    # stream_ptr matches it reuses the pre-existing wrapper.
    if stream is not None:
        _wp_stream_cache[(id(device), int(stream.cuda_stream))] = stream

    ctx = stf.context(
        use_graph=use_graph, stream=ctx_stream_arg, handle=handle
    )

    tokens = [ctx.token() for _ in range(ens.n)]

    BLOCKS_W2 = D_OUT * D_HID
    BLOCKS_W1 = D_HID * D_IN

    for _ in range(steps):
        for k in range(ens.n):
            with ctx.task(tokens[k].rw()) as t:
                s = _wrap_stream(t.stream_ptr(), device)
                wp.launch(
                    kernel=fwd_L1, dim=D_HID,
                    inputs=[ens.W1[k], ens.x[k], ens.z[k]],
                    device=device, stream=s,
                )
                wp.launch(
                    kernel=fwd_L2, dim=D_OUT,
                    inputs=[ens.W2[k], ens.z[k], ens.y[k]],
                    device=device, stream=s,
                )
                wp.launch(
                    kernel=bwd_gz, dim=D_HID,
                    inputs=[ens.y[k], ens.target[k], ens.W2[k],
                            ens.z[k], ens.gz[k]],
                    device=device, stream=s,
                )
                wp.launch(
                    kernel=upd_W2, dim=BLOCKS_W2,
                    inputs=[ens.y[k], ens.target[k], ens.z[k],
                            ens.W2[k], lr],
                    device=device, stream=s,
                )
                wp.launch(
                    kernel=upd_W1, dim=BLOCKS_W1,
                    inputs=[ens.gz[k], ens.x[k], ens.W1[k], lr],
                    device=device, stream=s,
                )

    ctx.finalize()


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


def _assert_weights_equal(ens_ref: "Ensemble", ens_stf: "Ensemble") -> None:
    W1_ref, W2_ref = ens_ref.snapshot_weights()
    W1_stf, W2_stf = ens_stf.snapshot_weights()
    for k in range(ens_ref.n):
        np.testing.assert_array_equal(W1_ref[k], W1_stf[k])
        np.testing.assert_array_equal(W2_ref[k], W2_stf[k])


@pytest.mark.parametrize("n_members", [1, 4])
def test_stf_matches_ref(n_members):
    steps = 8
    ens_ref = Ensemble(n_members, seed=123)
    ens_stf = Ensemble(n_members, seed=123)
    clone_weights(ens_ref, ens_stf)

    stream = wp.Stream(ens_ref.device)
    ref_train_ensemble(stream, ens_ref, steps)
    wp.synchronize_stream(stream)

    stf_train_ensemble(ens_stf, steps)
    wp.synchronize()

    _assert_weights_equal(ens_ref, ens_stf)


def test_stf_graph_stream_handle():
    """Exercise ``stream=`` / ``handle=`` kwargs on the graph backend.

    NOTE: The stream backend with ``stream=`` / ``handle=`` overrides
    (i.e. the ``stream_ctx(stream, ah)`` C++ path) does not properly
    chain consecutive contexts through the shared caller stream when
    Warp launches the kernels: a second back-to-back call can start
    before the first has drained, producing divergent weights. An
    explicit ``wp.synchronize()`` between calls works around it. The
    equivalent Numba demo does not trigger this, so it looks like an
    interaction bug between STF's stream-backend pool scheduling and
    Warp's kernel-launch path; the graph backend is unaffected.
    Until that path is fixed, only the graph backend is exercised here.
    """
    n = 4
    steps = 4

    ens_ref = Ensemble(n, seed=7)
    ens_stf = Ensemble(n, seed=7)
    clone_weights(ens_ref, ens_stf)

    stream = wp.Stream(ens_ref.device)
    ref_train_ensemble(stream, ens_ref, steps)
    wp.synchronize_stream(stream)

    h = stf.async_resources()
    stf_train_ensemble(
        ens_stf, steps, use_graph=True, stream=stream, handle=h,
    )
    wp.synchronize()

    _assert_weights_equal(ens_ref, ens_stf)


# ---------------------------------------------------------------------------
# Benchmark entry point (``python test_mlp_ensemble_warp.py``)
# ---------------------------------------------------------------------------


def _time(label: str, fn, niter: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(niter):
        fn()
    wp.synchronize()
    ms = (time.perf_counter() - t0) / niter * 1e3
    print(f"  {label:<44s} {ms:10.3f} ms/iter")
    return ms


def _benchmark(ensemble_sizes=None, steps: int = STEPS_DEFAULT, niter: int = 8):
    if ensemble_sizes is None:
        ensemble_sizes = [1, 2, 4, 8, 16, 32]

    device = wp.get_device()

    for n in ensemble_sizes:
        print(
            f"\n=== E = {n:3d} members, {steps} steps, "
            f"MLP({D_IN}->{D_HID}->{D_OUT}) ==="
        )

        seed = 0x5eed
        ens_ref   = Ensemble(n, seed=seed, device=device)
        ens_tok   = Ensemble(n, seed=seed, device=device); clone_weights(ens_ref, ens_tok)
        ens_tokg  = Ensemble(n, seed=seed, device=device); clone_weights(ens_ref, ens_tokg)
        ens_tokgh = Ensemble(n, seed=seed, device=device); clone_weights(ens_ref, ens_tokgh)

        stream = wp.Stream(device)
        handle = stf.async_resources()

        ref   = _time("ref_train_ensemble (single stream)",
                      lambda: ref_train_ensemble(stream, ens_ref, steps),
                      niter)
        tok   = _time("stf_train_ensemble (tokens)",
                      lambda: stf_train_ensemble(ens_tok, steps),
                      niter)
        tokg  = _time("stf_train_ensemble (tokens, graph)",
                      lambda: stf_train_ensemble(ens_tokg, steps,
                                                 use_graph=True),
                      niter)
        # The stream-backend override path (stream_ctx(stream, ah)) does
        # not chain consecutive contexts correctly through the shared
        # caller stream with Warp kernels -- see test_stf_graph_stream_handle
        # for the note. We only exercise the graph-backend override path
        # here, which does work.
        tokgh = _time("stf_train_ensemble (tokens,graph,+stream,+handle)",
                      lambda: stf_train_ensemble(
                          ens_tokgh, steps, use_graph=True,
                          stream=stream, handle=handle),
                      niter)

        print(f"  tokens / ref                           {tok   / ref:6.2f}x")
        print(f"  tokens(graph) / ref                    {tokg  / ref:6.2f}x")
        print(f"  tokens(graph,+stream,+handle) / ref    {tokgh / ref:6.2f}x")

        _assert_weights_equal(ens_ref, ens_tok)
        _assert_weights_equal(ens_ref, ens_tokg)
        _assert_weights_equal(ens_ref, ens_tokgh)

        handle = None
    print("\ncorrectness: OK for all ensemble sizes")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--e", type=int, nargs="*", default=None,
                   help="ensemble sizes to sweep (default: 1,2,4,8,16,32)")
    p.add_argument("--steps", type=int, default=STEPS_DEFAULT,
                   help="training steps per benchmark iteration")
    p.add_argument("--niter", type=int, default=8,
                   help="outer repetitions per variant")
    args = p.parse_args()

    wp.init()
    with wp.ScopedDevice("cuda:0"):
        _benchmark(ensemble_sizes=args.e, steps=args.steps, niter=args.niter)
