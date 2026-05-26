# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Ensemble training of tiny 2-layer MLPs -- token-based concurrency demo
(Numba backend). See ``test_mlp_ensemble_warp.py`` for the same demo
with Warp-kernel launches instead; the STF token logic is identical.

Motivation
----------
``warp/examples/tile/example_tile_mlp.py`` trains *one* coordinate-based MLP
per run. A single training step is strictly sequential (forward -> backward
-> optimizer), so there is no intra-step concurrency to extract. The
realistic pattern that *does* expose concurrency -- and that shows up in
Newton/Warp-style robotics codebases -- is **ensemble training**: train E
independent MLPs on the same data, e.g. value-function ensembles for RL,
per-agent policies, or hyperparameter-search sweeps.

With a plain single-stream Warp/Numba loop, the E training pipelines
serialize on one CUDA stream even though they touch *entirely disjoint*
parameter and scratch buffers. With STF tokens, we declare one
``ctx.token()`` per ensemble member and let STF
discover the E-way concurrency automatically -- the runtime lays out the
E chains on separate streams of its pool, and all members train in
parallel on one GPU.

Variants
--------
1. ``ref_train_ensemble`` - legacy baseline: a Python loop over steps and
                            members, every kernel launched on one
                            caller-provided stream. E members serialize.

2. ``stf_train_ensemble`` - token form: one ``ctx.token()`` per
                            ensemble member, E chains overlap on STF's
                            internal stream pool. Same ``use_graph`` /
                            ``stream=`` / ``handle=`` kwargs as
                            ``lib_call_token`` in ``test_legacy_to_stf.py``.

Correctness
-----------
Both variants start from the same parameters and see the same data in the
same order per member, and inside a member the 5 per-step kernels run in
a fixed order. Members are independent, so the *final* parameters after S
steps must match bit-for-bit between the reference and STF variants.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

numba = pytest.importorskip("numba")
pytest.importorskip("numba.cuda")
from numba import cuda

import cuda.stf._experimental as stf

numba.cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


# ---------------------------------------------------------------------------
# MLP geometry -- small enough to keep per-kernel work modest so the
# per-launch overhead / concurrency-win trade-off is visible.
# ---------------------------------------------------------------------------

D_IN = 4096  # input dim
D_HID = 4096  # hidden dim
D_OUT = 64  # output dim
LR = np.float32(0.01)

# Default training length per benchmark iteration. Larger `steps` amortizes
# the STF context-build cost over more token-scheduled launches, which is
# exactly what a training inner-loop does in practice.
STEPS_DEFAULT = 64


# ---------------------------------------------------------------------------
# Kernels. Each kernel operates on *one member's* arrays; the grid is
# sized per-member. That keeps launches small enough that launch overhead
# is non-trivial, which is exactly what E-way concurrency amortizes.
# ---------------------------------------------------------------------------


@cuda.jit
def fwd_L1(W1, x, z):
    """z[h] = relu(sum_d W1[h, d] * x[d])."""
    h = cuda.grid(1)
    if h >= z.size:
        return
    acc = np.float32(0.0)
    D = x.size
    for d in range(D):
        acc += W1[h, d] * x[d]
    z[h] = acc if acc > np.float32(0.0) else np.float32(0.0)


@cuda.jit
def fwd_L2(W2, z, y):
    """y[o] = sum_h W2[o, h] * z[h]."""
    o = cuda.grid(1)
    if o >= y.size:
        return
    acc = np.float32(0.0)
    H = z.size
    for h in range(H):
        acc += W2[o, h] * z[h]
    y[o] = acc


@cuda.jit
def bwd_gz(y, target, W2, z, gz):
    """gz[h] = (z[h] > 0) * sum_o (y[o] - target[o]) * W2[o, h].

    Must run *before* ``upd_W2`` so that it sees the pre-update W2.
    """
    h = cuda.grid(1)
    if h >= gz.size:
        return
    if z[h] <= np.float32(0.0):
        gz[h] = np.float32(0.0)
        return
    acc = np.float32(0.0)
    O = y.size
    for o in range(O):
        acc += (y[o] - target[o]) * W2[o, h]
    gz[h] = acc


@cuda.jit
def upd_W2(y, target, z, W2, lr):
    """W2[o, h] -= lr * (y[o] - target[o]) * z[h]."""
    tid = cuda.grid(1)
    total = W2.shape[0] * W2.shape[1]
    if tid >= total:
        return
    H = W2.shape[1]
    o = tid // H
    h = tid % H
    W2[o, h] -= lr * (y[o] - target[o]) * z[h]


@cuda.jit
def upd_W1(gz, x, W1, lr):
    """W1[h, d] -= lr * gz[h] * x[d] (zero for dead-ReLU rows via gz)."""
    tid = cuda.grid(1)
    total = W1.shape[0] * W1.shape[1]
    if tid >= total:
        return
    D = W1.shape[1]
    h = tid // D
    d = tid % D
    W1[h, d] -= lr * gz[h] * x[d]


# Launch configurations -- one block each for the small per-layer kernels,
# multiple blocks for the two big update kernels.
THREADS = 128

BLOCKS_H = (D_HID + THREADS - 1) // THREADS  # for fwd_L1 / bwd_gz
BLOCKS_O = (D_OUT + THREADS - 1) // THREADS  # for fwd_L2
BLOCKS_W2 = (D_OUT * D_HID + THREADS - 1) // THREADS  # for upd_W2
BLOCKS_W1 = (D_HID * D_IN + THREADS - 1) // THREADS  # for upd_W1


# ---------------------------------------------------------------------------
# Ensemble state
# ---------------------------------------------------------------------------


class Ensemble:
    """Per-member parameters, inputs, targets and scratch buffers.

    All arrays are contiguous device arrays. We keep a Python list per
    tensor type so indexing by member is O(1) and each kernel launch
    binds exactly one member's arrays via closure -- the same pattern
    used by ``lib_call_token`` in ``test_legacy_to_stf.py``.
    """

    def __init__(self, n_members: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Xavier-ish init for stable training.
        s1 = np.float32(1.0 / np.sqrt(D_IN))
        s2 = np.float32(1.0 / np.sqrt(D_HID))

        self.n = n_members
        self.W1 = []
        self.W2 = []
        self.x = []
        self.target = []
        self.z = []
        self.y = []
        self.gz = []

        for _ in range(n_members):
            W1 = rng.uniform(-1.0, 1.0, (D_HID, D_IN)).astype(np.float32) * s1
            W2 = rng.uniform(-1.0, 1.0, (D_OUT, D_HID)).astype(np.float32) * s2
            x = rng.standard_normal(D_IN).astype(np.float32)
            tg = rng.standard_normal(D_OUT).astype(np.float32)

            self.W1.append(cuda.to_device(W1))
            self.W2.append(cuda.to_device(W2))
            self.x.append(cuda.to_device(x))
            self.target.append(cuda.to_device(tg))
            self.z.append(cuda.device_array(D_HID, dtype=np.float32))
            self.y.append(cuda.device_array(D_OUT, dtype=np.float32))
            self.gz.append(cuda.device_array(D_HID, dtype=np.float32))

    def snapshot_weights(self):
        """Copy (W1, W2) of every member back to host for comparison."""
        cuda.synchronize()
        return (
            [W1.copy_to_host() for W1 in self.W1],
            [W2.copy_to_host() for W2 in self.W2],
        )


def clone_weights(src: "Ensemble", dst: "Ensemble") -> None:
    """Copy src's (W1, W2) into dst's (W1, W2). Used to put two ensembles
    into identical starting states before comparing training trajectories.
    """
    assert src.n == dst.n
    for k in range(src.n):
        dst.W1[k].copy_to_device(src.W1[k])
        dst.W2[k].copy_to_device(src.W2[k])
    cuda.synchronize()


# ---------------------------------------------------------------------------
# Variant 1: reference single-stream ensemble trainer
# ---------------------------------------------------------------------------


def ref_train_ensemble(stream, ens: "Ensemble", steps: int, lr=LR) -> None:
    """All E members trained back-to-back on one caller-provided stream.

    The runtime sees 5 * E launches per step on a single queue with no
    concurrency hint, so the E per-member chains serialize even though
    they touch disjoint buffers.
    """
    for _ in range(steps):
        for k in range(ens.n):
            fwd_L1[BLOCKS_H, THREADS, stream](ens.W1[k], ens.x[k], ens.z[k])
            fwd_L2[BLOCKS_O, THREADS, stream](ens.W2[k], ens.z[k], ens.y[k])
            bwd_gz[BLOCKS_H, THREADS, stream](
                ens.y[k], ens.target[k], ens.W2[k], ens.z[k], ens.gz[k]
            )
            upd_W2[BLOCKS_W2, THREADS, stream](
                ens.y[k], ens.target[k], ens.z[k], ens.W2[k], lr
            )
            upd_W1[BLOCKS_W1, THREADS, stream](ens.gz[k], ens.x[k], ens.W1[k], lr)


# ---------------------------------------------------------------------------
# Variant 2: STF tokens, one per ensemble member
# ---------------------------------------------------------------------------


def stf_train_ensemble(
    ens: "Ensemble",
    steps: int,
    lr=LR,
    use_graph: bool = False,
    stream=None,
    handle=None,
) -> None:
    """One ``ctx.token()`` per ensemble member -> E-way concurrency.

    The unit of STF work ("task") is the full 5-kernel chain for one
    member at one training step: all five kernels are independent only
    *across* members, not within a member, so there is nothing to gain
    by splitting them into separate tasks -- and splitting them would
    just multiply Python/Cython per-task bookkeeping. Each task grabs
    the member's buffers by closure (STF does not touch the data,
    identical pattern to ``lib_call_token`` in ``test_legacy_to_stf.py``).
    Because the tasks of member ``k`` take ``tok[k]`` as their only
    logical_data, STF concludes member chains are mutually independent
    and schedules them on separate streams of its pool.

    Parameters
    ----------
    use_graph : bool
        If True, build the context on the CUDA-graph backend. With a
        shared ``handle`` this lets the second and subsequent calls skip
        graph instantiation.
    stream, handle :
        Same semantics as ``lib_call_token``: inherit a caller stream
        and/or share a resources handle across contexts.
    """
    ctx = stf.context(use_graph=use_graph, stream=stream, handle=handle)

    tokens = [ctx.token() for _ in range(ens.n)]

    # One task per (step, member): the 5-kernel chain is a single unit of
    # work from STF's perspective (it all shares tok[k].rw()), so we launch
    # the whole chain on the task's stream. This keeps STF's per-task
    # bookkeeping costs from dominating while still expressing the cross-
    # member independence that gives us E-way concurrency.
    for _ in range(steps):
        for k in range(ens.n):
            with ctx.task(tokens[k].rw()) as t:
                s = cuda.external_stream(t.stream_ptr())
                fwd_L1[BLOCKS_H, THREADS, s](ens.W1[k], ens.x[k], ens.z[k])
                fwd_L2[BLOCKS_O, THREADS, s](ens.W2[k], ens.z[k], ens.y[k])
                bwd_gz[BLOCKS_H, THREADS, s](
                    ens.y[k], ens.target[k], ens.W2[k], ens.z[k], ens.gz[k]
                )
                upd_W2[BLOCKS_W2, THREADS, s](
                    ens.y[k], ens.target[k], ens.z[k], ens.W2[k], lr
                )
                upd_W1[BLOCKS_W1, THREADS, s](ens.gz[k], ens.x[k], ens.W1[k], lr)

    ctx.finalize()


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


def _assert_weights_equal(ens_ref: "Ensemble", ens_stf: "Ensemble") -> None:
    W1_ref, W2_ref = ens_ref.snapshot_weights()
    W1_stf, W2_stf = ens_stf.snapshot_weights()
    for k in range(ens_ref.n):
        # Per-member chains are deterministic and run identical op
        # sequences in both variants, so the results must be bit-equal.
        np.testing.assert_array_equal(W1_ref[k], W1_stf[k])
        np.testing.assert_array_equal(W2_ref[k], W2_stf[k])


@pytest.mark.parametrize("n_members", [1, 4])
def test_stf_matches_ref(n_members):
    steps = 8
    ens_ref = Ensemble(n_members, seed=123)
    ens_stf = Ensemble(n_members, seed=123)
    clone_weights(ens_ref, ens_stf)

    stream = cuda.stream()
    ref_train_ensemble(stream, ens_ref, steps)
    stream.synchronize()

    stf_train_ensemble(ens_stf, steps)
    cuda.synchronize()

    _assert_weights_equal(ens_ref, ens_stf)


@pytest.mark.parametrize("use_graph", [False, True])
def test_stf_graph_and_stream_handle(use_graph):
    """Exercise ``stream=`` / ``handle=`` kwargs on both backends."""
    n = 4
    steps = 4

    ens_ref = Ensemble(n, seed=7)
    ens_stf = Ensemble(n, seed=7)
    clone_weights(ens_ref, ens_stf)

    stream = cuda.stream()
    ref_train_ensemble(stream, ens_ref, steps)
    stream.synchronize()

    h = stf.async_resources()
    stf_train_ensemble(ens_stf, steps, use_graph=use_graph, stream=stream, handle=h)
    stream.synchronize()

    _assert_weights_equal(ens_ref, ens_stf)


# ---------------------------------------------------------------------------
# Benchmark entry point (``python test_mlp_ensemble.py``)
#
# Sweeps ensemble size E to make the token-concurrency win visible: as E
# grows, the reference cost scales ~E (everything serialized on one
# stream), whereas the STF variants stay closer to constant up to the GPU
# occupancy limit, since each member's chain runs on its own stream.
# ---------------------------------------------------------------------------


def _time(label: str, fn, niter: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(niter):
        fn()
    cuda.synchronize()
    ms = (time.perf_counter() - t0) / niter * 1e3
    print(f"  {label:<40s} {ms:10.3f} ms/iter")
    return ms


def _benchmark(ensemble_sizes=None, steps: int = STEPS_DEFAULT, niter: int = 8):
    """Compare ref vs STF token variants across a range of ensemble sizes."""
    if ensemble_sizes is None:
        ensemble_sizes = [1, 2, 4, 8, 16, 32]

    for n in ensemble_sizes:
        print(
            f"\n=== E = {n:3d} members, {steps} steps, "
            f"MLP({D_IN}->{D_HID}->{D_OUT}) ==="
        )

        # Each variant gets its own ensemble so they don't fight over
        # weights -- all cloned from a common seed for fairness.
        seed = 0x5EED
        ens_ref = Ensemble(n, seed=seed)
        ens_tok = Ensemble(n, seed=seed)
        clone_weights(ens_ref, ens_tok)
        ens_tokg = Ensemble(n, seed=seed)
        clone_weights(ens_ref, ens_tokg)
        ens_tokh = Ensemble(n, seed=seed)
        clone_weights(ens_ref, ens_tokh)
        ens_tokgh = Ensemble(n, seed=seed)
        clone_weights(ens_ref, ens_tokgh)

        stream = cuda.stream()
        handle = stf.async_resources()

        ref = _time(
            "ref_train_ensemble (single stream)",
            lambda: ref_train_ensemble(stream, ens_ref, steps),
            niter,
        )
        tok = _time(
            "stf_train_ensemble (tokens)",
            lambda: stf_train_ensemble(ens_tok, steps),
            niter,
        )
        tokg = _time(
            "stf_train_ensemble (tokens, graph)",
            lambda: stf_train_ensemble(ens_tokg, steps, use_graph=True),
            niter,
        )
        tokh = _time(
            "stf_train_ensemble (tokens,+stream,+handle)",
            lambda: stf_train_ensemble(ens_tokh, steps, stream=stream, handle=handle),
            niter,
        )
        tokgh = _time(
            "stf_train_ensemble (tokens,graph,+stream,+handle)",
            lambda: stf_train_ensemble(
                ens_tokgh, steps, use_graph=True, stream=stream, handle=handle
            ),
            niter,
        )

        print(f"  tokens / ref                           {tok / ref:6.2f}x")
        print(f"  tokens(graph) / ref                    {tokg / ref:6.2f}x")
        print(f"  tokens(+stream,+handle) / ref          {tokh / ref:6.2f}x")
        print(f"  tokens(graph,+stream,+handle) / ref    {tokgh / ref:6.2f}x")

        # Correctness spot-check at the largest-timed state.
        _assert_weights_equal(ens_ref, ens_tok)
        _assert_weights_equal(ens_ref, ens_tokg)
        _assert_weights_equal(ens_ref, ens_tokh)
        _assert_weights_equal(ens_ref, ens_tokgh)

        handle = None
    print("\ncorrectness: OK for all ensemble sizes")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--e",
        type=int,
        nargs="*",
        default=None,
        help="ensemble sizes to sweep (default: 1,2,4,8,16,32)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=STEPS_DEFAULT,
        help="training steps per benchmark iteration",
    )
    p.add_argument("--niter", type=int, default=8, help="outer repetitions per variant")
    args = p.parse_args()
    _benchmark(ensemble_sizes=args.e, steps=args.steps, niter=args.niter)
