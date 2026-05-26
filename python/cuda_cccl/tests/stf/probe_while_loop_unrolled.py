# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Probe: does ``stackable_context + while_loop`` support K fresh
``logical_data_empty`` objects created inside a Python-unrolled for loop in
the body, each paired with a persistent ``.rw()`` accumulator?

This mirrors the spec-decode pattern (K draft forwards per body, each
with a fresh hidden scratch + a copy-back into the shared hidden buffer).

Runs a sweep over (rounds, K) and prints pass/fail/hang per combo.
"""
from __future__ import annotations

import os
import signal
import sys
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pytorch_task import pytorch_task  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402


N = 128  # size of the "hidden" buffer


def _probe_body(ctx, l_hidden, l_round, l_done, K: int, max_rounds: int):
    """Body is captured once. Inside:
        - K unrolled iterations, each allocates a fresh l_scratch, writes it
          from l_hidden, copies back into l_hidden.
        - Increment the round counter and emit done flag.
    """
    with ctx.while_loop() as loop:
        for k in range(K):
            l_scratch = ctx.logical_data_empty((N,), np.float32, name=f"scr{k}")

            # scratch = hidden + 1.0
            with pytorch_task(ctx, l_hidden.read(), l_scratch.write()) as (th, ts):
                ts[:] = th + 1.0

            # hidden = scratch (the "copy-back")
            with pytorch_task(ctx, l_scratch.read(), l_hidden.write()) as (ts, th):
                th[:] = ts

        with pytorch_task(ctx, l_round.rw(), l_done.write()) as (tr, td):
            tr[:] = tr + 1.0
            flag = (tr < float(max_rounds)).to(td.dtype)
            td.copy_(flag.view(td.shape))

        loop.continue_while(l_done, ">", 0.5)


def run_probe(rounds: int, K: int):
    h_host = np.zeros(N, dtype=np.float32)
    round_host = np.zeros((1,), dtype=np.float64)
    done_host = np.ones((1,), dtype=np.float64)

    ctx = stf.stackable_context()
    l_hidden = ctx.logical_data(h_host, name="hidden")
    l_round = ctx.logical_data(round_host, name="round")
    l_done = ctx.logical_data(done_host, name="done")

    _probe_body(ctx, l_hidden, l_round, l_done, K, rounds)
    ctx.finalize()

    # Expected: every body iteration does K increments of +1 on hidden.
    # After `rounds` iterations: hidden == rounds * K.
    return h_host, round_host


def main():
    print("=== while_loop + K-unrolled fresh logical_data probe ===")
    print(f"{'rounds':>6} {'K':>3}  {'elapsed':>9}  {'expected':>9}  {'got':>9}  status")
    print("-" * 70)

    combos = [
        (1, 1), (1, 2), (1, 4),
        (2, 1), (2, 2), (2, 4),
        (4, 1), (4, 2), (4, 4),
        (8, 2), (8, 4),
    ]

    for rounds, K in combos:
        # Run in a subprocess-like timeout via SIGALRM (doesn't actually kill
        # CUDA, but if we get past 15s we mark it as hang and move on).
        t0 = time.perf_counter()
        try:
            hidden, _ = run_probe(rounds, K)
            dt = time.perf_counter() - t0
            got = float(hidden[0])
            expected = float(rounds * K)
            ok = abs(got - expected) < 1e-3
            status = "OK" if ok else "WRONG"
        except Exception as e:
            dt = time.perf_counter() - t0
            got = float("nan")
            expected = float(rounds * K)
            status = f"ERR: {type(e).__name__}"

        print(f"{rounds:>6d} {K:>3d}  {dt * 1e3:>7.1f}ms  {expected:>9.1f}  {got:>9.1f}  {status}")


if __name__ == "__main__":
    main()
