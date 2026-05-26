# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Probe: same K-unrolled "fresh scratch + copy-back" body from
``probe_while_loop_unrolled.py``, but driven by three alternative control
structures instead of ``stackable_context + while_loop``:

- host_eager:  plain ``stf.context()``, host Python loop of ``rounds``
  unrolled bodies into one context, finalize once.
- host_graph:  ``stf.context(use_graph=True)``, same host unroll, one graph.
- host_per_round: one fresh ``stf.context()`` per round (N contexts total).

Each variant should produce ``hidden == rounds * K`` if the K-unrolled
fresh-scratch body works when not inside a captured while-loop body.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pytorch_task import pytorch_task  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402

N = 128


def _body(ctx, l_hidden, K):
    for k in range(K):
        l_s = ctx.logical_data_empty((N,), np.float32, name=f"scr{k}")
        with pytorch_task(ctx, l_hidden.read(), l_s.write()) as (th, ts):
            ts[:] = th + 1.0
        with pytorch_task(ctx, l_s.read(), l_hidden.write()) as (ts, th):
            th[:] = ts


def run_host_eager(rounds, K):
    h = np.zeros(N, dtype=np.float32)
    ctx = stf.context()
    lh = ctx.logical_data(h, name="h")
    for _ in range(rounds):
        _body(ctx, lh, K)
    ctx.finalize()
    return float(h[0])


def run_host_graph(rounds, K):
    h = np.zeros(N, dtype=np.float32)
    ctx = stf.context(use_graph=True)
    lh = ctx.logical_data(h, name="h")
    for _ in range(rounds):
        _body(ctx, lh, K)
    ctx.finalize()
    return float(h[0])


def run_host_per_round(rounds, K):
    h = np.zeros(N, dtype=np.float32)
    for _ in range(rounds):
        ctx = stf.context()
        lh = ctx.logical_data(h, name="h")
        _body(ctx, lh, K)
        ctx.finalize()
    return float(h[0])


def main():
    print(f"{'variant':<18} {'rounds':>6} {'K':>3}  {'exp':>6} {'got':>6}  ok")
    print("-" * 55)
    combos = [(1, 1), (1, 2), (1, 4), (2, 2), (2, 4), (4, 2), (4, 4), (8, 4), (16, 4)]
    for name, fn in [
        ("host_eager", run_host_eager),
        ("host_graph", run_host_graph),
        ("host_per_round", run_host_per_round),
    ]:
        for rounds, K in combos:
            try:
                t0 = time.perf_counter()
                got = fn(rounds, K)
                dt = time.perf_counter() - t0
                exp = rounds * K
                ok = "OK" if abs(got - exp) < 1e-3 else "!!"
            except Exception as e:
                got = float("nan")
                exp = rounds * K
                ok = f"ERR {type(e).__name__}"
                dt = float("nan")
            print(
                f"{name:<18} {rounds:>6d} {K:>3d}  {exp:>6d} {got:>6.1f}  {ok}  ({dt * 1e3:.1f}ms)"
            )


if __name__ == "__main__":
    main()
