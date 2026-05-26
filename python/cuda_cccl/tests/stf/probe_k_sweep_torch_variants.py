# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Three PyTorch body shapes to localise the mod-4 bug that only surfaces with
``pytorch_task`` inside ``stackable_context.while_loop``.

Same outer plumbing as probe_k_sweep.py; only the body differs:

- ``add_temp``  : ``ta[:] = ta + 1.0`` (creates a temporary, then copy_ into ta)
- ``add_inplace``: ``ta.add_(1.0)``   (pure in-place, no temporary)
- ``fill_plus`` : ``ta.fill_(ta[0].item() + 1.0)`` (intentionally bad: host sync)
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pytorch_task import pytorch_task  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402

N = 128


def run(K, mode, rounds=1):
    acc = np.zeros(N, dtype=np.float32)
    r = np.zeros((1,), dtype=np.float64)
    d = np.ones((1,), dtype=np.float64)
    scratch = np.zeros((1,), dtype=np.float64)
    ctx = stf.stackable_context()
    l_acc = ctx.logical_data(acc, name="acc")
    l_r = ctx.logical_data(r, name="round")
    l_d = ctx.logical_data(d, name="done")
    l_s = ctx.logical_data(scratch, name="scratch")
    with ctx.while_loop() as loop:
        for _ in range(K):
            with pytorch_task(ctx, l_acc.rw()) as (ta,):
                ta[:] = ta + 1.0
        if mode == "fill_only":
            with pytorch_task(ctx, l_d.write()) as (td,):
                td.fill_(0.0)
        elif mode == "r_plus_fill":
            with pytorch_task(ctx, l_r.rw(), l_d.write()) as (tr, td):
                tr.add_(1.0)
                td.fill_(0.0)
        elif mode == "r_cast_copy":
            with pytorch_task(ctx, l_r.rw(), l_d.write()) as (tr, td):
                tr[:] = tr + 1.0
                flag = (tr < float(rounds)).to(td.dtype)
                td.copy_(flag.view(td.shape))
        elif mode == "r_cast_copy_no_view":
            with pytorch_task(ctx, l_r.rw(), l_d.write()) as (tr, td):
                tr[:] = tr + 1.0
                flag = (tr < float(rounds)).to(td.dtype)
                td.copy_(flag)
        elif mode == "r_cast_copy_del":
            # Same as r_cast_copy but explicitly release temp refs before task.end()
            with pytorch_task(ctx, l_r.rw(), l_d.write()) as (tr, td):
                tr[:] = tr + 1.0
                flag = (tr < float(rounds)).to(td.dtype)
                td.copy_(flag.view(td.shape))
                del flag
        elif mode == "r_single_cast":
            # Comparison + cast but no copy_
            with pytorch_task(ctx, l_r.rw(), l_d.write()) as (tr, td):
                tr[:] = tr + 1.0
                td[:] = (tr < float(rounds)).to(td.dtype)
        elif mode == "r_just_copy":
            # Skip the add + comparison; just copy_ from a fresh temp
            with pytorch_task(ctx, l_r.rw(), l_d.write()) as (tr, td):
                tr.add_(1.0)
                temp = torch.zeros_like(td)
                td.copy_(temp)
        elif mode == "r_cmp_only":
            # comparison only, result stored into a bool-typed td-shaped temp,
            # no cast. td itself is not touched (we still declare write to keep
            # the task structurally similar).
            with pytorch_task(ctx, l_r.rw(), l_d.write()) as (tr, td):
                tr.add_(1.0)
                _ = tr < float(rounds)
                td.fill_(0.0)
        elif mode == "r_to_only":
            # Cast tr to td.dtype without any comparison.
            with pytorch_task(ctx, l_r.rw(), l_d.write()) as (tr, td):
                tr.add_(1.0)
                td[:] = tr.to(td.dtype)
        elif mode == "r_to_only_float_to_float":
            # Cast float64 -> float64 (no-op dtype), still triggers .to() allocator path?
            with pytorch_task(ctx, l_r.rw(), l_d.write()) as (tr, td):
                tr.add_(1.0)
                tmp = tr.to(torch.float64)
                td[:] = tmp
        elif mode == "r_bool_to_float":
            # Comparison then cast, but don't assign to td.
            with pytorch_task(ctx, l_r.rw(), l_d.write()) as (tr, td):
                tr.add_(1.0)
                _ = (tr < float(rounds)).to(td.dtype)
                td.fill_(0.0)
        elif mode == "r_single_cast_scratch":
            # Same semantics as r_single_cast but route the cast through an
            # STF-owned scratch buffer to eliminate *any* PyTorch allocation
            # inside the captured while_loop body.
            with pytorch_task(ctx, l_r.rw(), l_s.rw(), l_d.write()) as (tr, ts, td):
                tr.add_(1.0)
                ts[:] = (tr < float(rounds)).to(ts.dtype)
                td.copy_(ts)
        else:
            raise ValueError(mode)
        loop.continue_while(l_d, ">", 0.5)
    ctx.finalize()
    return float(acc[0])


def main():
    for mode in (
        "r_cmp_only",
        "r_to_only",
        "r_to_only_float_to_float",
        "r_bool_to_float",
        "r_single_cast",
    ):
        print(f"\n== mode={mode} ==")
        print(f"{'K':>3} {'exp':>5} {'got':>7}  status")
        for K in (1, 2, 3, 4, 5, 7, 8):
            got = run(K, mode)
            exp = K
            ok = "OK" if abs(got - exp) < 1e-3 else f"!! off by {got - exp:+.1f}"
            print(f"{K:>3} {exp:>5} {got:>7.1f}  {ok}")


if __name__ == "__main__":
    main()
