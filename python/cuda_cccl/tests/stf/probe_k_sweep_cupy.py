# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Same shape as probe_k_sweep.py, but drive the task body with CuPy instead of
PyTorch. If this probe still shows the mod-4 drop, the bug is in the Cython
stackable_task glue; if it passes, the bug is specific to the PyTorch bridge
(ExternalStream context manager or caching allocator interacting with
while_graph_scope capture).
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import cuda.stf._experimental as stf  # noqa: E402

N = 128


def _as_cupy(cai_obj):
    """Wrap stf_cai -> cupy.ndarray by sharing the underlying buffer."""
    cai = cai_obj.__cuda_array_interface__
    return cp.ndarray(
        shape=tuple(cai["shape"]),
        dtype=np.dtype(cai["typestr"]),
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(cai["data"][0], int(np.prod(cai["shape"]) * np.dtype(cai["typestr"]).itemsize), None),
            0,
        ),
    )


def run(K, rounds=1):
    acc = np.zeros(N, dtype=np.float32)
    d = np.ones((1,), dtype=np.float64)
    ctx = stf.stackable_context()
    l_acc = ctx.logical_data(acc, name="acc")
    l_d = ctx.logical_data(d, name="done")
    with ctx.while_loop() as loop:
        for _ in range(K):
            t = ctx.task(l_acc.rw())
            t.start()
            try:
                s = cp.cuda.ExternalStream(t.stream_ptr())
                with s:
                    a = _as_cupy(t.get_arg_cai(0))
                    a += cp.float32(1.0)
            finally:
                t.end()
        t = ctx.task(l_d.write())
        t.start()
        try:
            s = cp.cuda.ExternalStream(t.stream_ptr())
            with s:
                a = _as_cupy(t.get_arg_cai(0))
                a.fill(0.0)
        finally:
            t.end()
        loop.continue_while(l_d, ">", 0.5)
    ctx.finalize()
    return float(acc[0])


def main():
    print(f"{'K':>3} {'exp':>5} {'got':>7}  status")
    for K in range(1, 17):
        got = run(K, rounds=1)
        exp = K
        ok = "OK" if abs(got - exp) < 1e-3 else f"!! off by {got - exp:+.1f}"
        print(f"{K:>3} {exp:>5} {got:>7.1f}  {ok}")


if __name__ == "__main__":
    main()
