# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""First-call latency benchmark: JIT vs AoT deserialize.

Mirrors c/parallel/test/test_bench_aot.cpp at the Python user level.

For each algorithm, measures wall-clock time for the complete first execution:
    JIT = make_<algo>(...) + execute() + sync()
    AoT = deserialize(blob) + execute() + sync()

The AoT blob is built once up front (untimed, simulating build-server work).
The in-memory algorithm cache is cleared between the two timed paths so each
measures a true cold start within the process.

Run:
    python tests/compute/bench_aot.py
"""

from __future__ import annotations

import time

import cupy as cp
import numpy as np

from cuda.compute import (
    OpKind,
    clear_all_caches,
    deserialize,
    make_exclusive_scan,
    make_merge_sort,
    make_reduce_into,
    serialize,
)
from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer

N = 1 << 20


def _sync() -> None:
    cp.cuda.runtime.deviceSynchronize()


def _bench(jit_setup, aot_setup) -> tuple[float, float]:
    """Run JIT cold-start and AoT cold-start, return (jit_ms, aot_ms)."""
    clear_all_caches()
    _sync()
    t0 = time.perf_counter()
    jit_setup()
    _sync()
    jit_ms = (time.perf_counter() - t0) * 1000.0

    clear_all_caches()
    _sync()
    t0 = time.perf_counter()
    aot_setup()
    _sync()
    aot_ms = (time.perf_counter() - t0) * 1000.0
    return jit_ms, aot_ms


def bench_reduce() -> tuple[float, float]:
    d_in = cp.ones(N, dtype=cp.int32)
    d_out = cp.zeros(1, dtype=cp.int32)
    h_init = np.array([0], dtype=np.int32)

    # Pre-build the AoT blob (untimed).
    blob = serialize(
        make_reduce_into(d_in=d_in, d_out=d_out, op=OpKind.PLUS, h_init=h_init)
    )

    def jit():
        reducer = make_reduce_into(
            d_in=d_in, d_out=d_out, op=OpKind.PLUS, h_init=h_init
        )
        nbytes = reducer(
            temp_storage=None,
            d_in=d_in,
            d_out=d_out,
            num_items=N,
            op=OpKind.PLUS,
            h_init=h_init,
        )
        tmp = TempStorageBuffer(nbytes, None)
        reducer(
            temp_storage=tmp,
            d_in=d_in,
            d_out=d_out,
            num_items=N,
            op=OpKind.PLUS,
            h_init=h_init,
        )

    def aot():
        reducer = deserialize(blob)
        nbytes = reducer(
            temp_storage=None,
            d_in=d_in,
            d_out=d_out,
            num_items=N,
            op=OpKind.PLUS,
            h_init=h_init,
        )
        tmp = TempStorageBuffer(nbytes, None)
        reducer(
            temp_storage=tmp,
            d_in=d_in,
            d_out=d_out,
            num_items=N,
            op=OpKind.PLUS,
            h_init=h_init,
        )

    return _bench(jit, aot)


def bench_exclusive_scan() -> tuple[float, float]:
    d_in = cp.ones(N, dtype=cp.int32)
    d_out = cp.empty_like(d_in)
    init_value = np.array([0], dtype=np.int32)

    blob = serialize(
        make_exclusive_scan(
            d_in=d_in, d_out=d_out, op=OpKind.PLUS, init_value=init_value
        )
    )

    def _execute(scanner):
        nbytes = scanner(
            temp_storage=None,
            d_in=d_in,
            d_out=d_out,
            op=OpKind.PLUS,
            init_value=init_value,
            num_items=N,
        )
        tmp = TempStorageBuffer(nbytes, None)
        scanner(
            temp_storage=tmp,
            d_in=d_in,
            d_out=d_out,
            op=OpKind.PLUS,
            init_value=init_value,
            num_items=N,
        )

    def jit():
        _execute(
            make_exclusive_scan(
                d_in=d_in, d_out=d_out, op=OpKind.PLUS, init_value=init_value
            )
        )

    def aot():
        _execute(deserialize(blob))

    return _bench(jit, aot)


def bench_merge_sort() -> tuple[float, float]:
    h = np.arange(N, 0, -1, dtype=np.int32)
    d_in_keys = cp.asarray(h)
    d_out_keys = cp.empty_like(d_in_keys)

    blob = serialize(
        make_merge_sort(
            d_in_keys=d_in_keys,
            d_in_values=None,
            d_out_keys=d_out_keys,
            d_out_values=None,
            op=OpKind.LESS,
        )
    )

    def _execute(sorter):
        nbytes = sorter(
            temp_storage=None,
            d_in_keys=d_in_keys,
            d_in_values=None,
            d_out_keys=d_out_keys,
            d_out_values=None,
            num_items=N,
            op=OpKind.LESS,
        )
        tmp = TempStorageBuffer(nbytes, None)
        sorter(
            temp_storage=tmp,
            d_in_keys=d_in_keys,
            d_in_values=None,
            d_out_keys=d_out_keys,
            d_out_values=None,
            num_items=N,
            op=OpKind.LESS,
        )

    def jit():
        _execute(
            make_merge_sort(
                d_in_keys=d_in_keys,
                d_in_values=None,
                d_out_keys=d_out_keys,
                d_out_values=None,
                op=OpKind.LESS,
            )
        )

    def aot():
        _execute(deserialize(blob))

    return _bench(jit, aot)


def main() -> None:
    cc_major, cc_minor = cp.cuda.Device().compute_capability
    print(f"\n--- AoT vs JIT first-call latency  (SM {cc_major}{cc_minor}, N={N})  ---")
    print(f"  {'algorithm':<25}  {'JIT':>10}     {'AoT':>20}")
    print("  " + "-" * 64)

    bench_fns = [
        ("reduce (int32)", bench_reduce),
        ("exclusive_scan (int32)", bench_exclusive_scan),
        ("merge_sort (int32, keys)", bench_merge_sort),
    ]
    for name, fn in bench_fns:
        jit_ms, aot_ms = fn()
        speedup = jit_ms / aot_ms if aot_ms > 0 else float("inf")
        print(
            f"  {name:<25}  {jit_ms:>7.1f} ms     {aot_ms:>7.3f} ms ({speedup:>5.0f}×)"
        )
    print()


if __name__ == "__main__":
    main()
