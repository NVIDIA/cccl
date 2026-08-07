# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Localization A/B benchmarks (GB300-class parts, locality domains).

The controlled experiments behind the 2026-08-08 findings, consolidated:

  streaming   1 GiB fused compiled pointwise. vanilla (plain alloc, whole
              device) vs localized+map near/far. At the power wall vanilla
              wins (~0.82x near/vanilla: default interleave aggregates both
              memories); under power caps near wins (1.27x @ 250-400 W,
              crossover ~650 W).
  gather      random index_select, DRAM-resident (256 MiB table, 8M
              lookups) and L2-resident (16 MiB, 16M): placement-neutral at
              nn concurrency (near == far == vanilla).
  energy      sustained streaming with nvidia-smi power sampling: near
              ~0.95x J/GB uncapped.

Requires the locality-domain bindings (PR #10703 family); skips otherwise.
Confinement uses domain-pool streams via exec_place.pick_stream. Power
caps, if desired, are external (e.g. the powerbot host daemon).

Usage: python benchmarks/localization_ab.py [streaming|gather|energy|all]
"""

import subprocess
import sys
import threading
import time

import torch

import cuda.stf._experimental as stf
from cuda.stf._experimental.interop import pytorch as tp


def domain_setup():
    stf.machine_init()
    eg = stf.exec_place_grid
    if not hasattr(eg, "locality_domains"):
        print("locality-domain bindings not available (PR #10703): skipping")
        sys.exit(0)
    grid = eg.machine(granularity="locality_domain") if hasattr(eg, "machine") else eg.locality_domains(0)
    res = stf.exec_place_resources()
    streams = []
    for d in range(stf.locality_domain_count(0)):
        p = stf.exec_place.locality_domain(0, d)
        with p:
            s = p.pick_stream(res)
        streams.append(torch.cuda.ExternalStream(int(s)))
    return grid, streams, res  # res must outlive the streams


def sustained_gbps(run, nbytes_per_iter, seconds=4.0):
    run()
    torch.cuda.synchronize()
    t0 = time.time()
    it = 0
    while time.time() - t0 < seconds:
        run()
        it += 1
    torch.cuda.synchronize()
    return nbytes_per_iter * it / (time.time() - t0) * 1e-9


def bench_streaming(grid, streams):
    shape = (262144, 1024)  # 1 GiB fp32
    nbytes = 2 * 262144 * 1024 * 4  # in-place read+write

    def body(t):
        t.mul_(1.0001).add_(0.5)

    fn = torch.compile(body)
    v = torch.empty(shape, dtype=torch.float32, device="cuda")
    v.normal_()
    x = tp.localized_empty(shape, torch.float32, grid)
    x.normal_()

    rows = [
        ("vanilla (plain, whole-device)", lambda: fn(v)),
        ("localized, whole-device", lambda: fn(x)),
        ("localized+map, plain streams", lambda: tp.map(fn, x)),
        ("localized+map, NEAR", lambda: tp.map(fn, x, streams=streams)),
        ("localized+map, FAR", lambda: tp.map(fn, x, streams=list(reversed(streams)))),
    ]
    print("== streaming (1 GiB fused compiled pointwise) ==")
    for label, run in rows:
        print(f"  {label:30s}: {sustained_gbps(run, nbytes):6.0f} GB/s")
    tp.release(x)


def bench_gather(grid, streams, rows_pow, lookups_pow, label):
    rows, dim = 2**rows_pow, 32
    lookups = 2**lookups_pow
    torch.manual_seed(0)
    idx = torch.randint(0, rows, (lookups,), device="cuda")
    half = rows // 2
    idx_lo, idx_hi = idx[idx < half], idx[idx >= half]
    idx_cat = torch.cat([idx_lo, idx_hi]).contiguous()  # same order everywhere
    out = torch.empty(lookups, dim, dtype=torch.float32, device="cuda")
    nbytes = lookups * dim * 4 * 2

    vt = torch.empty(rows, dim, dtype=torch.float32, device="cuda")
    vt.normal_()
    lt = tp.localized_empty((rows, dim), torch.float32, grid)
    lt.copy_(vt)
    views = tp.views(lt)
    n_lo = idx_lo.numel()
    idx_local = [idx_lo, idx_hi - half]
    out_parts = [out[:n_lo], out[n_lo:]]

    def near(ss):
        cur = torch.cuda.current_stream()
        fork = torch.cuda.Event()
        fork.record(cur)
        evs = []
        for d, s in enumerate(ss):
            s.wait_event(fork)
            with torch.cuda.stream(s):
                torch.index_select(views[d], 0, idx_local[d], out=out_parts[d])
            e = torch.cuda.Event()
            e.record(s)
            evs.append(e)
        for e in evs:
            cur.wait_event(e)

    print(f"== gather, {label} (table {rows * dim * 4 >> 20} MiB, {lookups >> 20}M lookups) ==")
    print(f"  {'vanilla':10s}: {sustained_gbps(lambda: torch.index_select(vt, 0, idx_cat, out=out), nbytes):6.0f} GB/s")
    print(f"  {'near':10s}: {sustained_gbps(lambda: near(streams), nbytes):6.0f} GB/s")
    print(f"  {'far':10s}: {sustained_gbps(lambda: near(list(reversed(streams))), nbytes):6.0f} GB/s")
    near(streams)
    torch.cuda.synchronize()
    assert torch.equal(out, torch.index_select(vt, 0, idx_cat)), "gather mismatch"
    tp.release(lt)


def bench_energy(grid, streams, seconds=8.0):
    shape = (262144, 1024)
    nbytes = 2 * 262144 * 1024 * 4

    def body(t):
        t.mul_(1.0001).add_(0.5)

    fn = torch.compile(body)
    v = torch.empty(shape, dtype=torch.float32, device="cuda")
    v.normal_()
    x = tp.localized_empty(shape, torch.float32, grid)
    x.normal_()

    def measure(run, label):
        samples, stop = [], [False]

        def sampler():
            while not stop[0]:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                )
                try:
                    samples.append(float(r.stdout.strip()))
                except ValueError:
                    pass
                time.sleep(0.1)

        run()
        torch.cuda.synchronize()
        th = threading.Thread(target=sampler)
        th.start()
        gbps = sustained_gbps(run, nbytes, seconds)
        stop[0] = True
        th.join()
        watts = sum(samples) / max(len(samples), 1)
        print(f"  {label:10s}: {gbps:6.0f} GB/s  {watts:5.0f} W  {watts / gbps:.4f} J/GB")

    print("== energy (sustained streaming, power sampled at 10 Hz) ==")
    measure(lambda: fn(v), "vanilla")
    measure(lambda: tp.map(fn, x, streams=streams), "near")
    tp.release(x)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    grid, streams, _res = domain_setup()
    if which in ("streaming", "all"):
        bench_streaming(grid, streams)
    if which in ("gather", "all"):
        bench_gather(grid, streams, 21, 23, "DRAM-resident")
        bench_gather(grid, streams, 17, 24, "L2-resident")
    if which in ("energy", "all"):
        bench_energy(grid, streams)
