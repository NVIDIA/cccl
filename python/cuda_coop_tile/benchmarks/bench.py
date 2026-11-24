# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


"""Time composed cuda.coop.tile kernels after checking exact integer results.

Run with a qualified Tile frontend and provider environment, for example:
    python benchmarks/bench.py --threads 128 256 --output results.json

The frontend must support every requested num_worker_warps value. This script
never patches frontend validation and does not exercise automatic ranking.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import cuda.tile as ct
import torch

import cuda.coop.tile as coop

SCENARIOS = ("scan_affine", "reduce_affine", "scan_reduce")


def make_kernel(scenario: str, extent: int, threads: int):
    @ct.kernel(num_worker_warps=threads // 32)
    def scan_affine(inp: ct.Array, out: ct.Array):
        block = ct.bid(0)
        values = ct.load(inp, (block,), shape=(extent,))
        transformed = values * 3 + 1
        result = coop.exclusive_sum(transformed) + values
        ct.store(out, (block,), result)

    @ct.kernel(num_worker_warps=threads // 32)
    def reduce_affine(inp: ct.Array, out: ct.Array):
        block = ct.bid(0)
        values = ct.load(inp, (block,), shape=(extent,))
        transformed = values * 3 + 1
        result = coop.sum(transformed) + values
        ct.store(out, (block,), result)

    @ct.kernel(num_worker_warps=threads // 32)
    def scan_reduce(inp: ct.Array, out: ct.Array):
        block = ct.bid(0)
        values = ct.load(inp, (block,), shape=(extent,))
        transformed = values * 3 + 1
        result = coop.sum(coop.exclusive_sum(transformed)) + values
        ct.store(out, (block,), result)

    return {
        "scan_affine": scan_affine,
        "reduce_affine": reduce_affine,
        "scan_reduce": scan_reduce,
    }[scenario]


def prepare(scenario: str, extent: int, threads: int, values, output):
    rows = values.numel() // extent
    kernel = make_kernel(scenario, extent, threads)

    def launch():
        ct.launch(
            torch.cuda.current_stream(), (rows, 1, 1), kernel, (values, output)
        )

    launch()
    torch.cuda.synchronize()
    source = values.reshape(rows, extent).to(torch.int64)
    transformed = source * 3 + 1
    prefix = torch.cumsum(transformed, dim=1) - transformed
    if scenario == "scan_affine":
        expected = prefix + source
    elif scenario == "reduce_affine":
        expected = transformed.sum(dim=1, keepdim=True) + source
    else:
        expected = prefix.sum(dim=1, keepdim=True) + source
    low, high = expected.aminmax()
    bounds = (int(low.item()), int(high.item()))
    if bounds[0] < -(2**31) or bounds[1] >= 2**31:
        raise AssertionError("reference result exceeds int32 range")
    if not torch.equal(output.reshape(rows, extent).to(torch.int64), expected):
        raise AssertionError(
            f"incorrect {scenario}, N={extent}, threads={threads}"
        )
    return launch, bounds


def capture(launch, launches: int):
    # Compile and initialize before capture; a graph removes Python dispatch
    # gaps from device timing while retaining every full load/compute/store.
    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(launches):
            launch()
    return graph


def measure(graph, launches: int, warmup: int, samples: int):
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(samples):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / launches)
    quartiles = statistics.quantiles(times, n=4)
    return {
        "samples_us": times,
        "median_us": statistics.median(times),
        "q1_us": quartiles[0],
        "q3_us": quartiles[2],
        "min_us": min(times),
        "max_us": max(times),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", nargs="+", type=int, default=[128, 256])
    parser.add_argument(
        "--extents", nargs="+", type=int, default=[256, 1024, 4096]
    )
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS
    )
    parser.add_argument("--elements", type=int, default=1048576)
    parser.add_argument("--launches", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(t not in (32, 64, 128, 256, 512, 1024) for t in args.threads):
        parser.error("threads must be a power of two between 32 and 1024")
    if args.elements <= 0 or any(
        n <= 0 or args.elements % n for n in args.extents
    ):
        parser.error("every extent must be positive and divide elements")
    if min(args.launches, args.warmup, args.rounds) <= 0 or args.samples < 4:
        parser.error("launches/warmup/rounds must be positive and samples >= 4")
    report = {
        "frontend": "cuda.tile Python",
        "selection": (
            "explicit num_worker_warps; automatic ranking not measured"
        ),
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch_version": torch.__version__,
        "cuda_tile_version": ct.__version__,
        "settings": {**vars(args), "output": str(args.output)},
        "cases": [],
    }
    cases = [
        (scenario, n, t)
        for scenario in args.scenarios
        for n in args.extents
        for t in args.threads
        if n % t == 0
    ]
    random.Random(args.seed).shuffle(cases)
    # Every case uses the same device addresses, avoiding placement-dependent
    # cache behavior when comparing worker widths. No timed launch allocates.
    values = (torch.arange(args.elements, device="cuda") % 17 - 8).to(
        torch.int32
    )
    output = torch.empty_like(values)
    report["working_set_bytes"] = 2 * values.numel() * values.element_size()
    report["shared_buffers_across_cases"] = True
    prepared = []
    for scenario, extent, threads in cases:
        print(
            f"prepare {scenario} extent={extent} threads={threads}", flush=True
        )
        launch, bounds = prepare(scenario, extent, threads, values, output)
        graph = capture(launch, args.launches)
        # Keep the launch closure alive: it owns the graph's tensor arguments.
        prepared.append((scenario, extent, threads, graph, launch, bounds))
    for round_index in range(args.rounds):
        random.Random(args.seed + round_index + 1).shuffle(prepared)
        for scenario, extent, threads, graph, _, bounds in prepared:
            timing = measure(graph, args.launches, args.warmup, args.samples)
            result = {
                "scenario": scenario,
                "extent": extent,
                "threads": threads,
                "items_per_thread": extent // threads,
                "blocks": args.elements // extent,
                "round": round_index,
                "correct": True,
                "reference_range": bounds,
                **timing,
            }
            report["cases"].append(result)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k != "samples_us"}
                ),
                flush=True,
            )

    torch.cuda.synchronize()
    # Release executable graphs while their kernel dispatchers/tensors live.
    for _, _, _, graph, _, _ in prepared:
        graph.reset()


if __name__ == "__main__":
    main()
