# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete normalization and unchanged first-thread consumers on resident inputs."""

import argparse
import json
from pathlib import Path

import torch
from common import digest, load_consumer, provenance

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--variant", choices=["baseline", "patch"], required=True)
parser.add_argument("--build", type=Path, required=True)
parser.add_argument("--rows", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(8)
torch.manual_seed(11504)
consumer = load_consumer(args.build, args.variant)
record = provenance(args.source, consumer) | {
    "variant": args.variant,
    "rows": args.rows,
    "cases": [],
}
correctness = []
for columns in [1, 7, 31, 33, 65, 127, 129, 257, 511, 513, 543]:
    for masked in [False, True]:
        input = torch.randn(19, columns, device="cuda") * 4
        if masked:
            valid = torch.arange(19, device="cuda") % columns + 1
            input = input.masked_fill(
                torch.arange(columns, device="cuda")[None] >= valid[:, None], -torch.inf
            )
        output = consumer.normalize(input, False, args.variant == "patch")
        expected = input.double().softmax(dim=-1).float()
        torch.testing.assert_close(output, expected, rtol=2e-6, atol=2e-7)
        correctness.append(
            {
                "columns": columns,
                "masked": masked,
                "max_abs_error": (output - expected).abs().max().item(),
                "output_sha256": digest(output),
            }
        )
record["correctness"] = correctness
cases = []
for columns in [32, 128, 256, 512]:
    input = torch.rand(args.rows, columns, device="cuda") * 8 - 4
    reference = input.double().softmax(dim=-1).float()
    for first_thread in [False, True]:
        if first_thread:
            # Dyadic inputs make every FP32 intermediate sum exactly representable.
            input = (
                torch.randint(-64, 65, (args.rows, columns), device="cuda").float() / 16
            )
        output = consumer.normalize(input, first_thread, args.variant == "patch")
        expected = (
            torch.stack(
                [input.double().sum(dim=-1).float(), input.max(dim=-1).values], dim=-1
            )
            if first_thread
            else reference
        )
        error = (output - expected).abs().max().item()
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)
        output_hash = digest(output)
        warm_stream = torch.cuda.Stream()
        warm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm_stream):
            for _ in range(5):
                consumer.normalize(input, first_thread, args.variant == "patch")
        torch.cuda.current_stream().wait_stream(warm_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = consumer.normalize(input, first_thread, args.variant == "patch")
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, output, rtol=0, atol=0)
        samples = []
        for _ in range(20):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 10)
        cases.append(
            {
                "kind": "first-thread" if first_thread else "normalization",
                "columns": columns,
                "max_abs_error": error,
                "input_sha256": digest(input),
                "output_sha256": output_hash,
                "input_address": input.data_ptr(),
                "output_address": captured.data_ptr(),
                "samples_us": samples,
            }
        )
        graph.reset()
record["cases"] = cases
args.output.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
