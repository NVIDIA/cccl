# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check warp reductions before benchmarking them.

For each type, one launch of 256-thread blocks reduces every warp's 32 inputs
with cuda.coop. The per-warp results are compared with a NumPy reference and,
when ``--cub`` names the ``check_warp_reduce`` executable, bit for bit with
CUB on the same inputs:

    cuda.coop sum             vs  WarpReduce::Sum
    cuda.coop reduce(add_op)  vs  WarpReduce::Reduce(x, cuda::std::plus<>)

Integer results must match the modular reference exactly. Floating-point
results must be within ``32 * eps * sum(|x|)`` of a float64 reference.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from numba_cuda_mlir import cuda

import cuda.coop.numba_mlir as coop

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from device_side_benchmark import NUMPY_TYPES, add_op

BLOCK_SIZE = 256
WARP_SIZE = 32
UNSUPPORTED_TYPES = {"F16"}
CUB_OPERATIONS = {"sum": "sum", "reduce_add": "reduce_plus"}


@cuda.jit(launch_bounds=BLOCK_SIZE)
def warp_sum_kernel(source, result):
    thread = cuda.grid(1)
    total = coop.sum(coop.this_warp(), source[thread])
    if cuda.threadIdx.x % WARP_SIZE == 0:
        result[thread // WARP_SIZE] = total


@cuda.jit(launch_bounds=BLOCK_SIZE)
def warp_reduce_add_kernel(source, result):
    thread = cuda.grid(1)
    total = coop.reduce(coop.this_warp(), source[thread], binary_op=add_op)
    if cuda.threadIdx.x % WARP_SIZE == 0:
        result[thread // WARP_SIZE] = total


KERNELS = {"sum": warp_sum_kernel, "reduce_add": warp_reduce_add_kernel}


def make_input(type_name, num_items, rng):
    dtype = np.dtype(NUMPY_TYPES[type_name])
    if dtype.kind == "i":
        info = np.iinfo(dtype)
        # Full range, so sums wrap the way CUB's do.
        return rng.integers(
            info.min, info.max, num_items, dtype=dtype, endpoint=True
        )
    values = rng.standard_normal(num_items) * 10.0 ** rng.integers(
        -3, 4, num_items
    )
    return values.astype(dtype)


def reference(type_name, values):
    dtype = np.dtype(NUMPY_TYPES[type_name])
    warps = values.reshape(-1, WARP_SIZE)
    if dtype.kind == "i":
        bits = dtype.itemsize * 8
        exact = warps.astype(object).sum(axis=1)
        wrapped = [
            (int(v) + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))
            for v in exact
        ]
        return np.array(wrapped, dtype=dtype), None
    exact = warps.astype(np.float64).sum(axis=1)
    bound = (
        WARP_SIZE
        * np.finfo(dtype).eps
        * np.abs(warps.astype(np.float64)).sum(axis=1)
    )
    return exact, bound


def run_cub(cub, type_name, algorithm, values):
    with tempfile.TemporaryDirectory() as scratch:
        source = os.path.join(scratch, "in.bin")
        result = os.path.join(scratch, "out.bin")
        values.tofile(source)
        subprocess.run(
            [
                cub,
                type_name,
                CUB_OPERATIONS[algorithm],
                source,
                result,
                str(values.size),
            ],
            check=True,
        )
        return np.fromfile(result, dtype=values.dtype)


def check(type_name, algorithm, num_items, cub, rng):
    values = make_input(type_name, num_items, rng)
    expected, bound = reference(type_name, values)
    record = {"type": type_name, "algorithm": algorithm, "num_items": num_items}

    if type_name in UNSUPPORTED_TYPES:
        record["coop"] = "unsupported"
    else:
        observed = np.zeros(num_items // WARP_SIZE, dtype=values.dtype)
        KERNELS[algorithm][num_items // BLOCK_SIZE, BLOCK_SIZE](
            values, observed
        )
        record["coop"] = _verdict(observed, expected, bound)
    if cub:
        cub_result = run_cub(cub, type_name, algorithm, values)
        record["cub"] = _verdict(cub_result, expected, bound)
        if type_name not in UNSUPPORTED_TYPES:
            record["coop_cub_bitwise_equal"] = bool(
                np.array_equal(
                    observed.view(np.uint8), cub_result.view(np.uint8)
                )
            )
    return record


def _verdict(observed, expected, bound):
    if bound is None:
        mismatches = int(np.count_nonzero(observed != expected))
        return {"ok": mismatches == 0, "mismatches": mismatches}
    error = np.abs(observed.astype(np.float64) - expected)
    return {
        "ok": bool(np.all(error <= bound)),
        "mismatches": int(np.count_nonzero(error > bound)),
        "max_error_over_bound": float(
            np.max(error / np.maximum(bound, np.finfo(np.float64).tiny))
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cub", help="path to the check_warp_reduce executable"
    )
    parser.add_argument("--num-items", type=int, default=BLOCK_SIZE * 64)
    parser.add_argument("--seed", type=int, default=943)
    parser.add_argument("--json", help="write the records to this file")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    records = [
        check(type_name, algorithm, args.num_items, args.cub, rng)
        for algorithm in KERNELS
        for type_name in NUMPY_TYPES
    ]
    failed = False
    for record in records:
        coop_ok = record["coop"] == "unsupported" or record["coop"]["ok"]
        cub_ok = "cub" not in record or record["cub"]["ok"]
        failed |= not (coop_ok and cub_ok)
        print(json.dumps(record))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"seed": args.seed, "records": records}, f, indent=2)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
