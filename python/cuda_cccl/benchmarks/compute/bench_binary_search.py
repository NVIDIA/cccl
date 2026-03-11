# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import cupy as cp
import numpy as np
import pytest

import cuda.compute


def lower_bound_run(d_data, d_values, d_out, build_only):
    searcher = cuda.compute.make_lower_bound(d_data, d_values, d_out)
    if not build_only:
        searcher(d_data, d_values, d_out, None, len(d_data), len(d_values))
    cp.cuda.runtime.deviceSynchronize()


def upper_bound_run(d_data, d_values, d_out, build_only):
    searcher = cuda.compute.make_upper_bound(d_data, d_values, d_out)
    if not build_only:
        searcher(d_data, d_values, d_out, None, len(d_data), len(d_values))
    cp.cuda.runtime.deviceSynchronize()


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_lower_bound(bench_fixture, request, size):
    actual_size = 100 if bench_fixture == "compile_benchmark" else size
    d_data = cp.sort(cp.random.randint(0, 1000, actual_size, dtype=np.int32))
    d_values = cp.random.randint(0, 1000, actual_size, dtype=np.int32)
    d_out = cp.empty_like(d_values, dtype=np.uintp)

    def run():
        lower_bound_run(
            d_data, d_values, d_out, build_only=(bench_fixture == "compile_benchmark")
        )

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_upper_bound(bench_fixture, request, size):
    actual_size = 100 if bench_fixture == "compile_benchmark" else size
    d_data = cp.sort(cp.random.randint(0, 1000, actual_size, dtype=np.int32))
    d_values = cp.random.randint(0, 1000, actual_size, dtype=np.int32)
    d_out = cp.empty_like(d_values, dtype=np.uintp)

    def run():
        upper_bound_run(
            d_data, d_values, d_out, build_only=(bench_fixture == "compile_benchmark")
        )

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)
