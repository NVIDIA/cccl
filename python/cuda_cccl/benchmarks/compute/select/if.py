# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for select_if using cuda.compute.

C++ equivalent: cub/benchmarks/bench/select/if.cu

Notes:
- The C++ benchmark uses a `less_then_t<T>` predicate with threshold based on entropy
- Entropy controls what fraction of elements are selected:
  - 1.000 → selects ~100% (threshold = max value)
  - 0.544 → selects ~54.4% (threshold at 54.4% of range)
  - 0.000 → selects ~0% (threshold = min value)
- InPlace axis controls whether output can alias input (not exposed in Python API)
- Migration: Python cannot expose InPlace axis; output is sized to num_elements but metrics use actual selected count.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import (
    ENTROPY_TO_PROB,
    as_cupy_stream,
    generate_data_with_entropy,
    lerp_min_max,
)
from utils import (
    FUNDAMENTAL_TYPES as TYPE_MAP,
)

import cuda.bench as bench
from cuda.compute import make_select

# Entropy values from C++ benchmark
# These control the selection threshold and thus how many elements are selected


def bench_select_if(state: bench.State):
    type_str = state.get_string("T{ct}")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements{io}"))
    entropy_str = state.get_string("Entropy")

    probability = ENTROPY_TO_PROB[entropy_str]
    threshold = lerp_min_max(dtype, probability)

    alloc_stream = as_cupy_stream(state.get_stream())

    # Match C++ benchmark: input data generation is independent of Entropy.
    # Entropy only controls the selection threshold.
    d_in = generate_data_with_entropy(num_elements, dtype, "1.000", alloc_stream)
    with alloc_stream:
        selected_elements = int(cp.count_nonzero(d_in < threshold).get())
        d_out = cp.empty(selected_elements, dtype=dtype)

        d_num_selected = cp.zeros(1, dtype=np.int64)

    alloc_stream.synchronize()

    # Create predicate: select elements less than threshold
    # For numba device functions, we need to use the value directly in closure
    thresh_val = threshold

    def less_than_threshold(x):
        return x < thresh_val

    selector = make_select(d_in, d_out, d_num_selected, less_than_threshold)

    temp_storage_bytes = selector(
        None,
        d_in,
        d_out,
        d_num_selected,
        less_than_threshold,
        num_elements,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in.dtype.itemsize)
    state.add_global_memory_writes(selected_elements * d_out.dtype.itemsize)
    state.add_global_memory_writes(1 * d_num_selected.dtype.itemsize)

    def launcher(launch: bench.Launch):
        selector(
            temp_storage,
            d_in,
            d_out,
            d_num_selected,
            less_than_threshold,
            num_elements,
            launch.get_stream(),
        )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_select_if)
    b.set_name("base")

    b.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 29, 4))
    b.add_string_axis("Entropy", ["1.000", "0.544", "0.000"])
    # Note: InPlace axis is not exposed in Python API, so we skip it

    bench.run_all_benchmarks(sys.argv)
