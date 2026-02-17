# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from numba import cuda as numba_cuda
from utils import as_cupy_stream

import cuda.bench as bench
from cuda.compute import clear_all_caches, make_select

# Type mapping: match C++ fundamental_types (excluding int128 and complex which Python doesn't support)
TYPE_MAP = {
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "F32": np.float32,
    "F64": np.float64,
}

# Entropy values from C++ benchmark
# These control the selection threshold and thus how many elements are selected
ENTROPY_VALUES = ["1.000", "0.544", "0.000"]

# Entropy to probability mapping (from nvbench_helper.cuh)
ENTROPY_TO_PROB = {
    "1.000": 1.0,
    "0.811": 0.811,
    "0.544": 0.544,
    "0.337": 0.337,
    "0.201": 0.201,
    "0.000": 0.0,
}


def lerp_min_max(dtype, probability):
    """
    Compute threshold value by interpolating between min and max for dtype.
    Mirrors C++ lerp_min_max() from nvbench_helper.cuh.

    Args:
        dtype: NumPy dtype
        probability: Value between 0.0 and 1.0

    Returns:
        Threshold value of the given dtype
    """
    if probability == 1.0:
        if np.issubdtype(dtype, np.integer):
            return np.iinfo(dtype).max
        else:
            return np.finfo(dtype).max

    if np.issubdtype(dtype, np.integer):
        min_val = float(np.iinfo(dtype).min)
        max_val = float(np.iinfo(dtype).max)
    else:
        min_val = float(np.finfo(dtype).min)
        max_val = float(np.finfo(dtype).max)

    # Linear interpolation
    result = min_val + probability * (max_val - min_val)
    return dtype(result)


def make_less_than_predicate(threshold):
    """
    Create a less-than predicate function for the given threshold.
    This mirrors C++ less_then_t<T>.
    """
    # Capture threshold in closure
    thresh = threshold

    @numba_cuda.jit(device=True)
    def less_than(x):
        return x < thresh

    return less_than


def bench_select_if(state: bench.State):
    """
    Benchmark select_if operation.
    """
    # WORKAROUND: Clear caches to avoid caching bug with different predicates
    # See BUG_REPORT_CACHING.md for details
    clear_all_caches()

    # Get parameters from axes
    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements"))
    entropy_str = state.get_string("Entropy")

    # Calculate threshold based on entropy
    probability = ENTROPY_TO_PROB[entropy_str]
    threshold = lerp_min_max(dtype, probability)

    # Allocate arrays
    alloc_stream = as_cupy_stream(state.get_stream())

    with alloc_stream:
        # Generate random input data (uniform distribution across full range)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            # Use int64 for randint to avoid overflow issues with large ranges
            d_in = cp.random.randint(
                int(info.min), int(info.max) + 1, size=num_elements, dtype=np.int64
            ).astype(dtype)
        else:
            # For floats, generate in [0, 1) and scale to full range
            info = np.finfo(dtype)
            d_in = cp.random.uniform(0, 1, size=num_elements).astype(dtype)
            # Scale to approximately [-max, max] range
            d_in = d_in * 2 * info.max - info.max

        selected_elements = int(cp.count_nonzero(d_in < threshold).get())
        d_out = cp.empty(selected_elements, dtype=dtype)

        # Number of selected elements output
        d_num_selected = cp.zeros(1, dtype=np.uint64)

    # Synchronize to ensure data is ready
    alloc_stream.synchronize()

    # Create predicate: select elements less than threshold
    # For numba device functions, we need to use the value directly in closure
    thresh_val = threshold

    def less_than_threshold(x):
        return x < thresh_val

    # Build select operation
    selector = make_select(
        d_in=d_in,
        d_out=d_out,
        d_num_selected_out=d_num_selected,
        cond=less_than_threshold,
    )

    # Get temp storage size and allocate
    temp_storage_bytes = selector(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        d_num_selected_out=d_num_selected,
        cond=less_than_threshold,
        num_items=num_elements,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    # Warmup run to catch any CUDA errors before benchmarking
    try:
        selector(
            temp_storage=temp_storage,
            d_in=d_in,
            d_out=d_out,
            d_num_selected_out=d_num_selected,
            cond=less_than_threshold,
            num_items=num_elements,
        )
        cp.cuda.Device().synchronize()
    except Exception as e:
        state.skip(f"CUDA error during warmup: {e}")
        return

    # Get actual number of selected elements for metrics
    num_selected = int(d_num_selected.get()[0])

    # Match C++ metrics
    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in.dtype.itemsize)
    state.add_global_memory_writes(num_selected * d_out.dtype.itemsize)
    state.add_global_memory_writes(1 * d_num_selected.dtype.itemsize)

    # Execute benchmark
    def launcher(launch: bench.Launch):
        selector(
            temp_storage=temp_storage,
            d_in=d_in,
            d_out=d_out,
            d_num_selected_out=d_num_selected,
            cond=less_than_threshold,
            num_items=num_elements,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(bench_select_if)
    b.set_name("base")  # Match C++ benchmark name

    # Match C++ axes
    b.add_string_axis("T", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 29, 4))  # [16, 20, 24, 28]
    b.add_string_axis("Entropy", ENTROPY_VALUES)
    # Note: InPlace axis is not exposed in Python API, so we skip it

    bench.run_all_benchmarks(sys.argv)
