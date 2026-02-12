# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for select flagged using cuda.compute.select.

C++ equivalent: cub/benchmarks/bench/select/flagged.cu

Notes:
- Uses a boolean flag array to select elements
- Entropy controls the selection probability
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import as_cupy_stream

import cuda.bench as bench
from cuda.compute import (
    TransformOutputIterator,
    ZipIterator,
    clear_all_caches,
    make_select,
)

# Type mapping: match C++ fundamental_types (excluding int128 and complex)
TYPE_MAP = {
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
    "U16": np.uint16,
    "U32": np.uint32,
    "U64": np.uint64,
    "F32": np.float32,
    "F64": np.float64,
}

ENTROPY_VALUES = ["1.000", "0.544", "0.000"]

ENTROPY_TO_PROB = {
    "1.000": 1.0,
    "0.811": 0.811,
    "0.544": 0.544,
    "0.337": 0.337,
    "0.201": 0.201,
    "0.000": 0.0,
}


def bench_select_flagged(state: bench.State):
    # WORKAROUND: Clear caches to avoid caching bug
    # See BUG_REPORT_CACHING.md for details
    clear_all_caches()

    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements"))
    entropy_str = state.get_string("Entropy")

    probability = ENTROPY_TO_PROB[entropy_str]

    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            d_in = cp.random.randint(
                int(info.min), int(info.max) + 1, size=num_elements
            ).astype(dtype)
        else:
            d_in = cp.random.uniform(-1, 1, size=num_elements).astype(dtype)

        # Flags: select with probability p
        flags = (cp.random.random(num_elements) < probability).astype(np.uint8)

        zip_it = ZipIterator(d_in, flags)
        selected_elements = int(cp.count_nonzero(flags).get())
        d_out = cp.empty(selected_elements, dtype=dtype)
        d_num_selected = cp.empty(1, dtype=np.uint64)

    def take_value(pair):
        return pair[0]

    def flag_predicate(pair):
        return np.uint8(pair[1] != 0)

    take_value.__annotations__ = {"pair": zip_it.value_type}
    flag_predicate.__annotations__ = {
        "pair": zip_it.value_type,
        "return": np.uint8,
    }

    d_out_it = TransformOutputIterator(d_out, take_value)

    selector = make_select(
        d_in=zip_it,
        d_out=d_out_it,
        d_num_selected_out=d_num_selected,
        cond=flag_predicate,
    )

    temp_storage_bytes = selector(
        temp_storage=None,
        d_in=zip_it,
        d_out=d_out_it,
        d_num_selected_out=d_num_selected,
        cond=flag_predicate,
        num_items=num_elements,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    # Warmup run to catch any CUDA errors before benchmarking
    try:
        selector(
            temp_storage=temp_storage,
            d_in=zip_it,
            d_out=d_out_it,
            d_num_selected_out=d_num_selected,
            cond=flag_predicate,
            num_items=num_elements,
        )
        cp.cuda.Device().synchronize()
    except Exception as e:
        state.skip(f"CUDA error during warmup: {e}")
        return

    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in.dtype.itemsize)
    state.add_global_memory_reads(num_elements * flags.dtype.itemsize)
    state.add_global_memory_writes(selected_elements * d_out.dtype.itemsize)
    state.add_global_memory_writes(d_num_selected.dtype.itemsize)

    def launcher(launch: bench.Launch):
        selector(
            temp_storage=temp_storage,
            d_in=zip_it,
            d_out=d_out_it,
            d_num_selected_out=d_num_selected,
            cond=flag_predicate,
            num_items=num_elements,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(bench_select_flagged)
    b.set_name("base")  # Match C++ benchmark name
    b.add_string_axis("T", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 29, 4))
    b.add_string_axis("Entropy", ENTROPY_VALUES)
    bench.run_all_benchmarks(sys.argv)
