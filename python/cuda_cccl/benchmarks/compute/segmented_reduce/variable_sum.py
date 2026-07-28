# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for segmented_reduce (sum) with variable-size segments.

C++ equivalent: cub/benchmarks/bench/segmented_reduce/variable_sum.cu (uses variable_base.cuh)

Notes:
- Implements four sub-benchmarks: variable_default, variable_small_dynamic,
  variable_medium_dynamic, variable_large_dynamic.
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import (
    ALL_TYPES,
    as_cupy_stream,
    generate_data_with_entropy,
    generate_uniform_segment_offsets,
)

import cuda.bench as bench
from cuda.compute import OpKind, make_segmented_reduce

TYPE_MAP = {k: ALL_TYPES[k] for k in ("I32", "I64", "F32", "F64")}


def run_segmented_reduce(
    state: bench.State,
    d_in,
    d_out,
    h_init,
    start_offsets,
    end_offsets,
    num_segments,
    guaranteed_max_seg_size,
):
    reducer = make_segmented_reduce(
        d_in=d_in,
        d_out=d_out,
        start_offsets_in=start_offsets,
        end_offsets_in=end_offsets,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    temp_storage_bytes = reducer(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        num_segments=num_segments,
        start_offsets_in=start_offsets,
        end_offsets_in=end_offsets,
        op=OpKind.PLUS,
        h_init=h_init,
        max_segment_size=guaranteed_max_seg_size,
    )
    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    def launcher(launch: bench.Launch):
        reducer(
            temp_storage=temp_storage,
            d_in=d_in,
            d_out=d_out,
            num_segments=num_segments,
            start_offsets_in=start_offsets,
            end_offsets_in=end_offsets,
            op=OpKind.PLUS,
            h_init=h_init,
            max_segment_size=guaranteed_max_seg_size,
            stream=launch.get_stream(),
        )

    state.exec(launcher, batched=False, sync=True)


def bench_variable_segmented_reduce(state: bench.State):
    type_str = state.get_string("T{ct}")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements{io}"))
    max_segment_size = int(state.get_int64("MaxSegmentSize"))
    guaranteed_max_seg_size = int(state.get_int64("GuaranteedMaxSegSize"))

    # Skip cases where hint would be incorrect (max > guaranteed)
    if guaranteed_max_seg_size != 0 and max_segment_size > guaranteed_max_seg_size:
        state.skip("max_segment_size > guaranteed_max_seg_size")
        return

    min_segment_size = 1
    offsets = generate_uniform_segment_offsets(
        num_elements, min_segment_size, max_segment_size
    )

    alloc_stream = as_cupy_stream(state.get_stream())
    h_init = np.zeros(1, dtype=dtype)
    d_in = generate_data_with_entropy(num_elements, dtype, "1.000", alloc_stream)
    with alloc_stream:
        start_offsets = cp.asarray(offsets[:-1], dtype=np.int64)
        end_offsets = cp.asarray(offsets[1:], dtype=np.int64)
        d_out = cp.empty(int(start_offsets.size), dtype=dtype)

    alloc_stream.synchronize()
    num_segments = int(start_offsets.size)

    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in.dtype.itemsize)
    state.add_global_memory_writes(num_segments * d_out.dtype.itemsize)
    state.add_global_memory_reads((num_segments + 1) * start_offsets.dtype.itemsize)

    run_segmented_reduce(
        state,
        d_in,
        d_out,
        h_init,
        start_offsets,
        end_offsets,
        num_segments,
        guaranteed_max_seg_size,
    )


if __name__ == "__main__":
    # Default: no size hint — uses generic large-reduce kernel regardless of segment size
    b_default = bench.register(bench_variable_segmented_reduce)
    b_default.set_name("variable_default")
    b_default.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b_default.add_int64_power_of_two_axis("Elements{io}", range(16, 28, 4))
    b_default.add_int64_power_of_two_axis("MaxSegmentSize", range(1, 17, 1))
    b_default.add_int64_axis("GuaranteedMaxSegSize", [0])

    # Small segments (1–16 items): hint enables warp-level reduction
    b_small = bench.register(bench_variable_segmented_reduce)
    b_small.set_name("variable_small_dynamic")
    b_small.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b_small.add_int64_power_of_two_axis("Elements{io}", range(16, 28, 4))
    b_small.add_int64_power_of_two_axis("MaxSegmentSize", range(1, 5, 1))
    b_small.add_int64_power_of_two_axis("GuaranteedMaxSegSize", range(1, 5, 1))

    # Medium segments (32–256 items): hint enables warp-level reduction
    b_medium = bench.register(bench_variable_segmented_reduce)
    b_medium.set_name("variable_medium_dynamic")
    b_medium.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b_medium.add_int64_power_of_two_axis("Elements{io}", range(16, 28, 4))
    b_medium.add_int64_power_of_two_axis("MaxSegmentSize", range(5, 9, 1))
    b_medium.add_int64_power_of_two_axis("GuaranteedMaxSegSize", range(5, 9, 1))

    # Large segments (512+ items): hint enables block-level reduction
    b_large = bench.register(bench_variable_segmented_reduce)
    b_large.set_name("variable_large_dynamic")
    b_large.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b_large.add_int64_power_of_two_axis("Elements{io}", range(16, 28, 4))
    b_large.add_int64_power_of_two_axis("MaxSegmentSize", range(9, 17, 1))
    b_large.add_int64_power_of_two_axis("GuaranteedMaxSegSize", range(9, 17, 1))

    bench.run_all_benchmarks(sys.argv)
