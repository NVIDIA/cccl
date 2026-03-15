# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for segmented_reduce sum using cuda.compute.

C++ equivalent: cub/benchmarks/bench/segmented_reduce/sum.cu

Notes:
- The C++ benchmark uses fixed-size segments (SegmentSize axis)
- Python API uses start/end offset arrays, so we generate uniform segment offsets
- Three sub-benchmarks matching C++:
  - small: SegmentSize 2^0 to 2^4 (1, 2, 4, 8, 16)
  - medium: SegmentSize 2^5 to 2^8 (32, 64, 128, 256)
  - large: SegmentSize 2^9 to 2^16 (512, 1024, ..., 65536)
- Uses OpKind.PLUS for sum reduction
- Migration: Python builds explicit offsets and clamps input range; C++ uses fixed-size segments.
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import (
    SIGNED_TYPES as TYPE_MAP,
)
from utils import (
    as_cupy_stream,
    generate_data_with_entropy,
    generate_fixed_segment_offsets,
)

import cuda.bench as bench
from cuda.compute import OpKind, make_segmented_reduce

# Segment size ranges from C++ benchmark (power of 2):
# - small: 2^0 to 2^4 (1, 2, 4, 8, 16)
# - medium: 2^5 to 2^8 (32, 64, 128, 256)
# - large: 2^9 to 2^16 (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)


def bench_segmented_reduce_sum(state: bench.State):
    type_str = state.get_string("T{ct}")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements{io}"))
    segment_size = int(state.get_int64("SegmentSize"))

    alloc_stream = as_cupy_stream(state.get_stream())

    # Generate segment offsets
    start_offsets, end_offsets, num_segments, actual_elements = (
        generate_fixed_segment_offsets(num_elements, segment_size, alloc_stream)
    )

    with alloc_stream:
        d_in = generate_data_with_entropy(actual_elements, dtype, "1.000", alloc_stream)

        # Output array (one result per segment)
        d_out = cp.empty(num_segments, dtype=dtype)

    # Initial value for sum reduction
    h_init = np.array(0, dtype=dtype)

    alloc_stream.synchronize()

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
        op=OpKind.PLUS,
        num_segments=num_segments,
        start_offsets_in=start_offsets,
        end_offsets_in=end_offsets,
        h_init=h_init,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    state.add_element_count(actual_elements)
    state.add_global_memory_reads(actual_elements * d_in.dtype.itemsize, "Size")
    state.add_global_memory_writes(num_segments * d_out.dtype.itemsize)

    def launcher(launch: bench.Launch):
        reducer(
            temp_storage=temp_storage,
            d_in=d_in,
            d_out=d_out,
            op=OpKind.PLUS,
            num_segments=num_segments,
            start_offsets_in=start_offsets,
            end_offsets_in=end_offsets,
            h_init=h_init,
            stream=launch.get_stream(),
        )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    # Register three benchmarks matching C++ structure:
    # - small: SegmentSize 2^0 to 2^4
    # - medium: SegmentSize 2^5 to 2^8
    # - large: SegmentSize 2^9 to 2^16

    # Small segments benchmark
    b_small = bench.register(bench_segmented_reduce_sum)
    b_small.set_name("small")
    b_small.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b_small.add_int64_power_of_two_axis("Elements{io}", range(16, 29, 4))
    b_small.add_int64_power_of_two_axis("SegmentSize", range(0, 5))

    # Medium segments benchmark
    b_medium = bench.register(bench_segmented_reduce_sum)
    b_medium.set_name("medium")
    b_medium.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b_medium.add_int64_power_of_two_axis("Elements{io}", range(16, 29, 4))
    b_medium.add_int64_power_of_two_axis("SegmentSize", range(5, 9))

    # Large segments benchmark
    b_large = bench.register(bench_segmented_reduce_sum)
    b_large.set_name("large")
    b_large.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b_large.add_int64_power_of_two_axis("Elements{io}", range(16, 29, 4))
    b_large.add_int64_power_of_two_axis("SegmentSize", range(9, 17))

    bench.run_all_benchmarks(sys.argv)
