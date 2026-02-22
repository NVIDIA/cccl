# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import as_cupy_stream

import cuda.bench as bench
from cuda.compute import OpKind, clear_all_caches, make_segmented_reduce

# Type mapping: match C++ all_types (fundamental types)
TYPE_MAP = {
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "F32": np.float32,
    "F64": np.float64,
}

# Segment size ranges from C++ benchmark (power of 2):
# - small: 2^0 to 2^4 (1, 2, 4, 8, 16)
# - medium: 2^5 to 2^8 (32, 64, 128, 256)
# - large: 2^9 to 2^16 (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
SEGMENT_SIZES_SMALL = [2**i for i in range(0, 5)]  # 1, 2, 4, 8, 16
SEGMENT_SIZES_MEDIUM = [2**i for i in range(5, 9)]  # 32, 64, 128, 256
SEGMENT_SIZES_LARGE = [2**i for i in range(9, 17)]  # 512 to 65536


def generate_segment_offsets(num_elements, segment_size, stream):
    """
    Generate uniform segment offsets for fixed-size segments.

    Args:
        num_elements: Total number of elements
        segment_size: Size of each segment
        stream: CuPy stream

    Returns:
        Tuple of (start_offsets, end_offsets, num_segments, actual_elements)
    """
    # Calculate number of complete segments
    num_segments = max(1, num_elements // segment_size)
    actual_elements = num_segments * segment_size

    with stream:
        # Generate start offsets: [0, segment_size, 2*segment_size, ...]
        start_offsets = cp.arange(0, actual_elements, segment_size, dtype=np.int64)
        # Generate end offsets: [segment_size, 2*segment_size, ...]
        end_offsets = cp.arange(
            segment_size, actual_elements + 1, segment_size, dtype=np.int64
        )

    return start_offsets, end_offsets, num_segments, actual_elements


def bench_segmented_reduce_sum(state: bench.State):
    """
    Benchmark segmented_reduce sum operation.
    """
    # WORKAROUND: Clear caches to avoid caching bug
    # See BUG_REPORT_CACHING.md for details
    clear_all_caches()

    # Get parameters from axes
    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements"))
    segment_size = int(state.get_int64("SegmentSize"))

    # Allocate arrays
    alloc_stream = as_cupy_stream(state.get_stream())

    # Generate segment offsets
    start_offsets, end_offsets, num_segments, actual_elements = (
        generate_segment_offsets(num_elements, segment_size, alloc_stream)
    )

    # Generate input data
    with alloc_stream:
        # Random data in a reasonable range to avoid overflow
        if np.issubdtype(dtype, np.integer):
            # Limit range to avoid overflow during reduction
            max_val = min(100, np.iinfo(dtype).max)
            min_val = max(-100, np.iinfo(dtype).min)
            d_in = cp.random.randint(
                min_val, max_val + 1, size=actual_elements, dtype=dtype
            )
        else:
            d_in = cp.random.uniform(-100, 100, size=actual_elements).astype(dtype)

        # Output array (one result per segment)
        d_out = cp.empty(num_segments, dtype=dtype)

    # Initial value for sum reduction
    h_init = np.array(0, dtype=dtype)

    # Synchronize to ensure data is ready
    alloc_stream.synchronize()

    # Build segmented reduce operation
    reducer = make_segmented_reduce(
        d_in=d_in,
        d_out=d_out,
        start_offsets_in=start_offsets,
        end_offsets_in=end_offsets,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    # Get temp storage size and allocate
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

    # Warmup run to catch any CUDA errors before benchmarking
    try:
        reducer(
            temp_storage=temp_storage,
            d_in=d_in,
            d_out=d_out,
            op=OpKind.PLUS,
            num_segments=num_segments,
            start_offsets_in=start_offsets,
            end_offsets_in=end_offsets,
            h_init=h_init,
        )
        cp.cuda.Device().synchronize()
    except Exception as e:
        state.skip(f"CUDA error during warmup: {e}")
        return

    # Match C++ metrics
    state.add_element_count(actual_elements)
    state.add_global_memory_reads(actual_elements * d_in.dtype.itemsize)
    state.add_global_memory_writes(num_segments * d_out.dtype.itemsize)

    # Execute benchmark
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

    state.exec(launcher)


if __name__ == "__main__":
    # Register three benchmarks matching C++ structure:
    # - small: SegmentSize 2^0 to 2^4
    # - medium: SegmentSize 2^5 to 2^8
    # - large: SegmentSize 2^9 to 2^16

    # Small segments benchmark
    b_small = bench.register(bench_segmented_reduce_sum)
    b_small.set_name("small")
    b_small.add_string_axis("T", list(TYPE_MAP.keys()))
    b_small.add_int64_power_of_two_axis("Elements", range(16, 29, 4))
    b_small.add_int64_axis("SegmentSize", SEGMENT_SIZES_SMALL)

    # Medium segments benchmark
    b_medium = bench.register(bench_segmented_reduce_sum)
    b_medium.set_name("medium")
    b_medium.add_string_axis("T", list(TYPE_MAP.keys()))
    b_medium.add_int64_power_of_two_axis("Elements", range(16, 29, 4))
    b_medium.add_int64_axis("SegmentSize", SEGMENT_SIZES_MEDIUM)

    # Large segments benchmark
    b_large = bench.register(bench_segmented_reduce_sum)
    b_large.set_name("large")
    b_large.add_string_axis("T", list(TYPE_MAP.keys()))
    b_large.add_int64_power_of_two_axis("Elements", range(16, 29, 4))
    b_large.add_int64_axis("SegmentSize", SEGMENT_SIZES_LARGE)

    bench.run_all_benchmarks(sys.argv)
