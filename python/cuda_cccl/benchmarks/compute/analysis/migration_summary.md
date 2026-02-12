# Python -> C++ CUB Benchmark Migration Summary

Notes:
- Each entry compares the Python benchmark against the C++ CUB benchmark under `cub/benchmarks/bench`.
- Focus is on API usage parity, axes alignment, and notable behavior differences.

## transform/fill
- Python: `python/cuda_cccl/benchmarks/compute/transform/fill.py`
- C++: `cub/benchmarks/bench/transform/fill.cu`
- API parity: Uses `ConstantIterator` + `make_unary_transform` with `OpKind.IDENTITY`, matching `return_constant<T>` and `bench_transform`.
- Differences: Python covers only signed integers (I8-I64) like C++ integral_types; no tune parameters (expected).

## transform/babelstream (mul/add/triad/nstream)
- Python: `python/cuda_cccl/benchmarks/compute/transform/babelstream.py`
- C++: `cub/benchmarks/bench/transform/babelstream.cu`
- API parity: Uses `make_unary_transform` and `make_binary_transform`, plus `ZipIterator` for 3-input nstream, matching C++ functors.
- Differences: C++ has `OffsetT` axis and optional int128; Python omits OffsetT and int128. Entropy not used in either.

## transform/heavy
- Python: `python/cuda_cccl/benchmarks/compute/transform/heavy.py`
- C++: `cub/benchmarks/bench/transform/heavy.cu`
- API parity: Uses `make_unary_transform` with heavy functor variants (32/64/128/256 regs), matching C++ `heavy_functor<N>`.
- Differences: Python uses Numba local arrays to emulate register pressure; C++ uses local arrays directly.

## transform/fib (fibonacci)
- Python: `python/cuda_cccl/benchmarks/compute/transform/fib.py`
- C++: `cub/benchmarks/bench/transform/fib.cu`
- API parity: `make_unary_transform` with fib function and uint32 outputs, matching C++ `fib_t`.
- Differences: C++ uses `OffsetT` axis; Python fixed to int64 offsets. Input generation uses CuPy random vs C++ `generate` entropy helper.

## transform/grayscale
- Python: `python/cuda_cccl/benchmarks/compute/transform/grayscale.py`
- C++: `cub/benchmarks/bench/transform/grayscale.cu`
- API parity: `gpu_struct` for RGB + `make_unary_transform`, matching C++ `rgb_t` + transform functor.
- Differences: Python constructs host RGB then memcpy to device; C++ builds on device with zip + transform.

## transform/complex_cmp (compare_complex)
- Python: `python/cuda_cccl/benchmarks/compute/transform/complex_cmp.py`
- C++: `cub/benchmarks/bench/transform/complex_cmp.cu`
- API parity: `make_binary_transform` over overlapping ranges (in[0:n-1], in[1:n]) with lexicographic compare; matches C++ overlap behavior.
- Differences: C++ uses `less_t` on `complex` type; Python implements explicit lexicographic compare for `complex64`.

## transform_reduce/sum
- Python: `python/cuda_cccl/benchmarks/compute/transform_reduce/sum.py`
- C++: `cub/benchmarks/bench/transform_reduce/sum.cu`
- API parity: Uses `TransformIterator` + `make_reduce_into` with `OpKind.PLUS`, matching C++ transform-reduce path.
- Differences: C++ includes `OffsetT` axis and tuning; Python fixes offsets and omits int128/complex types.

## reduce/sum
- Python: `python/cuda_cccl/benchmarks/compute/reduce/sum.py`
- C++: `cub/benchmarks/bench/reduce/sum.cu` (via `base.cuh`)
- API parity: `make_reduce_into` with `OpKind.PLUS`, matches C++ use of `cuda::std::plus<>`.
- Differences: Python excludes int128/complex32; C++ supports more types and tuning paths.

## reduce/min
- Python: `python/cuda_cccl/benchmarks/compute/reduce/min.py`
- C++: `cub/benchmarks/bench/reduce/min.cu` (via `base.cuh`)
- API parity: `make_reduce_into` with `OpKind.MINIMUM`, matching C++ `cuda::minimum<>` (DPX path).
- Differences: Python uses manual init with max value; C++ uses `base.cuh` defaults. Cache-clear workaround in Python.

## reduce/custom
- Python: `python/cuda_cccl/benchmarks/compute/reduce/custom.py`
- C++: `cub/benchmarks/bench/reduce/custom.cu`
- API parity: Custom `max_op` passed to `make_reduce_into`, matching C++ custom operator path.
- Differences: C++ includes int128/complex; Python limits to basic numeric types.

## reduce/deterministic
- Python: `python/cuda_cccl/benchmarks/compute/reduce/deterministic.py`
- C++: `cub/benchmarks/bench/reduce/deterministic.cu`
- API parity: Uses `Determinism.RUN_TO_RUN` + `OpKind.PLUS`, matching deterministic reduce path.
- Differences: Both restrict to float/double; Python uses int64 offsets by default (C++ uses int).

## reduce/nondeterministic
- Python: `python/cuda_cccl/benchmarks/compute/reduce/nondeterministic.py`
- C++: `cub/benchmarks/bench/reduce/nondeterministic.cu`
- API parity: Uses `Determinism.NOT_GUARANTEED` + `OpKind.PLUS`, matching nondeterministic reduce path.
- Differences: C++ includes `OffsetT` axis; Python fixes offsets. Type coverage matches (I32/I64/F32/F64).

## scan/exclusive/sum
- Python: `python/cuda_cccl/benchmarks/compute/scan/exclusive/sum.py`
- C++: `cub/benchmarks/bench/scan/exclusive/sum.cu` (via `base.cuh`)
- API parity: `make_exclusive_scan` with `OpKind.PLUS`, matching C++ `cuda::std::plus<>` scan.
- Differences: C++ includes `OffsetT` axis; Python fixes offsets. No tune parameters in Python.

## scan/exclusive/custom
- Python: `python/cuda_cccl/benchmarks/compute/scan/exclusive/custom.py`
- C++: `cub/benchmarks/bench/scan/exclusive/custom.cu`
- API parity: Custom `max_op` passed to `make_exclusive_scan`, matching C++ custom op benchmark.
- Differences: Same as sum (no OffsetT axis in Python).

## histogram/even
- Python: `python/cuda_cccl/benchmarks/compute/histogram/even.py`
- C++: `cub/benchmarks/bench/histogram/even.cu`
- API parity: Uses `make_histogram_even` with `h_num_output_levels`, `h_lower_level`, `h_upper_level`, matching C++ DispatchEven signature.
- Differences: Python approximates entropy distributions; C++ uses `generate` helper. Python skips some (I8/I16 + large bins) due to CUDA errors; C++ does not.

## select/if
- Python: `python/cuda_cccl/benchmarks/compute/select/if.py`
- C++: `cub/benchmarks/bench/select/if.cu`
- API parity: Uses `make_select` with predicate, matching C++ `SelectIf` path.
- Differences: C++ includes `InPlace` axis; Python cannot expose it. Output sized to `num_elements` in Python, but writes are measured using actual `d_num_selected`.

## select/flagged
- Python: `python/cuda_cccl/benchmarks/compute/select/flagged.py`
- C++: `cub/benchmarks/bench/select/flagged.cu`
- API parity: Uses `make_select` with boolean flags, matching C++ `SelectFlagged` path.
- Differences: C++ uses same generator for input/flags; Python samples flags independently. Output sized to selected count as in C++.

## select/unique_by_key
- Python: `python/cuda_cccl/benchmarks/compute/select/unique_by_key.py`
- C++: `cub/benchmarks/bench/select/unique_by_key.cu`
- API parity: Uses `make_unique_by_key` with `OpKind.EQUAL_TO`, matching C++ `equal_to` comparison.
- Differences: C++ has `OffsetT` axis; Python fixes offsets. Python generates key segments on CPU to mirror `generate.uniform.key_segments`.

## radix_sort/keys
- Python: `python/cuda_cccl/benchmarks/compute/radix_sort/keys.py`
- C++: `cub/benchmarks/bench/radix_sort/keys.cu`
- API parity: Uses `make_radix_sort` with ascending order and full bit range, matching C++ dispatch.
- Differences: C++ includes `OffsetT` axis; Python fixes offsets and excludes int128. Entropy generation approximates C++ `generate`.

## radix_sort/pairs
- Python: `python/cuda_cccl/benchmarks/compute/radix_sort/pairs.py`
- C++: `cub/benchmarks/bench/radix_sort/pairs.cu`
- API parity: Uses `make_radix_sort` for key/value pairs, ascending order, full bit range.
- Differences: C++ supports int128 values; Python limits to I8-I64. OffsetT axis omitted in Python.

## merge_sort/keys
- Python: `python/cuda_cccl/benchmarks/compute/merge_sort/keys.py`
- C++: `cub/benchmarks/bench/merge_sort/keys.cu`
- API parity: Uses `make_merge_sort` with `OpKind.LESS`, matching C++ `less_t`.
- Differences: C++ includes `OffsetT` axis; Python fixes offsets. Entropy generation approximated.

## merge_sort/pairs
- Python: `python/cuda_cccl/benchmarks/compute/merge_sort/pairs.py`
- C++: `cub/benchmarks/bench/merge_sort/pairs.cu`
- API parity: Uses `make_merge_sort` with key/value pairs and `OpKind.LESS`, matching C++ path.
- Differences: C++ includes int128 values and `OffsetT` axis; Python omits both.

## segmented_sort/keys (power/small/large)
- Python: `python/cuda_cccl/benchmarks/compute/segmented_sort/keys.py`
- C++: `cub/benchmarks/bench/segmented_sort/keys.cu`
- API parity: Uses `make_segmented_sort` with start/end offsets and ascending order, matching C++ DispatchSegmentedSort.
- Differences: Power-law offsets use Zipf approximation vs C++ generator. Uniform offsets use min_segment_size ~ max/2, consistent with C++ `1 << (log2(max)-1)` for power-of-two sizes.

## segmented_reduce/sum (small/medium/large)
- Python: `python/cuda_cccl/benchmarks/compute/segmented_reduce/sum.py`
- C++: `cub/benchmarks/bench/segmented_reduce/sum.cu` (via `base.cuh`)
- API parity: Uses `make_segmented_reduce` with fixed-size segments and `OpKind.PLUS`, matching C++ fixed-size segmented reduce.
- Differences: C++ uses implicit fixed-size segments; Python builds explicit start/end offsets. Input range is clamped in Python to avoid overflow.

## segmented_reduce/custom (small/medium/large)
- Python: `python/cuda_cccl/benchmarks/compute/segmented_reduce/custom.py`
- C++: `cub/benchmarks/bench/segmented_reduce/custom.cu` (via `base.cuh`)
- API parity: Custom `max_op` passed to `make_segmented_reduce`, matching C++ custom operator path.
- Differences: Same as sum (explicit offsets in Python, fixed-size in C++).

## partition/three_way
- Python: `python/cuda_cccl/benchmarks/compute/partition/three_way.py`
- C++: `cub/benchmarks/bench/partition/three_way.cu`
- API parity: Uses `make_three_way_partition` with two predicates, matching C++ `DispatchThreeWayPartitionIf`.
- Differences: C++ uses `min_val{}` (0) and `max_val` for borders; Python mirrors this for signed/unsigned types. OffsetT axis omitted in Python.

## binary_search (lower_bound/upper_bound)
- Python: `python/cuda_cccl/benchmarks/compute/bench_binary_search.py`
- C++: No CUB benchmark found under `cub/benchmarks/bench`.
- API parity: Uses `make_lower_bound` / `make_upper_bound` on sorted inputs; matches expected CUB device search API usage.
- Notes: Not listed in `run_benchmarks.py` supported set; appears to be standalone Python-only benchmark.

## coop/warp_reduce (warp_sum/warp_min)
- Python: `python/cuda_cccl/benchmarks/coop/bench_warp_reduce.py`
- C++: `cub/benchmarks/bench/reduce/warp_reduce_sum.cu`, `cub/benchmarks/bench/reduce/warp_reduce_min.cu`
- API parity: Uses `cuda.coop.warp.sum` and `cuda.coop.warp.reduce` inside a device-side kernel with block=256 and unroll=128, matching C++ device-side benchmark structure.
- Differences: Python uses Numba-generated kernel and local random data via inline PTX; C++ uses `cub::WarpReduce` in `warp_reduce_base.cuh`.
