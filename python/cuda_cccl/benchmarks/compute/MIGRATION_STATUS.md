# CUB to Python Benchmark Migration Status

Source of truth: C++ benchmarks under `cub/benchmarks/bench/**/*.cu`.
Python implementations live under `python/cuda_cccl/benchmarks/compute/`.

## Summary

- Total C++ benchmarks: 58
- Implemented in Python: 25
- Not implemented in Python: 33

## Mapping

Status key:
- Implemented: Python benchmark exists
- Not implemented: No Python benchmark yet

| C++ Benchmark | Python Benchmark | Status | Notes |
| --- | --- | --- | --- |
| `adjacent_difference/subtract_left.cu` | - | Not implemented | No Python API equivalent |
| `copy/memcpy.cu` | - | Not implemented | No Python API equivalent |
| `find_if/base.cu` | - | Not implemented | No Python API equivalent |
| `for_each/base.cu` | - | Not implemented | Possible workaround via `unary_transform` + `DiscardIterator` |
| `for_each/copy.cu` | - | Not implemented | Possible workaround via `unary_transform` |
| `for_each/extents.cu` | - | Not implemented | No Python API equivalent |
| `histogram/even.cu` | `histogram/even.py` | Implemented | |
| `histogram/multi/even.cu` | - | Not implemented | No Python API equivalent |
| `histogram/multi/range.cu` | - | Not implemented | No Python API equivalent |
| `histogram/range.cu` | - | Not implemented | No Python API equivalent |
| `merge/keys.cu` | - | Not implemented | No Python API equivalent |
| `merge/pairs.cu` | - | Not implemented | No Python API equivalent |
| `merge_sort/keys.cu` | `merge_sort/keys.py` | Implemented | |
| `merge_sort/pairs.cu` | `merge_sort/pairs.py` | Implemented | |
| `partition/flagged.cu` | - | Not implemented | Possible workaround via `select` (not a true partition) |
| `partition/if.cu` | - | Not implemented | Possible workaround via `select` (not a true partition) |
| `partition/three_way.cu` | `partition/three_way.py` | Implemented | |
| `radix_sort/keys.cu` | `radix_sort/keys.py` | Implemented | |
| `radix_sort/pairs.cu` | `radix_sort/pairs.py` | Implemented | |
| `reduce/arg_extrema.cu` | - | Not implemented | No Python API equivalent |
| `reduce/by_key.cu` | - | Not implemented | No Python API equivalent |
| `reduce/custom.cu` | `reduce/custom.py` | Implemented | |
| `reduce/deterministic.cu` | `reduce/deterministic.py` | Implemented | |
| `reduce/min.cu` | `reduce/min.py` | Implemented | |
| `reduce/nondeterministic.cu` | `reduce/nondeterministic.py` | Implemented | |
| `reduce/sum.cu` | `reduce/sum.py` | Implemented | |
| `reduce/warp_reduce_min.cu` | - | Not implemented | No Python API equivalent |
| `reduce/warp_reduce_sum.cu` | - | Not implemented | No Python API equivalent |
| `run_length_encode/encode.cu` | - | Not implemented | No Python API equivalent |
| `run_length_encode/non_trivial_runs.cu` | - | Not implemented | No Python API equivalent |
| `scan/applications/P1/log-cdf-from-log-pdfs.cu` | - | Not implemented | Complex custom operator |
| `scan/applications/P1/non-commutative-bicyclic-monoid.cu` | - | Not implemented | Complex custom operator |
| `scan/applications/P1/rabin-karp-second-fingerprinting.cu` | - | Not implemented | Complex custom operator |
| `scan/applications/P1/running-min-max.cu` | - | Not implemented | Complex custom operator |
| `scan/applications/P1/scan-over-unitriangular-group.cu` | - | Not implemented | Complex custom operator |
| `scan/exclusive/by_key.cu` | - | Not implemented | No Python API equivalent |
| `scan/exclusive/custom.cu` | `scan/exclusive/custom.py` | Implemented | |
| `scan/exclusive/sum.cu` | `scan/exclusive/sum.py` | Implemented | |
| `segmented_radix_sort/keys.cu` | - | Not implemented | No Python API equivalent |
| `segmented_reduce/argmin.cu` | - | Not implemented | No Python API equivalent |
| `segmented_reduce/custom.cu` | `segmented_reduce/custom.py` | Implemented | |
| `segmented_reduce/sum.cu` | `segmented_reduce/sum.py` | Implemented | |
| `segmented_sort/keys.cu` | `segmented_sort/keys.py` | Implemented | |
| `segmented_topk/keys.cu` | - | Not implemented | No Python API equivalent |
| `select/flagged.cu` | `select/flagged.py` | Implemented | |
| `select/if.cu` | `select/if.py` | Implemented | |
| `select/unique.cu` | - | Not implemented | No Python API equivalent |
| `select/unique_by_key.cu` | `select/unique_by_key.py` | Implemented | |
| `topk/keys.cu` | - | Not implemented | No Python API equivalent |
| `topk/pairs.cu` | - | Not implemented | No Python API equivalent |
| `transform/babelstream.cu` | `transform/babelstream.py` | Implemented | |
| `transform/complex_cmp.cu` | `transform/complex_cmp.py` | Implemented | |
| `transform/fib.cu` | `transform/fib.py` | Implemented | |
| `transform/fill.cu` | `transform/fill.py` | Implemented | |
| `transform/grayscale.cu` | `transform/grayscale.py` | Implemented | |
| `transform/heavy.cu` | `transform/heavy.py` | Implemented | |
| `transform/pytorch.cu` | - | Not implemented | No Python API equivalent |
| `transform_reduce/sum.cu` | - | Not implemented | Possible via `TransformIterator` + `reduce_into` |
