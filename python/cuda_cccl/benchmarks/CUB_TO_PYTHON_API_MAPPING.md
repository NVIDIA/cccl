# CUB Benchmarks to Python cuda.compute API Mapping

This document maps all 55 CUB benchmark files to their Python `cuda.compute` equivalents, indicating which are **implementable**, **partially implementable**, or **not yet available**.

## Legend
- ✅ **Fully Supported** - Direct Python API exists
- ⚠️ **Partially Supported** - Can implement with workarounds/combinations
- ❌ **Not Supported** - No Python API available yet
- 🔧 **Requires Iterator Combo** - Need to combine iterators creatively

---

## Transform Benchmarks (7 files) - 86% Coverage

| CUB Benchmark | CUB API | Python API | Status | Notes |
|---------------|---------|------------|--------|-------|
| `transform/fill.cu` | `cub::detail::transform::dispatch` with `return_constant` | `unary_transform` + `ConstantIterator` | ✅ **DONE** | Using ConstantIterator(42) as 0-ary generator |
| `transform/babelstream.cu` | `cub::detail::transform::dispatch` (mul, add, triad, nstream) | `unary_transform`, `binary_transform` | ✅ **Ready** | Multiple operations: a*scalar, a+b, a+b*scalar, a+b+c*scalar |
| `transform/heavy.cu` | `cub::detail::transform::dispatch` with heavy compute | `unary_transform` | ✅ **DONE** | Uses `cuda.local.array` to mimic register pressure |
| `transform/fib.cu` | `cub::detail::transform::dispatch` | `unary_transform` | ✅ **Ready** | Fibonacci computation |
| `transform/grayscale.cu` | `cub::detail::transform::dispatch` with struct input | `unary_transform` + `@gpu_struct` | ✅ **Ready** | RGB struct → grayscale, need gpu_struct for RGB type |
| `transform/complex_cmp.cu` | `cub::detail::transform::dispatch` | `unary_transform` | ✅ **Ready** | Complex comparison logic |
| `transform_reduce/sum.cu` | `cub::detail::reduce::dispatch` with transform | `reduce_into` with `TransformIterator` | ⚠️ **Workaround** | Need to use TransformIterator or pre-transform |

---

## Reduce Benchmarks (9 files) - 56% Coverage

| CUB Benchmark | CUB API | Python API | Status | Notes |
|---------------|---------|------------|--------|-------|
| `reduce/sum.cu` | `cub::DispatchReduce` with `cuda::std::plus<>` | `reduce_into` + `OpKind.PLUS` | ✅ **DONE** | Direct match |
| `reduce/min.cu` | `cub::DispatchReduce` with `cuda::minimum<>` | `reduce_into` + `OpKind.MINIMUM` | ✅ **DONE** | Direct match, DPX optimized on Hopper+ |
| `reduce/custom.cu` | `cub::DispatchReduce` with custom `max_t` | `reduce_into` + custom function | ✅ **Ready** | Python supports custom operators |
| `reduce/deterministic.cu` | `cub::detail::rfa::dispatch_t` | `reduce_into(determinism=Determinism.RUN_TO_RUN)` | ✅ **Ready** | Python has determinism kwarg |
| `reduce/nondeterministic.cu` | `cub::detail::reduce::dispatch_nondeterministic` | `reduce_into(determinism=Determinism.NOT_GUARANTEED)` | ✅ **Ready** | Default behavior |
| `reduce/by_key.cu` | `cub::DispatchReduceByKey` | ❌ **Not Available** | ❌ | No reduce_by_key in Python yet |
| `reduce/arg_extrema.cu` | `cub::detail::reduce::dispatch_streaming_arg_reduce_t` | ❌ **Not Available** | ❌ | No ArgMin/ArgMax in Python yet |
| `reduce/warp_reduce_sum.cu` | `cub::WarpReduce` | ❌ **Not Available** | ❌ | Block/warp primitives not exposed in Python |
| `reduce/warp_reduce_min.cu` | `cub::WarpReduce` | ❌ **Not Available** | ❌ | Block/warp primitives not exposed in Python |

---

## Scan Benchmarks (8 files) - 25% Coverage

| CUB Benchmark | CUB API | Python API | Status | Notes |
|---------------|---------|------------|--------|-------|
| `scan/exclusive/sum.cu` | `cub::DispatchScan` with `cuda::std::plus<>` | `exclusive_scan` + `OpKind.PLUS` | ✅ **Ready** | Direct match |
| `scan/exclusive/custom.cu` | `cub::DispatchScan` with custom `max_t` | `exclusive_scan` + custom function | ✅ **Ready** | Python supports custom operators |
| `scan/exclusive/by_key.cu` | `cub::DispatchScanByKey` | ❌ **Not Available** | ❌ | No scan_by_key in Python yet |
| `scan/applications/P1/scan-over-unitriangular-group.cu` | Custom scan operator | `inclusive_scan` + custom op | ⚠️ **Complex** | Need to implement matrix group operations |
| `scan/applications/P1/running-min-max.cu` | Custom scan operator | `inclusive_scan` + `@gpu_struct` | ⚠️ **Complex** | Track min/max with struct |
| `scan/applications/P1/log-cdf-from-log-pdfs.cu` | Custom scan operator | `inclusive_scan` + custom op | ⚠️ **Complex** | Logarithmic operations |
| `scan/applications/P1/non-commutative-bicyclic-monoid.cu` | Custom scan operator | `inclusive_scan` + custom op | ⚠️ **Complex** | Non-commutative operator |
| `scan/applications/P1/rabin-karp-second-fingerprinting.cu` | Custom scan operator | `inclusive_scan` + custom op | ⚠️ **Complex** | String hashing operations |

---

## Select Benchmarks (4 files) - 75% Coverage

| CUB Benchmark | CUB API | Python API | Status | Notes |
|---------------|---------|------------|--------|-------|
| `select/if.cu` | `cub::DispatchSelectIf` with predicate | `select` + condition | ✅ **DONE** | Direct match, Entropy axis controls selection % |
| `select/flagged.cu` | `cub::DispatchSelectIf` with flags | `select` + boolean array | ✅ **Ready** | Pass flag array as condition |
| `select/unique_by_key.cu` | `cub::DispatchUniqueByKey` | `unique_by_key` | ✅ **DONE** | Direct match, MaxSegSize axis |
| `select/unique.cu` | `cub::DispatchSelectIf` with equality | ❌ **Not Available** | ❌ | No unique() without keys in Python |

---

## Sort Benchmarks (6 files) - 83% Coverage

| CUB Benchmark | CUB API | Python API | Status | Notes |
|---------------|---------|------------|--------|-------|
| `radix_sort/keys.cu` | `cub::DispatchRadixSort` | `radix_sort` | ✅ **DONE** | Keys only, ascending, Entropy axis |
| `radix_sort/pairs.cu` | `cub::DispatchRadixSort` pairs | `radix_sort(values=...)` | ✅ **DONE** | Key-value sort, KeyT+ValueT axes |
| `merge_sort/keys.cu` | `cub::DispatchMergeSort` | `merge_sort` | ✅ **DONE** | Direct match, Entropy axis |
| `merge_sort/pairs.cu` | `cub::DispatchMergeSort` pairs | `merge_sort(values=...)` | ✅ **Ready** | Pass values parameter |
| `segmented_sort/keys.cu` | `cub::DispatchSegmentedSort` | `segmented_sort` | ✅ **Ready** | Direct match (hybrid radix+merge) |
| `segmented_radix_sort/keys.cu` | `cub::DispatchSegmentedRadixSort` | ❌ **Not Available** | ❌ | No segmented_radix_sort in Python |

---

## Histogram Benchmarks (4 files) - 25% Coverage

| CUB Benchmark | CUB API | Python API | Status | Notes |
|---------------|---------|------------|--------|-------|
| `histogram/even.cu` | `cub::DispatchHistogram::DispatchEven` | `histogram_even` | ✅ **DONE** | Direct match (single channel) |
| `histogram/range.cu` | `cub::DispatchHistogram::DispatchRange` | ❌ **Not Available** | ❌ | No histogram_range in Python |
| `histogram/multi/even.cu` | Multi-channel histogram even | ❌ **Not Available** | ❌ | Python only supports single channel |
| `histogram/multi/range.cu` | Multi-channel histogram range | ❌ **Not Available** | ❌ | Python only supports single channel |

---

## Partition Benchmarks (3 files) - 33% Coverage

| CUB Benchmark | CUB API | Python API | Status | Notes |
|---------------|---------|------------|--------|-------|
| `partition/three_way.cu` | `cub::DispatchThreeWayPartitionIf` | `three_way_partition` | ✅ **DONE** | Direct match, Entropy axis |
| `partition/if.cu` | `cub::DispatchSelectIf` with Partition mode | `select` | ⚠️ **Workaround** | Select compacts, doesn't partition in-place |
| `partition/flagged.cu` | `cub::DispatchSelectIf` with Partition mode | `select` | ⚠️ **Workaround** | Same as above |

---

## Segmented Reduce Benchmarks (3 files) - 67% Coverage

| CUB Benchmark | CUB API | Python API | Status | Notes |
|---------------|---------|------------|--------|-------|
| `segmented_reduce/sum.cu` | Segmented reduce dispatch | `segmented_reduce` + `OpKind.PLUS` | ✅ **DONE** | Fixed-size segments, SegmentSize axis |
| `segmented_reduce/custom.cu` | Segmented reduce dispatch | `segmented_reduce` + custom function | ✅ **Ready** | Direct match |
| `segmented_reduce/argmin.cu` | Segmented ArgMin dispatch | ❌ **Not Available** | ❌ | No ArgMin in Python |

---

## Other Algorithm Benchmarks (11 files) - 0% Coverage

| CUB Benchmark | CUB API | Python API | Status | Notes |
|---------------|---------|------------|--------|-------|
| `merge/keys.cu` | `cub::detail::merge::dispatch_t` | ❌ **Not Available** | ❌ | No standalone merge in Python |
| `merge/pairs.cu` | `cub::detail::merge::dispatch_t` pairs | ❌ **Not Available** | ❌ | No standalone merge in Python |
| `run_length_encode/encode.cu` | `cub::detail::reduce::DispatchStreamingReduceByKey` | ❌ **Not Available** | ❌ | No RLE in Python |
| `run_length_encode/non_trivial_runs.cu` | `cub::DeviceRleDispatch` | ❌ **Not Available** | ❌ | No RLE in Python |
| `for_each/base.cu` | `cub::DeviceFor::ForEachN` | `unary_transform` + `DiscardIterator` | ⚠️ **Workaround** | Use transform with discard output |
| `for_each/copy.cu` | `cub::DeviceFor::ForEachCopyN` | `unary_transform` | ⚠️ **Workaround** | Use unary_transform directly |
| `for_each/extents.cu` | `cub::DeviceFor::ForEachInExtents` (mdspan) | ❌ **Not Available** | ❌ | No mdspan support in Python |
| `find_if/base.cu` | `cub::DeviceFind::FindIf` | ❌ **Not Available** | ❌ | No find in Python |
| `copy/memcpy.cu` | `cub::detail::DispatchBatchMemcpy` | ❌ **Not Available** | ❌ | No batch copy in Python |
| `adjacent_difference/subtract_left.cu` | `cub::DispatchAdjacentDifference` | ❌ **Not Available** | ❌ | No adjacent_difference in Python |
| `topk/keys.cu` | `cub::detail::topk::DispatchTopK` | ❌ **Not Available** | ❌ | No top-k in Python |
| `topk/pairs.cu` | `cub::detail::topk::DispatchTopK` pairs | ❌ **Not Available** | ❌ | No top-k in Python |

---

## Python cuda.compute API Reference

### Available Algorithms (19 functions + 19 make_* factories)

#### Transform
- `unary_transform(d_in, d_out, op, num_items, stream=None)`
- `binary_transform(d_in1, d_in2, d_out, op, num_items, stream=None)`
- `make_unary_transform(d_in, d_out, op)` - Returns reusable object
- `make_binary_transform(d_in1, d_in2, d_out, op)` - Returns reusable object

#### Reduce
- `reduce_into(d_in, d_out, op, num_items, h_init, stream=None, determinism=...)`
- `make_reduce_into(d_in, d_out, op, h_init, determinism=...)`
- `segmented_reduce(d_in, d_out, start_offsets, end_offsets, op, num_segments, h_init, stream=None)`
- `make_segmented_reduce(d_in, d_out, start_offsets, end_offsets, op, h_init)`

#### Scan
- `exclusive_scan(d_in, d_out, op, num_items, init_value=None, stream=None)`
- `make_exclusive_scan(d_in, d_out, op, init_value=None)`
- `inclusive_scan(d_in, d_out, op, num_items, init_value=None, stream=None)`
- `make_inclusive_scan(d_in, d_out, op, init_value=None)`

#### Histogram
- `histogram_even(d_samples, d_histogram, h_num_output_levels, h_lower_level, h_upper_level, num_samples, stream=None)`
- `make_histogram_even(d_samples, d_histogram, h_num_output_levels, h_lower_level, h_upper_level, num_samples)`

#### Sort
- `radix_sort(keys, num_items, values=None, begin_bit=0, end_bit=None, order=SortOrder.ASCENDING, stream=None)`
- `make_radix_sort(keys, values=None, begin_bit=0, end_bit=None, order=SortOrder.ASCENDING)`
- `merge_sort(keys, num_items, values=None, op=OpKind.LESS, stream=None)`
- `make_merge_sort(keys, values=None, op=OpKind.LESS)`
- `segmented_sort(keys, values, start_offsets, end_offsets, num_segments, op=OpKind.LESS, stream=None)`
- `make_segmented_sort(keys, values, start_offsets, end_offsets, op=OpKind.LESS)`

#### Selection & Partitioning
- `select(d_in, d_out, d_num_selected_out, cond, num_items, stream=None)`
- `make_select(d_in, d_out, d_num_selected_out, cond)`
- `three_way_partition(d_in, first_out, second_out, unselected_out, d_num_selected_out, select_first_op, select_second_op, num_items, stream=None)`
- `make_three_way_partition(d_in, first_out, second_out, unselected_out, d_num_selected_out, select_first_op, select_second_op)`

#### Unique
- `unique_by_key(d_keys_in, d_values_in, d_keys_out, d_values_out, d_num_selected_out, num_items, comparison_op=OpKind.EQUAL_TO, stream=None)`
- `make_unique_by_key(d_keys_in, d_values_in, d_keys_out, d_values_out, d_num_selected_out, comparison_op=OpKind.EQUAL_TO)`

### Available Iterators (8 types)

- **`ConstantIterator(value)`** - Generates infinite sequence of constant values (0-ary)
- **`CountingIterator(offset)`** - Generates sequence [offset, offset+1, offset+2, ...]
- **`DiscardIterator(reference_iterator=None)`** - Output iterator that discards all writes
- **`TransformIterator(it, op)`** - Applies function on-the-fly during reads
- **`TransformOutputIterator(it, op)`** - Applies function on-the-fly during writes
- **`ReverseIterator(sequence)`** - Iterates in reverse order
- **`PermutationIterator(values, indices)`** - Access via indirection: values[indices[i]]
- **`ZipIterator(*iterators)`** - Combines multiple iterators into tuples
- **`CacheModifiedInputIterator(device_array, modifier)`** - Cache hint for reads (modifier="stream")

### Available Operators (OpKind)

**Arithmetic:** `PLUS`, `MINUS`, `MULTIPLIES`, `DIVIDES`, `MODULUS`, `NEGATE`  
**Comparison:** `EQUAL_TO`, `NOT_EQUAL_TO`, `GREATER`, `LESS`, `GREATER_EQUAL`, `LESS_EQUAL`  
**Logical:** `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT`  
**Bitwise:** `BIT_AND`, `BIT_OR`, `BIT_XOR`, `BIT_NOT`  
**Min/Max:** `MINIMUM`, `MAXIMUM`  
**Special:** `IDENTITY`, `STATELESS`, `STATEFUL`

### Utility Classes

- **`@gpu_struct`** - Decorator for creating GPU-compatible struct types
- **`SortOrder`** - Enum: `ASCENDING`, `DESCENDING`
- **`DoubleBuffer`** - Container for double-buffered sort operations
- **`Determinism`** - Enum: `RUN_TO_RUN`, `NOT_GUARANTEED`, `GPU_TO_GPU`
- **`clear_all_caches()`** - Clear internal JIT caches

---

## Coverage Summary

### By Status

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ **Fully Supported** | 28 | 51% |
| ⚠️ **Partially Supported** | 9 | 16% |
| ❌ **Not Supported** | 18 | 33% |
| **Total Benchmarks** | **55** | **100%** |

### By Algorithm Family

| Family | Supported | Partial | Not Supported | Total | Coverage |
|--------|-----------|---------|---------------|-------|----------|
| **Transform** | 6 | 1 | 0 | 7 | **86%** |
| **Sort** | 5 | 0 | 1 | 6 | **83%** |
| **Select** | 3 | 0 | 1 | 4 | **75%** |
| **Segmented Reduce** | 2 | 0 | 1 | 3 | **67%** |
| **Reduce** | 5 | 0 | 4 | 9 | **56%** |
| **Partition** | 1 | 2 | 0 | 3 | **33%** |
| **Scan** | 2 | 5 | 1 | 8 | **25%** |
| **Histogram** | 1 | 0 | 3 | 4 | **25%** |
| **Other** | 0 | 2 | 9 | 11 | **0%** |

---

## Recommended Implementation Priority

### Phase 1: High Priority (Easy Wins - Direct API Matches) ✅ COMPLETE

1. ✅ **transform/fill.cu** - **COMPLETED**
2. ✅ **transform/babelstream.cu** - **COMPLETED** - Industry standard streaming benchmark (4 variants)
3. ✅ **reduce/sum.cu** - **COMPLETED** - Basic reduction
4. ✅ **scan/exclusive/sum.cu** - **COMPLETED** - Basic scan
5. ✅ **histogram/even.cu** - **COMPLETED** - Histogram with Bins and Entropy axes
6. ✅ **select/if.cu** - **COMPLETED** - Selection with Entropy axis (InPlace not in Python API)
7. ✅ **radix_sort/keys.cu** - **COMPLETED** - Radix sort keys with Entropy axis
8. ✅ **segmented_reduce/sum.cu** - **COMPLETED** - Segmented reduction with SegmentSize axis
9. **unique_by_key.cu** - Unique operation (moved to Phase 2)

**Phase 1 Status:** 8/8 core benchmarks complete!

### Phase 2: Medium Priority (More Complex but Supported)

10. ✅ **transform/heavy.cu** - **COMPLETED** - Compute-intensive (shows Python overhead on heavy compute)
11. ✅ **unique_by_key.cu** - **COMPLETED** - Unique by key operation with MaxSegSize axis
12. ✅ **merge_sort/keys.cu** - **COMPLETED** - Comparison-based sort with Entropy axis
13. ✅ **partition/three_way.cu** - **COMPLETED** - Three-way partitioning with Entropy axis
14. **transform/fib.cu** - Fibonacci (branching/divergence patterns)
15. **transform/grayscale.cu** - Struct-based (demonstrates gpu_struct usage)
16. **reduce/deterministic.cu** - Determinism overhead
17. **reduce/custom.cu** - Custom operator patterns
18. **merge_sort/pairs.cu** - Key-value merge sort
19. **radix_sort/pairs.cu** - Key-value radix sort (DONE in Phase 1)
20. **segmented_sort/keys.cu** - Segmented operations

**Estimated Time:** ~1.5 hours per benchmark  
**Value:** Comprehensive coverage of major algorithm families

### Phase 3: Advanced (Requires Workarounds)

21. **transform_reduce/sum.cu** - Via TransformIterator
22. **scan P1 applications** (5 benchmarks) - Complex custom operators
23. **partition/if.cu** - Via select (not true partition)
24. **for_each benchmarks** (2) - Via transform workarounds

**Estimated Time:** ~2-3 hours per benchmark  
**Value:** Edge cases and advanced patterns

---

## Missing Python APIs (Feature Requests)

High-value missing APIs that would improve Python coverage:

### Critical Missing APIs
1. **`reduce_by_key`** - Common pattern, CUB has it
2. **`ArgMin/ArgMax`** - Find index of min/max element
3. **`scan_by_key`** - Segmented scan
4. **`histogram_range`** - Custom bin boundaries
5. **`unique`** (without keys) - Simpler than unique_by_key

### Medium Priority
6. **`segmented_radix_sort`** - Segmented radix sort
7. **`find_if`** - Find first matching element
8. **`adjacent_difference`** - Common pattern
9. **Multi-channel histogram** - Image processing use case

### Low Priority (Specialized)
10. **`merge`** - Standalone merge operation
11. **`run_length_encode`** - RLE compression
12. **`top_k`** - Select K largest/smallest
13. **`batch_memcpy`** - Variable-size batch copies
14. **Block/Warp primitives** - Lower-level primitives

---

## ConstantIterator vs @gpu_struct for Fill: Final Answer

### For Fill Benchmark: ConstantIterator is Better ✅

**Why ConstantIterator wins:**
1. **Semantic match:** It's a true 0-ary generator (like C++'s `return_constant<T>`)
2. **Zero memory reads:** No input allocation or memory access
3. **Simplicity:** 2 lines vs 5+ for struct definition
4. **Performance:** Identical - both compile to same CUB dispatch code

**Code Comparison:**
```python
# ✅ ConstantIterator (what we used)
constant_it = ConstantIterator(dtype(42))
transform = make_unary_transform(constant_it, out, identity)
# → 0 reads, N writes

# ❌ @gpu_struct approach (unnecessary complexity)
@gpu_struct
class Generator:
    value: np.int32

gen = Generator(42)
# Still need to pass as state/input, not truly 0-ary
```

### When @gpu_struct IS Better

**Use gpu_struct when you need:**
1. **Multi-field data structures** (Point2D, RGB, KeyValuePair)
2. **Complex state tracking** (running averages, min/max pairs)
3. **Custom types in reductions** (reducing to struct type)
4. **Real structured data** (not just constant generation)

**Example from grayscale benchmark:**
```python
@gpu_struct
class RGB:
    r: np.uint8
    g: np.uint8
    b: np.uint8

def to_grayscale(pixel: RGB) -> np.uint8:
    return 0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b

# Here gpu_struct is essential - ConstantIterator can't represent RGB
```

### The Rule

- **Constant scalar value?** → Use `ConstantIterator`
- **Structured data or state?** → Use `@gpu_struct`

---

## Implementation Status

- ✅ **Completed:** 14/55 benchmarks
  - transform/fill.cu
  - transform/heavy.cu
  - transform/babelstream.cu
  - reduce/sum.cu
  - reduce/min.cu
  - scan/exclusive/sum.cu
  - histogram/even.cu
  - select/if.cu
  - select/unique_by_key.cu
  - radix_sort/keys.cu
  - radix_sort/pairs.cu
  - merge_sort/keys.cu
  - segmented_reduce/sum.cu
  - partition/three_way.cu
- 📋 **Phase 1 Ready:** All Phase 1 complete!
- 📋 **Phase 2 Ready:** 8 more with moderate complexity
- 📋 **Phase 3 Ready:** 9 more with workarounds
- ❌ **Not Achievable:** 18 benchmarks (33%)

**Total Achievable:** 37 out of 55 benchmarks (67% coverage)

---

## File Organization

```
python/cuda_cccl/benchmarks/compute/
├── run_benchmarks.sh                   # Main runner script
├── Makefile                            # Convenient targets
├── utils.py                            # Shared utilities
│
├── transform/
│   ├── fill.py                         # ✅ DONE
│   ├── heavy.py                        # ✅ DONE
│   └── babelstream.py                  # ✅ DONE
│
├── reduce/
│   ├── sum.py                          # ✅ DONE
│   └── min.py                          # ✅ DONE
│
├── scan/
│   └── exclusive/
│       └── sum.py                      # ✅ DONE
│
├── histogram/
│   └── even.py                         # ✅ DONE
│
├── select/
│   ├── if.py                           # ✅ DONE
│   └── unique_by_key.py                # ✅ DONE
│
├── radix_sort/
│   ├── keys.py                         # ✅ DONE
│   └── pairs.py                        # ✅ DONE
│
├── merge_sort/
│   └── keys.py                         # ✅ DONE
│
├── segmented_reduce/
│   └── sum.py                          # ✅ DONE
│
├── partition/
│   └── three_way.py                    # ✅ DONE
│
├── results/                            # Benchmark output JSON files
│   ├── transform/
│   ├── reduce/
│   ├── scan/exclusive/
│   ├── histogram/
│   ├── select/
│   ├── radix_sort/
│   ├── merge_sort/
│   ├── segmented_reduce/
│   └── partition/
│
└── analysis/
    ├── python_vs_cpp_summary.py        # Comparison script
    └── utils.py
```

---

## Next Actions

1. **Phase 1 Complete!** All high-priority direct API matches implemented
2. **Phase 2 Partial:** 3 more Phase 2 benchmarks implemented (unique_by_key, merge_sort/keys, partition/three_way)
3. **Continue Phase 2:** Implement merge_sort/pairs, transform/fib, transform/grayscale, reduce/deterministic, etc.
4. **Build comparison database:** Track Python overhead across all algorithms
5. **Identify patterns:** Which algorithms show most/least overhead?
6. **Document findings:** Help Python CCCL users understand performance characteristics

---

**Generated:** Step 3 Complete  
**Last Updated:** After implementing merge_sort/keys, select/unique_by_key, partition/three_way
