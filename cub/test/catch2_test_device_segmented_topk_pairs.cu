// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Defer the unsupported-architecture diagnosis to the dispatch's runtime check (not a compile-time static_assert)
// so this test compiles across all target architectures, including pre-SM90, for the full configuration space. See
// _CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT in cub/device/dispatch/dispatch_batched_topk.cuh. Precedes CUB includes.
#define _CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_batched_topk.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh> // topk_policy / make_baseline_policy (cross-tuning test)

#include <thrust/count.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/__execution/tune.h>
#include <cuda/iterator>

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "catch2_test_device_topk_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/extended_types.h>
#include <catch2/generators/catch_generators.hpp>

// Maps an item index to its segment id for fixed-size segments
struct fixed_stride_segment_id_op
{
  cuda::std::int64_t stride;

  template <typename IndexT>
  __device__ IndexT operator()(IndexT idx) const
  {
    return static_cast<IndexT>(idx / stride);
  }
};

// Flags adjacent duplicate items that belong to the same segment
template <typename ItemItT, typename SegIdItT>
struct flag_intra_segment_duplicates
{
  ItemItT d_sorted_items;
  SegIdItT d_segment_ids;

  template <typename IndexT>
  __device__ bool operator()(IndexT idx) const
  {
    return d_segment_ids[idx] == d_segment_ids[idx + 1] && d_sorted_items[idx] == d_sorted_items[idx + 1];
  }
};

template <typename ItemItT, typename SegIdItT>
flag_intra_segment_duplicates(ItemItT, SegIdItT) -> flag_intra_segment_duplicates<ItemItT, SegIdItT>;

// Routes the key-value (pairs) top-k through the public `cub::DeviceBatchedTopK` API, threading the requested
// determinism/tie-break into the environment via `require`. The dispatch selects the backend from the architecture and
// the statically-known maximum segment size (a deterministic request routes to the cluster backend on SM90+).
template <cub::detail::topk::select SelectDirection,
          cuda::execution::determinism::__determinism_t Determinism =
            cuda::execution::determinism::__determinism_t::__not_guaranteed,
          cuda::execution::tie_break::__tie_break_t TieBreak = cuda::execution::tie_break::__tie_break_t::__unspecified,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename NumSegmentsParameterT>
CUB_RUNTIME_FUNCTION static cudaError_t dispatch_batched_topk_pairs(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  ValueInputItItT d_value_segments_it,
  ValueOutputItItT d_value_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k,
  NumSegmentsParameterT num_segments,
  cudaStream_t stream = nullptr)
{
  auto env = cuda::std::execution::env{
    cuda::stream_ref{stream},
    cuda::execution::require(cuda::execution::determinism::__determinism_holder_t<Determinism>{},
                             cuda::execution::tie_break::__tie_break_holder_t<TieBreak>{},
                             cuda::execution::output_ordering::unsorted)};
  if constexpr (SelectDirection == cub::detail::topk::select::max)
  {
    return cub::DeviceBatchedTopK::MaxPairs(
      d_temp_storage,
      temp_storage_bytes,
      d_key_segments_it,
      d_key_segments_out_it,
      d_value_segments_it,
      d_value_segments_out_it,
      segment_sizes,
      k,
      num_segments,
      env);
  }
  else
  {
    return cub::DeviceBatchedTopK::MinPairs(
      d_temp_storage,
      temp_storage_bytes,
      d_key_segments_it,
      d_key_segments_out_it,
      d_value_segments_it,
      d_value_segments_out_it,
      segment_sizes,
      k,
      num_segments,
      env);
  }
}

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_TMPL_LAUNCH_WRAPPER(
  dispatch_batched_topk_pairs,
  batched_topk_pairs,
  ESCAPE_LIST(
    cub::detail::topk::select SelectDirection,
    cuda::execution::determinism::__determinism_t Determinism =
      cuda::execution::determinism::__determinism_t::__not_guaranteed,
    cuda::execution::tie_break::__tie_break_t TieBreak = cuda::execution::tie_break::__tie_break_t::__unspecified),
  ESCAPE_LIST(SelectDirection, Determinism, TieBreak));

// Total segment size
using max_segment_size_list = c2h::enum_type_list<cuda::std::size_t, 4 * 1024>;

// Segment size: static, uniform
using max_num_k_list = c2h::enum_type_list<cuda::std::size_t, 32, 4 * 1024>;

// %PARAM% TEST_TYPES types 0:1:2

#if TEST_TYPES == 0
using key_types =
  c2h::type_list<cuda::std::uint8_t
// clang-format off
  #if TEST_HALF_T()
  , half_t
  #endif // TEST_HALF_T()
  >;
// clang-format on
#elif TEST_TYPES == 1
using key_types = c2h::type_list<float>;
#elif TEST_TYPES == 2
using key_types =
  c2h::type_list<cuda::std::uint64_t
// clang-format off
  #if TEST_BF_T()
  , bfloat16_t
  #endif // TEST_BF_T()
  >;
// clang-format on
#endif

// Unsigned integer types used for the radix-pass boundary distribution test
using uint_key_types = c2h::type_list<cuda::std::uint8_t, cuda::std::uint16_t, cuda::std::uint64_t>;

// Selection direction is a compile-time option; cover both as a static test axis.
using select_direction_list =
  c2h::enum_type_list<cub::detail::topk::select, cub::detail::topk::select::min, cub::detail::topk::select::max>;

// Determinism/tie-break combinations used as a single compile-time axis by the determinism-aware pairs tests. The
// selected multiset is invariant to the tie-break preference, so every combo is verified the same way; tie-break
// preferences only pair with a deterministic requirement.
template <cuda::execution::determinism::__determinism_t Determinism, cuda::execution::tie_break::__tie_break_t TieBreak>
struct det_tie_pair
{
  static constexpr auto determinism = Determinism;
  static constexpr auto tie_break   = TieBreak;
};
using det_tie_pair_combos =
  c2h::type_list<det_tie_pair<cuda::execution::determinism::__determinism_t::__not_guaranteed,
                              cuda::execution::tie_break::__tie_break_t::__unspecified>,
                 det_tie_pair<cuda::execution::determinism::__determinism_t::__gpu_to_gpu,
                              cuda::execution::tie_break::__tie_break_t::__unspecified>,
                 det_tie_pair<cuda::execution::determinism::__determinism_t::__gpu_to_gpu,
                              cuda::execution::tie_break::__tie_break_t::__prefer_smaller_index>,
                 det_tie_pair<cuda::execution::determinism::__determinism_t::__gpu_to_gpu,
                              cuda::execution::tie_break::__tie_break_t::__prefer_larger_index>>;

// Consistency check: ensures values remain associated with their corresponding keys
template <typename KeyT, typename ValueT>
bool verify_pairs_consistency(const c2h::device_vector<KeyT>& keys_in,
                              const c2h::device_vector<KeyT>& keys_out,
                              const c2h::device_vector<ValueT>& values_out)
{
  auto d_keys_in    = thrust::raw_pointer_cast(keys_in.data());
  auto d_values_out = thrust::raw_pointer_cast(values_out.data());

  // permutation_it[i] -> d_keys_in[d_values_out[i]] to verify that keys and values remained associated
  auto permutation_it = cuda::make_permutation_iterator(d_keys_in, d_values_out);

  return thrust::equal(keys_out.cbegin(), keys_out.cend(), permutation_it);
}

// Uniqueness check: ensures there are no duplicate values within the top-k items of each segment
template <typename ValueT>
bool verify_unique_indices(c2h::device_vector<ValueT>& values_out, cuda::std::int64_t num_segments, cuda::std::int64_t k)
{
  // Make a copy & sort
  c2h::device_vector<ValueT> sorted_values{values_out};
  fixed_size_segmented_sort_keys(sorted_values, num_segments, k, cub::detail::topk::select::min);

  auto num_items   = sorted_values.size();
  auto counting_it = cuda::make_counting_iterator(cuda::std::int64_t{0});
  auto seg_ids     = cuda::make_transform_iterator(counting_it, fixed_stride_segment_id_op{k});
  flag_intra_segment_duplicates flag_op{sorted_values.cbegin(), seg_ids};
  auto num_duplicates = thrust::count_if(counting_it, counting_it + (num_items - 1), flag_op);

  return num_duplicates == 0;
}

// Overload for variable-size segments: sorts compacted values within each segment and checks for duplicates
template <typename ValueT, typename OffsetT>
bool verify_unique_indices(const c2h::device_vector<ValueT>& values_compacted,
                           const c2h::device_vector<OffsetT>& compacted_offsets,
                           cuda::std::int64_t num_segments)
{
  c2h::device_vector<ValueT> sorted_values = values_compacted;
  segmented_sort_keys(
    sorted_values,
    num_segments,
    compacted_offsets.cbegin(),
    compacted_offsets.cbegin() + 1,
    cub::detail::topk::select::min);

  auto num_items = sorted_values.size();

  // Generate segment ids via scatter + inclusive_scan: scatter a 1 at each interior segment
  // boundary, then prefix-sum to produce monotonic group ids
  c2h::device_vector<OffsetT> segment_ids(num_items, OffsetT{0});
  thrust::scatter(cuda::constant_iterator<OffsetT>(1),
                  cuda::constant_iterator<OffsetT>(1) + (num_segments - 1),
                  compacted_offsets.cbegin() + 1,
                  segment_ids.begin());
  thrust::inclusive_scan(segment_ids.begin(), segment_ids.end(), segment_ids.begin());

  flag_intra_segment_duplicates flag_op{sorted_values.cbegin(), segment_ids.cbegin()};

  auto num_duplicates =
    thrust::count_if(cuda::make_counting_iterator(size_t{0}), cuda::make_counting_iterator(num_items - 1), flag_op);

  return num_duplicates == 0;
}

C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs work with small fixed-size segments",
         "[pairs][segmented][topk][device]",
         key_types,
         max_segment_size_list,
         max_num_k_list,
         select_direction_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using key_t = c2h::get<0, TestType>;
  using val_t = cuda::std::int32_t;

  // Statically constrained maximum segment size and k
  constexpr segment_size_t static_max_segment_size = c2h::get<1, TestType>::value;
  constexpr segment_size_t static_max_k            = c2h::get<2, TestType>::value;

  // Selection direction comes from the compile-time test axis.
  constexpr auto direction = c2h::get<3, TestType>::value;

  // Generate segment size
  constexpr segment_size_t min_segment_size = 1;
  constexpr auto max_segment_size           = static_max_segment_size;
  const segment_size_t segment_size = GENERATE_COPY(values({min_segment_size, segment_size_t{3}, max_segment_size}),
                                                    take(1, random(min_segment_size, max_segment_size)));
  const segment_size_t max_k        = (cuda::std::min) (static_max_k, segment_size);

  // Skip invalid combinations
  if (segment_size > max_segment_size)
  {
    SKIP("The given segment size may not exceed the maximum segment size, we statically constrained the algorithm on.");
  }

  // Set the k value
  const segment_size_t k = GENERATE_COPY(values({segment_size_t{1}, max_k}), take(1, random(segment_size_t{1}, max_k)));

  // Generate number of segments
  const segment_index_t num_segments = GENERATE_COPY(
    values({segment_index_t{1}, segment_index_t{42}}), take(1, random(segment_index_t{1}, segment_index_t{1000})));

  // Capture test parameters
  CAPTURE(c2h::type_name<key_t>(),
          c2h::type_name<segment_size_t>(),
          c2h::type_name<segment_index_t>(),
          static_max_segment_size,
          static_max_k,
          segment_size,
          k,
          num_segments,
          direction);

  // Prepare keys input & output
  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), keys_in_buffer);
  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  // Prepare values input & output
  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  c2h::device_vector<val_t> values_out_buffer(num_segments * k, thrust::no_init);
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_values_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), segment_size);
  auto d_values_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_values_out_ptr), k);

  // Copy input for verification
  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Run the top-k algorithm
  batched_topk_pairs<direction>(
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments});

  // Verification:
  // - We verify correct top-k selection through the keys
  // - We verify that values were permuted along correctly by making sure values remain associated with their keys and
  // making sure we do not duplicate values Verify values remain associated with their corresponding keys
  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);

  // Verify values don't appear more than once in the returned results
  // This catches the case where we just returned a valid value multiple times
  REQUIRE(verify_unique_indices(values_out_buffer, num_segments, k) == true);

  // Verify keys are returned correctly
  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);

  // Since the results of top-k are unordered, sort output segments before comparison.
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

#if TEST_TYPES == 0
// A `k` larger than the segment is clamped to `segment_size` (every pair is selected), routing through the select-all
// fast path. The output then holds exactly `segment_size` pairs per segment -- identical to a `k == segment_size`
// request -- so we verify with a `segment_size`-wide output and a key/value-consistency check on the full segment.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs clamp k larger than the segment size",
         "[pairs][segmented][topk][device]",
         key_types,
         select_direction_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;
  using key_t           = c2h::get<0, TestType>;
  using val_t           = cuda::std::int32_t;

  constexpr auto direction = c2h::get<1, TestType>::value;

  constexpr segment_size_t segment_size = 384;
  // Deliberately request more than the segment holds; the algorithm must clamp to `segment_size`.
  const segment_size_t k_requested =
    GENERATE_COPY(values({segment_size + 1, segment_size + 100, 2 * segment_size, 10 * segment_size}));
  const segment_size_t effective_k   = (cuda::std::min) (k_requested, segment_size); // == segment_size
  const segment_index_t num_segments = GENERATE_COPY(
    values({segment_index_t{1}, segment_index_t{37}}), take(2, random(segment_index_t{1}, segment_index_t{500})));

  CAPTURE(c2h::type_name<key_t>(), segment_size, k_requested, effective_k, num_segments, direction);

  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(num_segments * effective_k, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);
  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), effective_k);

  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  c2h::device_vector<val_t> values_out_buffer(num_segments * effective_k, thrust::no_init);
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_values_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), segment_size);
  auto d_values_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_values_out_ptr), effective_k);

  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  batched_topk_pairs<direction>(
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    cuda::args::constant<segment_size>{},
    cuda::args::immediate{k_requested, cuda::args::bounds<segment_size_t{1}, 10 * segment_size>()},
    cuda::args::immediate{num_segments});

  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);
  REQUIRE(verify_unique_indices(values_out_buffer, num_segments, effective_k) == true);

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, effective_k); // clamped k == segment_size keeps everything
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, effective_k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

// Segment-size types narrower than the internal `offset_t`, both signed and unsigned: a signed type that narrows a
// too-large index goes negative (indexing before the segment); an unsigned one wraps to a small in-range index
// (duplicate racing stores).
using narrow_seg_size_list = c2h::type_list<cuda::std::int8_t, cuda::std::uint8_t>;

// Regression for a segment-size type narrower than the internal `offset_t`. The select-all copy must bound-check in
// `offset_t` *before* narrowing: a block launches 512 threads -- far past an 8-bit type's range -- so any path that
// narrows a thread/element index to `segment_size_val_t` before its bound check (or wraps an unsigned one) would
// read/write out of bounds (a signed type indexes before the segment; an unsigned one wraps into duplicate racing
// stores -- caught by racecheck even when the value is coincidentally right). We sweep `segment_size` and `k` with
// each narrow size type to exercise every path under it: `k == segment_size` hits the select-all copy, while
// `k < segment_size` drives the radix-select / cluster-histogram / output-ordering paths (and, for pairs, the value
// gather). Running over `det_tie_pair_combos` additionally covers the deterministic output-ordering scan/atomics.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs handle a segment-size type narrower than the internal offset",
         "[pairs][segmented][topk][device][cluster][determinism]",
         narrow_seg_size_list,
         det_tie_pair_combos)
{
  using seg_size_t      = c2h::get<0, TestType>;
  using segment_index_t = cuda::std::int64_t;
  using key_t           = cuda::std::uint8_t; // key type is immaterial to this index-arithmetic regression
  using val_t           = cuda::std::int32_t;

  using combo                = c2h::get<1, TestType>;
  constexpr auto determinism = combo::determinism;
  constexpr auto tie_break   = combo::tie_break;
  // Baseline-coverable segment sizes (statically bounded to 127 below): only the deterministic / tie-break requirements
  // route to the SM90+ cluster backend.
  skip_if_batched_topk_backend_unavailable<determinism, tie_break>(/*static_max_segment_size=*/127);
  constexpr auto direction = cub::detail::topk::select::max;
  // Sizes fit both 8-bit types but sit far below the 512 threads a block launches; `127` probes the signed type's max.
  const seg_size_t segment_size = static_cast<seg_size_t>(GENERATE(values({3, 100, 127})));
  const seg_size_t k            = static_cast<seg_size_t>(
    GENERATE_COPY(values({1, int{segment_size} / 2, int{segment_size}}), take(2, random(1, int{segment_size}))));
  const segment_index_t num_segments = GENERATE_COPY(
    values({segment_index_t{1}, segment_index_t{37}}), take(2, random(segment_index_t{1}, segment_index_t{500})));

  CAPTURE(c2h::type_name<seg_size_t>(), int{segment_size}, int{k}, num_segments);

  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);
  auto d_keys_in_ptr = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_in     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), int{segment_size});
  auto values_in_it  = cuda::make_counting_iterator(val_t{0});
  auto d_values_in   = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), int{segment_size});

  // Key/value output buffers with front/back canary padding: a signed-narrowing underrun lands in the front guard, an
  // overrun in the back guard. The in-range checks below miss both (the underrun also stays inside the allocation,
  // hiding it from memcheck), so we assert the guards stay untouched. `guard` covers the worst-case 8-bit offset
  // (<= 256); the value canary `-1` is impossible for a real (non-negative) payload, so the value guard is exact.
  constexpr segment_index_t guard = 256;
  const key_t key_canary          = static_cast<key_t>(0x5A);
  constexpr val_t val_canary      = -1;
  c2h::device_vector<key_t> keys_out_storage(guard + num_segments * k + guard, key_canary);
  c2h::device_vector<val_t> values_out_storage(guard + num_segments * k + guard, val_canary);
  auto d_keys_out_ptr   = thrust::raw_pointer_cast(keys_out_storage.data()) + guard;
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_storage.data()) + guard;
  auto d_keys_out       = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), int{k});
  auto d_values_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_values_out_ptr), int{k});

  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  batched_topk_pairs<direction, determinism, tie_break>(
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<seg_size_t{1}, seg_size_t{127}>()},
    cuda::args::immediate{k, cuda::args::bounds<seg_size_t{1}, seg_size_t{127}>()},
    cuda::args::immediate{num_segments});

  // Guards must be untouched by the algorithm.
  const c2h::device_vector<key_t> expected_kguard(guard, key_canary);
  const c2h::device_vector<val_t> expected_vguard(guard, val_canary);
  CHECK(c2h::device_vector<key_t>(keys_out_storage.begin(), keys_out_storage.begin() + guard) == expected_kguard);
  CHECK(c2h::device_vector<key_t>(keys_out_storage.end() - guard, keys_out_storage.end()) == expected_kguard);
  CHECK(c2h::device_vector<val_t>(values_out_storage.begin(), values_out_storage.begin() + guard) == expected_vguard);
  CHECK(c2h::device_vector<val_t>(values_out_storage.end() - guard, values_out_storage.end()) == expected_vguard);

  // Extract the in-range outputs for the standard pair verification.
  c2h::device_vector<key_t> keys_out_buffer(keys_out_storage.begin() + guard, keys_out_storage.end() - guard);
  c2h::device_vector<val_t> values_out_buffer(values_out_storage.begin() + guard, values_out_storage.end() - guard);

  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);
  REQUIRE(verify_unique_indices(values_out_buffer, num_segments, k) == true);

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}
#endif // TEST_TYPES == 0

// Large-segment cluster pair tests, built only in the float build (`TEST_TYPES == 1`): they fix their own key/value
// types, so repeating them per key-type axis would only waste time on the expensive 1 Mi-element runs. They run on
// every launch id, including device launch (`lid_1`): the CDP static config's small resident capacity is what streams
// big segments and peels the unaligned tail edge, so it is the path that must cover them.
#if TEST_TYPES == 1
template <typename KeyT>
struct cast_to_key_op
{
  template <typename T>
  __host__ __device__ KeyT operator()(T x) const
  {
    return static_cast<KeyT>(x);
  }
};

// Yields, for each segment, a *non-contiguous* iterator over that segment's keys (an integral counting iterator cast
// to the key type). Feeding the cluster top-k a non-contiguous key iterator makes `use_block_load_to_shared` false, so
// the agent takes its generic (non-BlockLoadToShared) path. Segment `seg` produces keys
// [seg * segment_size, (seg + 1) * segment_size), so the flattened input equals the identity sequence.
template <typename KeyT, typename SegmentSizeT>
struct counting_segment_keys_op
{
  SegmentSizeT segment_size;

  template <typename IndexT>
  __host__ __device__ auto operator()(IndexT seg) const
  {
    return cuda::make_transform_iterator(
      cuda::make_counting_iterator(static_cast<SegmentSizeT>(seg) * segment_size), cast_to_key_op<KeyT>{});
  }
};

C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs stream large segments through a non-contiguous key iterator",
         "[pairs][segmented][topk][device][cluster]",
         select_direction_list)
{
  using key_t           = float;
  using val_t           = cuda::std::int32_t;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  // Selection direction comes from the compile-time test axis.
  constexpr auto direction = c2h::get<0, TestType>::value;

  // The counting-iterator key source is non-contiguous, so the agent uses its generic path rather than
  // BlockLoadToShared - the only place the pair value writes flow through the generic resident/overflow code.
  // `static_max_segment_size` exceeds the largest all-resident cluster coverage, so the 1 Mi-element segments stream
  // (incl. an unaligned `- 31` tail), while the 128 Ki-element segment validates the generic resident path (no
  // streaming) through the same code. Keeping the largest total below 2^24 makes every key an exact float, and every
  // value index an exact `int32`.
  constexpr segment_size_t static_max_segment_size = 1024 * 1024;
  constexpr segment_size_t static_max_k            = 4 * 1024;
  constexpr segment_index_t num_segments           = 3;
  // Oversize segments always route to the SM90+ cluster backend.
  skip_if_batched_topk_backend_unavailable(static_max_segment_size);

  const segment_size_t segment_size =
    GENERATE_COPY(values({static_max_segment_size, static_max_segment_size - 31, segment_size_t{128 * 1024}}));
  const segment_size_t max_k     = (cuda::std::min) (static_max_k, segment_size);
  const segment_size_t k         = GENERATE_COPY(values({segment_size_t{1}, max_k / 2, max_k}));
  const segment_size_t num_items = num_segments * segment_size;

  CAPTURE(static_max_segment_size, static_max_k, segment_size, k, num_segments, direction);

  // Non-contiguous key input: segment `seg` is the counting iterator [seg * segment_size, (seg + 1) * segment_size).
  auto d_keys_in = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}), counting_segment_keys_op<key_t, segment_size_t>{segment_size});

  // Value payload = global flattened index, so each output value indexes back into the identity `expected_keys`. Only
  // the key iterator drives the resident/streaming path; the value iterator type is immaterial (values are fetched
  // lazily per selected key), so a strided counting iterator suffices.
  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  auto d_values_in  = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), segment_size);

  // Outputs are real buffers (the output iterators stay contiguous; only the key input drives the generic path).
  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  c2h::device_vector<val_t> values_out_buffer(num_segments * k, thrust::no_init);
  auto d_keys_out_ptr   = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_keys_out       = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);
  auto d_values_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_values_out_ptr), k);

  batched_topk_pairs<direction>(
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments});

  // The flattened input is the identity sequence, so materialize it for the standard pair verification.
  c2h::device_vector<key_t> expected_keys(num_items, thrust::no_init);
  thrust::sequence(expected_keys.begin(), expected_keys.end());

  // Verify (before sorting) that values stayed associated with their keys and that no index repeats within a segment.
  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);
  REQUIRE(verify_unique_indices(values_out_buffer, num_segments, k) == true);

  // Verify the selected keys are the correct top-k.
  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

// Pair analog of the keys "large fixed-size unaligned segments" test: contiguous keys offset by `pad` take the
// block-load path with an unaligned head edge, and the 1 Mi-element segments stream, so the value payloads exercise
// the boundary-edge value writes that the small and non-contiguous pair tests above do not. An unaligned tail suffix
// is always peeled into `edge_keys` (like the head prefix), so every launch config that owns such a tail exercises the
// head edge plus the persistent `tail_edge_len`/`process_tail_edge` value writes.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs work with large fixed-size unaligned segments",
         "[pairs][segmented][topk][device][cluster]",
         select_direction_list)
{
  using key_t           = float;
  using val_t           = cuda::std::int32_t;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  constexpr auto direction = c2h::get<0, TestType>::value;

  constexpr segment_size_t static_max_segment_size = 1024 * 1024;
  constexpr segment_size_t static_max_k            = 4 * 1024;
  constexpr segment_index_t num_segments           = 3;
  // Oversize segments always route to the SM90+ cluster backend.
  skip_if_batched_topk_backend_unavailable(static_max_segment_size);

  const int pad = GENERATE(0, 1, 3, 7);
  // The `+ 1` / `- 4095` sizes make the global-last chunk a single item, i.e. a pure-suffix tail with an empty aligned
  // bulk (`bulk == 0`) once `pad == 0` aligns the base, exercising the always-peeled tail edge value writes on top of a
  // zero-length resident/streamed tail chunk (resident `128 Ki + 1`, streamed `1 Mi - 4095`).
  const segment_size_t segment_size = GENERATE_COPY(values(
    {static_max_segment_size,
     static_max_segment_size - 31,
     static_max_segment_size - 4095,
     segment_size_t{128 * 1024},
     segment_size_t{128 * 1024 + 1}}));
  const segment_size_t max_k        = (cuda::std::min) (static_max_k, segment_size);
  const segment_size_t k            = GENERATE_COPY(values({segment_size_t{1}, max_k / 2, max_k}));
  const segment_size_t num_items    = num_segments * segment_size;

  CAPTURE(pad, static_max_segment_size, static_max_k, segment_size, k, num_segments, direction);

  // Contiguous key storage offset by `pad` so each segment base is unaligned (forces the head boundary edge).
  c2h::device_vector<key_t> keys_in_buffer(pad + num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);
  auto d_keys_in_ptr = thrust::raw_pointer_cast(keys_in_buffer.data()) + pad;
  auto d_keys_in     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);

  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  // Value payload = global flattened index into the (pad-excluded) logical input, so each output value indexes back
  // into `expected_keys`.
  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  auto d_values_in  = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), segment_size);
  c2h::device_vector<val_t> values_out_buffer(num_segments * k, thrust::no_init);
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_values_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_values_out_ptr), k);

  // Logical input (pad excluded), flattened, for verification.
  c2h::device_vector<key_t> expected_keys(num_items, thrust::no_init);
  thrust::copy(keys_in_buffer.cbegin() + pad, keys_in_buffer.cend(), expected_keys.begin());

  batched_topk_pairs<direction>(
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments});

  // Verify (before sorting) that values stayed associated with their keys and that no index repeats within a segment.
  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);
  REQUIRE(verify_unique_indices(values_out_buffer, num_segments, k) == true);

  // Verify the selected keys are the correct top-k.
  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

// Streaming counterpart of the narrow-segment-size regression (pairs). Streaming needs segments larger than the
// resident cluster coverage (>128 Ki), which 8/16-bit types can't represent, so we use signed 32-bit -- same width as
// the (unsigned) internal `offset_t` but signed -- to exercise the streaming path's index arithmetic (including the
// value gather) across det/non-det.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs stream large segments with a signed 32-bit segment-size type",
         "[pairs][segmented][topk][device][cluster][determinism]",
         det_tie_pair_combos)
{
  using key_t           = float;
  using val_t           = cuda::std::int32_t;
  using seg_size_t      = int; // signed 32-bit: same width as offset_t, but signed
  using segment_index_t = cuda::std::int64_t;

  using combo                = c2h::get<0, TestType>;
  constexpr auto determinism = combo::determinism;
  constexpr auto tie_break   = combo::tie_break;
  constexpr auto direction   = cub::detail::topk::select::max;

  constexpr seg_size_t static_max_segment_size = 1024 * 1024;
  constexpr seg_size_t static_max_k            = 4 * 1024;
  constexpr segment_index_t num_segments       = 2;
  // Oversize segments always route to the SM90+ cluster backend, regardless of the determinism / tie-break requirement.
  skip_if_batched_topk_backend_unavailable<determinism, tie_break>(static_max_segment_size);

  const int pad                   = GENERATE(0, 7);
  const seg_size_t segment_size   = static_max_segment_size - 31; // unaligned -> forces streaming + unaligned tail edge
  const seg_size_t max_k          = (cuda::std::min) (static_max_k, segment_size);
  const seg_size_t k              = GENERATE_COPY(values({seg_size_t{1}, max_k}));
  const segment_index_t num_items = num_segments * segment_size;

  CAPTURE(pad, static_max_segment_size, static_max_k, segment_size, k, num_segments);

  c2h::device_vector<key_t> keys_in_buffer(pad + num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);
  auto d_keys_in_ptr = thrust::raw_pointer_cast(keys_in_buffer.data()) + pad;
  auto d_keys_in     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);

  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  auto d_values_in  = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), segment_size);
  c2h::device_vector<val_t> values_out_buffer(num_segments * k, thrust::no_init);
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_values_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_values_out_ptr), k);

  c2h::device_vector<key_t> expected_keys(num_items, thrust::no_init);
  thrust::copy(keys_in_buffer.cbegin() + pad, keys_in_buffer.cend(), expected_keys.begin());

  batched_topk_pairs<direction, determinism, tie_break>(
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<seg_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<seg_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments});

  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);
  REQUIRE(verify_unique_indices(values_out_buffer, num_segments, k) == true);

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

// Tie-break preferences exercised by the deterministic tests below (only meaningful with a deterministic requirement).
using tie_break_pref_list =
  c2h::enum_type_list<cuda::execution::tie_break::__tie_break_t,
                      cuda::execution::tie_break::__tie_break_t::__prefer_smaller_index,
                      cuda::execution::tie_break::__tie_break_t::__prefer_larger_index>;

// Deterministic requirements exercised with an *unspecified* tie-break: both must yield a reproducible result.
using determinism_list =
  c2h::enum_type_list<cuda::execution::determinism::__determinism_t,
                      cuda::execution::determinism::__determinism_t::__run_to_run,
                      cuda::execution::determinism::__determinism_t::__gpu_to_gpu>;

// Host reference for the deterministic top-k: per segment, stably picks the k best (key, global-index) pairs via a
// plain lexicographic sort -- max sorts descending (larger index wins ties), min ascending (smaller wins). When the
// requested preference is the opposite, the index is reversed (last_index - idx) so the natural order breaks ties the
// desired way without a custom comparator (and, unlike negation, stays non-negative for unsigned indices). Returns the
// selected indices sorted ascending per segment, to compare against the unordered device output as a set.
template <typename KeyT, typename IndexT, typename SegSizeT>
c2h::host_vector<IndexT> reference_deterministic_topk_indices(
  const c2h::host_vector<KeyT>& h_keys,
  SegSizeT num_segments,
  SegSizeT segment_size,
  SegSizeT k,
  cub::detail::topk::select direction,
  bool prefer_larger_index)
{
  const bool want_max      = direction == cub::detail::topk::select::max;
  const bool reverse_index = want_max != prefer_larger_index;
  const IndexT last_index  = static_cast<IndexT>(num_segments * segment_size - 1);
  // Tie-break sort key (an involution, so it also decodes): reverses index order when requested, else identity.
  const auto encode = [&](IndexT idx) {
    return reverse_index ? static_cast<IndexT>(last_index - idx) : idx;
  };

  c2h::host_vector<IndexT> selected(static_cast<std::size_t>(num_segments * k));
  std::vector<std::pair<KeyT, IndexT>> pairs(static_cast<std::size_t>(segment_size));
  for (SegSizeT seg = 0; seg < num_segments; ++seg)
  {
    const SegSizeT base = seg * segment_size;
    for (SegSizeT i = 0; i < segment_size; ++i)
    {
      const IndexT idx                   = static_cast<IndexT>(base + i);
      pairs[static_cast<std::size_t>(i)] = {h_keys[static_cast<std::size_t>(base + i)], encode(idx)};
    }
    if (want_max)
    {
      std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(), std::greater<std::pair<KeyT, IndexT>>{});
    }
    else
    {
      std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end());
    }
    const auto seg_begin = selected.begin() + static_cast<std::ptrdiff_t>(seg * k);
    for (SegSizeT i = 0; i < k; ++i)
    {
      seg_begin[i] = encode(pairs[static_cast<std::size_t>(i)].second);
    }
    std::sort(seg_begin, seg_begin + k);
  }
  return selected;
}

// Deterministic tie-break: a specified preference is `gpu_to_gpu` deterministic by definition, so the cluster path
// returns a uniquely defined top-k. Few distinct key values pack many ties into the k-th bucket so the preference (not
// the key comparison) drives the result; the value payload is the global index, so we compare per-segment index sets
// against the host reference (within-top-k order is unspecified). `lid_1` streams the 64 Ki segments.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs deterministic tie-break returns the index-ordered top-k",
         "[pairs][segmented][topk][device][cluster][determinism]",
         select_direction_list,
         tie_break_pref_list)
{
  using key_t           = cuda::std::uint32_t;
  using val_t           = cuda::std::int32_t;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  constexpr auto direction     = c2h::get<0, TestType>::value;
  constexpr auto tie_break     = c2h::get<1, TestType>::value;
  constexpr auto determinism   = cuda::execution::determinism::__determinism_t::__gpu_to_gpu;
  constexpr bool prefer_larger = tie_break == cuda::execution::tie_break::__tie_break_t::__prefer_larger_index;

  constexpr segment_size_t static_max_segment_size = 64 * 1024;
  constexpr segment_size_t static_max_k            = 64 * 1024;
  constexpr segment_index_t num_segments           = 2;
  // Deterministic and oversize: this configuration always routes to the SM90+ cluster backend.
  skip_if_batched_topk_backend_unavailable<determinism, tie_break>(static_max_segment_size);

  const segment_size_t segment_size = GENERATE_COPY(values({segment_size_t{4096}, segment_size_t{64 * 1024}}));
  const segment_size_t max_k        = (cuda::std::min) (static_max_k, segment_size);
  const segment_size_t k            = GENERATE_COPY(values({segment_size_t{1}, max_k / 2}));
  const segment_size_t num_items    = num_segments * segment_size;

  CAPTURE(segment_size, k, num_segments, direction, prefer_larger);

  // Few distinct key values -> many tied candidates in the k-th bucket. Contiguous -> resident BlockLoadToShared path.
  c2h::device_vector<key_t> keys_in_buffer(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer, key_t{0}, key_t{7});

  auto d_keys_in_ptr = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_in     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);

  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  // Values = global flattened index, so each selected value points back into the flattened input.
  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  auto d_values_in  = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), segment_size);
  c2h::device_vector<val_t> values_out_buffer(num_segments * k, thrust::no_init);
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_values_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_values_out_ptr), k);

  batched_topk_pairs<direction, determinism, tie_break>(
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments});

  // Values still belong to their keys, and no source index is selected twice.
  c2h::device_vector<key_t> expected_keys(keys_in_buffer);
  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);
  REQUIRE(verify_unique_indices(values_out_buffer, num_segments, k) == true);

  // The deterministic path must return *exactly* the index-ordered top-k. Compare per-segment selected index sets.
  c2h::host_vector<key_t> h_keys = keys_in_buffer;
  const c2h::host_vector<val_t> ref =
    reference_deterministic_topk_indices<key_t, val_t>(h_keys, num_segments, segment_size, k, direction, prefer_larger);

  c2h::host_vector<val_t> h_values_out = values_out_buffer;
  for (segment_index_t seg = 0; seg < num_segments; ++seg)
  {
    const auto seg_begin = h_values_out.begin() + static_cast<std::ptrdiff_t>(seg * k);
    std::sort(seg_begin, seg_begin + k);
  }

  REQUIRE(ref == h_values_out);
}

// Whole-`topk_policy` tuning overrides for the reproducibility test, threaded through the public API's single tuning
// query (`cuda::execution::tune`). Both force the cluster backend (which alone honors a deterministic request).
// `default_cluster_selector` uses the default cluster tuning; `alt_cluster_selector` starts from it and overrides the
// shape knobs (block size, items-per-thread, pipeline depth, tie-break granularity) to a second *valid* tuning, so the
// gpu_to_gpu config-independence check can compare two different launch configs.
struct default_cluster_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const
    -> cub::detail::batched_topk::topk_policy
  {
    return cub::detail::batched_topk::topk_policy{
      cub::detail::batched_topk::topk_backend::cluster,
      cub::detail::batched_topk::make_baseline_policy(),
      cub::detail::batched_topk::make_cluster_policy()};
  }
};

struct alt_cluster_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const
    -> cub::detail::batched_topk::topk_policy
  {
    auto cluster                       = cub::detail::batched_topk::make_cluster_policy();
    cluster.threads_per_block          = 256;
    cluster.histogram_items_per_thread = 2;
    cluster.pipeline_stages            = 2;
    cluster.tie_break_items_per_thread = 2;
    return cub::detail::batched_topk::topk_policy{
      cub::detail::batched_topk::topk_backend::cluster, cub::detail::batched_topk::make_baseline_policy(), cluster};
  }
};

// Reproducibility with an *unspecified* tie-break (which tied candidate wins is an implementation detail). We run twice
// and require the same selected index set, per each requirement's contract: `run_to_run` only promises repeated runs of
// the *same* config agree (so both runs share a tuning); `gpu_to_gpu` must be config-independent (so the second run
// uses a different valid tuning), mirroring the reduce/scan deterministic tests. Within-top-k order is unspecified, so
// we compare sorted sets. `gpu_to_gpu` cannot be checked more strictly without a second device.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs deterministic unspecified tie-break is reproducible",
         "[pairs][segmented][topk][device][cluster][determinism]",
         select_direction_list,
         determinism_list)
{
  using key_t           = cuda::std::uint32_t;
  using val_t           = cuda::std::int32_t;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  constexpr auto direction   = c2h::get<0, TestType>::value;
  constexpr auto determinism = c2h::get<1, TestType>::value;
  constexpr auto tie_break   = cuda::execution::tie_break::__tie_break_t::__unspecified;

  constexpr segment_size_t static_max_segment_size = 64 * 1024;
  constexpr segment_size_t static_max_k            = 64 * 1024;
  constexpr segment_index_t num_segments           = 2;
  // Deterministic and oversize: this configuration always routes to the SM90+ cluster backend.
  skip_if_batched_topk_backend_unavailable<determinism, tie_break>(static_max_segment_size);

  const segment_size_t segment_size = GENERATE_COPY(values({segment_size_t{4096}, segment_size_t{64 * 1024}}));
  const segment_size_t max_k        = (cuda::std::min) (static_max_k, segment_size);
  const segment_size_t k            = GENERATE_COPY(values({segment_size_t{1}, max_k / 2}));
  const segment_size_t num_items    = num_segments * segment_size;

  CAPTURE(segment_size, k, num_segments, direction);

  // Few distinct key values -> many tied candidates at the k-th bucket, so the deterministic scan actually does work.
  c2h::device_vector<key_t> keys_in_buffer(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer, key_t{0}, key_t{7});

  auto d_keys_in_ptr = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_in     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);

  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  auto d_values_in  = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), segment_size);

  // One output buffer per tuning.
  c2h::device_vector<key_t> keys_out_a(num_segments * k, thrust::no_init);
  c2h::device_vector<key_t> keys_out_b(num_segments * k, thrust::no_init);
  c2h::device_vector<val_t> values_out_a(num_segments * k, thrust::no_init);
  c2h::device_vector<val_t> values_out_b(num_segments * k, thrust::no_init);

  // Runs the same problem through the public `cub::DeviceBatchedTopK` API, tuning the whole `topk_policy` via
  // `cuda::execution::tune` so each run can pick a different valid cluster tuning (the override forces the cluster
  // backend). Drives the two-phase temp-storage protocol manually so both runs share this call site.
  const auto run_with_tuning =
    [&](auto selector, c2h::device_vector<key_t>& keys_out, c2h::device_vector<val_t>& values_out) {
      auto d_keys_out =
        cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);
      auto d_values_out =
        cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(values_out.data())), k);

      auto env = cuda::std::execution::env{
        cuda::stream_ref{cudaStream_t{0}},
        cuda::execution::require(cuda::execution::determinism::__determinism_holder_t<determinism>{},
                                 cuda::execution::tie_break::__tie_break_holder_t<tie_break>{},
                                 cuda::execution::output_ordering::unsorted),
        cuda::execution::tune(selector)};

      size_t temp_bytes = 0;
      const auto invoke = [&](void* d_temp) {
        auto seg_sizes =
          cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()};
        auto k_param = cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()};
        if constexpr (direction == cub::detail::topk::select::max)
        {
          return cub::DeviceBatchedTopK::MaxPairs(
            d_temp,
            temp_bytes,
            d_keys_in,
            d_keys_out,
            d_values_in,
            d_values_out,
            seg_sizes,
            k_param,
            cuda::args::immediate{num_segments},
            env);
        }
        else
        {
          return cub::DeviceBatchedTopK::MinPairs(
            d_temp,
            temp_bytes,
            d_keys_in,
            d_keys_out,
            d_values_in,
            d_values_out,
            seg_sizes,
            k_param,
            cuda::args::immediate{num_segments},
            env);
        }
      };
      REQUIRE(invoke(nullptr) == cudaSuccess);
      c2h::device_vector<std::uint8_t> temp_storage(temp_bytes, thrust::no_init);
      REQUIRE(invoke(thrust::raw_pointer_cast(temp_storage.data())) == cudaSuccess);
      REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    };

  run_with_tuning(default_cluster_selector{}, keys_out_a, values_out_a);
  // run_to_run: same tuning twice. gpu_to_gpu: a different valid tuning for the stronger config-independent check.
  if constexpr (determinism == cuda::execution::determinism::__determinism_t::__gpu_to_gpu)
  {
    run_with_tuning(alt_cluster_selector{}, keys_out_b, values_out_b);
  }
  else
  {
    run_with_tuning(default_cluster_selector{}, keys_out_b, values_out_b);
  }

  // Sanity: values still belong to their keys and no source index repeats within a segment.
  c2h::device_vector<key_t> expected_keys(keys_in_buffer);
  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_a, values_out_a) == true);
  REQUIRE(verify_unique_indices(values_out_a, num_segments, k) == true);

  // Determinism: both runs must select the same per-segment index set. Sort each segment (within-top-k order is free).
  c2h::host_vector<val_t> h_values_a = values_out_a;
  c2h::host_vector<val_t> h_values_b = values_out_b;
  for (segment_index_t seg = 0; seg < num_segments; ++seg)
  {
    const auto a_begin = h_values_a.begin() + static_cast<std::ptrdiff_t>(seg * k);
    const auto b_begin = h_values_b.begin() + static_cast<std::ptrdiff_t>(seg * k);
    std::sort(a_begin, a_begin + k);
    std::sort(b_begin, b_begin + k);
  }
  REQUIRE(h_values_a == h_values_b);
}

// Effective-cluster-width coverage for pairs: the large `deferred_sequence` bound sizes a wide (16-CTA) launch, but the
// actual segments are small, so a single launch mixes a fully-resident multi-CTA segment (96 Ki + 17, every rank
// works), two medium segments that leave surplus cluster CTAs idle (the runtime effective-width path, one unaligned),
// and a tiny single-CTA-collapsed segment (257). Confirms each value payload stays attached to its key through that
// path, for every determinism requirement. Streaming (overflowing) pairs are covered by the large-segment tests above,
// so this one stays small enough for `compute-sanitizer --tool racecheck`.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs work with mixed effective-width variable-size segments",
         "[pairs][segmented][topk][device][cluster][determinism]",
         det_tie_pair_combos)
{
  using key_t           = float;
  using val_t           = cuda::std::int32_t;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using combo                = c2h::get<0, TestType>;
  constexpr auto determinism = combo::determinism;
  constexpr auto tie_break   = combo::tie_break;
  // The bound far exceeds the actual segments, so the launch picks a wide cluster while the segments stay small.
  constexpr segment_size_t static_max_segment_size = 1100 * 1024;
  constexpr segment_size_t static_max_k            = 4 * 1024;
  constexpr auto direction                         = cub::detail::topk::select::max;
  // Oversize bound always routes to the SM90+ cluster backend, regardless of the determinism / tie-break requirement.
  skip_if_batched_topk_backend_unavailable<determinism, tie_break>(static_max_segment_size);
  const segment_size_t k = GENERATE_COPY(values({segment_size_t{1}, static_max_k / 2, static_max_k}));

  constexpr segment_size_t full_segment_size = 96 * 1024 + 17; // enough chunks to fill the launched cluster (no idle)
  constexpr segment_size_t med_segment_a     = 12 * 1024 + 1; // a few chunks -> most cluster CTAs idle (unaligned)
  constexpr segment_size_t med_segment_b     = 40 * 1024; // more chunks, still below the launched width
  c2h::host_vector<segment_size_t> h_segment_offsets{
    0,
    full_segment_size,
    full_segment_size + med_segment_a,
    full_segment_size + med_segment_a + med_segment_b,
    full_segment_size + med_segment_a + med_segment_b + 257};
  c2h::device_vector<segment_size_t> segment_offsets = h_segment_offsets;
  const segment_index_t num_segments                 = static_cast<segment_index_t>(h_segment_offsets.size() - 1);
  const segment_size_t num_items                     = h_segment_offsets.back();

  auto segment_offsets_it = thrust::raw_pointer_cast(segment_offsets.data());
  auto segment_size_it    = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}), segment_size_op<segment_size_t*>{segment_offsets_it});

  CAPTURE(static_max_segment_size, static_max_k, k, num_segments, num_items);

  // Each output segment holds exactly min(k, segment_size[i]) items, tightly packed.
  auto compacted_output_sizes_it = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}),
    get_output_size_op{segment_offsets.cbegin(), cuda::constant_iterator(k), num_segments});
  c2h::device_vector<segment_size_t> compacted_offsets(num_segments + 1, thrust::no_init);
  thrust::exclusive_scan(
    compacted_output_sizes_it, compacted_output_sizes_it + num_segments + 1, compacted_offsets.begin());
  const segment_size_t total_output_size = compacted_offsets.back();

  c2h::device_vector<key_t> keys_in_buffer(num_items, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(total_output_size, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);
  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_offsets.cbegin());
  auto d_keys_out =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(d_keys_out_ptr), compacted_offsets.cbegin());

  // Values = global flattened index, so each selected value points back into the flattened input.
  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  c2h::device_vector<val_t> values_out_buffer(total_output_size, thrust::no_init);
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_values_in =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(values_in_it), segment_offsets.cbegin());
  auto d_values_out =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(d_values_out_ptr), compacted_offsets.cbegin());

  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  batched_topk_pairs<direction, determinism, tie_break>(
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    cuda::args::deferred_sequence{segment_size_it, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments});

  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);
  REQUIRE(verify_unique_indices(values_out_buffer, compacted_offsets, num_segments) == true);

  segmented_sort_keys(expected_keys, num_segments, segment_offsets.cbegin(), segment_offsets.cbegin() + 1, direction);
  expected_keys = compact_to_topk_batched(expected_keys, segment_offsets, k);
  segmented_sort_keys(
    keys_out_buffer, num_segments, compacted_offsets.cbegin(), compacted_offsets.cbegin() + 1, direction);
  REQUIRE(expected_keys == keys_out_buffer);
}
#endif // TEST_TYPES == 1

C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs work with small variable-size segments",
         "[pairs][segmented][topk][device]",
         key_types,
         max_segment_size_list,
         max_num_k_list,
         select_direction_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using key_t = c2h::get<0, TestType>;
  using val_t = cuda::std::int32_t;

  // Statically constrained maximum segment size and k
  constexpr segment_size_t static_max_segment_size = c2h::get<1, TestType>::value;
  constexpr segment_size_t static_max_k            = c2h::get<2, TestType>::value;

  // Selection direction comes from the compile-time test axis.
  constexpr auto direction = c2h::get<3, TestType>::value;

  constexpr segment_size_t min_items = 1;
  constexpr segment_size_t max_items = 1'000'000;

  // Number of items
  const segment_size_t num_items = GENERATE_COPY(
    take(2, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  // Generate segment sizes
  constexpr segment_size_t min_segment_size = 1;
  constexpr auto max_segment_size           = static_max_segment_size;
  c2h::device_vector<segment_size_t> segment_offsets =
    c2h::gen_uniform_offsets<segment_size_t>(C2H_SEED(3), num_items, min_segment_size, max_segment_size);
  const segment_index_t num_segments = static_cast<segment_index_t>(segment_offsets.size() - 1);
  auto segment_offsets_it            = thrust::raw_pointer_cast(segment_offsets.data());
  auto segment_size_it               = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}), segment_size_op<segment_size_t*>{segment_offsets_it});

  // Set the k value
  const segment_size_t k =
    GENERATE_COPY(values({segment_size_t{1}, static_max_k}), take(3, random(segment_size_t{1}, static_max_k)));

  // Capture test parameters
  CAPTURE(c2h::type_name<key_t>(),
          c2h::type_name<segment_size_t>(),
          c2h::type_name<segment_index_t>(),
          static_max_segment_size,
          static_max_k,
          k,
          num_segments,
          direction);

  // Compute compacted output offsets:
  // Each output segment holds exactly min(k, segment_size[i]) items, tightly packed.
  auto compacted_output_sizes_it = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}),
    get_output_size_op{segment_offsets.cbegin(), cuda::constant_iterator(k), num_segments});
  c2h::device_vector<segment_size_t> compacted_offsets(num_segments + 1, thrust::no_init);
  thrust::exclusive_scan(
    compacted_output_sizes_it, compacted_output_sizes_it + num_segments + 1, compacted_offsets.begin());
  segment_size_t total_output_size = compacted_offsets.back();

  // Prepare keys input & output
  c2h::device_vector<key_t> keys_in_buffer(num_items, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(total_output_size, thrust::no_init);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), keys_in_buffer);
  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_offsets.cbegin());
  auto d_keys_out =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(d_keys_out_ptr), compacted_offsets.cbegin());

  // Prepare values input & output
  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  c2h::device_vector<val_t> values_out_buffer(total_output_size, thrust::no_init);
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_values_in =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(values_in_it), segment_offsets.cbegin());
  auto d_values_out =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(d_values_out_ptr), compacted_offsets.cbegin());

  // Copy input for verification
  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Run the top-k algorithm
  batched_topk_pairs<direction>(
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    cuda::args::deferred_sequence{segment_size_it, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments});

  // Verification:
  // - We verify correct top-k selection through the keys
  // - We verify that values were permuted along correctly by making sure values remain associated with their keys and
  //   making sure we do not duplicate values
  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);

  // Verify values don't appear more than once in the returned results
  REQUIRE(verify_unique_indices(values_out_buffer, compacted_offsets, num_segments) == true);

  // Verify keys are returned correctly: sort each segment of the expected input, then compact the top-k
  segmented_sort_keys(expected_keys, num_segments, segment_offsets.cbegin(), segment_offsets.cbegin() + 1, direction);
  expected_keys = compact_to_topk_batched(expected_keys, segment_offsets, k);

  // Since the results of top-k are unordered, sort compacted output segments before comparison
  segmented_sort_keys(
    keys_out_buffer, num_segments, compacted_offsets.cbegin(), compacted_offsets.cbegin() + 1, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs work with fixed-size segments and per-segment k",
         "[pairs][segmented][topk][device]",
         key_types,
         max_segment_size_list,
         max_num_k_list,
         select_direction_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using key_t = c2h::get<0, TestType>;
  using val_t = cuda::std::int32_t;

  // Statically constrained maximum segment size and k
  constexpr segment_size_t static_max_segment_size = c2h::get<1, TestType>::value;
  constexpr segment_size_t static_max_k            = c2h::get<2, TestType>::value;

  // Selection direction comes from the compile-time test axis.
  constexpr auto direction = c2h::get<3, TestType>::value;

  // Generate the (uniform) input segment size. Unlike the uniform-k tests, k still varies per segment below.
  constexpr segment_size_t min_segment_size = 1;
  constexpr auto max_segment_size           = static_max_segment_size;
  const segment_size_t segment_size = GENERATE_COPY(values({min_segment_size, segment_size_t{3}, max_segment_size}),
                                                    take(1, random(min_segment_size, max_segment_size)));

  // Skip invalid combinations
  if (segment_size > max_segment_size)
  {
    SKIP("The given segment size may not exceed the maximum segment size, we statically constrained the algorithm on.");
  }

  // Generate number of segments
  const segment_index_t num_segments = GENERATE_COPY(
    values({segment_index_t{1}, segment_index_t{42}}), take(1, random(segment_index_t{1}, segment_index_t{1000})));

  // Generate a per-segment k in [1, static_max_k]
  c2h::device_vector<segment_size_t> segment_k(num_segments, thrust::no_init);
  c2h::gen(C2H_SEED(1), segment_k, segment_size_t{1}, static_max_k);

  // Capture test parameters
  CAPTURE(c2h::type_name<key_t>(),
          c2h::type_name<segment_size_t>(),
          c2h::type_name<segment_index_t>(),
          static_max_segment_size,
          static_max_k,
          segment_size,
          num_segments,
          direction);

  // Materialize fixed-size input offsets: [0, segment_size, 2 * segment_size, ...]
  auto fixed_offsets_it = cuda::make_strided_iterator(cuda::make_counting_iterator<segment_size_t>(0), segment_size);
  c2h::device_vector<segment_size_t> segment_offsets(num_segments + 1, thrust::no_init);
  thrust::copy(fixed_offsets_it, fixed_offsets_it + (num_segments + 1), segment_offsets.begin());

  // Compute compacted output offsets: each output segment holds exactly min(k[i], segment_size) items, tightly packed.
  auto compacted_output_sizes_it = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}),
    get_output_size_op{segment_offsets.cbegin(), segment_k.cbegin(), num_segments});
  c2h::device_vector<segment_size_t> compacted_offsets(num_segments + 1, thrust::no_init);
  thrust::exclusive_scan(
    compacted_output_sizes_it, compacted_output_sizes_it + num_segments + 1, compacted_offsets.begin());
  segment_size_t total_output_size = compacted_offsets.back();

  // Prepare keys input & output. Input segments are fixed-size (strided); output segments are compacted (variable).
  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(total_output_size, thrust::no_init);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), keys_in_buffer);
  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(d_keys_out_ptr), compacted_offsets.cbegin());

  // Prepare values input & output
  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  c2h::device_vector<val_t> values_out_buffer(total_output_size, thrust::no_init);
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_values_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), segment_size);
  auto d_values_out =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(d_values_out_ptr), compacted_offsets.cbegin());

  // Copy input for verification
  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Run the top-k algorithm with a per-segment k passed as a deferred sequence
  batched_topk_pairs<direction>(
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, max_segment_size>()},
    cuda::args::deferred_sequence{
      thrust::raw_pointer_cast(segment_k.data()), cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments});

  // Verification:
  // - We verify correct top-k selection through the keys
  // - We verify that values were permuted along correctly by making sure values remain associated with their keys and
  //   making sure we do not duplicate values
  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);

  // Verify values don't appear more than once in the returned results
  REQUIRE(verify_unique_indices(values_out_buffer, compacted_offsets, num_segments) == true);

  // Verify keys are returned correctly: sort each fixed-size input segment, then compact each to its per-segment top-k.
  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  expected_keys = compact_to_topk_batched(expected_keys, segment_offsets, segment_k.cbegin());

  // Since the results of top-k are unordered, sort compacted output segments before comparison.
  segmented_sort_keys(
    keys_out_buffer, num_segments, compacted_offsets.cbegin(), compacted_offsets.cbegin() + 1, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs work with variable-size segments and per-segment k",
         "[pairs][segmented][topk][device]",
         key_types,
         max_segment_size_list,
         max_num_k_list,
         select_direction_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using key_t = c2h::get<0, TestType>;
  using val_t = cuda::std::int32_t;

  // Statically constrained maximum segment size and k
  constexpr segment_size_t static_max_segment_size = c2h::get<1, TestType>::value;
  constexpr segment_size_t static_max_k            = c2h::get<2, TestType>::value;

  // Selection direction comes from the compile-time test axis.
  constexpr auto direction = c2h::get<3, TestType>::value;

  constexpr segment_size_t min_items = 1;
  constexpr segment_size_t max_items = 1'000'000;

  // Number of items
  const segment_size_t num_items = GENERATE_COPY(
    take(2, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  // Generate segment sizes
  constexpr segment_size_t min_segment_size = 1;
  constexpr auto max_segment_size           = static_max_segment_size;
  c2h::device_vector<segment_size_t> segment_offsets =
    c2h::gen_uniform_offsets<segment_size_t>(C2H_SEED(3), num_items, min_segment_size, max_segment_size);
  const segment_index_t num_segments = static_cast<segment_index_t>(segment_offsets.size() - 1);
  auto segment_offsets_it            = thrust::raw_pointer_cast(segment_offsets.data());
  auto segment_size_it               = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}), segment_size_op<segment_size_t*>{segment_offsets_it});

  // Generate a per-segment k in [1, static_max_k]
  c2h::device_vector<segment_size_t> segment_k(num_segments, thrust::no_init);
  c2h::gen(C2H_SEED(1), segment_k, segment_size_t{1}, static_max_k);

  // Capture test parameters
  CAPTURE(c2h::type_name<key_t>(),
          c2h::type_name<segment_size_t>(),
          c2h::type_name<segment_index_t>(),
          static_max_segment_size,
          static_max_k,
          num_segments,
          direction);

  // Compute compacted output offsets:
  // Each output segment holds exactly min(k[i], segment_size[i]) items, tightly packed.
  auto compacted_output_sizes_it = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}),
    get_output_size_op{segment_offsets.cbegin(), segment_k.cbegin(), num_segments});
  c2h::device_vector<segment_size_t> compacted_offsets(num_segments + 1, thrust::no_init);
  thrust::exclusive_scan(
    compacted_output_sizes_it, compacted_output_sizes_it + num_segments + 1, compacted_offsets.begin());
  segment_size_t total_output_size = compacted_offsets.back();

  // Prepare keys input & output
  c2h::device_vector<key_t> keys_in_buffer(num_items, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(total_output_size, thrust::no_init);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), keys_in_buffer);
  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_offsets.cbegin());
  auto d_keys_out =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(d_keys_out_ptr), compacted_offsets.cbegin());

  // Prepare values input & output
  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  c2h::device_vector<val_t> values_out_buffer(total_output_size, thrust::no_init);
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_values_in =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(values_in_it), segment_offsets.cbegin());
  auto d_values_out =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(d_values_out_ptr), compacted_offsets.cbegin());

  // Copy input for verification
  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Run the top-k algorithm with a per-segment k passed as a deferred sequence
  batched_topk_pairs<direction>(
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    cuda::args::deferred_sequence{segment_size_it, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::deferred_sequence{
      thrust::raw_pointer_cast(segment_k.data()), cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments});

  // Verification:
  // - We verify correct top-k selection through the keys
  // - We verify that values were permuted along correctly by making sure values remain associated with their keys and
  //   making sure we do not duplicate values
  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);

  // Verify values don't appear more than once in the returned results
  REQUIRE(verify_unique_indices(values_out_buffer, compacted_offsets, num_segments) == true);

  // Verify keys are returned correctly: sort each segment of the expected input, then compact the per-segment top-k
  segmented_sort_keys(expected_keys, num_segments, segment_offsets.cbegin(), segment_offsets.cbegin() + 1, direction);
  expected_keys = compact_to_topk_batched(expected_keys, segment_offsets, segment_k.cbegin());

  // Since the results of top-k are unordered, sort compacted output segments before comparison
  segmented_sort_keys(
    keys_out_buffer, num_segments, compacted_offsets.cbegin(), compacted_offsets.cbegin() + 1, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}
