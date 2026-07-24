// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Defer the unsupported-architecture diagnosis to the dispatch's runtime check (not a compile-time static_assert)
// so this test compiles across all target architectures, including pre-SM90, for the full configuration space. See
// CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT in cub/device/device_batched_topk.cuh. Precedes CUB includes.
#define CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT

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
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

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
    return d_segment_ids[idx] == d_segment_ids[idx + 1] // NOLINT(bugprone-misplaced-widening-cast)
        && d_sorted_items[idx] == d_sorted_items[idx + 1];
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
_CCCL_HOST_API static cudaError_t dispatch_batched_topk_pairs(
  void* d_temp_storage,
  cuda::std::size_t& temp_storage_bytes,
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

// %PARAM% TEST_LAUNCH lid 0:2
DECLARE_TMPL_LAUNCH_WRAPPER(
  dispatch_batched_topk_pairs,
  batched_topk_pairs,
  ESCAPE_LIST(
    cub::detail::topk::select SelectDirection,
    cuda::execution::determinism::__determinism_t Determinism =
      cuda::execution::determinism::__determinism_t::__not_guaranteed,
    cuda::execution::tie_break::__tie_break_t TieBreak = cuda::execution::tie_break::__tie_break_t::__unspecified),
  ESCAPE_LIST(SelectDirection, Determinism, TieBreak));

// Wrapper-test companion to expect_batched_topk_unsupported_and_skip: when the request's backend is unavailable in this
// build, dispatch it directly (host), verify the runtime cudaErrorNotSupported, and skip the correctness checks;
// otherwise return so the caller runs its normal batched_topk_pairs<...> launch + checks. Pass the same trailing
// arguments (and Direction / Determinism / TieBreak) as that launch.
template <cub::detail::topk::select Direction,
          cuda::execution::determinism::__determinism_t Determinism =
            cuda::execution::determinism::__determinism_t::__not_guaranteed,
          cuda::execution::tie_break::__tie_break_t TieBreak = cuda::execution::tie_break::__tie_break_t::__unspecified,
          typename... Args>
void skip_unless_batched_topk_pairs_supported(cuda::std::int64_t static_max_segment_size, Args... args)
{
  if (batched_topk_backend_unavailable<Determinism, TieBreak>(static_max_segment_size))
  {
    expect_batched_topk_unsupported_and_skip([&](void* d_temp_storage, cuda::std::size_t& temp_storage_bytes) {
      return dispatch_batched_topk_pairs<Direction, Determinism, TieBreak>(d_temp_storage, temp_storage_bytes, args...);
    });
  }
}

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

  auto num_duplicates = thrust::count_if(
    cuda::make_counting_iterator(cuda::std::size_t{0}), cuda::make_counting_iterator(num_items - 1), flag_op);

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

  // Verify values stayed associated with their keys.
  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);

  // Verify no value appears more than once (catches returning a valid value multiple times).
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
  constexpr auto direction   = cub::detail::topk::select::max;
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

  // Baseline-coverable segment sizes (bounded to 127); only a deterministic / tie-break requirement routes to the SM90+
  // cluster backend.
  const auto seg_arg = cuda::args::immediate{segment_size, cuda::args::bounds<seg_size_t{1}, seg_size_t{127}>()};
  const auto k_arg   = cuda::args::immediate{k, cuda::args::bounds<seg_size_t{1}, seg_size_t{127}>()};
  const auto ns_arg  = cuda::args::immediate{num_segments};
  skip_unless_batched_topk_pairs_supported<direction, determinism, tie_break>(
    /*static_max_segment_size=*/127, d_keys_in, d_keys_out, d_values_in, d_values_out, seg_arg, k_arg, ns_arg);
  batched_topk_pairs<direction, determinism, tie_break>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, seg_arg, k_arg, ns_arg);

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
// types, so repeating them per key-type axis would only waste time on the expensive 1 Mi-element runs. The large
// segments stream from gmem and peel the unaligned tail edge -- the path these tests must cover.
#if TEST_TYPES == 1
// Launch-wrapper-compatible pairs cluster dispatch that also pins a whole-`topk_policy` tune override (`Selector`). The
// env (require + tune) is built internally from the threaded stream, so -- unlike a direct-API call that owns its env
// -- a test using it runs under every launch mode. Used only by the interface-level determinism/reproducibility test
// below (the path-pinning tune tests stay host-only direct-API).
template <typename Selector,
          cub::detail::topk::select SelectDirection,
          cuda::execution::determinism::__determinism_t Determinism,
          cuda::execution::tie_break::__tie_break_t TieBreak,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename NumSegmentsParameterT>
_CCCL_HOST_API static cudaError_t dispatch_cluster_topk_pairs(
  void* d_temp_storage,
  cuda::std::size_t& temp_storage_bytes,
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
                             cuda::execution::output_ordering::unsorted),
    cuda::execution::tune(Selector{})};
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

DECLARE_TMPL_LAUNCH_WRAPPER(
  dispatch_cluster_topk_pairs,
  cluster_topk_pairs,
  ESCAPE_LIST(typename Selector,
              cub::detail::topk::select SelectDirection,
              cuda::execution::determinism::__determinism_t Determinism,
              cuda::execution::tie_break::__tie_break_t TieBreak),
  ESCAPE_LIST(Selector, SelectDirection, Determinism, TieBreak));

// Runs the direct-API cluster top-k twice (temp-size query, then the real call) and syncs, requiring success at each
// step. `Direction` selects Min/Max at compile time. The path-pinning cluster tests below hand it an env carrying a
// `cluster_tuning_selector` tune override, which forces the SM90+ cluster backend; where no SM90+ target can serve it
// this verifies the runtime cudaErrorNotSupported and skips the correctness checks instead.
template <cub::detail::topk::select Direction,
          typename KeyInItT,
          typename KeyOutItT,
          typename ValueInItT,
          typename ValueOutItT,
          typename SegSizesT,
          typename KParamT,
          typename NumSegT,
          typename EnvT>
void run_cluster_topk_pairs(
  KeyInItT d_keys_in,
  KeyOutItT d_keys_out,
  ValueInItT d_values_in,
  ValueOutItT d_values_out,
  SegSizesT seg_sizes,
  KParamT k_param,
  NumSegT num_seg,
  EnvT env)
{
  const auto dispatch = [&](void* d_temp, cuda::std::size_t& temp_bytes) {
    if constexpr (Direction == cub::detail::topk::select::max)
    {
      return cub::DeviceBatchedTopK::MaxPairs(
        d_temp, temp_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, seg_sizes, k_param, num_seg, env);
    }
    else
    {
      return cub::DeviceBatchedTopK::MinPairs(
        d_temp, temp_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, seg_sizes, k_param, num_seg, env);
    }
  };
  if (batched_topk_cluster_backend_unavailable(/*needs_cluster=*/true))
  {
    expect_batched_topk_unsupported_and_skip(dispatch);
  }
  cuda::std::size_t temp_bytes = 0;
  REQUIRE(cudaSuccess == dispatch(nullptr, temp_bytes));
  c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_bytes, thrust::no_init);
  REQUIRE(cudaSuccess == dispatch(thrust::raw_pointer_cast(temp_storage.data()), temp_bytes));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

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

  // Oversize segments always route to the SM90+ cluster backend.
  const auto seg_arg =
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()};
  const auto k_arg  = cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()};
  const auto ns_arg = cuda::args::immediate{num_segments};
  skip_unless_batched_topk_pairs_supported<direction>(
    static_max_segment_size, d_keys_in, d_keys_out, d_values_in, d_values_out, seg_arg, k_arg, ns_arg);
  batched_topk_pairs<direction>(d_keys_in, d_keys_out, d_values_in, d_values_out, seg_arg, k_arg, ns_arg);

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
// head edge plus the persistent `tail_edge_len_items`/`process_tail_edge` value writes.
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
  const int pad                                    = GENERATE(0, 1, 3, 7);
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

  // Oversize segments always route to the SM90+ cluster backend.
  const auto seg_arg =
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()};
  const auto k_arg  = cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()};
  const auto ns_arg = cuda::args::immediate{num_segments};
  skip_unless_batched_topk_pairs_supported<direction>(
    static_max_segment_size, d_keys_in, d_keys_out, d_values_in, d_values_out, seg_arg, k_arg, ns_arg);
  batched_topk_pairs<direction>(d_keys_in, d_keys_out, d_values_in, d_values_out, seg_arg, k_arg, ns_arg);

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
// resident cluster coverage (>128 Ki), which 8/16-bit types can't represent, so we use the same signed 32-bit type as
// the internal `offset_t` to exercise the streaming path's index arithmetic (including the value gather) across
// det/non-det.
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
  const int pad                                = GENERATE(0, 7);
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

  // Oversize segments always route to the SM90+ cluster backend, regardless of the determinism / tie-break requirement.
  const auto seg_arg =
    cuda::args::immediate{segment_size, cuda::args::bounds<seg_size_t{1}, static_max_segment_size>()};
  const auto k_arg  = cuda::args::immediate{k, cuda::args::bounds<seg_size_t{1}, static_max_k>()};
  const auto ns_arg = cuda::args::immediate{num_segments};
  skip_unless_batched_topk_pairs_supported<direction, determinism, tie_break>(
    static_max_segment_size, d_keys_in, d_keys_out, d_values_in, d_values_out, seg_arg, k_arg, ns_arg);
  batched_topk_pairs<direction, determinism, tie_break>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, seg_arg, k_arg, ns_arg);

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

  c2h::host_vector<IndexT> selected(static_cast<cuda::std::size_t>(num_segments * k));
  std::vector<std::pair<KeyT, IndexT>> pairs(static_cast<cuda::std::size_t>(segment_size));
  for (SegSizeT seg = 0; seg < num_segments; ++seg)
  {
    const SegSizeT base = seg * segment_size;
    for (SegSizeT i = 0; i < segment_size; ++i)
    {
      const IndexT idx                         = static_cast<IndexT>(base + i);
      pairs[static_cast<cuda::std::size_t>(i)] = {h_keys[static_cast<cuda::std::size_t>(base + i)], encode(idx)};
    }
    if (want_max)
    {
      std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(), std::greater<std::pair<KeyT, IndexT>>{});
    }
    else
    {
      std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end());
    }
    const auto seg_begin = selected.begin() + static_cast<cuda::std::ptrdiff_t>(seg * k);
    for (SegSizeT i = 0; i < k; ++i)
    {
      seg_begin[i] = encode(pairs[static_cast<cuda::std::size_t>(i)].second);
    }
    std::sort(seg_begin, seg_begin + k);
  }
  return selected;
}

// Deterministic tie-break: a specified preference is `gpu_to_gpu` deterministic by definition, so the cluster path
// returns a uniquely defined top-k. Few distinct key values pack many ties into the k-th bucket so the preference (not
// the key comparison) drives the result; the value payload is the global index, so we compare per-segment index sets
// against the host reference (within-top-k order is unspecified).
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

  // Deterministic and oversize: this configuration always routes to the SM90+ cluster backend.
  const auto seg_arg =
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()};
  const auto k_arg  = cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()};
  const auto ns_arg = cuda::args::immediate{num_segments};
  skip_unless_batched_topk_pairs_supported<direction, determinism, tie_break>(
    static_max_segment_size, d_keys_in, d_keys_out, d_values_in, d_values_out, seg_arg, k_arg, ns_arg);
  batched_topk_pairs<direction, determinism, tie_break>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, seg_arg, k_arg, ns_arg);

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
    const auto seg_begin = h_values_out.begin() + static_cast<cuda::std::ptrdiff_t>(seg * k);
    std::sort(seg_begin, seg_begin + k);
  }

  REQUIRE(ref == h_values_out);
}

#  if TEST_LAUNCH == 0
// Tiny multi-CTA resident cross-CTA scan (pairs): `single_block = 0` disables the single-CTA fast path and cap 2 pins a
// 2-CTA cluster, so a small *fully-resident* segment (no overflow) runs the real cross-CTA prefix scan
// (`prime_placement_counters` / remote `red.add`), cluster barriers, and DSMEM histogram fold that previously only ran
// on 64 Ki+ segments too large for `compute-sanitizer racecheck`. Heavy ties (keys in [0, 7]) fill the k-th bucket so
// the deterministic scan/tie-break drives the result; the value payload (global index) is checked against the host
// index reference, so a wrong scan result is observable. Direct-API, so built once for `TEST_LAUNCH == 0`.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs run a tiny multi-CTA segment through the deterministic cross-CTA scan",
         "[pairs][segmented][topk][device][cluster][determinism]",
         select_direction_list,
         tie_break_pref_list)
{
  using key_t           = cuda::std::uint32_t;
  using val_t           = cuda::std::int32_t;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  constexpr auto direction                    = c2h::get<0, TestType>::value;
  constexpr auto tie_break                    = c2h::get<1, TestType>::value;
  [[maybe_unused]] constexpr auto determinism = cuda::execution::determinism::__determinism_t::__gpu_to_gpu; // nvhpc
                                                                                                             // warns,
                                                                                                             // only
                                                                                                             // used in
                                                                                                             // nttp
  constexpr bool prefer_larger = tie_break == cuda::execution::tie_break::__tie_break_t::__prefer_larger_index;
  constexpr segment_size_t static_max_segment_size = 2048;
  constexpr segment_size_t static_max_k            = 1024;
  constexpr segment_index_t num_segments           = 2;

  const segment_size_t segment_size = static_max_segment_size;
  const segment_size_t k            = GENERATE_COPY(values({segment_size_t{1}, static_max_k / 2}));
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

  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  auto d_values_in  = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), segment_size);
  c2h::device_vector<val_t> values_out_buffer(num_segments * k, thrust::no_init);
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_values_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_values_out_ptr), k);

  // Force the cluster backend and a resident 2-CTA cluster (single-CTA fast path disabled) through the tune query.
  auto env = cuda::std::execution::env{
    cuda::stream_ref{cudaStream_t{0}},
    cuda::execution::require(cuda::execution::determinism::__determinism_holder_t<determinism>{},
                             cuda::execution::tie_break::__tie_break_holder_t<tie_break>{},
                             cuda::execution::output_ordering::unsorted),
    cuda::execution::tune(cluster_tuning_selector<2, /*slots=*/0, /*single_block=*/0, cluster_test_chunk_bytes>{})};

  auto seg_sizes =
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()};
  auto k_param = cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()};

  run_cluster_topk_pairs<direction>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, seg_sizes, k_param, cuda::args::immediate{num_segments}, env);

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
    const auto seg_begin = h_values_out.begin() + static_cast<cuda::std::ptrdiff_t>(seg * k);
    std::sort(seg_begin, seg_begin + k);
  }

  REQUIRE(ref == h_values_out);
}
#  endif // TEST_LAUNCH == 0

// The streaming tie-break regressions below need both `num_passes` parities so they exercise both ping-pong toggle
// counts. Enforce that the two widths actually straddle the parity against the real cluster `bits_per_pass` (rather
// than hard-coding "uint32 -> odd, uint64 -> even"): if tuning ever makes both widths share a parity, this fails to
// compile.
using streaming_tie_pair_key_types = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;
template <typename KeyT>
inline constexpr int cluster_num_passes =
  cub::detail::topk::calc_num_passes<KeyT>(cub::detail::batched_topk::make_cluster_policy().bits_per_pass);
static_assert(cluster_num_passes<cuda::std::uint32_t> % 2 != cluster_num_passes<cuda::std::uint64_t> % 2,
              "streaming_tie_pair_key_types must cover both num_passes parities");

// Deterministic tie-break *while streaming*: the value-observable (index-set) test for the preselected ping-pong
// direction. Heavy ties (keys in [0, 7]) straddle the k-th bucket while a 1 Mi segment (>> resident cluster coverage)
// forces the straddling CTA onto the overflow-streaming path, so a wrong preselected direction picks the wrong tied
// indices -- observable here (unlike keys-only, where all tied keys are equal) via the value payload checked against
// the host index reference. Two key widths cover both `num_passes` parities; `tie_break_pref_list` both tie-break
// directions.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs deterministic tie-break streams the index-ordered top-k",
         "[pairs][segmented][topk][device][cluster][determinism]",
         streaming_tie_pair_key_types,
         tie_break_pref_list)
{
  using key_t           = c2h::get<0, TestType>;
  using val_t           = cuda::std::int32_t;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  constexpr auto direction     = cub::detail::topk::select::max;
  constexpr auto tie_break     = c2h::get<1, TestType>::value;
  constexpr auto determinism   = cuda::execution::determinism::__determinism_t::__gpu_to_gpu;
  constexpr bool prefer_larger = tie_break == cuda::execution::tie_break::__tie_break_t::__prefer_larger_index;

  constexpr segment_size_t static_max_segment_size = 1024 * 1024;
  constexpr segment_size_t static_max_k            = 512 * 1024;
  constexpr segment_index_t num_segments           = 2;
  const segment_size_t segment_size = static_max_segment_size - 31; // unaligned -> streaming + peeled tail edge
  // With 8 near-uniform values each bucket holds ~segment_size/8 (~128 Ki) keys: k == 1 straddles the top bucket (empty
  // front), k == 300 Ki spans full buckets plus a mid-bucket straddle (non-empty front).
  const segment_size_t k = GENERATE_COPY(values({segment_size_t{1}, segment_size_t{300 * 1024}}));

  CAPTURE(c2h::type_name<key_t>(), static_max_segment_size, static_max_k, segment_size, k, num_segments, prefer_larger);

  // Few distinct key values -> many tied candidates in the k-th bucket. Contiguous -> BlockLoadToShared (streaming, as
  // the 1 Mi segment exceeds resident coverage).
  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
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

  // Deterministic and oversize: this configuration always routes to the SM90+ cluster backend.
  const auto seg_arg =
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()};
  const auto k_arg  = cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()};
  const auto ns_arg = cuda::args::immediate{num_segments};
  skip_unless_batched_topk_pairs_supported<direction, determinism, tie_break>(
    static_max_segment_size, d_keys_in, d_keys_out, d_values_in, d_values_out, seg_arg, k_arg, ns_arg);
  batched_topk_pairs<direction, determinism, tie_break>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, seg_arg, k_arg, ns_arg);

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
    const auto seg_begin = h_values_out.begin() + static_cast<cuda::std::ptrdiff_t>(seg * k);
    std::sort(seg_begin, seg_begin + k);
  }

  REQUIRE(ref == h_values_out);
}

// Maps a flattened element index to one of 8 values (host/device consistent), so a large segment has many duplicates
// and the k-th bucket holds a large tied set. Same twiddle as the keys test's heavy-tie op.
template <typename KeyT>
struct heavy_tie_key_op
{
  template <typename IndexT>
  __host__ __device__ KeyT operator()(IndexT i) const
  {
    const unsigned h = static_cast<unsigned>(static_cast<cuda::std::uint64_t>(i) * 2654435761u);
    return static_cast<KeyT>(static_cast<int>((h >> 13) & 7u));
  }
};

// Non-contiguous per-segment key iterator (like `counting_segment_keys_op`, but heavy-tie instead of identity): the
// transform iterator makes `use_block_load_to_shared` false, forcing the generic streaming path, while the heavy-tie
// op on the global index injects the boundary straddle. Segment `seg` -> keys at [seg*segment_size, ...).
template <typename KeyT, typename SegmentSizeT>
struct heavy_tie_segment_keys_op
{
  SegmentSizeT segment_size;

  template <typename IndexT>
  __host__ __device__ auto operator()(IndexT seg) const
  {
    return cuda::make_transform_iterator(
      cuda::make_counting_iterator(static_cast<SegmentSizeT>(seg) * segment_size), heavy_tie_key_op<KeyT>{});
  }
};

// Generic-path counterpart of the deterministic streaming tie-break test above: a non-contiguous key iterator forces
// the agent's generic overflow-streaming path (not BlockLoadToShared) while still straddling the k-th bucket, so it
// covers the preselected ping-pong direction on the generic streamer for both `num_passes` parities and both tie-break
// directions. Value-observable: the value payload (global index) is checked against the host index reference.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs deterministic tie-break streams the index-ordered top-k (generic path)",
         "[pairs][segmented][topk][device][cluster][determinism]",
         streaming_tie_pair_key_types,
         tie_break_pref_list)
{
  using key_t           = c2h::get<0, TestType>;
  using val_t           = cuda::std::int32_t;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  constexpr auto direction     = cub::detail::topk::select::max;
  constexpr auto tie_break     = c2h::get<1, TestType>::value;
  constexpr auto determinism   = cuda::execution::determinism::__determinism_t::__gpu_to_gpu;
  constexpr bool prefer_larger = tie_break == cuda::execution::tie_break::__tie_break_t::__prefer_larger_index;

  constexpr segment_size_t static_max_segment_size = 1024 * 1024;
  constexpr segment_size_t static_max_k            = 512 * 1024;
  constexpr segment_index_t num_segments           = 2;
  // Non-round size -> streaming (the generic fallback reads any trailing items straight from gmem; it never peels).
  const segment_size_t segment_size = static_max_segment_size - 31;
  const segment_size_t k            = GENERATE_COPY(values({segment_size_t{1}, segment_size_t{300 * 1024}}));
  const segment_size_t num_items    = num_segments * segment_size;

  CAPTURE(c2h::type_name<key_t>(), static_max_segment_size, static_max_k, segment_size, k, num_segments, prefer_larger);

  // Non-contiguous key input (transform iterator) -> generic streaming path; heavy-tie values straddle the k-th bucket.
  auto d_keys_in = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}), heavy_tie_segment_keys_op<key_t, segment_size_t>{segment_size});

  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  // Values = global flattened index, so each selected value points back into the flattened input.
  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  auto d_values_in  = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), segment_size);
  c2h::device_vector<val_t> values_out_buffer(num_segments * k, thrust::no_init);
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_values_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_values_out_ptr), k);

  // Deterministic and oversize: this configuration always routes to the SM90+ cluster backend.
  const auto seg_arg =
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()};
  const auto k_arg  = cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()};
  const auto ns_arg = cuda::args::immediate{num_segments};
  skip_unless_batched_topk_pairs_supported<direction, determinism, tie_break>(
    static_max_segment_size, d_keys_in, d_keys_out, d_values_in, d_values_out, seg_arg, k_arg, ns_arg);
  batched_topk_pairs<direction, determinism, tie_break>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, seg_arg, k_arg, ns_arg);

  // Materialize the (non-contiguous) key input for verification: same heavy-tie function of the global index.
  c2h::host_vector<key_t> h_keys(static_cast<cuda::std::size_t>(num_items));
  heavy_tie_key_op<key_t> key_op{};
  for (segment_size_t idx = 0; idx < num_items; ++idx)
  {
    h_keys[static_cast<cuda::std::size_t>(idx)] = key_op(idx);
  }
  const c2h::device_vector<key_t> keys_materialized = h_keys;

  // Values still belong to their keys, and no source index is selected twice.
  REQUIRE(verify_pairs_consistency(keys_materialized, keys_out_buffer, values_out_buffer) == true);
  REQUIRE(verify_unique_indices(values_out_buffer, num_segments, k) == true);

  // The deterministic path must return *exactly* the index-ordered top-k. Compare per-segment selected index sets.
  const c2h::host_vector<val_t> ref =
    reference_deterministic_topk_indices<key_t, val_t>(h_keys, num_segments, segment_size, k, direction, prefer_larger);

  c2h::host_vector<val_t> h_values_out = values_out_buffer;
  for (segment_index_t seg = 0; seg < num_segments; ++seg)
  {
    const auto seg_begin = h_values_out.begin() + static_cast<cuda::std::ptrdiff_t>(seg * k);
    std::sort(seg_begin, seg_begin + k);
  }

  REQUIRE(ref == h_values_out);
}

#  if TEST_LAUNCH == 0
// Reproducibility with an *unspecified* tie-break (which tied candidate wins is an implementation detail). We run twice
// and require the same selected index set, per each requirement's contract: `run_to_run` only promises repeated runs of
// the *same* config agree (so both runs share a tuning); `gpu_to_gpu` must be config-independent (so the second run
// uses a different valid tuning), mirroring the reduce/scan deterministic tests. Within-top-k order is unspecified, so
// we compare sorted sets. `gpu_to_gpu` cannot be checked more strictly without a second device.
//
// Both configs reuse tiny `cluster_tuning_selector` shapes already instantiated by the tests above (no extra kernels):
// config A is the 2-CTA fully-resident shape (P1's cross-CTA-scan tuning); config B is the single-CTA streaming shape
// (cap 1 from the schedule sweep). The CTA-count/streaming contrast is the second valid config the gpu_to_gpu check
// needs, while forcing the cluster backend (which alone honors a deterministic request) at a racecheck-tiny footprint.
//
// Asserts an *interface-level* contract -- a deterministic request stays reproducible. Routed through the
// `cluster_topk_pairs` launch wrapper, which builds the require+tune env internally so both runs share one call site
// with different tunings. Built once for `TEST_LAUNCH == 0`: graph launch (`TEST_LAUNCH == 2`) re-runs the same host
// dispatch arm, adding only redundant kernel instantiations.
using repro_config_a = cluster_tuning_selector<2, /*slots=*/0, /*single_block=*/0, cluster_test_chunk_bytes>;
using repro_config_b =
  cluster_tuning_selector<1, /*slots=*/4, /*single_block=*/0, cluster_test_chunk_bytes, /*stages=*/4>;
C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs deterministic unspecified tie-break is reproducible",
         "[pairs][segmented][topk][device][cluster][determinism]",
         select_direction_list,
         determinism_list)
{
  using key_t           = cuda::std::uint32_t;
  using val_t           = cuda::std::int32_t;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  static constexpr auto direction   = c2h::get<0, TestType>::value;
  static constexpr auto determinism = c2h::get<1, TestType>::value;
  static constexpr auto tie_break   = cuda::execution::tie_break::__tie_break_t::__unspecified;

  // Tiny footprint: config B (cap 1, 4 resident slots = 512 keys) overflows this 2048-key segment and streams, while
  // config A (cap 2, unrestricted slots) keeps it resident across a 2-CTA cluster -- the streaming-vs-resident contrast
  // the gpu_to_gpu config-independence check needs, at a racecheck-tiny size.
  constexpr segment_size_t static_max_segment_size = 2048;
  constexpr segment_size_t static_max_k            = 1024;
  constexpr segment_index_t num_segments           = 2;
  const segment_size_t segment_size                = static_max_segment_size;
  const segment_size_t max_k                       = (cuda::std::min) (static_max_k, segment_size);
  const segment_size_t k                           = GENERATE_COPY(values({segment_size_t{1}, max_k / 2}));
  const segment_size_t num_items                   = num_segments * segment_size;

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
  // backend). `selector` is only used for its type; the `cluster_topk_pairs` wrapper drives the two-phase
  // temp-storage protocol so both runs share this call site.
  const auto run_with_tuning =
    [&](auto selector, c2h::device_vector<key_t>& keys_out, c2h::device_vector<val_t>& values_out) {
      auto d_keys_out =
        cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);
      auto d_values_out =
        cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(values_out.data())), k);

      cluster_topk_pairs<decltype(selector), direction, determinism, tie_break>(
        d_keys_in,
        d_keys_out,
        d_values_in,
        d_values_out,
        cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
        cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
        cuda::args::immediate{num_segments});
    };

  // Deterministic request always routes to the SM90+ cluster backend (the tune override forces it at this tiny size).
  {
    auto d_keys_out_a =
      cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out_a.data())), k);
    auto d_values_out_a =
      cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(values_out_a.data())), k);
    skip_unless_batched_topk_pairs_supported<direction, determinism, tie_break>(
      static_max_segment_size,
      d_keys_in,
      d_keys_out_a,
      d_values_in,
      d_values_out_a,
      cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
      cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
      cuda::args::immediate{num_segments});
  }

  run_with_tuning(repro_config_a{}, keys_out_a, values_out_a);
  // run_to_run: same tuning twice. gpu_to_gpu: a different valid tuning for the stronger config-independent check.
  if constexpr (determinism == cuda::execution::determinism::__determinism_t::__gpu_to_gpu)
  {
    run_with_tuning(repro_config_b{}, keys_out_b, values_out_b);
  }
  else
  {
    run_with_tuning(repro_config_a{}, keys_out_b, values_out_b);
  }

  // Sanity: values still belong to their keys and no source index repeats within a segment.
  c2h::device_vector<key_t> expected_keys(keys_in_buffer);
  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_a, values_out_a) == true);
  REQUIRE(verify_unique_indices(values_out_a, num_segments, k) == true);
  // The second run uses config B (the streaming path under gpu_to_gpu); validate its pairing/uniqueness too so a
  // B-only key/value mismatch cannot hide behind the index-set comparison below.
  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_b, values_out_b) == true);
  REQUIRE(verify_unique_indices(values_out_b, num_segments, k) == true);

  // Determinism: both runs must select the same per-segment index set. Sort each segment (within-top-k order is free).
  c2h::host_vector<val_t> h_values_a = values_out_a;
  c2h::host_vector<val_t> h_values_b = values_out_b;
  for (segment_index_t seg = 0; seg < num_segments; ++seg)
  {
    const auto a_begin = h_values_a.begin() + static_cast<cuda::std::ptrdiff_t>(seg * k);
    const auto b_begin = h_values_b.begin() + static_cast<cuda::std::ptrdiff_t>(seg * k);
    std::sort(a_begin, a_begin + k);
    std::sort(b_begin, b_begin + k);
  }
  REQUIRE(h_values_a == h_values_b);
}
#  endif // TEST_LAUNCH == 0

#  if TEST_LAUNCH == 0
// Cluster-width cap + tiny streaming (pairs): force the cluster backend, cap the launch to 1 or 2 CTAs, and cap the
// resident slots so a tiny segment overflows and streams -- deterministically reaching the overflow/streaming path
// (single-CTA at cap 1, a fixed 2-CTA cluster at cap 2) with the value payload carried along, at a small footprint
// rather than the 1 Mi a hardware-derived wide cluster would otherwise need. Both `stream_stages` and `prologue` are
// `min(PipelineStages, .)`, so sweeping `PipelineStages` {2,4,8} across the two caps (cap 1 -> ~8 overflow chunks/CTA,
// cap 2 -> ~2) spans the `stream_stages` <, ==, > `prologue` trichotomy and reaches the `stage_base` up-front prime
// (the `>` case). The 4-slot cap leaves an aligned resident tail here; the complementary misaligned-tail `stage_rot`
// reorder is covered by the dedicated test below (reached through the slot cap, not a distinct `PipelineStages`).
// Swept over `det_tie_pair_combos` (both drivers); the deterministic combos additionally reach the reverse first-wave
// direction. Direct-API, so built once for `TEST_LAUNCH == 0`.
using cluster_cap_list = c2h::enum_type_list<int, 1, 2>;
using stage_list       = c2h::enum_type_list<int, 2, 4, 8>;

C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs stream a tiny oversize segment across the pipeline-stage schedule",
         "[pairs][segmented][topk][device][cluster][determinism]",
         cluster_cap_list,
         stage_list,
         det_tie_pair_combos)
{
  using key_t           = float;
  using val_t           = cuda::std::int32_t;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  constexpr int cluster_cap  = c2h::get<0, TestType>::value;
  constexpr int stages       = c2h::get<1, TestType>::value;
  using combo                = c2h::get<2, TestType>;
  constexpr auto determinism = combo::determinism;
  constexpr auto tie_break   = combo::tie_break;
  constexpr auto direction   = cub::detail::topk::select::max;

  constexpr segment_size_t static_max_segment_size = 1536;
  constexpr segment_size_t static_max_k            = 512;
  constexpr segment_index_t num_segments           = 2;
  const segment_size_t segment_size = static_max_segment_size - 31; // unaligned -> peeled overflow tail edge
  const segment_size_t max_k        = (cuda::std::min) (static_max_k, segment_size);
  const segment_size_t k            = GENERATE_COPY(values({segment_size_t{1}, max_k}));

  CAPTURE(cluster_cap, stages, static_max_segment_size, static_max_k, segment_size, k, num_segments);

  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);
  auto d_keys_in_ptr = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_in     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);

  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  auto d_values_in  = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), segment_size);
  c2h::device_vector<val_t> values_out_buffer(num_segments * k, thrust::no_init);
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_values_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_values_out_ptr), k);

  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Force the cluster backend at the requested width cap, cap resident slots to 4, and set the pipeline depth, all
  // through the public API's single tuning query.
  auto env = cuda::std::execution::env{
    cuda::stream_ref{cudaStream_t{0}},
    cuda::execution::require(cuda::execution::determinism::__determinism_holder_t<determinism>{},
                             cuda::execution::tie_break::__tie_break_holder_t<tie_break>{},
                             cuda::execution::output_ordering::unsorted),
    cuda::execution::tune(
      cluster_tuning_selector<cluster_cap, /*slots=*/4, /*single_block=*/0, cluster_test_chunk_bytes, stages>{})};

  auto seg_sizes =
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()};
  auto k_param = cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()};

  run_cluster_topk_pairs<direction>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, seg_sizes, k_param, cuda::args::immediate{num_segments}, env);

  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);
  REQUIRE(verify_unique_indices(values_out_buffer, num_segments, k) == true);

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

// Deterministic counterpart of the keys misaligned-tail (`stage_rot`) test: reuses the schedule sweep's cap-1 /
// stages-2 kernel (same `cluster_tuning_selector<1, 4, 0, .., 2>` type and same size/k bounds -> no new instantiation);
// only the *runtime* segment size differs. Sized to 5 chunks so, at 4 slots minus 1 reserved stream slot, 3 resident
// chunks remain and `3 % prologue(2) == 1` misaligns the tail. Swept over `det_tie_pair_combos` so the deterministic
// combos flip the first-wave direction, rotating both the forward and the reverse `first_wave_chunk_for_stage`
// mapping. The unaligned size (`609 = 5 chunks - 31`) also peels the unaligned tail edge under the rotation.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs stream a tiny oversize segment with a misaligned resident tail",
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
  constexpr auto direction   = cub::detail::topk::select::max;

  constexpr segment_size_t static_max_segment_size = 1536;
  constexpr segment_size_t static_max_k            = 512;
  constexpr segment_index_t num_segments           = 2;
  const segment_size_t segment_size = 640 - 31; // 5 chunks, unaligned -> misaligned tail + peeled overflow tail edge
  const segment_size_t max_k        = (cuda::std::min) (static_max_k, segment_size);
  const segment_size_t k            = GENERATE_COPY(values({segment_size_t{1}, max_k}));

  CAPTURE(static_max_segment_size, static_max_k, segment_size, k, num_segments);

  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);
  auto d_keys_in_ptr = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_in     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);

  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  auto d_values_in  = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), segment_size);
  c2h::device_vector<val_t> values_out_buffer(num_segments * k, thrust::no_init);
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_values_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_values_out_ptr), k);

  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Same selector as the schedule sweep's cap-1 / stages-2 case; the 5-chunk runtime segment overflows 4 slots and
  // leaves 3 resident chunks -> a misaligned reload tail that triggers `stage_rot`.
  auto env = cuda::std::execution::env{
    cuda::stream_ref{cudaStream_t{0}},
    cuda::execution::require(cuda::execution::determinism::__determinism_holder_t<determinism>{},
                             cuda::execution::tie_break::__tie_break_holder_t<tie_break>{},
                             cuda::execution::output_ordering::unsorted),
    cuda::execution::tune(
      cluster_tuning_selector<1, /*slots=*/4, /*single_block=*/0, cluster_test_chunk_bytes, /*stages=*/2>{})};

  auto seg_sizes =
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()};
  auto k_param = cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()};

  run_cluster_topk_pairs<direction>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, seg_sizes, k_param, cuda::args::immediate{num_segments}, env);

  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);
  REQUIRE(verify_unique_indices(values_out_buffer, num_segments, k) == true);

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

// Forward first-wave `stage_base` prime (`stream_stages > prologue` *with* a resident chunk present). Reuses the
// schedule sweep's cap-1 / stages-4 kernel (same `cluster_tuning_selector<1, 4, 0, .., 4>` type and static bounds -> no
// new instantiation); only the runtime segment size differs. Sized to 7 chunks so, at 4 slots, streaming reserves 3 (=
// min(stages, excess=3)) and 1 resident chunk remains: `stream_stages(3) > prologue(1)` yet `my_resident > 0`, so a
// forward wave sets `stage_base = 2` and the merged loop's stream re-arm must seed its ring counter at
// `fw(stage_base)`, not `fw(0)`. The schedule sweep never reaches this (its `>` configs all collapse to `my_resident ==
// 0` pure streaming, which guards `stage_base` back to 0). The combos that select a forward first wave drive the new
// coverage; the reverse ones here land on `stage_base == 0` (`overflow_chunks(6) % stream_stages(3) == 0`), the
// simplest reverse case (the wrapping non-zero reverse `stage_base` is pinned by the C = 5 test below).
C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs stream a tiny oversize segment with a resident chunk below a wider stream",
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
  constexpr auto direction   = cub::detail::topk::select::max;

  constexpr segment_size_t static_max_segment_size = 1536;
  constexpr segment_size_t static_max_k            = 512;
  constexpr segment_index_t num_segments           = 2;
  const segment_size_t segment_size = 896 - 31; // 7 chunks (128 items each), unaligned -> peeled overflow tail edge
  const segment_size_t max_k        = (cuda::std::min) (static_max_k, segment_size);
  const segment_size_t k            = GENERATE_COPY(values({segment_size_t{1}, max_k}));

  CAPTURE(static_max_segment_size, static_max_k, segment_size, k, num_segments);

  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);
  auto d_keys_in_ptr = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_in     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);

  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  auto d_values_in  = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), segment_size);
  c2h::device_vector<val_t> values_out_buffer(num_segments * k, thrust::no_init);
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_values_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_values_out_ptr), k);

  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  auto env = cuda::std::execution::env{
    cuda::stream_ref{cudaStream_t{0}},
    cuda::execution::require(cuda::execution::determinism::__determinism_holder_t<determinism>{},
                             cuda::execution::tie_break::__tie_break_holder_t<tie_break>{},
                             cuda::execution::output_ordering::unsorted),
    cuda::execution::tune(
      cluster_tuning_selector<1, /*slots=*/4, /*single_block=*/0, cluster_test_chunk_bytes, /*stages=*/4>{})};

  auto seg_sizes =
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()};
  auto k_param = cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()};

  run_cluster_topk_pairs<direction>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, seg_sizes, k_param, cuda::args::immediate{num_segments}, env);

  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);
  REQUIRE(verify_unique_indices(values_out_buffer, num_segments, k) == true);

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

// Reverse first-wave with a non-zero cyclic `stage_base` and a wrapping resident window. Reuses the schedule sweep's
// cap-1 / slots-4 / stages-4 kernel (no new instantiation); only the runtime segment size differs. Sized to 5 chunks so
// the single CTA reserves 1 stream slot (`excess = 1`), leaving 3 resident chunks: sub-case B (`stream_stages(1) <=
// prologue(3)`) with `overflow_chunks = 2`, so a reverse first wave (`first_wave_is_forward == false`) sets
// `stage_base = (s0+1) % stage_cycle = 1` over `stage_cycle = 3`, giving the cyclic window `[1,4) mod 3`. Resident
// chunks 0,1,2 then map to stages 0,2,1 (descending cycle position from `reverse_cycle_seed`), and the `-1` re-arm must
// free stream stage 0 in consume order. The `stage_base == 0` reverse case is covered by the wider-stream test above (C
// = 7); this pins the wrapping, non-zero-`stage_base` reverse path. Only the deterministic combos whose (tie-break,
// pass-parity) select a reverse first wave exercise it; the rest re-drive forward paths.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs stream a tiny oversize segment with a wrapping reverse resident window",
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
  constexpr auto direction   = cub::detail::topk::select::max;

  constexpr segment_size_t static_max_segment_size = 1536;
  constexpr segment_size_t static_max_k            = 512;
  constexpr segment_index_t num_segments           = 2;
  const segment_size_t segment_size = 640 - 31; // 5 chunks (128 items each), unaligned -> peeled overflow tail edge
  const segment_size_t max_k        = (cuda::std::min) (static_max_k, segment_size);
  const segment_size_t k            = GENERATE_COPY(values({segment_size_t{1}, max_k}));

  CAPTURE(static_max_segment_size, static_max_k, segment_size, k, num_segments);

  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);
  auto d_keys_in_ptr = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_in     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);

  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  auto d_values_in  = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), segment_size);
  c2h::device_vector<val_t> values_out_buffer(num_segments * k, thrust::no_init);
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_values_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_values_out_ptr), k);

  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  auto env = cuda::std::execution::env{
    cuda::stream_ref{cudaStream_t{0}},
    cuda::execution::require(cuda::execution::determinism::__determinism_holder_t<determinism>{},
                             cuda::execution::tie_break::__tie_break_holder_t<tie_break>{},
                             cuda::execution::output_ordering::unsorted),
    cuda::execution::tune(
      cluster_tuning_selector<1, /*slots=*/4, /*single_block=*/0, cluster_test_chunk_bytes, /*stages=*/4>{})};

  auto seg_sizes =
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()};
  auto k_param = cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()};

  run_cluster_topk_pairs<direction>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, seg_sizes, k_param, cuda::args::immediate{num_segments}, env);

  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);
  REQUIRE(verify_unique_indices(values_out_buffer, num_segments, k) == true);

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

// Partial final wave: `overflow_chunks % stream_stages != 0`, so the streaming reload branch wraps its ring on an
// uneven tail. Reuses the misaligned-tail test's cap-1 / slots-4 / stages-2 kernel (no new instantiation). Sized to 7
// chunks so, at 4 slots / 2 stages, streaming reserves 2 (`excess = 3`, `min(stages,excess) = 2`) and 2 resident chunks
// remain: sub-case B with `overflow_chunks = 5`, `stream_stages = 2` -> `5 % 2 == 1`. The other tune tests all keep
// `overflow_chunks` a multiple of `stream_stages`, so this is the only tiny check of the uneven-tail reload for both
// directions (deterministic combos drive reverse, non-deterministic forward).
C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs stream a tiny oversize segment with a partial final overflow wave",
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
  constexpr auto direction   = cub::detail::topk::select::max;

  constexpr segment_size_t static_max_segment_size = 1536;
  constexpr segment_size_t static_max_k            = 512;
  constexpr segment_index_t num_segments           = 2;
  const segment_size_t segment_size = 896 - 63; // 7 chunks (128 items each), unaligned -> peeled overflow tail edge
  const segment_size_t max_k        = (cuda::std::min) (static_max_k, segment_size);
  const segment_size_t k            = GENERATE_COPY(values({segment_size_t{1}, max_k}));

  CAPTURE(static_max_segment_size, static_max_k, segment_size, k, num_segments);

  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);
  auto d_keys_in_ptr = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_in     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);

  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  auto d_values_in  = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), segment_size);
  c2h::device_vector<val_t> values_out_buffer(num_segments * k, thrust::no_init);
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_values_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_values_out_ptr), k);

  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  auto env = cuda::std::execution::env{
    cuda::stream_ref{cudaStream_t{0}},
    cuda::execution::require(cuda::execution::determinism::__determinism_holder_t<determinism>{},
                             cuda::execution::tie_break::__tie_break_holder_t<tie_break>{},
                             cuda::execution::output_ordering::unsorted),
    cuda::execution::tune(
      cluster_tuning_selector<1, /*slots=*/4, /*single_block=*/0, cluster_test_chunk_bytes, /*stages=*/2>{})};

  auto seg_sizes =
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()};
  auto k_param = cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()};

  run_cluster_topk_pairs<direction>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, seg_sizes, k_param, cuda::args::immediate{num_segments}, env);

  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);
  REQUIRE(verify_unique_indices(values_out_buffer, num_segments, k) == true);

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}
#  endif // TEST_LAUNCH == 0

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

  // Oversize bound always routes to the SM90+ cluster backend, regardless of the determinism / tie-break requirement.
  const auto seg_arg =
    cuda::args::deferred_sequence{segment_size_it, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()};
  const auto k_arg  = cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()};
  const auto ns_arg = cuda::args::immediate{num_segments};
  skip_unless_batched_topk_pairs_supported<direction, determinism, tie_break>(
    static_max_segment_size, d_keys_in, d_keys_out, d_values_in, d_values_out, seg_arg, k_arg, ns_arg);
  batched_topk_pairs<direction, determinism, tie_break>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, seg_arg, k_arg, ns_arg);

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
