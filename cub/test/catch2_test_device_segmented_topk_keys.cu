// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_batched_topk.cuh>
#include <cub/device/dispatch/dispatch_batched_topk_cluster.cuh>
#include <cub/util_type.cuh>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/iterator>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__cmath/signbit.h>

#include "catch2_test_device_topk_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/extended_types.h>
#include <catch2/generators/catch_generators.hpp>

struct is_minus_zero
{
  __device__ bool operator()(float x) const
  {
    return x == 0.0f && cuda::std::signbit(x);
  }
};

// Maps a flat element index to one of only 8 distinct key values, so a large segment has many duplicates and the k-th
// key's bucket holds a large tied-candidate set. Stresses the cluster agent's candidate/tie-break path.
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

enum class topk_backend
{
  baseline,
  cluster,
};

inline constexpr topk_backend selected_backend = topk_backend::cluster;

template <cub::detail::topk::select SelectDirection,
          cuda::execution::determinism::__determinism_t Determinism =
            cuda::execution::determinism::__determinism_t::__not_guaranteed,
          cuda::execution::tie_break::__tie_break_t TieBreak = cuda::execution::tie_break::__tie_break_t::__unspecified,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename SegmentSizeParamT,
          typename KParamT,
          typename NumSegmentsParameterT>
CUB_RUNTIME_FUNCTION static cudaError_t dispatch_batched_topk_keys(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  SegmentSizeParamT segment_sizes,
  KParamT k,
  NumSegmentsParameterT num_segments,
  cudaStream_t stream = nullptr)
{
  if constexpr (selected_backend == topk_backend::cluster)
  {
    return cub::detail::batched_topk_cluster::dispatch<Determinism, TieBreak>(
      d_temp_storage,
      temp_storage_bytes,
      d_key_segments_it,
      d_key_segments_out_it,
      static_cast<cub::NullType**>(nullptr),
      static_cast<cub::NullType**>(nullptr),
      segment_sizes,
      k,
      cuda::args::constant<SelectDirection>{},
      num_segments,
      stream);
  }
  else
  {
    // Baseline backend routes through the public API; the cluster backend keeps using the lower-level dispatch above.
    // The public API takes no total-items guarantee and always runs nondeterministic (the determinism-aware tests skip
    // the deterministic combos for non-cluster backends).
    auto env = cuda::std::execution::env{
      cuda::stream_ref{stream},
      cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                               cuda::execution::tie_break::unspecified,
                               cuda::execution::output_ordering::unsorted)};
    if constexpr (SelectDirection == cub::detail::topk::select::max)
    {
      return cub::DeviceBatchedTopK::MaxKeys(
        d_temp_storage,
        temp_storage_bytes,
        d_key_segments_it,
        d_key_segments_out_it,
        segment_sizes,
        k,
        num_segments,
        env);
    }
    else
    {
      return cub::DeviceBatchedTopK::MinKeys(
        d_temp_storage,
        temp_storage_bytes,
        d_key_segments_it,
        d_key_segments_out_it,
        segment_sizes,
        k,
        num_segments,
        env);
    }
  }
}

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_TMPL_LAUNCH_WRAPPER(
  dispatch_batched_topk_keys,
  batched_topk_keys,
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

// Selection direction is a compile-time option; cover both as a static test axis.
using select_direction_list =
  c2h::enum_type_list<cub::detail::topk::select, cub::detail::topk::select::min, cub::detail::topk::select::max>;

// Segment-size argument form for the single-value (uniform) fixed-size tests. The host-value forms (`immediate` and an
// un-annotated raw scalar, which auto-wraps as an `immediate` with loose bounds) are distributed across the TEST_TYPES
// variants so each variant compiles only one extra dispatch instantiation (balancing build time). The un-annotated form
// reports a `numeric_limits<T>::max()` upper bound, so the types_2 variant also exercises the loose-bound
// oversize/streaming fallback (and its int-narrowing overflow guard) on the cluster backend. The remaining public forms
// get their own dedicated tests below (`constant` needs a compile-time size; `deferred` needs a device-accessible
// handle), and `deferred_sequence` is covered by the variable-size tests.
enum class seg_size_arg
{
  immediate_form,
  unannotated_form,
};

template <seg_size_arg Form, auto Lo, auto Hi, typename SizeT>
auto make_segment_size_arg(SizeT segment_size)
{
  if constexpr (Form == seg_size_arg::immediate_form)
  {
    return cuda::args::immediate{segment_size, cuda::args::bounds<Lo, Hi>()};
  }
  else
  {
    return segment_size; // un-annotated: auto-wrapped as an immediate with loose bounds
  }
}

#if TEST_TYPES == 2
inline constexpr seg_size_arg fixed_seg_size_arg = seg_size_arg::unannotated_form;
#else
inline constexpr seg_size_arg fixed_seg_size_arg = seg_size_arg::immediate_form;
#endif

// A (determinism, tie-break) requirement pair, used as a single compile-time test axis. Not a full cross product: a
// tie-break preference is only meaningful with a deterministic requirement and is `gpu_to_gpu` by definition.
template <cuda::execution::determinism::__determinism_t Determinism, cuda::execution::tie_break::__tie_break_t TieBreak>
struct det_tie
{
  static constexpr auto determinism = Determinism;
  static constexpr auto tie_break   = TieBreak;
};

// The 5 valid determinism/tie-break combinations. The selected key multiset is identical for all of them (which tied
// index wins is not observable in keys-only output), so each is verified against the same reference -- confirming every
// code path computes the correct top-k, including the boundary tie count.
using det_tie_combos =
  c2h::type_list<det_tie<cuda::execution::determinism::__determinism_t::__not_guaranteed,
                         cuda::execution::tie_break::__tie_break_t::__unspecified>,
                 det_tie<cuda::execution::determinism::__determinism_t::__run_to_run,
                         cuda::execution::tie_break::__tie_break_t::__unspecified>,
                 det_tie<cuda::execution::determinism::__determinism_t::__gpu_to_gpu,
                         cuda::execution::tie_break::__tie_break_t::__unspecified>,
                 det_tie<cuda::execution::determinism::__determinism_t::__gpu_to_gpu,
                         cuda::execution::tie_break::__tie_break_t::__prefer_smaller_index>,
                 det_tie<cuda::execution::determinism::__determinism_t::__gpu_to_gpu,
                         cuda::execution::tie_break::__tie_break_t::__prefer_larger_index>>;

C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys work with small fixed-size segments",
         "[keys][segmented][topk][device]",
         key_types,
         max_segment_size_list,
         max_num_k_list,
         select_direction_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using key_t = c2h::get<0, TestType>;

  // Statically constrained maximum segment size and k
  constexpr segment_size_t static_max_segment_size = c2h::get<1, TestType>::value;
  constexpr segment_size_t static_max_k            = c2h::get<2, TestType>::value;

  // Selection direction comes from the compile-time test axis.
  constexpr auto direction = c2h::get<3, TestType>::value;

  // Generate segment size
  constexpr segment_size_t min_segment_size = 1;
  constexpr auto max_segment_size           = static_max_segment_size;
  const segment_size_t segment_size = GENERATE_COPY(values({min_segment_size, segment_size_t{3}, max_segment_size}),
                                                    take(4, random(min_segment_size, max_segment_size)));
  const segment_size_t max_k        = (cuda::std::min) (static_max_k, segment_size);

  // Skip invalid combinations
  if (segment_size > max_segment_size)
  {
    SKIP("The given segment size may not exceed the maximum segment size, we statically constrained the algorithm on.");
  }

  // Set the k value
  const segment_size_t k = GENERATE_COPY(values({segment_size_t{1}, max_k}), take(3, random(segment_size_t{1}, max_k)));

  // Generate number of segments
  const segment_index_t num_segments = GENERATE_COPY(
    values({segment_index_t{1}, segment_index_t{42}}), take(4, random(segment_index_t{1}, segment_index_t{1000})));

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

  // Prepare input & output
  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), keys_in_buffer);
  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  // Copy input for verification
  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Run the top-k algorithm. The segment-size argument form (immediate / un-annotated) is selected per TEST_TYPES so
  // the suite covers each form without any single variant compiling all of them.
  batched_topk_keys<direction>(
    d_keys_in,
    d_keys_out,
    make_segment_size_arg<fixed_seg_size_arg, segment_size_t{1}, max_segment_size>(segment_size),
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments});
  // Prepare expected results
  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);

  // Since the results of top-k are unordered, sort output segments before comparison.
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

#if TEST_TYPES == 0
// `constant` is the one public arg form whose segment size must be a compile-time constant expression, so it gets its
// own fixed-size test. Gated to a single TEST_TYPES variant to balance build time across the suite.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys work with a compile-time-constant segment size",
         "[keys][segmented][topk][device]",
         key_types,
         select_direction_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;
  using key_t           = c2h::get<0, TestType>;

  constexpr auto direction = c2h::get<1, TestType>::value;

  constexpr segment_size_t segment_size = 384;
  const segment_size_t k                = GENERATE_COPY(values({segment_size_t{1}, segment_size_t{17}, segment_size}));
  const segment_index_t num_segments    = GENERATE_COPY(
    values({segment_index_t{1}, segment_index_t{37}}), take(2, random(segment_index_t{1}, segment_index_t{500})));

  CAPTURE(c2h::type_name<key_t>(), segment_size, k, num_segments, direction);

  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);
  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  batched_topk_keys<direction>(
    d_keys_in,
    d_keys_out,
    cuda::args::constant<segment_size>{},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, segment_size>()},
    cuda::args::immediate{num_segments});

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

// A `k` larger than the segment is clamped to `segment_size` (every element selected), routing through the select-all
// fast path. The output then holds exactly `segment_size` items per segment, so we verify against the full sorted
// segment.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys clamp k larger than the segment size",
         "[keys][segmented][topk][device]",
         key_types,
         select_direction_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;
  using key_t           = c2h::get<0, TestType>;

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

  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  batched_topk_keys<direction>(
    d_keys_in,
    d_keys_out,
    cuda::args::constant<segment_size>{},
    cuda::args::immediate{k_requested, cuda::args::bounds<segment_size_t{1}, 10 * segment_size>()},
    cuda::args::immediate{num_segments});

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, effective_k); // clamped k == segment_size keeps everything
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, effective_k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

// Segment-size types narrower than the internal `offset_t`: a signed type narrows a too-large index to a negative one
// (indexing before the segment); an unsigned one wraps to a small in-range index (duplicate racing stores).
using narrow_seg_size_list = c2h::type_list<cuda::std::int8_t, cuda::std::uint8_t>;

// Regression for a narrow segment-size type. A block launches 512 threads -- past an 8-bit type's range -- so any path
// that narrows an index to `segment_size_val_t` before its bound check would go out of bounds. We sweep `segment_size`
// and `k` to hit every path (`k == segment_size` -> select-all copy; `k < segment_size` -> radix / histogram /
// output-ordering) and run over `det_tie_combos` to also cover the deterministic scan/atomics under the narrow type.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys handle a segment-size type narrower than the internal offset",
         "[keys][segmented][topk][device][cluster][determinism]",
         narrow_seg_size_list,
         det_tie_combos)
{
  using seg_size_t      = c2h::get<0, TestType>;
  using segment_index_t = cuda::std::int64_t;
  using key_t           = cuda::std::uint8_t; // key type is immaterial to this index-arithmetic regression

  using combo                = c2h::get<1, TestType>;
  constexpr auto determinism = combo::determinism;
  constexpr auto tie_break   = combo::tie_break;
  constexpr auto direction   = cub::detail::topk::select::max;

  // Only the cluster backend honors determinism; elsewhere the deterministic combinations just rerun the
  // nondeterministic path, so skip them (the base nondeterministic combo still runs).
  if constexpr (selected_backend != topk_backend::cluster
                && determinism != cuda::execution::determinism::__determinism_t::__not_guaranteed)
  {
    SKIP("Determinism requirements are only provided by the cluster backend");
  }

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

  // Output buffer with front/back canary padding: a signed-narrowing underrun lands in the front guard, an overrun in
  // the back guard. The in-range comparison below misses both (the underrun also stays inside the allocation, hiding it
  // from memcheck), so we assert the guards stay untouched. `guard` covers the worst-case 8-bit offset (<= 256).
  constexpr segment_index_t guard = 256;
  const key_t key_canary          = static_cast<key_t>(0x5A);
  c2h::device_vector<key_t> keys_out_storage(guard + num_segments * k + guard, key_canary);
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_storage.data()) + guard;
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), int{k});

  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  batched_topk_keys<direction, determinism, tie_break>(
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<seg_size_t{1}, seg_size_t{127}>()},
    cuda::args::immediate{k, cuda::args::bounds<seg_size_t{1}, seg_size_t{127}>()},
    cuda::args::immediate{num_segments});

  // Guards must be untouched by the algorithm.
  const c2h::device_vector<key_t> expected_guard(guard, key_canary);
  CHECK(c2h::device_vector<key_t>(keys_out_storage.begin(), keys_out_storage.begin() + guard) == expected_guard);
  CHECK(c2h::device_vector<key_t>(keys_out_storage.end() - guard, keys_out_storage.end()) == expected_guard);

  // Extract the in-range output for the standard top-k comparison.
  c2h::device_vector<key_t> keys_out_buffer(keys_out_storage.begin() + guard, keys_out_storage.end() - guard);

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}
#endif // TEST_TYPES == 0

#if TEST_TYPES == 2
// `deferred` wraps a device-accessible handle whose value is read in stream order on the device, so unlike the
// host-value forms it cannot be produced from a plain scalar; the uniform segment size lives in a 1-element device
// buffer. Gated to a single TEST_TYPES variant to balance build time across the suite.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys work with a deferred (device-resident) segment size",
         "[keys][segmented][topk][device]",
         key_types,
         select_direction_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;
  using key_t           = c2h::get<0, TestType>;

  constexpr auto direction = c2h::get<1, TestType>::value;

  constexpr segment_size_t segment_size     = 384;
  constexpr segment_size_t max_segment_size = 512;
  const segment_size_t k             = GENERATE_COPY(values({segment_size_t{1}, segment_size_t{17}, segment_size}));
  const segment_index_t num_segments = GENERATE_COPY(
    values({segment_index_t{1}, segment_index_t{37}}), take(2, random(segment_index_t{1}, segment_index_t{500})));

  CAPTURE(c2h::type_name<key_t>(), segment_size, k, num_segments, direction);

  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);
  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  // The uniform segment size is read by the device through a device-resident pointer; the static upper bound drives the
  // host-side launch sizing.
  c2h::device_vector<segment_size_t> d_segment_size(1, segment_size);
  auto d_segment_size_ptr = thrust::raw_pointer_cast(d_segment_size.data());

  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  batched_topk_keys<direction>(
    d_keys_in,
    d_keys_out,
    cuda::args::deferred{d_segment_size_ptr, cuda::args::bounds<segment_size_t{1}, max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, segment_size>()},
    cuda::args::immediate{num_segments});

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}
#endif // TEST_TYPES == 2

#if TEST_TYPES == 1
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys work with large fixed-size unaligned segments",
         "[keys][segmented][topk][device][cluster][determinism]",
         det_tie_combos)
{
  using key_t           = float;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  // Requirement under test. Random keys rarely tie at the boundary, so this mainly exercises the blocked-partition +
  // streaming/edge interaction; the key result is invariant to the requirement, so all combinations match the
  // reference.
  using combo                = c2h::get<0, TestType>;
  constexpr auto determinism = combo::determinism;
  constexpr auto tie_break   = combo::tie_break;

  // Only the cluster backend honors determinism; elsewhere the deterministic combinations just rerun the
  // nondeterministic path, so skip them (the base nondeterministic combo still runs).
  if constexpr (selected_backend != topk_backend::cluster
                && determinism != cuda::execution::determinism::__determinism_t::__not_guaranteed)
  {
    SKIP("Determinism requirements are only provided by the cluster backend");
  }

  // `static_max_segment_size` is chosen to exceed the largest all-resident cluster coverage (~16 blocks worth of
  // resident SMEM), so the 1 Mi-element segments force the agent's gmem-streaming overflow path (including an
  // unaligned overflow tail via `- 31`), while the 128 Ki-element segment still runs fully resident under the same
  // streaming-capable launch configuration. The `+ 1` / `- 4095` sizes make the global-last chunk a single item, i.e.
  // a pure-suffix tail with an empty aligned bulk (`bulk == 0`) once `pad == 0` aligns the base, exercising the
  // always-peeled tail edge on top of a zero-length resident/streamed tail chunk (resident `128 Ki + 1`, streamed
  // `1 Mi - 4095`).
  constexpr segment_size_t static_max_segment_size = 1024 * 1024;
  constexpr segment_size_t static_max_k            = 4 * 1024;
  constexpr segment_index_t num_segments           = 3;

  constexpr auto direction          = cub::detail::topk::select::max;
  const int pad                     = GENERATE(0, 1, 3, 7);
  const segment_size_t segment_size = GENERATE_COPY(values(
    {static_max_segment_size,
     static_max_segment_size - 31,
     static_max_segment_size - 4095,
     segment_size_t{128 * 1024},
     segment_size_t{128 * 1024 + 1}}));
  const segment_size_t max_k        = (cuda::std::min) (static_max_k, segment_size);
  const segment_size_t k            = GENERATE_COPY(values({segment_size_t{1}, max_k / 2, max_k}));

  CAPTURE(pad, static_max_segment_size, static_max_k, segment_size, k, num_segments, direction);

  c2h::device_vector<key_t> keys_in_buffer(pad + num_segments * segment_size, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);

  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data()) + pad;
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  c2h::device_vector<key_t> expected_keys(num_segments * segment_size, thrust::no_init);
  thrust::copy(keys_in_buffer.cbegin() + pad, keys_in_buffer.cend(), expected_keys.begin());

  batched_topk_keys<direction, determinism, tie_break>(
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments});

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);

  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

// Streaming counterpart of the narrow-segment-size regression. Streaming needs segments larger than the resident
// cluster coverage (>128 Ki), which 8/16-bit types can't represent, so we use signed 32-bit -- same width as the
// (unsigned) internal `offset_t` but signed -- to exercise the streaming path's index arithmetic across det/non-det.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys stream large segments with a signed 32-bit segment-size type",
         "[keys][segmented][topk][device][cluster][determinism]",
         det_tie_combos)
{
  using key_t           = float;
  using seg_size_t      = int; // signed 32-bit: same width as offset_t, but signed
  using segment_index_t = cuda::std::int64_t;

  using combo                = c2h::get<0, TestType>;
  constexpr auto determinism = combo::determinism;
  constexpr auto tie_break   = combo::tie_break;
  constexpr auto direction   = cub::detail::topk::select::max;

  if constexpr (selected_backend != topk_backend::cluster
                && determinism != cuda::execution::determinism::__determinism_t::__not_guaranteed)
  {
    SKIP("Determinism requirements are only provided by the cluster backend");
  }

  constexpr seg_size_t static_max_segment_size = 1024 * 1024;
  constexpr seg_size_t static_max_k            = 4 * 1024;
  constexpr segment_index_t num_segments       = 2;

  const int pad                 = GENERATE(0, 7);
  const seg_size_t segment_size = static_max_segment_size - 31; // unaligned -> forces streaming + unaligned tail edge
  const seg_size_t max_k        = (cuda::std::min) (static_max_k, segment_size);
  const seg_size_t k            = GENERATE_COPY(values({seg_size_t{1}, max_k}));

  CAPTURE(pad, static_max_segment_size, static_max_k, segment_size, k, num_segments);

  c2h::device_vector<key_t> keys_in_buffer(pad + num_segments * segment_size, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);

  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data()) + pad;
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  c2h::device_vector<key_t> expected_keys(num_segments * segment_size, thrust::no_init);
  thrust::copy(keys_in_buffer.cbegin() + pad, keys_in_buffer.cend(), expected_keys.begin());

  batched_topk_keys<direction, determinism, tie_break>(
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<seg_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<seg_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments});

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
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

// Yields, for each segment, a *non-contiguous* iterator over that segment's keys (an integral counting iterator
// cast to the key type). Feeding the cluster top-k a non-contiguous key iterator makes `use_block_load_to_shared`
// false, so the agent takes its generic (non-BlockLoadToShared) overflow-streaming path. Segment `seg` produces
// keys [seg * segment_size, (seg + 1) * segment_size), so the flattened input equals the identity sequence and the
// expected top-k is exact.
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

C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys stream large segments through a non-contiguous key iterator",
         "[keys][segmented][topk][device][cluster][determinism]",
         det_tie_combos)
{
  using key_t           = float;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  // Requirement under test (the key result is invariant to it; exercises the generic streaming path).
  using combo                = c2h::get<0, TestType>;
  constexpr auto determinism = combo::determinism;
  constexpr auto tie_break   = combo::tie_break;

  // Only the cluster backend honors determinism; elsewhere the deterministic combinations just rerun the
  // nondeterministic path, so skip them (the base nondeterministic combo still runs).
  if constexpr (selected_backend != topk_backend::cluster
                && determinism != cuda::execution::determinism::__determinism_t::__not_guaranteed)
  {
    SKIP("Determinism requirements are only provided by the cluster backend");
  }

  // The counting-iterator key source is non-contiguous, so the agent uses its generic overflow-streaming path rather
  // than BlockLoadToShared. `static_max_segment_size` exceeds the largest all-resident cluster coverage, so the 1 Mi
  // -element segments stream (incl. an unaligned `- 31` tail), while the 128 Ki-element segment validates the generic
  // resident path (no streaming) through the same code. Keeping the largest total below 2^24 makes every key an exact
  // float.
  constexpr segment_size_t static_max_segment_size = 1024 * 1024;
  constexpr segment_size_t static_max_k            = 4 * 1024;
  constexpr segment_index_t num_segments           = 3;

  constexpr auto direction = cub::detail::topk::select::max;
  const segment_size_t segment_size =
    GENERATE_COPY(values({static_max_segment_size, static_max_segment_size - 31, segment_size_t{128 * 1024}}));
  const segment_size_t max_k     = (cuda::std::min) (static_max_k, segment_size);
  const segment_size_t k         = GENERATE_COPY(values({segment_size_t{1}, max_k / 2, max_k}));
  const segment_size_t num_items = num_segments * segment_size;

  CAPTURE(static_max_segment_size, static_max_k, segment_size, k, num_segments, direction);

  // Non-contiguous input: segment `seg` is the counting iterator [seg * segment_size, (seg + 1) * segment_size).
  auto d_keys_in = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}), counting_segment_keys_op<key_t, segment_size_t>{segment_size});

  // Output is a real buffer (the output iterator stays contiguous; only the input drives the streaming path).
  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  batched_topk_keys<direction, determinism, tie_break>(
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments});

  // The flattened input is the identity sequence, so build the expected keys directly and reuse the standard
  // sort + compact verification.
  c2h::device_vector<key_t> expected_keys(num_items, thrust::no_init);
  thrust::sequence(expected_keys.begin(), expected_keys.end());

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);

  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}
#endif // TEST_TYPES == 1

C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys work with small variable-size segments",
         "[keys][segmented][topk][device]",
         key_types,
         max_segment_size_list,
         max_num_k_list,
         select_direction_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using key_t = c2h::get<0, TestType>;

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

  // Copy input for verification
  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Run the top-k algorithm
  batched_topk_keys<direction>(
    d_keys_in,
    d_keys_out,
    cuda::args::deferred_sequence{segment_size_it, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments});

  // Verify keys are returned correctly: sort each segment of the expected input, then compact the top-k
  segmented_sort_keys(expected_keys, num_segments, segment_offsets.cbegin(), segment_offsets.cbegin() + 1, direction);
  expected_keys = compact_to_topk_batched(expected_keys, segment_offsets, k);

  // Since the results of top-k are unordered, sort compacted output segments before comparison
  segmented_sort_keys(
    keys_out_buffer, num_segments, compacted_offsets.cbegin(), compacted_offsets.cbegin() + 1, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

#if TEST_TYPES == 1
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys work with large variable-size unaligned segments",
         "[keys][segmented][topk][device][cluster][determinism]",
         det_tie_combos)
{
  using key_t           = float;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  // Requirement under test (the key result is invariant to it; exercises mixed resident/streaming segments).
  using combo                = c2h::get<0, TestType>;
  constexpr auto determinism = combo::determinism;
  constexpr auto tie_break   = combo::tie_break;

  // Only the cluster backend honors determinism; elsewhere the deterministic combinations just rerun the
  // nondeterministic path, so skip them (the base nondeterministic combo still runs).
  if constexpr (selected_backend != topk_backend::cluster
                && determinism != cuda::execution::determinism::__determinism_t::__not_guaranteed)
  {
    SKIP("Determinism requirements are only provided by the cluster backend");
  }

  // `static_max_segment_size` exceeds the largest all-resident cluster coverage, so a single per-segment launch sizes a
  // wide (16-CTA) cluster and mixes every effective-width regime: streaming segments (the 1 Mi-element ones, one with
  // an unaligned `- 31` overflow tail), a fully-resident multi-CTA segment (96 Ki + 17), two *medium* segments that
  // exceed the single-CTA threshold but need only a few chunks (so surplus cluster CTAs go idle -- the runtime
  // effective-cluster-width path), and a tiny segment that collapses onto a single CTA (257 elements).
  constexpr segment_size_t static_max_segment_size = 1100 * 1024;
  constexpr segment_size_t static_max_k            = 4 * 1024;

  constexpr auto direction = cub::detail::topk::select::max;
  const int pad            = GENERATE(1, 3, 7);
  const segment_size_t k   = GENERATE_COPY(values({segment_size_t{1}, static_max_k / 2, static_max_k}));

  constexpr segment_size_t big_segment_size = 1024 * 1024;
  constexpr segment_size_t med_segment_a    = 12 * 1024 + 1; // a few chunks, unaligned tail -> most cluster CTAs idle
  constexpr segment_size_t med_segment_b    = 40 * 1024; // more chunks, still well below the 16-CTA launch width
  c2h::host_vector<segment_size_t> h_segment_offsets{
    0,
    big_segment_size,
    big_segment_size + (big_segment_size - 31),
    big_segment_size + (big_segment_size - 31) + (96 * 1024 + 17),
    big_segment_size + (big_segment_size - 31) + (96 * 1024 + 17) + 257,
    big_segment_size + (big_segment_size - 31) + (96 * 1024 + 17) + 257 + med_segment_a,
    big_segment_size + (big_segment_size - 31) + (96 * 1024 + 17) + 257 + med_segment_a + med_segment_b};
  c2h::device_vector<segment_size_t> segment_offsets = h_segment_offsets;
  const segment_index_t num_segments                 = static_cast<segment_index_t>(h_segment_offsets.size() - 1);
  const segment_size_t num_items                     = h_segment_offsets.back();

  auto segment_offsets_it = thrust::raw_pointer_cast(segment_offsets.data());
  auto segment_size_it    = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}), segment_size_op<segment_size_t*>{segment_offsets_it});

  CAPTURE(pad, static_max_segment_size, static_max_k, k, num_segments, num_items, direction);

  // Each output segment holds exactly min(k, segment_size[i]) items, tightly packed.
  auto compacted_output_sizes_it = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}),
    get_output_size_op{segment_offsets.cbegin(), cuda::constant_iterator(k), num_segments});
  c2h::device_vector<segment_size_t> compacted_offsets(num_segments + 1, thrust::no_init);
  thrust::exclusive_scan(
    compacted_output_sizes_it, compacted_output_sizes_it + num_segments + 1, compacted_offsets.begin());
  segment_size_t total_output_size = compacted_offsets.back();

  c2h::device_vector<key_t> keys_in_buffer(pad + num_items, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(total_output_size, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);

  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data()) + pad;
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_offsets.cbegin());
  auto d_keys_out =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(d_keys_out_ptr), compacted_offsets.cbegin());

  c2h::device_vector<key_t> expected_keys(num_items, thrust::no_init);
  thrust::copy(keys_in_buffer.cbegin() + pad, keys_in_buffer.cend(), expected_keys.begin());

  batched_topk_keys<direction, determinism, tie_break>(
    d_keys_in,
    d_keys_out,
    cuda::args::deferred_sequence{segment_size_it, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments});

  segmented_sort_keys(expected_keys, num_segments, segment_offsets.cbegin(), segment_offsets.cbegin() + 1, direction);
  expected_keys = compact_to_topk_batched(expected_keys, segment_offsets, k);

  segmented_sort_keys(
    keys_out_buffer, num_segments, compacted_offsets.cbegin(), compacted_offsets.cbegin() + 1, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

#endif // TEST_TYPES == 1

C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys work with fixed-size segments and per-segment k",
         "[keys][segmented][topk][device]",
         key_types,
         max_segment_size_list,
         max_num_k_list,
         select_direction_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using key_t = c2h::get<0, TestType>;

  // Statically constrained maximum segment size and k
  constexpr segment_size_t static_max_segment_size = c2h::get<1, TestType>::value;
  constexpr segment_size_t static_max_k            = c2h::get<2, TestType>::value;

  // Selection direction comes from the compile-time test axis.
  constexpr auto direction = c2h::get<3, TestType>::value;

  // Generate the (uniform) input segment size. Unlike the uniform-k tests, k still varies per segment below.
  constexpr segment_size_t min_segment_size = 1;
  constexpr auto max_segment_size           = static_max_segment_size;
  const segment_size_t segment_size = GENERATE_COPY(values({min_segment_size, segment_size_t{3}, max_segment_size}),
                                                    take(2, random(min_segment_size, max_segment_size)));

  // Skip invalid combinations
  if (segment_size > max_segment_size)
  {
    SKIP("The given segment size may not exceed the maximum segment size, we statically constrained the algorithm on.");
  }

  // Generate number of segments
  const segment_index_t num_segments = GENERATE_COPY(
    values({segment_index_t{1}, segment_index_t{42}}), take(2, random(segment_index_t{1}, segment_index_t{1000})));

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

  // Prepare input & output. Input segments are fixed-size (strided); output segments are compacted (variable).
  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(total_output_size, thrust::no_init);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), keys_in_buffer);
  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(d_keys_out_ptr), compacted_offsets.cbegin());

  // Copy input for verification
  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Run the top-k algorithm with a per-segment k passed as a deferred sequence
  batched_topk_keys<direction>(
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, max_segment_size>()},
    cuda::args::deferred_sequence{
      thrust::raw_pointer_cast(segment_k.data()), cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments});

  // Prepare expected results: sort each fixed-size input segment, then compact each to its per-segment top-k.
  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  expected_keys = compact_to_topk_batched(expected_keys, segment_offsets, segment_k.cbegin());

  // Since the results of top-k are unordered, sort compacted output segments before comparison.
  segmented_sort_keys(
    keys_out_buffer, num_segments, compacted_offsets.cbegin(), compacted_offsets.cbegin() + 1, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys work with variable-size segments and per-segment k",
         "[keys][segmented][topk][device]",
         key_types,
         max_segment_size_list,
         max_num_k_list,
         select_direction_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using key_t = c2h::get<0, TestType>;

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

  // Copy input for verification
  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Run the top-k algorithm with a per-segment k passed as a deferred sequence
  batched_topk_keys<direction>(
    d_keys_in,
    d_keys_out,
    cuda::args::deferred_sequence{segment_size_it, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::deferred_sequence{
      thrust::raw_pointer_cast(segment_k.data()), cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments});

  // Verify keys are returned correctly: sort each segment of the expected input, then compact the per-segment top-k
  segmented_sort_keys(expected_keys, num_segments, segment_offsets.cbegin(), segment_offsets.cbegin() + 1, direction);
  expected_keys = compact_to_topk_batched(expected_keys, segment_offsets, segment_k.cbegin());

  // Since the results of top-k are unordered, sort compacted output segments before comparison
  segmented_sort_keys(
    keys_out_buffer, num_segments, compacted_offsets.cbegin(), compacted_offsets.cbegin() + 1, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}
#if TEST_TYPES == 2
// Heavy-tie stress/regression: collapse the keys to only a handful of distinct values so the k-th key's bucket holds a
// large set of tied candidates. Exercises the cluster agent's candidate path and, on a deterministic requirement, the
// cross-CTA tie-break scan (cand_prefix + BlockScan ranks); the boundary value counts must be exact in either mode.
//
// The tie-break path is key-type-independent (the 8 small values twiddle trivially), so we fix the key type and build
// this only in the matching `TEST_TYPES` variant. Float keys and per-type behavior are covered by the tests above.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys handle heavy ties at the k-th boundary",
         "[keys][segmented][topk][device][cluster][determinism]",
         select_direction_list,
         det_tie_combos)
{
  using key_t              = cuda::std::uint64_t;
  constexpr auto direction = c2h::get<0, TestType>::value;

  // Requirement under test. The key multiset is invariant to the preference, so every combination matches the same
  // reference; tie-rich data makes the deterministic scan do real work, so a boundary miscount would surface here.
  using combo                = c2h::get<1, TestType>;
  constexpr auto determinism = combo::determinism;
  constexpr auto tie_break   = combo::tie_break;

  // Only the cluster backend honors determinism; elsewhere the deterministic combinations just rerun the
  // nondeterministic path, so skip them (the base nondeterministic combo still runs).
  if constexpr (selected_backend != topk_backend::cluster
                && determinism != cuda::execution::determinism::__determinism_t::__not_guaranteed)
  {
    SKIP("Determinism requirements are only provided by the cluster backend");
  }

  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  constexpr segment_size_t static_max_segment_size = 64 * 1024;
  constexpr segment_size_t static_max_k            = 64 * 1024;
  constexpr segment_index_t num_segments           = 3;

  const segment_size_t segment_size =
    GENERATE_COPY(values({segment_size_t{257}, segment_size_t{4096}, segment_size_t{64 * 1024}}));
  const segment_size_t max_k     = (cuda::std::min) (static_max_k, segment_size);
  const segment_size_t k         = GENERATE_COPY(values({segment_size_t{1}, max_k / 2, max_k}));
  const segment_size_t num_items = num_segments * segment_size;

  CAPTURE(c2h::type_name<key_t>(), static_max_segment_size, static_max_k, segment_size, k, num_segments, direction);

  // Deterministic, duplicate-heavy input (8 distinct values), materialized into a contiguous buffer so the cluster
  // agent takes its resident BlockLoadToShared path.
  c2h::device_vector<key_t> keys_in_buffer(num_items, thrust::no_init);
  thrust::tabulate(keys_in_buffer.begin(), keys_in_buffer.end(), heavy_tie_key_op<key_t>{});
  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);

  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  batched_topk_keys<direction, determinism, tie_break>(
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments});

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);

  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}
#endif // TEST_TYPES == 2

// Regression test: top-k must preserve -0.0f in the output (not normalize to +0.0f).
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys preserve -0.0f in output",
         "[keys][segmented][topk][device][float]",
         select_direction_list)
{
  constexpr cuda::std::int64_t segment_size                      = 8;
  constexpr cuda::std::int64_t num_segments                      = 1;
  [[maybe_unused]] constexpr cuda::std::int64_t max_segment_size = 64; // msvc warns, only used in nttp

  constexpr auto direction = c2h::get<0, TestType>::value;

  c2h::device_vector<float> d_keys_in =
    (direction == cub::detail::topk::select::min)
      ? c2h::device_vector<float>{3.0f, -0.0f, 1.0f, 2.0f, 0.0f, -1.0f, 4.0f, 5.0f}
      : c2h::device_vector<float>{-2.0f, -0.0f, -3.0f, 0.0f, -1.0f, -4.0f, -5.0f, -6.0f};
  const cuda::std::int64_t k = (direction == cub::detail::topk::select::min) ? 5 : 3;

  c2h::device_vector<float> d_keys_out(k, thrust::no_init);

  auto d_keys_in_it =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(d_keys_in.data())), segment_size);
  auto d_keys_out_it =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(d_keys_out.data())), k);

  batched_topk_keys<direction>(
    d_keys_in_it,
    d_keys_out_it,
    cuda::args::immediate{segment_size, cuda::args::bounds<cuda::std::int64_t{1}, max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<cuda::std::int64_t{1}, k>()},
    cuda::args::immediate{num_segments});

  const int num_minus_zero = static_cast<int>(thrust::count_if(d_keys_out.begin(), d_keys_out.end(), is_minus_zero{}));
  REQUIRE(num_minus_zero >= 1);
}

// Users may pass `k` and `num_segments` un-annotated. A plain integral value is taken as a uniform immediate with no
// compile-time bound.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys accept unwrapped (plain integral) k and num_segments",
         "[keys][segmented][topk][device]",
         key_types,
         max_segment_size_list,
         max_num_k_list,
         select_direction_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using key_t = c2h::get<0, TestType>;

  constexpr segment_size_t static_max_segment_size = c2h::get<1, TestType>::value;
  constexpr segment_size_t static_max_k            = c2h::get<2, TestType>::value;
  constexpr auto direction                         = c2h::get<3, TestType>::value;

  // Fixed sizes: this test exercises the argument form, not the size matrix.
  const segment_size_t segment_size  = (cuda::std::min) (segment_size_t{256}, static_max_segment_size);
  const segment_size_t k             = (cuda::std::min) (static_max_k, segment_size);
  const segment_index_t num_segments = 42;

  CAPTURE(c2h::type_name<key_t>(), static_max_segment_size, static_max_k, segment_size, k, num_segments, direction);

  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);
  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // `k` and `num_segments` are passed as plain integral values (un-annotated immediate).
  batched_topk_keys<direction>(
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    k,
    num_segments);

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}
