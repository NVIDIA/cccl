// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Defer the unsupported-architecture diagnosis to the dispatch's runtime check (not a compile-time static_assert)
// so this test compiles across all target architectures, including pre-SM90, for the full configuration space. See
// CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT in cub/device/device_batched_topk.cuh. Precedes CUB includes.
#define CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_batched_topk.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh> // topk_policy / make_{baseline,cluster}_policy (cluster-cap tuning test)
#include <cub/util_type.cuh>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/__execution/tune.h>
#include <cuda/iterator>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__cmath/signbit.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>

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

// All tests drive the algorithm through the public `cub::DeviceBatchedTopK` API, threading the requested
// determinism/tie-break into the environment via `require`. The dispatch selects the backend from the architecture and
// the statically-known maximum segment size (a deterministic request routes to the cluster backend on SM90+).
template <cub::detail::topk::select SelectDirection,
          cuda::execution::determinism::__determinism_t Determinism =
            cuda::execution::determinism::__determinism_t::__not_guaranteed,
          cuda::execution::tie_break::__tie_break_t TieBreak = cuda::execution::tie_break::__tie_break_t::__unspecified,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename SegmentSizeParamT,
          typename KParamT,
          typename NumSegmentsParameterT>
_CCCL_HOST_API static cudaError_t dispatch_batched_topk_keys(
  void* d_temp_storage,
  cuda::std::size_t& temp_storage_bytes,
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  SegmentSizeParamT segment_sizes,
  KParamT k,
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
    return cub::DeviceBatchedTopK::MaxKeys(
      d_temp_storage, temp_storage_bytes, d_key_segments_it, d_key_segments_out_it, segment_sizes, k, num_segments, env);
  }
  else
  {
    return cub::DeviceBatchedTopK::MinKeys(
      d_temp_storage, temp_storage_bytes, d_key_segments_it, d_key_segments_out_it, segment_sizes, k, num_segments, env);
  }
}

// %PARAM% TEST_LAUNCH lid 0:2
DECLARE_TMPL_LAUNCH_WRAPPER(
  dispatch_batched_topk_keys,
  batched_topk_keys,
  ESCAPE_LIST(
    cub::detail::topk::select SelectDirection,
    cuda::execution::determinism::__determinism_t Determinism =
      cuda::execution::determinism::__determinism_t::__not_guaranteed,
    cuda::execution::tie_break::__tie_break_t TieBreak = cuda::execution::tie_break::__tie_break_t::__unspecified),
  ESCAPE_LIST(SelectDirection, Determinism, TieBreak));

// Wrapper-test companion to expect_batched_topk_unsupported_and_skip: when the request's backend is unavailable in this
// build, dispatch it directly (host), verify the runtime cudaErrorNotSupported, and skip the correctness checks;
// otherwise return so the caller runs its normal batched_topk_keys<...> launch + checks. Pass the same trailing
// arguments (and Direction / Determinism / TieBreak) as that launch.
template <cub::detail::topk::select Direction,
          cuda::execution::determinism::__determinism_t Determinism =
            cuda::execution::determinism::__determinism_t::__not_guaranteed,
          cuda::execution::tie_break::__tie_break_t TieBreak = cuda::execution::tie_break::__tie_break_t::__unspecified,
          typename... Args>
void skip_unless_batched_topk_keys_supported(cuda::std::int64_t static_max_segment_size, Args... args)
{
  if (batched_topk_backend_unavailable<Determinism, TieBreak>(static_max_segment_size))
  {
    expect_batched_topk_unsupported_and_skip([&](void* d_temp_storage, cuda::std::size_t& temp_storage_bytes) {
      return dispatch_batched_topk_keys<Direction, Determinism, TieBreak>(d_temp_storage, temp_storage_bytes, args...);
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

// Selection direction is a compile-time option; cover both as a static test axis.
using select_direction_list =
  c2h::enum_type_list<cub::detail::topk::select, cub::detail::topk::select::min, cub::detail::topk::select::max>;

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

  batched_topk_keys<direction>(
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, max_segment_size>()},
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

  // Baseline-coverable segment sizes (bounded to 127); only a deterministic / tie-break requirement routes to the SM90+
  // cluster backend.
  skip_unless_batched_topk_keys_supported<direction, determinism, tie_break>(
    /*static_max_segment_size=*/127,
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<seg_size_t{1}, seg_size_t{127}>()},
    cuda::args::immediate{k, cuda::args::bounds<seg_size_t{1}, seg_size_t{127}>()},
    cuda::args::immediate{num_segments});
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

// Automatic size-based backend crossover. The segment size is at the cluster crossover threshold
// (`cluster_beneficial_min_segment_size` == 8 Ki) yet still baseline-coverable (baseline top tile is 256*64 == 16 Ki),
// so on SM 10.0+ the automatic selector prefers the cluster backend purely on size, under a *non-deterministic* request
// -- a branch no other test hits (deterministic tests force cluster via the requirement, oversize tests via
// non-coverage). Below SM 10.0 the crossover does not apply, so the case is skipped. Checked vs a sorted reference.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys route a large baseline-coverable segment to the cluster backend on SM10+",
         "[keys][segmented][topk][device][cluster]",
         select_direction_list)
{
  // Gate on the compiled/JIT compute capability the dispatch resolves (cub::PtxVersion), not the physical device
  // (cub::SmVersion): the size-based crossover fires from the selector evaluated at the resolved target, so a build
  // whose highest applicable target is below SM 10.0 stays on the baseline backend even on an SM 10.0+ device.
  int ptx_version = 0;
  REQUIRE(cudaSuccess == cub::PtxVersion(ptx_version));
  constexpr int cluster_beneficial_min_ptx_version = 1000; // SM 10.0
  if (ptx_version < cluster_beneficial_min_ptx_version)
  {
    SKIP(
      "The size-based baseline->cluster crossover only applies on SM 10.0+; below it the automatic selector stays on "
      "the baseline backend (already covered by the small-segment tests).");
  }

  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;
  using key_t           = cuda::std::uint8_t; // small key so the baseline backend can cover the 8 Ki segment

  constexpr auto direction = c2h::get<0, TestType>::value;

  constexpr segment_size_t static_max_segment_size = 8 * 1024; // == cluster_beneficial_min_segment_size
  const segment_size_t segment_size                = static_max_segment_size;
  const segment_size_t k = GENERATE_COPY(values({segment_size_t{1}, segment_size_t{257}, static_max_segment_size}));
  const segment_index_t num_segments = GENERATE_COPY(values({segment_index_t{1}, segment_index_t{5}}));

  CAPTURE(static_max_segment_size, segment_size, k, num_segments, direction);

  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);
  auto d_keys_in_ptr = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_in     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);

  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Defaulted determinism/tie-break (not_guaranteed / unspecified): the statically-known maximum segment size, not the
  // requirement, is what routes this to the cluster backend.
  batched_topk_keys<direction>(
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{num_segments});

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

  constexpr segment_size_t segment_size                      = 384;
  [[maybe_unused]] constexpr segment_size_t max_segment_size = 512; // msvc warns, only used in nttp
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
template <typename KeyT>
struct cast_to_key_op
{
  template <typename T>
  __host__ __device__ KeyT operator()(T x) const
  {
    return static_cast<KeyT>(x);
  }
};

// Yields, for each segment, a *non-contiguous* iterator over that segment's keys (an integral counting iterator cast to
// the key type). Feeding the cluster top-k a non-contiguous key iterator makes `use_block_load_to_shared` false, so the
// agent takes its generic (non-BlockLoadToShared) overflow-streaming path. Segment `seg` produces keys
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
#endif // TEST_TYPES == 1

// The cluster-path coverage below pins the launch geometry through `cluster_tuning_selector` (a whole-`topk_policy`
// tune override), which is a direct-API construct -- so this whole block is `TEST_LAUNCH == 0` (built once) rather than
// going through the launch-wrapper macro. These tests exercise agent-internal cluster paths (multi-CTA scan / cluster
// barriers / gmem streaming / stage schedule / idle ranks) that are identical across launch modes, so host launch is
// sufficient; they run at a racecheck-tiny footprint instead of the ~1 Mi segments these paths used to require (small
// segments otherwise collapse to the single-CTA fast path). The chunk stride is shrunk to 512 B (128 floats), so every
// segment still spans several chunks per block rather than degenerating to one. On a device with a very small
// shared-memory budget the real resident capacity could dip below the pinned slot cap, but that only forces *more*
// streaming, so the paths under test stay covered.
#if TEST_TYPES == 1 && TEST_LAUNCH == 0
// Runs the direct-API cluster top-k twice (temp-size query, then the real call) and syncs, requiring success at each
// step. `Direction` selects Min/Max at compile time. Factored out because the tune-override tests here cannot use the
// launch-wrapper macro (it owns the env) and would otherwise repeat this boilerplate. The tune override forces the
// SM90+ cluster backend, so where no SM90+ target can serve it this verifies the runtime cudaErrorNotSupported and
// skips the correctness checks instead.
template <cub::detail::topk::select Direction,
          typename KeyInItT,
          typename KeyOutItT,
          typename SegSizesT,
          typename KParamT,
          typename NumSegT,
          typename EnvT>
void run_cluster_topk_keys(
  KeyInItT d_keys_in, KeyOutItT d_keys_out, SegSizesT seg_sizes, KParamT k_param, NumSegT num_seg, EnvT env)
{
  const auto dispatch = [&](void* d_temp, cuda::std::size_t& temp_bytes) {
    if constexpr (Direction == cub::detail::topk::select::max)
    {
      return cub::DeviceBatchedTopK::MaxKeys(
        d_temp, temp_bytes, d_keys_in, d_keys_out, seg_sizes, k_param, num_seg, env);
    }
    else
    {
      return cub::DeviceBatchedTopK::MinKeys(
        d_temp, temp_bytes, d_keys_in, d_keys_out, seg_sizes, k_param, num_seg, env);
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

// Builds the require/tune env pinning `Selector` as the whole-policy tune override.
template <cuda::execution::determinism::__determinism_t Determinism,
          cuda::execution::tie_break::__tie_break_t TieBreak,
          typename Selector>
auto make_cluster_tune_env(Selector selector)
{
  return cuda::std::execution::env{
    cuda::stream_ref{cudaStream_t{0}},
    cuda::execution::require(cuda::execution::determinism::__determinism_holder_t<Determinism>{},
                             cuda::execution::tie_break::__tie_break_holder_t<TieBreak>{},
                             cuda::execution::output_ordering::unsorted),
    cuda::execution::tune(selector)};
}

// A small multi-CTA segment that stays *fully resident* across a forced 2-CTA cluster: `single_block_max_seg_size = 0`
// disables the single-CTA fast path so even this tiny segment fans out, and cap 2 pins the width. This exercises the
// real cross-CTA prefix scan (`prime_placement_counters` / remote `red.add`), the cluster barriers, and the DSMEM
// histogram fold -- machinery that previously only ran on 64 Ki+ segments too large for `compute-sanitizer racecheck`.
// Non-deterministic here (striped load + early-stop driver, the path unique to keys); the deterministic driver's use of
// the same scan is verified against an index reference by the pairs "deterministic cross-CTA scan" test. No overflow
// (verifies the scan in isolation).
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys run a small multi-CTA segment through the cross-CTA scan",
         "[keys][segmented][topk][device][cluster]")
{
  using key_t           = float;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  constexpr auto determinism = cuda::execution::determinism::__determinism_t::__not_guaranteed;
  constexpr auto tie_break   = cuda::execution::tie_break::__tie_break_t::__unspecified;
  constexpr auto direction   = cub::detail::topk::select::max;

  // 2048 floats = 16 chunks; a 2-CTA cluster holds 8 chunks each -> fully resident, no streaming.
  constexpr segment_size_t static_max_segment_size = 2048;
  constexpr segment_size_t static_max_k            = 512;
  constexpr segment_index_t num_segments           = 2;
  constexpr segment_size_t segment_size            = static_max_segment_size;
  const segment_size_t k                           = GENERATE_COPY(values({segment_size_t{1}, static_max_k}));

  CAPTURE(segment_size, k, num_segments);

  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);

  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Force a 2-CTA cluster with the single-CTA fast path disabled; slots stay unrestricted (the segment is resident).
  auto env =
    make_cluster_tune_env<determinism, tie_break>(cluster_tuning_selector<2, 0, 0, cluster_test_chunk_bytes>{});
  run_cluster_topk_keys<direction>(
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments},
    env);

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

// Cluster-width cap (`cluster_cap_list`) plus a resident-slot cap force a tiny segment to overflow and stream: cap 1
// streams inside one CTA (barrier-free), cap 2 across a fixed 2-CTA cluster (cross-CTA scan while streaming). The
// resident capacity is `slots * chunk_items` (host-known), so 4 slots x 128 floats = 512 resident floats and the ~1.5 K
// segment spills several overflow chunks. Both `stream_stages` and `prologue` are `min(PipelineStages, .)` (of the
// per-CTA overflow count and the 4 resident chunks respectively), so sweeping `PipelineStages` {2,4,8} across the two
// caps (cap 1 -> ~8 overflow chunks/CTA, cap 2 -> ~2) spans the `stream_stages` <, ==, > `prologue` trichotomy and
// reaches the `stage_base` up-front prime (the `>` case). The 4-slot cap leaves an aligned resident tail here; the
// complementary misaligned-tail `stage_rot` reorder is covered by the dedicated test below (reached through the slot
// cap, not a distinct `PipelineStages`). An unaligned size (`- 31`) + `pad` also covers the peeled overflow tail edge.
// Non-deterministic (forward first wave only); the pairs schedule test streams the deterministic combos, which
// additionally reach the reverse first wave. Verified against a sorted reference.
using cluster_cap_list = c2h::enum_type_list<int, 1, 2>;
using stage_list       = c2h::enum_type_list<int, 2, 4, 8>;

C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys stream a tiny oversize segment across the pipeline-stage schedule",
         "[keys][segmented][topk][device][cluster]",
         cluster_cap_list,
         stage_list)
{
  using key_t           = float;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  constexpr int cluster_cap  = c2h::get<0, TestType>::value;
  constexpr int stages       = c2h::get<1, TestType>::value;
  constexpr auto determinism = cuda::execution::determinism::__determinism_t::__not_guaranteed;
  constexpr auto tie_break   = cuda::execution::tie_break::__tie_break_t::__unspecified;
  constexpr auto direction   = cub::detail::topk::select::max;

  constexpr segment_size_t static_max_segment_size = 1536;
  constexpr segment_size_t static_max_k            = 512;
  constexpr segment_index_t num_segments           = 2;

  const int pad = GENERATE(0, 7);
  const segment_size_t segment_size =
    GENERATE_COPY(values({static_max_segment_size, static_max_segment_size - 31})); // unaligned -> peeled overflow tail
                                                                                    // edge
  const segment_size_t max_k = (cuda::std::min) (static_max_k, segment_size);
  const segment_size_t k     = GENERATE_COPY(values({segment_size_t{1}, max_k}));

  CAPTURE(cluster_cap, stages, pad, segment_size, k, num_segments);

  c2h::device_vector<key_t> keys_in_buffer(pad + num_segments * segment_size, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);

  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data()) + pad;
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  c2h::device_vector<key_t> expected_keys(num_segments * segment_size, thrust::no_init);
  thrust::copy(keys_in_buffer.cbegin() + pad, keys_in_buffer.cend(), expected_keys.begin());

  // cap + 4 resident slots -> a tiny segment overflows and streams; `stages` varies the pipeline depth.
  auto env = make_cluster_tune_env<determinism, tie_break>(
    cluster_tuning_selector<cluster_cap, /*slots=*/4, /*single_block=*/0, cluster_test_chunk_bytes, stages>{});
  run_cluster_topk_keys<direction>(
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments},
    env);

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

// Misaligned-tail stage rotation (`stage_rot`): the interleaved overflow prime rotates the resident mbarrier-stage
// assignment when the resident chunk count is not a multiple of the pipeline depth. Reuses the schedule sweep's exact
// cap-1 / stages-2 kernel (same `cluster_tuning_selector<1, 4, 0, .., 2>` type and same size/k bounds -> no new
// instantiation); only the *runtime* segment size differs -- 5 chunks here vs 12 there. With 4 slots and one chunk of
// excess over them, 1 stream slot is reserved, leaving 3 resident chunks, and `3 % prologue(2) == 1` misaligns the tail
// (whereas the sweep's 12-chunk segment leaves an aligned 2-chunk tail). Single-CTA (cap 1) streaming,
// non-deterministic (forward first wave); the pairs stage-rotation test streams the deterministic combos, which
// additionally rotate the reverse first wave. Verified against a sorted reference.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys stream a tiny oversize segment with a misaligned resident tail",
         "[keys][segmented][topk][device][cluster]")
{
  using key_t           = float;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  constexpr auto determinism = cuda::execution::determinism::__determinism_t::__not_guaranteed;
  constexpr auto tie_break   = cuda::execution::tie_break::__tie_break_t::__unspecified;
  constexpr auto direction   = cub::detail::topk::select::max;

  // Bounds match the schedule sweep (same kernel); the runtime segment is sized to 5 chunks (640 floats) so, at 4 slots
  // minus 1 reserved stream slot, only 3 resident chunks remain -> a misaligned reload tail (`3 % prologue(2) == 1`).
  constexpr segment_size_t static_max_segment_size = 1536;
  constexpr segment_size_t static_max_k            = 512;
  constexpr segment_index_t num_segments           = 2;
  constexpr segment_size_t segment_size            = 640;
  const segment_size_t k                           = GENERATE_COPY(values({segment_size_t{1}, static_max_k}));

  CAPTURE(static_max_segment_size, segment_size, k, num_segments);

  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in_buffer);

  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Same selector as the schedule sweep's cap-1 / stages-2 case; the 5-chunk runtime segment overflows 4 slots and
  // leaves 3 resident chunks -> a misaligned reload tail that triggers `stage_rot`.
  auto env = make_cluster_tune_env<determinism, tie_break>(
    cluster_tuning_selector<1, /*slots=*/4, /*single_block=*/0, cluster_test_chunk_bytes, /*stages=*/2>{});
  run_cluster_topk_keys<direction>(
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments},
    env);

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

// Generic (non-BlockLoadToShared) overflow streaming at a tiny footprint: the non-contiguous counting-iterator key
// source makes `use_block_load_to_shared` false, and the slot cap forces a ~1.5 K segment to overflow, so the agent
// takes its generic gmem-streaming path. Non-deterministic (the load path is determinism-independent; the deterministic
// generic streamer is covered by the pairs generic index-order test). Verified against the identity sequence.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys stream a tiny oversize segment through a non-contiguous key iterator",
         "[keys][segmented][topk][device][cluster]")
{
  using key_t           = float;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  constexpr auto determinism = cuda::execution::determinism::__determinism_t::__not_guaranteed;
  constexpr auto tie_break   = cuda::execution::tie_break::__tie_break_t::__unspecified;
  constexpr auto direction   = cub::detail::topk::select::max;

  constexpr segment_size_t static_max_segment_size = 1536;
  constexpr segment_size_t static_max_k            = 512;
  constexpr segment_index_t num_segments           = 2;
  constexpr segment_size_t segment_size            = static_max_segment_size;
  const segment_size_t k                           = GENERATE_COPY(values({segment_size_t{1}, static_max_k}));
  const segment_size_t num_items                   = num_segments * segment_size;

  CAPTURE(segment_size, k, num_segments);

  // Non-contiguous input: segment `seg` is the counting iterator [seg * segment_size, (seg + 1) * segment_size).
  auto d_keys_in = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}), counting_segment_keys_op<key_t, segment_size_t>{segment_size});

  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  auto env = make_cluster_tune_env<determinism, tie_break>(
    cluster_tuning_selector<1, /*slots=*/4, /*single_block=*/0, cluster_test_chunk_bytes>{});
  run_cluster_topk_keys<direction>(
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments},
    env);

  // The flattened input is the identity sequence, so build the expected keys directly and reuse the standard
  // sort + compact verification.
  c2h::device_vector<key_t> expected_keys(num_items, thrust::no_init);
  thrust::sequence(expected_keys.begin(), expected_keys.end());

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

#endif // TEST_TYPES == 1 && TEST_LAUNCH == 0

// Big-segment general coverage (launch wrapper -> all launch modes, no tune override). The tiny tune-override tests
// above pin geometry to reach the multi-CTA / streaming / schedule / idle paths at a racecheck-tiny footprint; these
// keep real 1 Mi-scale coverage so the index/offset arithmetic (and the streaming path under device/graph launch) is
// exercised at scale. Their sweeps are trimmed (fewer sizes/k/pads) since the fine-grained path coverage now lives in
// the tiny tests.
#if TEST_TYPES == 1
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys work with large fixed-size unaligned segments",
         "[keys][segmented][topk][device][cluster][determinism]",
         det_tie_combos)
{
  using key_t           = float;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using combo                = c2h::get<0, TestType>;
  constexpr auto determinism = combo::determinism;
  constexpr auto tie_break   = combo::tie_break;
  // 1 Mi segments overflow the resident cluster and stream (one unaligned `- 31` tail; `- 4095` makes the global-last
  // chunk a single item -> pure-suffix tail with empty aligned bulk once `pad == 0`). The requirement is swept because
  // the deterministic (blocked-load) path has distinct large-offset arithmetic.
  constexpr segment_size_t static_max_segment_size = 1024 * 1024;
  constexpr segment_size_t static_max_k            = 4 * 1024;
  constexpr segment_index_t num_segments           = 3;

  constexpr auto direction = cub::detail::topk::select::max;
  const int pad            = GENERATE(0, 7);
  const segment_size_t segment_size =
    GENERATE_COPY(values({static_max_segment_size, static_max_segment_size - 31, static_max_segment_size - 4095}));
  const segment_size_t max_k = (cuda::std::min) (static_max_k, segment_size);
  const segment_size_t k     = GENERATE_COPY(values({segment_size_t{1}, max_k}));

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

  const auto seg_arg =
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()};
  const auto k_arg  = cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()};
  const auto ns_arg = cuda::args::immediate{num_segments};
  skip_unless_batched_topk_keys_supported<direction, determinism, tie_break>(
    static_max_segment_size, d_keys_in, d_keys_out, seg_arg, k_arg, ns_arg);
  batched_topk_keys<direction, determinism, tie_break>(d_keys_in, d_keys_out, seg_arg, k_arg, ns_arg);

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

// Big streaming with a signed 32-bit segment-size type (same width as the internal `offset_t`, but signed): the point
// is that the large-offset arithmetic stays correct on a signed type at real 1 Mi scale.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys stream large segments with a signed 32-bit segment-size type",
         "[keys][segmented][topk][device][cluster][determinism]",
         det_tie_combos)
{
  using key_t           = float;
  using seg_size_t      = int; // signed 32-bit
  using segment_index_t = cuda::std::int64_t;

  using combo                = c2h::get<0, TestType>;
  constexpr auto determinism = combo::determinism;
  constexpr auto tie_break   = combo::tie_break;
  constexpr auto direction   = cub::detail::topk::select::max;

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

  const auto seg_arg =
    cuda::args::immediate{segment_size, cuda::args::bounds<seg_size_t{1}, static_max_segment_size>()};
  const auto k_arg  = cuda::args::immediate{k, cuda::args::bounds<seg_size_t{1}, static_max_k>()};
  const auto ns_arg = cuda::args::immediate{num_segments};
  skip_unless_batched_topk_keys_supported<direction, determinism, tie_break>(
    static_max_segment_size, d_keys_in, d_keys_out, seg_arg, k_arg, ns_arg);
  batched_topk_keys<direction, determinism, tie_break>(d_keys_in, d_keys_out, seg_arg, k_arg, ns_arg);

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

// Big variable-size batch: a loose 1.1 Mi bound sizes a wide cluster while the actual segments mix a streaming 1 Mi
// segment, a fully-resident multi-CTA segment, and small segments that leave surplus CTAs idle -- large-offset
// coverage of every effective-width regime in one launch (the tiny idle-rank test above covers the idle path
// racecheck-clean; this covers it at scale).
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys work with large variable-size unaligned segments",
         "[keys][segmented][topk][device][cluster][determinism]",
         det_tie_combos)
{
  using key_t           = float;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using combo                                      = c2h::get<0, TestType>;
  constexpr auto determinism                       = combo::determinism;
  constexpr auto tie_break                         = combo::tie_break;
  constexpr segment_size_t static_max_segment_size = 1100 * 1024;
  constexpr segment_size_t static_max_k            = 4 * 1024;

  constexpr auto direction = cub::detail::topk::select::max;
  const int pad            = GENERATE(1, 7);
  const segment_size_t k   = GENERATE_COPY(values({segment_size_t{1}, static_max_k}));

  constexpr segment_size_t big_segment_size = 1024 * 1024;
  c2h::host_vector<segment_size_t> h_segment_offsets{
    0,
    big_segment_size,
    big_segment_size + (big_segment_size - 31),
    big_segment_size + (big_segment_size - 31) + (96 * 1024 + 17),
    big_segment_size + (big_segment_size - 31) + (96 * 1024 + 17) + 257,
    big_segment_size + (big_segment_size - 31) + (96 * 1024 + 17) + 257 + (12 * 1024 + 1)};
  c2h::device_vector<segment_size_t> segment_offsets = h_segment_offsets;
  const segment_index_t num_segments                 = static_cast<segment_index_t>(h_segment_offsets.size() - 1);
  const segment_size_t num_items                     = h_segment_offsets.back();

  auto segment_offsets_it = thrust::raw_pointer_cast(segment_offsets.data());
  auto segment_size_it    = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}), segment_size_op<segment_size_t*>{segment_offsets_it});

  CAPTURE(pad, static_max_segment_size, static_max_k, k, num_segments, num_items, direction);

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

  const auto seg_arg =
    cuda::args::deferred_sequence{segment_size_it, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()};
  const auto k_arg  = cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()};
  const auto ns_arg = cuda::args::immediate{num_segments};
  skip_unless_batched_topk_keys_supported<direction, determinism, tie_break>(
    static_max_segment_size, d_keys_in, d_keys_out, seg_arg, k_arg, ns_arg);
  batched_topk_keys<direction, determinism, tie_break>(d_keys_in, d_keys_out, seg_arg, k_arg, ns_arg);

  segmented_sort_keys(expected_keys, num_segments, segment_offsets.cbegin(), segment_offsets.cbegin() + 1, direction);
  expected_keys = compact_to_topk_batched(expected_keys, segment_offsets, k);

  segmented_sort_keys(
    keys_out_buffer, num_segments, compacted_offsets.cbegin(), compacted_offsets.cbegin() + 1, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

// Big generic (non-BlockLoadToShared) streaming under default tuning: the non-contiguous counting-iterator key source
// makes `use_block_load_to_shared` false, so the dispatch routes these 1 Mi segments to the cluster backend's generic
// overflow-streaming path (vs. the tiny forced-cluster generic test above, this exercises the real default routing at
// large-offset scale). The 1 Mi - 31 segment streams with an unaligned tail edge; the 128 Ki segment validates the
// generic resident path (no streaming) through the same code. Keeping totals below 2^24 makes every key an exact float.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys stream large segments through a non-contiguous key iterator",
         "[keys][segmented][topk][device][cluster][determinism]",
         det_tie_combos)
{
  using key_t           = float;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using combo                = c2h::get<0, TestType>;
  constexpr auto determinism = combo::determinism;
  constexpr auto tie_break   = combo::tie_break;
  constexpr auto direction   = cub::detail::topk::select::max;

  constexpr segment_size_t static_max_segment_size = 1024 * 1024;
  constexpr segment_size_t static_max_k            = 4 * 1024;
  constexpr segment_index_t num_segments           = 2;

  const segment_size_t segment_size = GENERATE_COPY(values({static_max_segment_size - 31, segment_size_t{128 * 1024}}));
  const segment_size_t max_k        = (cuda::std::min) (static_max_k, segment_size);
  const segment_size_t k            = GENERATE_COPY(values({segment_size_t{1}, max_k}));
  const segment_size_t num_items    = num_segments * segment_size;

  CAPTURE(static_max_segment_size, static_max_k, segment_size, k, num_segments, direction);

  auto d_keys_in = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}), counting_segment_keys_op<key_t, segment_size_t>{segment_size});

  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  const auto seg_arg =
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()};
  const auto k_arg  = cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()};
  const auto ns_arg = cuda::args::immediate{num_segments};
  skip_unless_batched_topk_keys_supported<direction, determinism, tie_break>(
    static_max_segment_size, d_keys_in, d_keys_out, seg_arg, k_arg, ns_arg);
  batched_topk_keys<direction, determinism, tie_break>(d_keys_in, d_keys_out, seg_arg, k_arg, ns_arg);

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

#if TEST_TYPES == 1 && TEST_LAUNCH == 0
// Small variable-size batch under a forced wide (4-CTA) cluster with the single-CTA fast path disabled. One per-batch
// launch sizes the cluster from the loose bound (4096 -> 4 CTAs), but each segment derives its own effective width from
// its actual chunk count, so the smaller segments leave surplus CTAs idle -- exercising the effective-cluster-width
// arithmetic and the `rank >= eff_cluster_blocks` idle early-out at a racecheck-tiny footprint (previously only
// reachable on a 64 Ki+ mixed batch). Non-deterministic (the effective-width arithmetic is determinism-independent; the
// deterministic idle path is covered by the pairs mixed-effective-width test). Verified against a sorted reference.
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys leave surplus cluster CTAs idle on small variable-size segments",
         "[keys][segmented][topk][device][cluster]")
{
  using key_t           = float;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  constexpr auto determinism = cuda::execution::determinism::__determinism_t::__not_guaranteed;
  constexpr auto tie_break   = cuda::execution::tie_break::__tie_break_t::__unspecified;
  constexpr auto direction   = cub::detail::topk::select::max;

  constexpr segment_size_t static_max_segment_size = 4096; // loose bound -> 4-CTA physical cluster (128 floats/chunk)
  constexpr segment_size_t static_max_k            = 512;
  const int pad                                    = GENERATE(0, 7);
  const segment_size_t k                           = GENERATE_COPY(values({segment_size_t{1}, static_max_k}));

  // Mixed effective widths under the fixed 4-CTA launch: 4096 -> full 4, 1024/512 -> 4, 256 -> 2 (2 idle), 128 -> 1
  // (3 idle).
  c2h::host_vector<segment_size_t> h_segment_offsets{0};
  for (const segment_size_t s :
       {segment_size_t{4096}, segment_size_t{128}, segment_size_t{512}, segment_size_t{256}, segment_size_t{1024}})
  {
    h_segment_offsets.push_back(h_segment_offsets.back() + s);
  }
  c2h::device_vector<segment_size_t> segment_offsets = h_segment_offsets;
  const segment_index_t num_segments                 = static_cast<segment_index_t>(h_segment_offsets.size() - 1);
  const segment_size_t num_items                     = h_segment_offsets.back();

  auto segment_offsets_it = thrust::raw_pointer_cast(segment_offsets.data());
  auto segment_size_it    = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}), segment_size_op<segment_size_t*>{segment_offsets_it});

  CAPTURE(pad, static_max_segment_size, static_max_k, k, num_segments, num_items);

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

  auto env = make_cluster_tune_env<determinism, tie_break>(
    cluster_tuning_selector<4, /*slots=*/0, /*single_block=*/0, cluster_test_chunk_bytes>{});
  run_cluster_topk_keys<direction>(
    d_keys_in,
    d_keys_out,
    cuda::args::deferred_sequence{segment_size_it, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments},
    env);

  segmented_sort_keys(expected_keys, num_segments, segment_offsets.cbegin(), segment_offsets.cbegin() + 1, direction);
  expected_keys = compact_to_topk_batched(expected_keys, segment_offsets, k);

  segmented_sort_keys(
    keys_out_buffer, num_segments, compacted_offsets.cbegin(), compacted_offsets.cbegin() + 1, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

#endif // TEST_TYPES == 1 && TEST_LAUNCH == 0

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
  using segment_size_t       = cuda::std::int64_t;
  using segment_index_t      = cuda::std::int64_t;

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

  // Oversize segments always route to the SM90+ cluster backend, regardless of the determinism / tie-break requirement.
  const auto seg_arg =
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()};
  const auto k_arg  = cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()};
  const auto ns_arg = cuda::args::immediate{num_segments};
  skip_unless_batched_topk_keys_supported<direction, determinism, tie_break>(
    static_max_segment_size, d_keys_in, d_keys_out, seg_arg, k_arg, ns_arg);
  batched_topk_keys<direction, determinism, tie_break>(d_keys_in, d_keys_out, seg_arg, k_arg, ns_arg);

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

// The following tests exercise public-API behavior that is independent of key type and launch mechanism, so they call
// the API directly (not through the templated launch wrapper) and are compiled once for the whole param matrix.
#if TEST_TYPES == 0 && TEST_LAUNCH == 0
// The temporary storage size requirement must not assume a particular base-pointer alignment (the public contract
// states that no special alignment is required). Over-allocate by one byte and offset the base pointer.
C2H_TEST("DeviceBatchedTopK::MaxKeys handles a misaligned temporary storage pointer", "[keys][segmented][topk][device]")
{
  constexpr int num_segments = 2;
  constexpr int segment_size = 8;
  constexpr int k            = 3;

  auto keys_in  = thrust::device_vector<int>{5, -3, 1, 7, 8, 2, 4, 6, /**/ 0, 9, 3, 2, 1, 8, 7, 4};
  auto keys_out = thrust::device_vector<int>(num_segments * k, thrust::no_init);

  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), segment_size);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);

  constexpr auto segment_sizes = cuda::args::constant<segment_size>{};
  constexpr auto k_arg         = cuda::args::constant<k>{};
  auto num_segs                = cuda::args::immediate{cuda::std::int64_t{num_segments}};
  auto env                     = cuda::std::execution::env{cuda::execution::require(
    cuda::execution::determinism::not_guaranteed,
    cuda::execution::tie_break::unspecified,
    cuda::execution::output_ordering::unsorted)};

  cuda::std::size_t temp_storage_bytes = 0;
  auto error                           = cub::DeviceBatchedTopK::MaxKeys(
    nullptr, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segs, env);
  REQUIRE(error == cudaSuccess);

  // Allocate one extra byte and offset the base pointer by one to misalign it.
  thrust::device_vector<char> temp_storage(temp_storage_bytes + 1, thrust::no_init);
  error = cub::DeviceBatchedTopK::MaxKeys(
    thrust::raw_pointer_cast(temp_storage.data()) + 1,
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    segment_sizes,
    k_arg,
    num_segs,
    env);
  REQUIRE(error == cudaSuccess);

  thrust::sort(keys_out.begin(), keys_out.begin() + k, cuda::std::greater<int>{});
  thrust::sort(keys_out.begin() + k, keys_out.begin() + 2 * k, cuda::std::greater<int>{});
  REQUIRE(keys_out == thrust::device_vector<int>{8, 7, 6, 9, 8, 7});
}

// The supported maximum segment size (2^21, about 2 million) is enforced at compile time from the statically-known
// maximum. An un-annotated segment size is accepted only when its element type's maximum already fits: a narrow type
// (e.g. uint16, [0, 65535]) qualifies with no cuda::args::bounds, while a type whose max exceeds 2^21 (int32, uint32,
// int64) needs one. Verify the un-annotated narrow path compiles and runs end-to-end.
C2H_TEST("DeviceBatchedTopK::MaxKeys accepts an un-annotated narrow-unsigned segment size",
         "[keys][segmented][topk][device]")
{
  constexpr int num_segments = 2;
  constexpr int segment_size = 8;
  constexpr int k            = 3;

  auto keys_in  = thrust::device_vector<int>{5, -3, 1, 7, 8, 2, 4, 6, /**/ 0, 9, 3, 2, 1, 8, 7, 4};
  auto keys_out = thrust::device_vector<int>(num_segments * k, thrust::no_init);

  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), segment_size);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);

  // Un-annotated narrow-unsigned segment size: a bare integral value (no cuda::args wrapper), taken as a uniform
  // immediate. With no wrapper/bounds the static maximum is the type maximum (uint16 -> 65535), which must itself fall
  // not exceed 2^21 to be accepted; the static_assert below pins that exposed bound. Any narrow-unsigned type
  // qualifies the same way (e.g. uint8, [0, 255]); uint16 is representative.
  auto segment_sizes = cuda::std::uint16_t{segment_size};
  static_assert(cuda::args::__traits<decltype(segment_sizes)>::highest == 65535,
                "expected the un-annotated uint16 segment size to expose its full type range as the static bound");
  [[maybe_unused]] constexpr auto k_arg = cuda::args::constant<k>{}; // gcc7 warns, only used in `dispatch` lambda
  auto num_segs                         = cuda::args::immediate{cuda::std::int64_t{num_segments}};
  auto env                              = cuda::std::execution::env{cuda::execution::require(
    cuda::execution::determinism::not_guaranteed,
    cuda::execution::tie_break::unspecified,
    cuda::execution::output_ordering::unsorted)};

  const auto dispatch = [&](void* d_temp_storage, cuda::std::size_t& temp_storage_bytes) {
    return cub::DeviceBatchedTopK::MaxKeys(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segs, env);
  };

  // The exposed static max (65535) exceeds the baseline backend's per-tile coverage, so this config routes to the SM90+
  // cluster backend (unavailable pre-SM90).
  if (batched_topk_backend_unavailable(
        /*static_max_segment_size=*/cuda::args::__traits<decltype(segment_sizes)>::highest))
  {
    expect_batched_topk_unsupported_and_skip(dispatch);
  }

  cuda::std::size_t temp_storage_bytes = 0;
  auto error                           = dispatch(nullptr, temp_storage_bytes);
  REQUIRE(error == cudaSuccess);

  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);
  error = dispatch(thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes);
  REQUIRE(error == cudaSuccess);

  thrust::sort(keys_out.begin(), keys_out.begin() + k, cuda::std::greater<int>{});
  thrust::sort(keys_out.begin() + k, keys_out.begin() + 2 * k, cuda::std::greater<int>{});
  REQUIRE(keys_out == thrust::device_vector<int>{8, 7, 6, 9, 8, 7});
}

// A negative statically-known lower bound (here an explicit `bounds<-8, ...>`, matching e.g. a bare `int16_t`) is
// accepted: the kernel clamps a negative runtime segment size up to 0, so that segment is treated as empty (skipped,
// no output) instead of indexing with a negative count. A non-negative lower bound is instead trusted (no clamp).
// Exercise the clamp with a mixed batch -- segment 0 declares a negative size, segment 1 a normal one. A small size
// and no determinism requirement route this to the baseline backend (the deterministic-requirement counterpart, which
// routes to the cluster backend, is below).
C2H_TEST("DeviceBatchedTopK::MaxKeys clamps a negative segment size to an empty segment (no determinism requirement)",
         "[keys][segmented][topk][device]")
{
  using seg_size_t           = cuda::std::int16_t; // negative-capable lower bound -> clamp path
  constexpr int num_segments = 2;
  constexpr int k            = 3;
  constexpr int stride       = 8;
  constexpr int sentinel     = -12345;

  // Segment 0: declared size -1 -> clamped to 0 -> skipped. Segment 1: 8 real keys, top-3 max = {9, 8, 7}.
  auto d_segment_sizes = thrust::device_vector<seg_size_t>{seg_size_t{-1}, seg_size_t{stride}};
  auto keys_in         = thrust::device_vector<int>{0, 0, 0, 0, 0, 0, 0, 0, /**/ 0, 9, 3, 2, 1, 8, 7, 4};
  auto keys_out        = thrust::device_vector<int>(num_segments * k, sentinel);

  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), stride);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);

  auto segment_sizes =
    cuda::args::deferred_sequence{thrust::raw_pointer_cast(d_segment_sizes.data()), cuda::args::bounds<-8, 100>()};
  constexpr auto k_arg = cuda::args::constant<k>{};
  auto num_segs        = cuda::args::immediate{cuda::std::int64_t{num_segments}};
  auto env             = cuda::std::execution::env{cuda::execution::require(
    cuda::execution::determinism::not_guaranteed,
    cuda::execution::tie_break::unspecified,
    cuda::execution::output_ordering::unsorted)};

  cuda::std::size_t temp_storage_bytes = 0;
  auto error                           = cub::DeviceBatchedTopK::MaxKeys(
    nullptr, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segs, env);
  REQUIRE(error == cudaSuccess);

  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);
  error = cub::DeviceBatchedTopK::MaxKeys(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    segment_sizes,
    k_arg,
    num_segs,
    env);
  REQUIRE(error == cudaSuccess);

  // Segment 0's output slots are left untouched (still the sentinel); segment 1 holds its top-3.
  thrust::sort(keys_out.begin() + k, keys_out.begin() + 2 * k, cuda::std::greater<int>{});
  REQUIRE(keys_out == thrust::device_vector<int>{sentinel, sentinel, sentinel, 9, 8, 7});
}

// Deterministic-requirement counterpart of the clamp test above: a gpu_to_gpu requirement routes to the SM90+ cluster
// backend even for small segments, pinning that backend's own empty-segment early-out on a clamped negative size.
C2H_TEST("DeviceBatchedTopK::MaxKeys clamps a negative segment size to an empty segment (deterministic requirement)",
         "[keys][segmented][topk][device][cluster][determinism]")
{
  using seg_size_t           = cuda::std::int16_t;
  constexpr auto determinism = cuda::execution::determinism::__determinism_t::__gpu_to_gpu;
  constexpr auto tie_break   = cuda::execution::tie_break::__tie_break_t::__prefer_smaller_index;
  constexpr int num_segments = 2;
  constexpr int k            = 3;
  constexpr int stride       = 8;
  constexpr int sentinel     = -12345;
  constexpr cuda::std::int64_t static_max_segment_size = 100;

  // Segment 0: declared size -1 -> clamped to 0 -> skipped. Segment 1: 8 real keys, top-3 max = {9, 8, 7}.
  auto d_segment_sizes = thrust::device_vector<seg_size_t>{seg_size_t{-1}, seg_size_t{stride}};
  auto keys_in         = thrust::device_vector<int>{0, 0, 0, 0, 0, 0, 0, 0, /**/ 0, 9, 3, 2, 1, 8, 7, 4};
  auto keys_out        = thrust::device_vector<int>(num_segments * k, sentinel);

  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), stride);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);

  auto segment_sizes =
    cuda::args::deferred_sequence{thrust::raw_pointer_cast(d_segment_sizes.data()), cuda::args::bounds<-8, 100>()};

  skip_unless_batched_topk_keys_supported<cub::detail::topk::select::max, determinism, tie_break>(
    static_max_segment_size,
    d_keys_in,
    d_keys_out,
    segment_sizes,
    cuda::args::constant<k>{},
    cuda::args::immediate{cuda::std::int64_t{num_segments}});
  batched_topk_keys<cub::detail::topk::select::max, determinism, tie_break>(
    d_keys_in,
    d_keys_out,
    segment_sizes,
    cuda::args::constant<k>{},
    cuda::args::immediate{cuda::std::int64_t{num_segments}});

  // Segment 0's output slots are left untouched (still the sentinel); segment 1 holds its top-3.
  thrust::sort(keys_out.begin() + k, keys_out.begin() + 2 * k, cuda::std::greater<int>{});
  REQUIRE(keys_out == thrust::device_vector<int>{sentinel, sentinel, sentinel, 9, 8, 7});
}

// A negative statically-known lower bound on `k` (a `bounds<-8, ...>`, as a bare `int16_t` would have) is accepted: the
// kernel clamps a negative runtime `k` up to 0 before widening it, so that segment selects nothing rather than
// reinterpreting the negative count as a huge unsigned "select all". Small segments and no determinism route to the
// baseline backend (the cluster-backend counterpart is below).
C2H_TEST("DeviceBatchedTopK::MaxKeys clamps a negative k to no selection (no determinism requirement)",
         "[keys][segmented][topk][device]")
{
  using k_t                  = cuda::std::int16_t; // negative-capable lower bound -> clamp path
  constexpr int num_segments = 2;
  constexpr int stride       = 8; // segment size (also the input stride between segments)
  constexpr int k            = 3; // segment 1's requested top-k
  constexpr int sentinel     = -12345;
  // Give each segment a full-segment-sized output region, wider than any k requested here: a regressed clamp that
  // "selected all" would then write in-bounds and fail the assertion cleanly instead of storing out of bounds.
  constexpr int out_stride = stride;

  // Segment 0: requests k = -1 -> clamped to 0 -> selects nothing. Segment 1: top-3 max of 8 keys = {9, 8, 7}.
  auto d_k      = thrust::device_vector<k_t>{k_t{-1}, k_t{3}};
  auto keys_in  = thrust::device_vector<int>{0, 0, 0, 0, 0, 0, 0, 0, /**/ 0, 9, 3, 2, 1, 8, 7, 4};
  auto keys_out = thrust::device_vector<int>(num_segments * out_stride, sentinel);

  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), stride);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), out_stride);

  auto segment_sizes = cuda::args::immediate{cuda::std::int64_t{stride}, cuda::args::bounds<0, 100>()};
  auto k_arg    = cuda::args::deferred_sequence{thrust::raw_pointer_cast(d_k.data()), cuda::args::bounds<-8, 100>()};
  auto num_segs = cuda::args::immediate{cuda::std::int64_t{num_segments}};
  auto env      = cuda::std::execution::env{cuda::execution::require(
    cuda::execution::determinism::not_guaranteed,
    cuda::execution::tie_break::unspecified,
    cuda::execution::output_ordering::unsorted)};

  cuda::std::size_t temp_storage_bytes = 0;
  auto error                           = cub::DeviceBatchedTopK::MaxKeys(
    nullptr, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segs, env);
  REQUIRE(error == cudaSuccess);

  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);
  error = cub::DeviceBatchedTopK::MaxKeys(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    segment_sizes,
    k_arg,
    num_segs,
    env);
  REQUIRE(error == cudaSuccess);

  // Segment 0 selects nothing, so its whole output region stays at the sentinel; segment 1 holds its top-3 in the first
  // k slots of its region, the rest untouched.
  thrust::sort(keys_out.begin() + out_stride, keys_out.begin() + out_stride + k, cuda::std::greater<int>{});
  auto expected            = thrust::device_vector<int>(num_segments * out_stride, sentinel);
  expected[out_stride + 0] = 9;
  expected[out_stride + 1] = 8;
  expected[out_stride + 2] = 7;
  REQUIRE(keys_out == expected);
}

// Deterministic-requirement counterpart of the negative-`k` clamp test above: a gpu_to_gpu requirement routes to the
// SM90+ cluster backend even for small segments, exercising that backend's negative-`k` clamp (a signed `k` widened to
// the cluster's 64-bit intermediate must not become a huge "select all").
C2H_TEST("DeviceBatchedTopK::MaxKeys clamps a negative k to no selection (deterministic requirement)",
         "[keys][segmented][topk][device][cluster][determinism]")
{
  using k_t                  = cuda::std::int16_t;
  constexpr auto determinism = cuda::execution::determinism::__determinism_t::__gpu_to_gpu;
  constexpr auto tie_break   = cuda::execution::tie_break::__tie_break_t::__prefer_smaller_index;
  constexpr int num_segments = 2;
  constexpr int stride       = 8; // segment size (also the input stride between segments)
  constexpr int k            = 3; // segment 1's requested top-k
  constexpr int sentinel     = -12345;
  constexpr int out_stride   = stride; // full-segment-sized region (see baseline test above)
  constexpr cuda::std::int64_t static_max_segment_size = 100;

  // Segment 0: requests k = -1 -> clamped to 0 -> selects nothing. Segment 1: top-3 max of 8 keys = {9, 8, 7}.
  auto d_k      = thrust::device_vector<k_t>{k_t{-1}, k_t{3}};
  auto keys_in  = thrust::device_vector<int>{0, 0, 0, 0, 0, 0, 0, 0, /**/ 0, 9, 3, 2, 1, 8, 7, 4};
  auto keys_out = thrust::device_vector<int>(num_segments * out_stride, sentinel);

  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), stride);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), out_stride);

  auto segment_sizes = cuda::args::immediate{cuda::std::int64_t{stride}, cuda::args::bounds<0, 100>()};
  auto k_arg    = cuda::args::deferred_sequence{thrust::raw_pointer_cast(d_k.data()), cuda::args::bounds<-8, 100>()};
  auto num_segs = cuda::args::immediate{cuda::std::int64_t{num_segments}};

  skip_unless_batched_topk_keys_supported<cub::detail::topk::select::max, determinism, tie_break>(
    static_max_segment_size, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segs);
  batched_topk_keys<cub::detail::topk::select::max, determinism, tie_break>(
    d_keys_in, d_keys_out, segment_sizes, k_arg, num_segs);

  // Segment 0 selects nothing, so its whole output region stays at the sentinel; segment 1 holds its top-3 in the first
  // k slots of its region, the rest untouched.
  thrust::sort(keys_out.begin() + out_stride, keys_out.begin() + out_stride + k, cuda::std::greater<int>{});
  auto expected            = thrust::device_vector<int>(num_segments * out_stride, sentinel);
  expected[out_stride + 0] = 9;
  expected[out_stride + 1] = 8;
  expected[out_stride + 2] = 7;
  REQUIRE(keys_out == expected);
}

// A uniform (host-known) negative segment size means every segment is empty. With a deterministic requirement (which
// routes to the cluster backend), this must be recognized on the host from the non-positive runtime maximum segment
// size and skipped without launching, leaving the output untouched -- rather than casting the negative maximum to
// unsigned and sizing an enormous launch. Where that backend is unavailable the request fails with
// cudaErrorNotSupported even though it is a no-op (the wrapper verifies that, then skips).
C2H_TEST("DeviceBatchedTopK::MaxKeys treats a uniform negative segment size as no work (deterministic requirement)",
         "[keys][segmented][topk][device][cluster][determinism]")
{
  using seg_size_t           = cuda::std::int16_t;
  constexpr auto determinism = cuda::execution::determinism::__determinism_t::__gpu_to_gpu;
  constexpr auto tie_break   = cuda::execution::tie_break::__tie_break_t::__prefer_smaller_index;
  constexpr int num_segments = 2;
  constexpr int k            = 3;
  constexpr int stride       = 8;
  constexpr int sentinel     = -12345;
  constexpr cuda::std::int64_t static_max_segment_size = 100;

  auto keys_in  = thrust::device_vector<int>{0, 9, 3, 2, 1, 8, 7, 4, /**/ 5, 6, 1, 0, 3, 2, 8, 7};
  auto keys_out = thrust::device_vector<int>(num_segments * k, sentinel);
  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), stride);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);

  skip_unless_batched_topk_keys_supported<cub::detail::topk::select::max, determinism, tie_break>(
    static_max_segment_size,
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{seg_size_t{-1}, cuda::args::bounds<-8, 100>()},
    cuda::args::constant<k>{},
    cuda::args::immediate{cuda::std::int64_t{num_segments}});
  batched_topk_keys<cub::detail::topk::select::max, determinism, tie_break>(
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{seg_size_t{-1}, cuda::args::bounds<-8, 100>()},
    cuda::args::constant<k>{},
    cuda::args::immediate{cuda::std::int64_t{num_segments}});

  // No segment had any items, so the whole output is left untouched.
  REQUIRE(keys_out == thrust::device_vector<int>(num_segments * k, sentinel));
}

// Zero segments is a no-op, but with a positive segment-size bound the `max_seg_size <= 0` guard does not fire, so the
// dispatch must elide the launch from `num_segments == 0` alone -- otherwise the grid would use `grid_dim == 0`, an
// invalid launch configuration. A small size and no determinism requirement route this to the baseline backend; the
// deterministic-requirement counterpart is below.
C2H_TEST("DeviceBatchedTopK::MaxKeys treats zero segments as no work (no determinism requirement)",
         "[keys][segmented][topk][device]")
{
  constexpr int k        = 3;
  constexpr int stride   = 8;
  constexpr int sentinel = -12345;
  auto keys_in           = thrust::device_vector<int>{};
  // Canary output slots that must stay untouched: with zero segments nothing may be written.
  auto keys_out = thrust::device_vector<int>(k, sentinel);
  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), stride);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);

  auto segment_sizes   = cuda::args::immediate{cuda::std::int16_t{stride}, cuda::args::bounds<0, 100>()};
  constexpr auto k_arg = cuda::args::constant<k>{};
  auto num_segs        = cuda::args::immediate{cuda::std::int64_t{0}};
  auto env             = cuda::std::execution::env{cuda::execution::require(
    cuda::execution::determinism::not_guaranteed,
    cuda::execution::tie_break::unspecified,
    cuda::execution::output_ordering::unsorted)};

  cuda::std::size_t temp_storage_bytes = 0;
  auto error                           = cub::DeviceBatchedTopK::MaxKeys(
    nullptr, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segs, env);
  REQUIRE(error == cudaSuccess);

  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);
  error = cub::DeviceBatchedTopK::MaxKeys(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    segment_sizes,
    k_arg,
    num_segs,
    env);
  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == thrust::device_vector<int>(k, sentinel));
}

// Deterministic-requirement counterpart: a gpu_to_gpu requirement routes to the SM90+ cluster backend. A positive size
// bound means the `max_seg_size <= 0` guard does not apply, so the launch must be elided from `num_segments == 0`
// alone -- by the dispatch's empty-batch guard, before the cluster arm launches. Where that backend is unavailable the
// request fails with cudaErrorNotSupported even though it is a no-op (the wrapper verifies that, then skips).
C2H_TEST("DeviceBatchedTopK::MaxKeys treats zero segments as no work (deterministic requirement)",
         "[keys][segmented][topk][device][cluster][determinism]")
{
  constexpr auto determinism = cuda::execution::determinism::__determinism_t::__gpu_to_gpu;
  constexpr auto tie_break   = cuda::execution::tie_break::__tie_break_t::__prefer_smaller_index;
  constexpr int k            = 3;
  constexpr int stride       = 8;
  constexpr int sentinel     = -12345;
  constexpr cuda::std::int64_t static_max_segment_size = 100;

  auto keys_in = thrust::device_vector<int>{};
  // Canary output slots that must stay untouched: with zero segments nothing may be written.
  auto keys_out = thrust::device_vector<int>(k, sentinel);
  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), stride);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);

  skip_unless_batched_topk_keys_supported<cub::detail::topk::select::max, determinism, tie_break>(
    static_max_segment_size,
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{cuda::std::int16_t{stride}, cuda::args::bounds<0, 100>()},
    cuda::args::constant<k>{},
    cuda::args::immediate{cuda::std::int64_t{0}});
  batched_topk_keys<cub::detail::topk::select::max, determinism, tie_break>(
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{cuda::std::int16_t{stride}, cuda::args::bounds<0, 100>()},
    cuda::args::constant<k>{},
    cuda::args::immediate{cuda::std::int64_t{0}});

  REQUIRE(keys_out == thrust::device_vector<int>(k, sentinel));
}
#endif // TEST_TYPES == 0 && TEST_LAUNCH == 0
