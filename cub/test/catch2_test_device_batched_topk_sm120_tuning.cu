// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Covers the SM 12.0 keys-only, non-deterministic DeviceBatchedTopK tunings: which requests the policy selector routes
// to them, and the correctness of the selected key set across the segment-size bounds that reach them.

// See CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT in cub/device/device_batched_topk.cuh. Precedes CUB includes.
#define CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_batched_topk.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/argument>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <algorithm>
#include <random>
#include <vector>

#include "catch2_test_device_topk_common.cuh"
#include "cub_test_macros.h"
#include <c2h/catch2_test_helper.h>

namespace bt = cub::detail::batched_topk;

using select_t = cub::detail::topk::select;
using det_t    = cuda::execution::determinism::__determinism_t;
using tie_t    = cuda::execution::tie_break::__tie_break_t;
using size_t64 = cuda::std::int64_t;

template <typename T>
using segment_it_t =
  decltype(cuda::make_strided_iterator(cuda::make_counting_iterator(static_cast<T*>(nullptr)), size_t64{}));

template <size_t64 MaxSegmentSize>
using segment_size_param_t =
  decltype(cuda::args::immediate{size_t64{}, cuda::args::bounds<size_t64{1}, MaxSegmentSize>()});

template <size_t64 MaxK>
using k_param_t = decltype(cuda::args::immediate{size_t64{}, cuda::args::bounds<size_t64{1}, MaxK>()});

using num_segments_param_t = decltype(cuda::args::immediate{size_t64{}});

template <typename ValueT>
struct value_it
{
  using type = segment_it_t<ValueT>;
};

template <>
struct value_it<cub::NullType>
{
  using type = cub::NullType**;
};

template <select_t Direction>
using select_direction_param_t = const decltype(bt::wrap_select_direction(cuda::args::constant<Direction>{}));

// The selector the dispatch instantiates for a request issued through the public API with the iterator and argument
// types used by the execution tests below. Keys-only requests pass `NullType**` value iterators, like
// DeviceBatchedTopK::{Max,Min}Keys.
template <typename KeyT,
          typename ValueT,
          size_t64 MaxSegmentSize,
          size_t64 MaxK,
          det_t Determinism  = det_t::__not_guaranteed,
          tie_t TieBreak     = tie_t::__unspecified,
          select_t Direction = select_t::max>
struct request
{
  using value_it_t = typename value_it<ValueT>::type;

  using selector_t = bt::policy_selector_from_types<
    KeyT,
    ValueT,
    MaxK,
    MaxSegmentSize,
    Determinism,
    TieBreak,
    segment_size_param_t<MaxSegmentSize>,
    segment_it_t<KeyT>,
    segment_it_t<KeyT>,
    value_it_t,
    value_it_t,
    k_param_t<MaxK>,
    select_direction_param_t<Direction>,
    num_segments_param_t,
    size_t64>;

  [[nodiscard]] static constexpr bt::topk_policy policy(cuda::compute_capability cc)
  {
    return selector_t{}(cc);
  }

  // The one-worker-per-segment policy the baseline kernel instantiates for this request on `cc` (index into
  // `policy(cc).baseline.worker_per_segment_policies`, or -1 when none covers the segment size).
  template <int Major, int Minor>
  struct policy_getter
  {
    [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bt::topk_policy operator()() const
    {
      return selector_t{}(cuda::compute_capability{Major, Minor});
    }
  };

  template <int Major, int Minor>
  static constexpr int baseline_worker_index = bt::find_covering_policy_index<
    policy_getter<Major, Minor>,
    segment_size_param_t<MaxSegmentSize>,
    segment_it_t<KeyT>,
    segment_it_t<KeyT>,
    value_it_t,
    value_it_t,
    segment_size_param_t<MaxSegmentSize>,
    k_param_t<MaxK>,
    select_direction_param_t<Direction>,
    num_segments_param_t,
    size_t64>::value;
};

inline constexpr cuda::compute_capability cc_120{12, 0};

template <typename Request>
[[nodiscard]] constexpr bool uses_sm120_keys_tables(cuda::compute_capability cc, size_t64 max_seg, size_t64 max_k)
{
  const auto policy = Request::policy(cc);
  return policy.baseline == bt::make_sm120_baseline_policy()
      && policy.cluster == bt::make_sm120_keys_cluster_policy(max_seg, max_k);
}

template <typename Request>
[[nodiscard]] constexpr bool uses_default_tables(cuda::compute_capability cc)
{
  const auto policy = Request::policy(cc);
  return policy.baseline == bt::make_baseline_policy() && policy.cluster == bt::make_cluster_policy();
}

// The execution tests target the SM 12.0 tunings, which the dispatch selects only when it resolves to CC 12.0.
inline void skip_unless_sm120()
{
  cuda::compute_capability cc{};
  REQUIRE(cub::detail::ptx_compute_cap(cc) == cudaSuccess);
  if (cc != cc_120)
  {
    SKIP("The SM 12.0 tunings apply only when the dispatch resolves to compute capability 12.0.");
  }
}

#if TEST_TYPES == 0
CUB_TEST("DeviceBatchedTopK selector routes keys-only non-deterministic requests on SM 12.0 to the SM 12.0 tables",
         "[keys][segmented][topk][device][tuning]",
         CUB_SMALL)
{
  // Preconditions that make the assertions below discriminating.
  STATIC_REQUIRE(bt::make_sm120_baseline_policy() != bt::make_baseline_policy());
  STATIC_REQUIRE(bt::make_sm120_keys_cluster_policy(16 * 1024, 8 * 1024) != bt::make_cluster_policy());
  STATIC_REQUIRE(bt::make_sm120_keys_cluster_policy(16 * 1024 + 1, 8 * 1024) != bt::make_cluster_policy());
  STATIC_REQUIRE(bt::make_sm120_keys_cluster_policy(16 * 1024, 8 * 1024)
                 != bt::make_sm120_keys_cluster_policy(16 * 1024 + 1, 8 * 1024));

  // Static segment-size bounds below the cluster crossover (8K) run the SM 12.0 baseline backend, the rest the cluster
  // backend. Up to 8K the SM 12.0 cluster table is the default cluster policy (pinned in the next test).
  using r1k_t   = request<float, cub::NullType, 1024, 512>;
  using r2k_t   = request<float, cub::NullType, 2048, 1024>;
  using r8k_t   = request<float, cub::NullType, 8 * 1024, 4 * 1024>;
  using r16k_t  = request<float, cub::NullType, 16 * 1024, 8 * 1024>;
  using r16k1_t = request<float, cub::NullType, 16 * 1024 + 1, 8 * 1024>;
  using r16k_min_t =
    request<float, cub::NullType, 16 * 1024, 8 * 1024, det_t::__not_guaranteed, tie_t::__unspecified, select_t::min>;

  STATIC_REQUIRE(uses_sm120_keys_tables<r1k_t>(cc_120, 1024, 512));
  STATIC_REQUIRE(uses_sm120_keys_tables<r2k_t>(cc_120, 2048, 1024));
  STATIC_REQUIRE(uses_sm120_keys_tables<r8k_t>(cc_120, 8 * 1024, 4 * 1024));
  STATIC_REQUIRE(uses_sm120_keys_tables<r16k_t>(cc_120, 16 * 1024, 8 * 1024));
  STATIC_REQUIRE(uses_sm120_keys_tables<r16k1_t>(cc_120, 16 * 1024 + 1, 8 * 1024));
  STATIC_REQUIRE(uses_sm120_keys_tables<r16k_min_t>(cc_120, 16 * 1024, 8 * 1024));

  STATIC_REQUIRE(r1k_t::policy(cc_120).backend == bt::topk_algorithm::baseline);
  STATIC_REQUIRE(r2k_t::policy(cc_120).backend == bt::topk_algorithm::baseline);
  STATIC_REQUIRE(r8k_t::policy(cc_120).backend == bt::topk_algorithm::cluster);
  STATIC_REQUIRE(r16k_t::policy(cc_120).backend == bt::topk_algorithm::cluster);
  STATIC_REQUIRE(r16k1_t::policy(cc_120).backend == bt::topk_algorithm::cluster);

  // The baseline kernel picks the SM 12.0 workers: 1K keys run the 64x16 worker with 6-bit digits, 2K the 128x16
  // worker.
  constexpr auto sm120_workers = bt::make_sm120_baseline_policy().worker_per_segment_policies;
  STATIC_REQUIRE(r1k_t::baseline_worker_index<12, 0> == 4);
  STATIC_REQUIRE(sm120_workers[4].threads_per_block == 64);
  STATIC_REQUIRE(sm120_workers[4].items_per_thread == 16);
  STATIC_REQUIRE(sm120_workers[4].radix_bits == 6);
  STATIC_REQUIRE(r2k_t::baseline_worker_index<12, 0> == 3);
  STATIC_REQUIRE(sm120_workers[3].threads_per_block == 128);
  STATIC_REQUIRE(sm120_workers[3].items_per_thread == 16);

  // The gate is independent of the key type.
  STATIC_REQUIRE(uses_sm120_keys_tables<request<cuda::std::uint8_t, cub::NullType, 16 * 1024, 8 * 1024>>(
    cc_120, 16 * 1024, 8 * 1024));
  STATIC_REQUIRE(uses_sm120_keys_tables<request<cuda::std::uint32_t, cub::NullType, 1024, 512>>(cc_120, 1024, 512));
  STATIC_REQUIRE(
    uses_sm120_keys_tables<request<double, cub::NullType, 16 * 1024 + 1, 8 * 1024>>(cc_120, 16 * 1024 + 1, 8 * 1024));
}

template <size_t64 MaxSegmentSize, size_t64 MaxK = MaxSegmentSize / 2>
inline constexpr bt::cluster_topk_policy sm120_cluster =
  request<float, cub::NullType, MaxSegmentSize, MaxK>::policy(cc_120).cluster;

template <size_t64 MaxSegmentSize, size_t64 MaxK = MaxSegmentSize / 2>
[[nodiscard]] constexpr bool
sm120_cluster_is(bool wave_aware, int chunk_bytes, int single_block_max_seg_size, int bits_per_pass)
{
  constexpr auto policy = sm120_cluster<MaxSegmentSize, MaxK>;
  return policy.wave_aware_cluster_width == wave_aware && policy.chunk_bytes == chunk_bytes
      && policy.single_block_max_seg_size == single_block_max_seg_size && policy.bits_per_pass == bits_per_pass;
}

[[nodiscard]] constexpr bool
sm120_worker_is(const bt::worker_policy& worker, int threads_per_block, int items_per_thread, int radix_bits)
{
  return worker.threads_per_block == threads_per_block && worker.items_per_thread == items_per_thread
      && worker.radix_bits == radix_bits && worker.load_algorithm == cub::BLOCK_LOAD_VECTORIZE
      && worker.store_algorithm == cub::BLOCK_STORE_DIRECT && worker.epilogue.items_per_thread == 16
      && worker.epilogue.load_algorithm == cub::BLOCK_LOAD_WARP_TRANSPOSE
      && worker.epilogue.store_algorithm == cub::BLOCK_STORE_WARP_TRANSPOSE
      && worker.epilogue.scan_algorithm == cub::BLOCK_SCAN_WARP_SCANS;
}

CUB_TEST("DeviceBatchedTopK SM 12.0 keys-only tables select the measured values",
         "[keys][segmented][topk][device][tuning]",
         CUB_SMALL)
{
  // Baseline: every worker's geometry and digit width, and the shared load/store/epilogue choices.
  constexpr auto baseline = request<float, cub::NullType, 1024, 512>::policy(cc_120).baseline;
  STATIC_REQUIRE(baseline.worker_per_segment_policies.size() == 6);
  STATIC_REQUIRE(sm120_worker_is(baseline.worker_per_segment_policies[0], 256, 64, 0));
  STATIC_REQUIRE(sm120_worker_is(baseline.worker_per_segment_policies[1], 256, 32, 0));
  STATIC_REQUIRE(sm120_worker_is(baseline.worker_per_segment_policies[2], 256, 16, 0));
  STATIC_REQUIRE(sm120_worker_is(baseline.worker_per_segment_policies[3], 128, 16, 0));
  STATIC_REQUIRE(sm120_worker_is(baseline.worker_per_segment_policies[4], 64, 16, 6));
  STATIC_REQUIRE(sm120_worker_is(baseline.worker_per_segment_policies[5], 128, 2, 0));
  STATIC_REQUIRE(baseline.multi_worker_per_segment_policy.threads_per_block == 256);
  STATIC_REQUIRE(baseline.multi_worker_per_segment_policy.items_per_thread == 64);

  // Cluster: (wave-aware width, chunk bytes, single-CTA threshold, digit bits) at each size-class boundary.
  // Up to 8K, and whenever the k bound reaches the segment bound, the default policy applies.
  STATIC_REQUIRE(sm120_cluster_is<8 * 1024>(false, 16 * 1024, 8 * 1024, 11));
  STATIC_REQUIRE(sm120_cluster_is<64 * 1024, 64 * 1024>(false, 16 * 1024, 8 * 1024, 11));
  STATIC_REQUIRE(sm120_cluster<8 * 1024> == bt::make_cluster_policy());
  STATIC_REQUIRE(sm120_cluster<64 * 1024, 64 * 1024> == bt::make_cluster_policy());
  // (8K, 16K]: single-CTA path up to 16K with 10-bit digits.
  STATIC_REQUIRE(sm120_cluster_is<8 * 1024 + 1>(true, 16 * 1024, 16 * 1024, 10));
  STATIC_REQUIRE(sm120_cluster_is<16 * 1024>(true, 16 * 1024, 16 * 1024, 10));
  // Above 16K: 32 KiB chunks; 9-bit digits below 512K, 8-bit in [512K, 1M), 11-bit in [1M, 2M), 8-bit at 2M.
  STATIC_REQUIRE(sm120_cluster_is<16 * 1024 + 1>(true, 32 * 1024, 8 * 1024, 9));
  STATIC_REQUIRE(sm120_cluster_is<512 * 1024 - 1>(true, 32 * 1024, 8 * 1024, 9));
  STATIC_REQUIRE(sm120_cluster_is<512 * 1024>(true, 32 * 1024, 8 * 1024, 8));
  STATIC_REQUIRE(sm120_cluster_is<1024 * 1024 - 1>(true, 32 * 1024, 8 * 1024, 8));
  STATIC_REQUIRE(sm120_cluster_is<1024 * 1024>(true, 32 * 1024, 8 * 1024, 11));
  STATIC_REQUIRE(sm120_cluster_is<2048 * 1024 - 1>(true, 32 * 1024, 8 * 1024, 11));
  STATIC_REQUIRE(sm120_cluster_is<2048 * 1024>(true, 32 * 1024, 8 * 1024, 8));
}

CUB_TEST("DeviceBatchedTopK selector keeps the default tables for pairs and deterministic requests on SM 12.0",
         "[pairs][keys][segmented][topk][device][tuning]",
         CUB_SMALL)
{
  // Pairs, non-deterministic.
  STATIC_REQUIRE(uses_default_tables<request<float, cuda::std::uint32_t, 1024, 512>>(cc_120));
  STATIC_REQUIRE(uses_default_tables<request<float, cuda::std::uint32_t, 16 * 1024, 8 * 1024>>(cc_120));
  STATIC_REQUIRE(uses_default_tables<request<double, cuda::std::int64_t, 16 * 1024 + 1, 8 * 1024>>(cc_120));

  // Keys-only, deterministic without a tie-break preference.
  STATIC_REQUIRE(
    uses_default_tables<request<float, cub::NullType, 1024, 512, det_t::__run_to_run, tie_t::__unspecified>>(cc_120));
  STATIC_REQUIRE(
    uses_default_tables<request<float, cub::NullType, 16 * 1024, 8 * 1024, det_t::__gpu_to_gpu, tie_t::__unspecified>>(
      cc_120));

  // Keys-only with an explicit tie-break.
  STATIC_REQUIRE(
    uses_default_tables<request<float, cub::NullType, 1024, 512, det_t::__gpu_to_gpu, tie_t::__prefer_smaller_index>>(
      cc_120));
  STATIC_REQUIRE(
    uses_default_tables<
      request<float, cub::NullType, 16 * 1024, 8 * 1024, det_t::__gpu_to_gpu, tie_t::__prefer_larger_index>>(cc_120));

  // Pairs, deterministic.
  STATIC_REQUIRE(
    uses_default_tables<
      request<float, cuda::std::uint32_t, 16 * 1024, 8 * 1024, det_t::__gpu_to_gpu, tie_t::__unspecified>>(cc_120));
}

CUB_TEST("DeviceBatchedTopK selector applies the SM 12.0 tables only to compute capability 12.0",
         "[keys][segmented][topk][device][tuning]",
         CUB_SMALL)
{
  using r1k_t         = request<float, cub::NullType, 1024, 512>;
  using r16k_t        = request<float, cub::NullType, 16 * 1024, 8 * 1024>;
  using r16k_double_t = request<double, cub::NullType, 16 * 1024, 8 * 1024>;

  constexpr cuda::compute_capability cc_121{12, 1};
  constexpr cuda::compute_capability cc_100{10, 0};
  constexpr cuda::compute_capability cc_103{10, 3};
  constexpr cuda::compute_capability cc_90{9, 0};

  STATIC_REQUIRE(uses_default_tables<r1k_t>(cc_121));
  STATIC_REQUIRE(uses_default_tables<r16k_t>(cc_121));
  STATIC_REQUIRE(uses_default_tables<r1k_t>(cc_90));
  STATIC_REQUIRE(uses_default_tables<r16k_t>(cc_90));

  // SM 10.x keep the default baseline; their own keys cluster tables apply to 4-byte keys only.
  STATIC_REQUIRE(r16k_t::policy(cc_100).baseline == bt::make_baseline_policy());
  STATIC_REQUIRE(r16k_t::policy(cc_100).cluster == bt::make_sm100_keys_cluster_policy(16 * 1024, 8 * 1024));
  STATIC_REQUIRE(r16k_t::policy(cc_103).baseline == bt::make_baseline_policy());
  STATIC_REQUIRE(r16k_t::policy(cc_103).cluster == bt::make_sm103_keys_cluster_policy(16 * 1024, 8 * 1024));
  STATIC_REQUIRE(uses_default_tables<r16k_double_t>(cc_100));
  STATIC_REQUIRE(uses_default_tables<r1k_t>(cc_100));

  STATIC_REQUIRE(r1k_t::baseline_worker_index<12, 1> == r1k_t::baseline_worker_index<9, 0>);
  STATIC_REQUIRE(
    r1k_t::policy(cc_121).baseline.worker_per_segment_policies[r1k_t::baseline_worker_index<12, 1>].radix_bits == 0);
}

CUB_TEST("DeviceBatchedTopK resolves to the SM 12.0 tables on an SM 12.0 device",
         "[keys][segmented][topk][device][tuning]",
         CUB_SMALL)
{
  skip_unless_sm120();
  cuda::compute_capability cc{};
  REQUIRE(cub::detail::ptx_compute_cap(cc) == cudaSuccess);
  REQUIRE(uses_sm120_keys_tables<request<float, cub::NullType, 1024, 512>>(cc, 1024, 512));
  REQUIRE(uses_sm120_keys_tables<request<float, cub::NullType, 16 * 1024 + 1, 8 * 1024>>(cc, 16 * 1024 + 1, 8 * 1024));
}
#endif // TEST_TYPES == 0

// ---------------------------------------------------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------------------------------------------------

template <select_t Direction, size_t64 MaxSegmentSize, size_t64 MaxK, typename KeyT>
void run_nondeterministic_keys(
  c2h::device_vector<KeyT>& keys_in,
  c2h::device_vector<KeyT>& keys_out,
  size_t64 segment_size,
  size_t64 k,
  size_t64 num_segments)
{
  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), segment_size);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);
  const auto segment_sizes = cuda::args::immediate{segment_size, cuda::args::bounds<size_t64{1}, MaxSegmentSize>()};
  const auto k_arg         = cuda::args::immediate{k, cuda::args::bounds<size_t64{1}, MaxK>()};
  const auto num_segs      = cuda::args::immediate{num_segments};
  static_assert(cuda::std::is_same_v<decltype(segment_sizes), const segment_size_param_t<MaxSegmentSize>>);
  static_assert(cuda::std::is_same_v<decltype(k_arg), const k_param_t<MaxK>>);

  const auto env = cuda::std::execution::env{cuda::execution::require(
    cuda::execution::determinism::not_guaranteed,
    cuda::execution::tie_break::unspecified,
    cuda::execution::output_ordering::unsorted)};

  const auto dispatch = [&](void* d_temp_storage, cuda::std::size_t& temp_storage_bytes) {
    if constexpr (Direction == select_t::max)
    {
      return cub::DeviceBatchedTopK::MaxKeys(
        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segs, env);
    }
    else
    {
      return cub::DeviceBatchedTopK::MinKeys(
        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segs, env);
    }
  };

  if (batched_topk_backend_unavailable(MaxSegmentSize))
  {
    expect_batched_topk_unsupported_and_skip(dispatch);
  }

  cuda::std::size_t temp_storage_bytes = 0;
  REQUIRE(dispatch(nullptr, temp_storage_bytes) == cudaSuccess);
  c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  REQUIRE(dispatch(thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes) == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
}

// %PARAM% TEST_TYPES types 0:1:2

#if TEST_TYPES == 0
using key_type = float;
#elif TEST_TYPES == 1
using key_type = cuda::std::uint32_t;
#elif TEST_TYPES == 2
using key_type = double;
#endif

// Static segment-size bounds covering the SM 12.0 baseline workers (1K: 64x16 with 6-bit digits, 2K: 128x16), the
// cluster backend with the default cluster policy (8K), and the SM 12.0 cluster table's classes (16K, >16K).
using max_segment_size_list = c2h::enum_type_list<size_t64, 1024, 2048, 8 * 1024, 16 * 1024, 16 * 1024 + 1>;

using select_direction_list = c2h::enum_type_list<select_t, select_t::min, select_t::max>;

CUB_TEST("DeviceBatchedTopK::{Min,Max}Keys non-deterministic selects the correct set across the SM 12.0 size classes",
         "[keys][segmented][topk][device][tuning]",
         CUB_SMALL,
         max_segment_size_list,
         select_direction_list)
{
  constexpr size_t64 max_segment_size = c2h::get<0, TestType>::value;
  // A k bound below the segment bound: `max_k >= max_segment_size` falls back to the default cluster table.
  constexpr size_t64 max_k     = max_segment_size / 2;
  constexpr select_t direction = c2h::get<1, TestType>::value;

  skip_unless_sm120();

  const size_t64 segment_size =
    GENERATE_COPY(values({max_segment_size, max_segment_size - 1}), take(1, random(size_t64{1}, max_segment_size)));
  const size_t64 k = GENERATE_COPY(values({size_t64{1}, max_k}), take(1, random(size_t64{1}, max_k)));
  // 700 segments make the wave-aware cluster width pick a narrower cluster than the fully resident one.
  const size_t64 num_segments = GENERATE(size_t64{3}, size_t64{700});
  const size_t64 out_k        = (cuda::std::min) (k, segment_size);

  CAPTURE(c2h::type_name<key_type>(), max_segment_size, max_k, segment_size, k, num_segments, direction);

  c2h::device_vector<key_type> keys_in(num_segments * segment_size, thrust::no_init);
  c2h::gen(C2H_SEED(2), keys_in);
  c2h::device_vector<key_type> keys_out(num_segments * out_k, thrust::no_init);

  run_nondeterministic_keys<direction, max_segment_size, max_k>(keys_in, keys_out, segment_size, out_k, num_segments);

  c2h::device_vector<key_type> expected(keys_in);
  fixed_size_segmented_sort_keys(expected, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected, segment_size, out_k);
  fixed_size_segmented_sort_keys(keys_out, num_segments, out_k, direction);

  REQUIRE(expected == keys_out);
}

#if TEST_TYPES != 1
template <typename T>
using bits_t = cuda::std::conditional_t<sizeof(T) == 4, cuda::std::uint32_t, cuda::std::uint64_t>;

template <typename T>
[[nodiscard]] bits_t<T> to_bits(T x)
{
  return cuda::std::bit_cast<bits_t<T>>(x);
}

template <typename T>
inline constexpr bits_t<T> sign_bit = bits_t<T>{1} << (sizeof(T) * 8 - 1);

// The key order of DeviceRadixSort, which the top-k reference sorts in this suite rely on: +0.0 and -0.0 are
// equivalent, and NaNs are ordered by their bit representation after the sign transform (a positive NaN ranks above
// +inf, a negative NaN below -inf).
template <typename T>
[[nodiscard]] bits_t<T> radix_order(T x)
{
  bits_t<T> b = to_bits(x);
  if (b == sign_bit<T>)
  {
    b = 0;
  }
  return (b & sign_bit<T>) ? static_cast<bits_t<T>>(~b) : static_cast<bits_t<T>>(b | sign_bit<T>);
}

template <typename T>
[[nodiscard]] bool is_zero(T x)
{
  return (to_bits(x) & ~sign_bit<T>) == 0;
}

// One segment's multiset of key bit patterns, with -0.0 mapped to +0.0 when `merge_zeros`.
template <typename T>
[[nodiscard]] std::vector<bits_t<T>> sorted_bits(const T* first, const T* last, bool merge_zeros)
{
  std::vector<bits_t<T>> bits;
  for (; first != last; ++first)
  {
    bits.push_back(merge_zeros && is_zero(*first) ? bits_t<T>{0} : to_bits(*first));
  }
  std::sort(bits.begin(), bits.end());
  return bits;
}

CUB_TEST("DeviceBatchedTopK::{Min,Max}Keys non-deterministic select the correct set with NaNs and signed zeros",
         "[keys][segmented][topk][device][tuning][float]",
         CUB_SMALL,
         max_segment_size_list,
         select_direction_list)
{
  skip_unless_sm120();

  using key_t                         = key_type;
  constexpr size_t64 max_segment_size = c2h::get<0, TestType>::value;
  constexpr size_t64 max_k            = max_segment_size - 1;
  constexpr select_t direction        = c2h::get<1, TestType>::value;
  constexpr size_t64 segment_size     = max_segment_size;
  constexpr size_t64 num_segments     = 3;

  constexpr int num_pos_nan  = 5;
  constexpr int num_neg_nan  = 3;
  constexpr int num_inf      = 2; // each sign
  constexpr int num_pos_zero = 4;
  constexpr int num_neg_zero = 6;

  const key_t pos_nan  = cuda::std::numeric_limits<key_t>::quiet_NaN();
  const key_t neg_nan  = cuda::std::bit_cast<key_t>(static_cast<bits_t<key_t>>(to_bits(pos_nan) | sign_bit<key_t>));
  const key_t pos_zero = key_t{0};
  const key_t neg_zero = cuda::std::bit_cast<key_t>(sign_bit<key_t>);
  REQUIRE(!(pos_nan == pos_nan));
  REQUIRE(!(neg_nan == neg_nan));
  REQUIRE(to_bits(neg_nan) != to_bits(pos_nan));
  REQUIRE(to_bits(neg_zero) != to_bits(pos_zero));

  // Every segment holds the same multiset in a different order, so the expected top-k set is the same for all.
  std::mt19937 rng(static_cast<unsigned>(max_segment_size));
  std::vector<key_t> base;
  base.insert(base.end(), num_pos_nan, pos_nan);
  base.insert(base.end(), num_neg_nan, neg_nan);
  base.insert(base.end(), num_inf, cuda::std::numeric_limits<key_t>::infinity());
  base.insert(base.end(), num_inf, -cuda::std::numeric_limits<key_t>::infinity());
  base.insert(base.end(), num_pos_zero, pos_zero);
  base.insert(base.end(), num_neg_zero, neg_zero);
  std::uniform_real_distribution<double> magnitude(1e-3, 1e3);
  std::bernoulli_distribution negative(0.5);
  while (static_cast<size_t64>(base.size()) < segment_size)
  {
    const auto m = static_cast<key_t>(magnitude(rng));
    base.push_back(negative(rng) ? -m : m);
  }

  c2h::host_vector<key_t> h_keys_in;
  for (size_t64 segment = 0; segment < num_segments; ++segment)
  {
    std::shuffle(base.begin(), base.end(), rng);
    h_keys_in.insert(h_keys_in.end(), base.begin(), base.end());
  }

  // Host reference order: best key first.
  std::vector<key_t> ordered(base);
  std::stable_sort(ordered.begin(), ordered.end(), [](key_t a, key_t b) {
    return direction == select_t::max ? radix_order(a) > radix_order(b) : radix_order(a) < radix_order(b);
  });
  const auto zero_begin =
    static_cast<size_t64>(std::find_if(ordered.begin(), ordered.end(), is_zero<key_t>) - ordered.begin());
  const auto zero_end = zero_begin + num_pos_zero + num_neg_zero;
  REQUIRE(std::all_of(ordered.begin() + zero_begin, ordered.begin() + zero_end, is_zero<key_t>));
  constexpr size_t64 num_head_nan = direction == select_t::max ? num_pos_nan : num_neg_nan;
  REQUIRE(to_bits(ordered[0]) == to_bits(direction == select_t::max ? pos_nan : neg_nan));
  REQUIRE(to_bits(ordered[num_head_nan - 1]) == to_bits(ordered[0]));
  REQUIRE(to_bits(ordered[num_head_nan]) != to_bits(ordered[0]));

  struct k_case
  {
    size_t64 k;
    bool zeros_split; // the k-th boundary falls between tied +0.0 / -0.0
  };
  const std::vector<k_case> k_cases = {
    {1, false},
    {num_head_nan, false}, // only the NaNs at the selected end
    {num_head_nan + 1, false},
    {zero_begin, false}, // every zero just outside
    {zero_end, false}, // every zero just inside
    {zero_begin + 3, true},
    {segment_size - num_pos_nan, false}, // for MinKeys, everything but the positive NaNs
    {max_k, false}};

  c2h::device_vector<key_t> keys_in(h_keys_in);
  for (const auto& kc : k_cases)
  {
    const size_t64 k = kc.k;
    CAPTURE(c2h::type_name<key_t>(), max_segment_size, direction, k, kc.zeros_split, zero_begin, zero_end);
    REQUIRE(k >= 1);
    REQUIRE(k <= max_k);

    c2h::device_vector<key_t> keys_out(num_segments * k, thrust::no_init);
    run_nondeterministic_keys<direction, max_segment_size, max_k>(keys_in, keys_out, segment_size, k, num_segments);
    c2h::host_vector<key_t> h_keys_out(keys_out);

    const auto expected = sorted_bits(ordered.data(), ordered.data() + k, kc.zeros_split);
    for (size_t64 segment = 0; segment < num_segments; ++segment)
    {
      CAPTURE(segment);
      const key_t* out = thrust::raw_pointer_cast(h_keys_out.data()) + segment * k;
      REQUIRE(sorted_bits(out, out + k, kc.zeros_split) == expected);
    }
  }
}
#endif // TEST_TYPES != 1
