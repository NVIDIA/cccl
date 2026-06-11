// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_batched_topk.cuh>
#include <cub/device/dispatch/dispatch_batched_topk_cluster.cuh>
#include <cub/util_type.cuh>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>

#include <cuda/iterator>
#include <cuda/std/__algorithm/min.h>

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

// Maps a flat element index to one of only 8 distinct key values, so a large segment contains many duplicates and the
// k-th key's bucket holds a large tied-candidate set. Used to stress the cluster agent's candidate/tie-break path (and
// the deterministic tie-break scan when CUB_ENABLE_CLUSTER_TOPK_DETERMINISM is defined).
template <typename KeyT>
struct heavy_tie_key_op
{
  template <typename IndexT>
  __host__ __device__ KeyT operator()(IndexT i) const
  {
    const unsigned h = static_cast<unsigned>(static_cast<cuda::std::uint64_t>(i) * 2654435761u);
    return static_cast<KeyT>((h >> 13) & 7u);
  }
};

enum class topk_backend
{
  baseline,
  cluster,
};

inline constexpr topk_backend selected_backend = topk_backend::cluster;

template <typename KeyInputItItT,
          typename KeyOutputItItT,
          typename SegmentSizeParamT,
          typename KParamT,
          typename SelectDirectionT,
          typename NumSegmentsParameterT,
          typename TotalNumItemsGuaranteeT>
CUB_RUNTIME_FUNCTION static cudaError_t dispatch_batched_topk_keys(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  SegmentSizeParamT segment_sizes,
  KParamT k,
  SelectDirectionT select_direction,
  NumSegmentsParameterT num_segments,
  TotalNumItemsGuaranteeT total_num_items_guarantee,
  cudaStream_t stream = nullptr)
{
  if constexpr (selected_backend == topk_backend::cluster)
  {
    return cub::detail::batched_topk_cluster::dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_key_segments_it,
      d_key_segments_out_it,
      static_cast<cub::NullType**>(nullptr),
      static_cast<cub::NullType**>(nullptr),
      segment_sizes,
      k,
      select_direction,
      num_segments,
      total_num_items_guarantee,
      stream);
  }
  else
  {
    auto values_it = static_cast<cub::NullType**>(nullptr);
    return cub::detail::batched_topk::dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_key_segments_it,
      d_key_segments_out_it,
      values_it,
      values_it,
      segment_sizes,
      k,
      select_direction,
      num_segments,
      total_num_items_guarantee,
      stream);
  }
}

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_LAUNCH_WRAPPER(dispatch_batched_topk_keys, batched_topk_keys);

// Total segment size
using max_segment_size_list = c2h::enum_type_list<cuda::std::size_t, 4 * 1024>;

// Segment size: static, uniform
using max_num_k_list = c2h::enum_type_list<cuda::std::size_t, 32, 4 * 1024>;

#if 0
using key_types =
  c2h::type_list<cuda::std::uint8_t,
                 float,
                 cuda::std::uint64_t
// clang-format off
#if TEST_HALF_T()
                , half_t
#endif // TEST_HALF_T()
#if TEST_BF_T()
                , bfloat16_t
#endif // TEST_BF_T()
>;
#else
using key_types = c2h::type_list<float>;
#endif
// clang-format on

// Selection direction is a compile-time option; cover both as a static test axis.
using select_direction_list =
  c2h::enum_type_list<cub::detail::topk::select, cub::detail::topk::select::min, cub::detail::topk::select::max>;

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

  // Run the top-k algorithm
  batched_topk_keys(
    d_keys_in,
    d_keys_out,
    ::cuda::__argument::__immediate{segment_size, ::cuda::__argument::__bounds<segment_size_t{1}, max_segment_size>()},
    ::cuda::__argument::__immediate{k, ::cuda::__argument::__bounds<segment_size_t{1}, static_max_k>()},
    ::cuda::__argument::__constant<direction>{},
    ::cuda::__argument::__immediate{num_segments},
    ::cuda::__argument::__immediate{num_segments * segment_size});
  // Prepare expected results
  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);

  // Since the results of top-k are unordered, sort output segments before comparison.
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

#if TEST_LAUNCH != 1
TEST_CASE("DeviceBatchedTopK::{Min,Max}Keys work with large fixed-size unaligned segments",
          "[keys][segmented][topk][device][cluster]")
{
  using key_t           = float;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  // `static_max_segment_size` is chosen to exceed the largest all-resident cluster coverage (~16 blocks worth of
  // resident SMEM), so the 1 Mi-element segments force the agent's gmem-streaming overflow path (including an
  // unaligned overflow tail via `- 31`), while the 128 Ki-element segment still runs fully resident under the same
  // streaming-capable launch configuration.
  constexpr segment_size_t static_max_segment_size = 1024 * 1024;
  constexpr segment_size_t static_max_k            = 4 * 1024;
  constexpr segment_index_t num_segments           = 3;

  const auto direction = cub::detail::topk::select::max;
  const int pad        = GENERATE(0, 1, 3, 7);
  const segment_size_t segment_size =
    GENERATE_COPY(values({static_max_segment_size, static_max_segment_size - 31, segment_size_t{128 * 1024}}));
  const segment_size_t max_k = (cuda::std::min) (static_max_k, segment_size);
  const segment_size_t k     = GENERATE_COPY(values({segment_size_t{1}, max_k / 2, max_k}));

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

  batched_topk_keys(
    d_keys_in,
    d_keys_out,
    ::cuda::__argument::__immediate{
      segment_size, ::cuda::__argument::__bounds<segment_size_t{1}, static_max_segment_size>()},
    ::cuda::__argument::__immediate{k, ::cuda::__argument::__bounds<segment_size_t{1}, static_max_k>()},
    ::cuda::__argument::__constant<cub::detail::topk::select::max>{},
    ::cuda::__argument::__immediate{num_segments},
    ::cuda::__argument::__immediate{num_segments * segment_size});

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

TEST_CASE("DeviceBatchedTopK::{Min,Max}Keys stream large segments through a non-contiguous key iterator",
          "[keys][segmented][topk][device][cluster]")
{
  using key_t           = float;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  // The counting-iterator key source is non-contiguous, so the agent uses its generic overflow-streaming path rather
  // than BlockLoadToShared. `static_max_segment_size` exceeds the largest all-resident cluster coverage, so the 1 Mi
  // -element segments stream (incl. an unaligned `- 31` tail), while the 128 Ki-element segment validates the generic
  // resident path (no streaming) through the same code. Keeping the largest total below 2^24 makes every key an exact
  // float.
  constexpr segment_size_t static_max_segment_size = 1024 * 1024;
  constexpr segment_size_t static_max_k            = 4 * 1024;
  constexpr segment_index_t num_segments           = 3;

  const auto direction = cub::detail::topk::select::max;
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

  batched_topk_keys(
    d_keys_in,
    d_keys_out,
    ::cuda::__argument::__immediate{
      segment_size, ::cuda::__argument::__bounds<segment_size_t{1}, static_max_segment_size>()},
    ::cuda::__argument::__immediate{k, ::cuda::__argument::__bounds<segment_size_t{1}, static_max_k>()},
    ::cuda::__argument::__constant<cub::detail::topk::select::max>{},
    ::cuda::__argument::__immediate{num_segments},
    ::cuda::__argument::__immediate{num_items});

  // The flattened input is the identity sequence, so build the expected keys directly and reuse the standard
  // sort + compact verification.
  c2h::device_vector<key_t> expected_keys(num_items, thrust::no_init);
  thrust::sequence(expected_keys.begin(), expected_keys.end());

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);

  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}
#endif // TEST_LAUNCH != 1

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
    get_output_size_op{segment_offsets.cbegin(), cuda::constant_iterator(k)});
  // Exclusive prefix sum of the per-segment output sizes. Scan only the `num_segments` valid sizes (each reads
  // `offset[seg]`/`offset[seg + 1]`, staying in bounds) into indices [1, num_segments]; index 0 stays 0.
  c2h::device_vector<segment_size_t> compacted_offsets(num_segments + 1);
  thrust::inclusive_scan(
    compacted_output_sizes_it, compacted_output_sizes_it + num_segments, compacted_offsets.begin() + 1);
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
  batched_topk_keys(
    d_keys_in,
    d_keys_out,
    ::cuda::__argument::__deferred_sequence{
      segment_size_it, ::cuda::__argument::__bounds<segment_size_t{1}, static_max_segment_size>()},
    ::cuda::__argument::__immediate{k, ::cuda::__argument::__bounds<segment_size_t{1}, static_max_k>()},
    ::cuda::__argument::__constant<direction>{},
    ::cuda::__argument::__immediate{num_segments},
    ::cuda::__argument::__immediate{num_items});

  // Verify keys are returned correctly: sort each segment of the expected input, then compact the top-k
  segmented_sort_keys(expected_keys, num_segments, segment_offsets.cbegin(), segment_offsets.cbegin() + 1, direction);
  expected_keys = compact_to_topk_batched(expected_keys, segment_offsets, k);

  // Since the results of top-k are unordered, sort compacted output segments before comparison
  segmented_sort_keys(
    keys_out_buffer, num_segments, compacted_offsets.cbegin(), compacted_offsets.cbegin() + 1, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

#if TEST_LAUNCH != 1
TEST_CASE("DeviceBatchedTopK::{Min,Max}Keys work with large variable-size unaligned segments",
          "[keys][segmented][topk][device][cluster]")
{
  using key_t           = float;
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  // `static_max_segment_size` exceeds the largest all-resident cluster coverage, so a single per-segment launch
  // mixes streaming segments (the 1 Mi-element ones, one with an unaligned `- 31` overflow tail) with fully-resident
  // segments (96 Ki + 17 and 257 elements).
  constexpr segment_size_t static_max_segment_size = 1100 * 1024;
  constexpr segment_size_t static_max_k            = 4 * 1024;

  const auto direction   = cub::detail::topk::select::max;
  const int pad          = GENERATE(1, 3, 7);
  const segment_size_t k = GENERATE_COPY(values({segment_size_t{1}, static_max_k / 2, static_max_k}));

  constexpr segment_size_t big_segment_size = 1024 * 1024;
  c2h::host_vector<segment_size_t> h_segment_offsets{
    0,
    big_segment_size,
    big_segment_size + (big_segment_size - 31),
    big_segment_size + (big_segment_size - 31) + (96 * 1024 + 17),
    big_segment_size + (big_segment_size - 31) + (96 * 1024 + 17) + 257};
  c2h::device_vector<segment_size_t> segment_offsets = h_segment_offsets;
  const segment_index_t num_segments                 = static_cast<segment_index_t>(h_segment_offsets.size() - 1);
  const segment_size_t num_items                     = h_segment_offsets.back();

  auto segment_offsets_it = thrust::raw_pointer_cast(segment_offsets.data());
  auto segment_size_it    = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}), segment_size_op<segment_size_t*>{segment_offsets_it});

  CAPTURE(pad, static_max_segment_size, static_max_k, k, num_segments, num_items, direction);

  auto compacted_output_sizes_it = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}),
    get_output_size_op{segment_offsets.cbegin(), cuda::constant_iterator(k)});
  // Exclusive prefix sum of the per-segment output sizes. Scan only the `num_segments` valid sizes (each reads
  // `offset[seg]`/`offset[seg + 1]`, staying in bounds) into indices [1, num_segments]; index 0 stays 0.
  c2h::device_vector<segment_size_t> compacted_offsets(num_segments + 1);
  thrust::inclusive_scan(
    compacted_output_sizes_it, compacted_output_sizes_it + num_segments, compacted_offsets.begin() + 1);
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

  batched_topk_keys(
    d_keys_in,
    d_keys_out,
    ::cuda::__argument::__deferred_sequence{
      segment_size_it, ::cuda::__argument::__bounds<segment_size_t{1}, static_max_segment_size>()},
    ::cuda::__argument::__immediate{k, ::cuda::__argument::__bounds<segment_size_t{1}, static_max_k>()},
    ::cuda::__argument::__constant<cub::detail::topk::select::max>{},
    ::cuda::__argument::__immediate{num_segments},
    ::cuda::__argument::__immediate{num_items});

  segmented_sort_keys(expected_keys, num_segments, segment_offsets.cbegin(), segment_offsets.cbegin() + 1, direction);
  expected_keys = compact_to_topk_batched(expected_keys, segment_offsets, k);

  segmented_sort_keys(
    keys_out_buffer, num_segments, compacted_offsets.cbegin(), compacted_offsets.cbegin() + 1, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}
#endif // TEST_LAUNCH != 1

#if TEST_LAUNCH != 1
// Heavy-tie stress/regression: collapse the keys to only a handful of distinct values so the k-th key's bucket holds a
// large set of tied candidates. This exercises the cluster agent's candidate path and, when built with
// CUB_ENABLE_CLUSTER_TOPK_DETERMINISM, the deterministic cross-CTA tie-break scan (cand_prefix + BlockScan ranks). The
// returned top-k *value* set must be correct in either mode (tied keys are equal, so which tied index is chosen is not
// observable in keys-only output, but the count of each value at the boundary must be exact).
C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys handle heavy ties at the k-th boundary",
         "[keys][segmented][topk][device][cluster]",
         key_types,
         select_direction_list)
{
  using key_t              = c2h::get<0, TestType>;
  constexpr auto direction = c2h::get<1, TestType>::value;

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

  batched_topk_keys(
    d_keys_in,
    d_keys_out,
    ::cuda::__argument::__immediate{
      segment_size, ::cuda::__argument::__bounds<segment_size_t{1}, static_max_segment_size>()},
    ::cuda::__argument::__immediate{k, ::cuda::__argument::__bounds<segment_size_t{1}, static_max_k>()},
    ::cuda::__argument::__constant<direction>{},
    ::cuda::__argument::__immediate{num_segments},
    ::cuda::__argument::__immediate{num_items});

  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);

  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}
#endif // TEST_LAUNCH != 1

// Regression test: top-k must preserve -0.0f in the output (not normalize to +0.0f).
C2H_TEST("DeviceBatchedTopK::MinKeys preserves -0.0f in output", "[keys][segmented][topk][device][float]")
{
  constexpr cuda::std::int64_t segment_size                      = 8;
  constexpr cuda::std::int64_t k                                 = 5;
  constexpr cuda::std::int64_t num_segments                      = 1;
  [[maybe_unused]] constexpr cuda::std::int64_t max_segment_size = 64; // msvc warns, only used in nttp

  // Input: one segment containing -0.0f and +0.0f; top-5 min should include both zeros.
  c2h::device_vector<float> d_keys_in{3.0f, -0.0f, 1.0f, 2.0f, 0.0f, -1.0f, 4.0f, 5.0f};
  c2h::device_vector<float> d_keys_out(k, thrust::no_init);

  auto d_keys_in_it =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(d_keys_in.data())), segment_size);
  auto d_keys_out_it =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(d_keys_out.data())), k);

  batched_topk_keys(
    d_keys_in_it,
    d_keys_out_it,
    ::cuda::__argument::__immediate{
      segment_size, ::cuda::__argument::__bounds<cuda::std::int64_t{1}, max_segment_size>()},
    ::cuda::__argument::__immediate{k, ::cuda::__argument::__bounds<cuda::std::int64_t{1}, k>()},
    ::cuda::__argument::__constant<cub::detail::topk::select::min>{},
    ::cuda::__argument::__immediate{num_segments},
    ::cuda::__argument::__immediate{num_segments * segment_size});

  const int num_minus_zero = static_cast<int>(thrust::count_if(d_keys_out.begin(), d_keys_out.end(), is_minus_zero{}));
  REQUIRE(num_minus_zero >= 1);
}
