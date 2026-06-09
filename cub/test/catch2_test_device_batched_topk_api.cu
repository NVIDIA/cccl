// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_batched_topk.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/argument>
#include <cuda/iterator>
#include <cuda/std/__execution/env.h>
#include <cuda/std/functional>

#include <c2h/catch2_test_helper.h>

// Two segments of eight keys each. The per-segment top-3 max are {8,7,6} and {9,8,7}; the top-3 min are {-3,1,2} and
// {0,1,2}.
static auto make_two_segments()
{
  return thrust::device_vector<int>{5, -3, 1, 7, 8, 2, 4, 6, /**/ 0, 9, 3, 2, 1, 8, 7, 4};
}

C2H_TEST("cub::DeviceBatchedTopK::MaxKeys temp-storage API example", "[batched_topk][device]")
{
  // example-begin batched-topk-max-keys
  constexpr int num_segments = 2;
  constexpr int segment_size = 8;
  constexpr int k            = 3;

  auto keys_in  = make_two_segments();
  auto keys_out = thrust::device_vector<int>(num_segments * k, thrust::no_init);

  // Per-segment iterators: d_keys_in[s] yields an iterator to the start of segment s.
  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), segment_size);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);

  // Argument annotations: a small, compile-time segment size and k, plus the runtime segment count and item-count
  // bound.
  auto segment_sizes = cuda::args::constant<segment_size>{};
  auto k_arg         = cuda::args::constant<k>{};
  auto num_segs      = cuda::args::immediate{cuda::std::int64_t{num_segments}};
  auto total_items   = cuda::args::immediate{cuda::std::int64_t{num_segments * segment_size}};

  // Top-k output is unordered and may be non-deterministic; this must be acknowledged via the environment.
  auto env = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted)};

  // Query temporary storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceBatchedTopK::MaxKeys(
    nullptr, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segs, total_items, env);

  // Allocate temporary storage and run
  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);
  cub::DeviceBatchedTopK::MaxKeys(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    segment_sizes,
    k_arg,
    num_segs,
    total_items,
    env);
  // example-end batched-topk-max-keys

  // The per-segment output is unordered; sort each segment (descending) before comparing.
  thrust::sort(keys_out.begin(), keys_out.begin() + k, cuda::std::greater<int>{});
  thrust::sort(keys_out.begin() + k, keys_out.begin() + 2 * k, cuda::std::greater<int>{});
  REQUIRE(keys_out == thrust::device_vector<int>{8, 7, 6, 9, 8, 7});
}

C2H_TEST("cub::DeviceBatchedTopK::MinKeys temp-storage API example", "[batched_topk][device]")
{
  // example-begin batched-topk-min-keys
  constexpr int num_segments = 2;
  constexpr int segment_size = 8;
  constexpr int k            = 3;

  auto keys_in  = make_two_segments();
  auto keys_out = thrust::device_vector<int>(num_segments * k, thrust::no_init);

  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), segment_size);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);

  auto segment_sizes = cuda::args::constant<segment_size>{};
  auto k_arg         = cuda::args::constant<k>{};
  auto num_segs      = cuda::args::immediate{cuda::std::int64_t{num_segments}};
  auto total_items   = cuda::args::immediate{cuda::std::int64_t{num_segments * segment_size}};
  auto env           = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted)};

  size_t temp_storage_bytes = 0;
  cub::DeviceBatchedTopK::MinKeys(
    nullptr, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segs, total_items, env);
  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);
  cub::DeviceBatchedTopK::MinKeys(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    segment_sizes,
    k_arg,
    num_segs,
    total_items,
    env);
  // example-end batched-topk-min-keys

  thrust::sort(keys_out.begin(), keys_out.begin() + k);
  thrust::sort(keys_out.begin() + k, keys_out.begin() + 2 * k);
  REQUIRE(keys_out == thrust::device_vector<int>{-3, 1, 2, 0, 1, 2});
}

C2H_TEST("cub::DeviceBatchedTopK::MaxPairs temp-storage API example", "[batched_topk][device]")
{
  // example-begin batched-topk-max-pairs
  constexpr int num_segments = 2;
  constexpr int segment_size = 8;
  constexpr int k            = 3;

  auto keys_in    = make_two_segments();
  auto keys_out   = thrust::device_vector<int>(num_segments * k, thrust::no_init);
  auto values_out = thrust::device_vector<int>(num_segments * k, thrust::no_init);

  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), segment_size);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);
  // Input values are the per-segment item indices [0, segment_size).
  auto d_values_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(cuda::make_counting_iterator(0)), segment_size);
  auto d_values_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(values_out.data())), k);

  auto segment_sizes = cuda::args::constant<segment_size>{};
  auto k_arg         = cuda::args::constant<k>{};
  auto num_segs      = cuda::args::immediate{cuda::std::int64_t{num_segments}};
  auto total_items   = cuda::args::immediate{cuda::std::int64_t{num_segments * segment_size}};
  auto env           = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted)};

  size_t temp_storage_bytes = 0;
  cub::DeviceBatchedTopK::MaxPairs(
    nullptr,
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    segment_sizes,
    k_arg,
    num_segs,
    total_items,
    env);
  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);
  cub::DeviceBatchedTopK::MaxPairs(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    segment_sizes,
    k_arg,
    num_segs,
    total_items,
    env);
  // example-end batched-topk-max-pairs

  // Each input value is the global item index, so each returned value indexes back to its returned key.
  thrust::host_vector<int> h_keys_in(keys_in);
  thrust::host_vector<int> h_keys_out(keys_out);
  thrust::host_vector<int> h_values_out(values_out);
  for (int s = 0; s < num_segments; ++s)
  {
    for (int j = 0; j < k; ++j)
    {
      REQUIRE(h_keys_in[h_values_out[s * k + j]] == h_keys_out[s * k + j]);
    }
  }
}

C2H_TEST("cub::DeviceBatchedTopK::MinPairs temp-storage API example", "[batched_topk][device]")
{
  // example-begin batched-topk-min-pairs
  constexpr int num_segments = 2;
  constexpr int segment_size = 8;
  constexpr int k            = 3;

  auto keys_in    = make_two_segments();
  auto keys_out   = thrust::device_vector<int>(num_segments * k, thrust::no_init);
  auto values_out = thrust::device_vector<int>(num_segments * k, thrust::no_init);

  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), segment_size);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);
  auto d_values_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(cuda::make_counting_iterator(0)), segment_size);
  auto d_values_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(values_out.data())), k);

  auto segment_sizes = cuda::args::constant<segment_size>{};
  auto k_arg         = cuda::args::constant<k>{};
  auto num_segs      = cuda::args::immediate{cuda::std::int64_t{num_segments}};
  auto total_items   = cuda::args::immediate{cuda::std::int64_t{num_segments * segment_size}};
  auto env           = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted)};

  size_t temp_storage_bytes = 0;
  cub::DeviceBatchedTopK::MinPairs(
    nullptr,
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    segment_sizes,
    k_arg,
    num_segs,
    total_items,
    env);
  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);
  cub::DeviceBatchedTopK::MinPairs(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    segment_sizes,
    k_arg,
    num_segs,
    total_items,
    env);
  // example-end batched-topk-min-pairs

  thrust::host_vector<int> h_keys_in(keys_in);
  thrust::host_vector<int> h_keys_out(keys_out);
  thrust::host_vector<int> h_values_out(values_out);
  for (int s = 0; s < num_segments; ++s)
  {
    for (int j = 0; j < k; ++j)
    {
      REQUIRE(h_keys_in[h_values_out[s * k + j]] == h_keys_out[s * k + j]);
    }
  }
}
