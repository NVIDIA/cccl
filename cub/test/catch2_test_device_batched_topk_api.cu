// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Defer the unsupported-architecture diagnosis to the dispatch's runtime check (not a compile-time static_assert)
// so this test compiles across all target architectures, including pre-SM90, for the full configuration space. See
// _CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT in cub/device/dispatch/dispatch_batched_topk.cuh. Precedes CUB includes.
#define _CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT

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

C2H_TEST("cub::DeviceBatchedTopK::MaxKeys temp-storage API example", "[batched_topk][device]")
{
  // example-begin batched-topk-max-keys
  constexpr int num_segments = 2;
  constexpr int segment_size = 8;
  constexpr int k            = 3;

  auto keys_in  = thrust::device_vector<int>{5, -3, 1, 7, 8, 2, 4, 6, /**/ 0, 9, 3, 2, 1, 8, 7, 4};
  auto keys_out = thrust::device_vector<int>(num_segments * k, thrust::no_init);

  // Per-segment iterators: d_keys_in[s] yields an iterator to the start of segment s.
  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), segment_size);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);

  // Argument annotations: a small, compile-time segment size and k, plus the runtime segment count and item-count
  // bound.
  constexpr auto segment_sizes = cuda::args::constant<segment_size>{};
  constexpr auto k_arg         = cuda::args::constant<k>{};
  auto num_segs                = cuda::args::immediate{cuda::std::int64_t{num_segments}};

  // Top-k output is unordered and may be non-deterministic; this must be acknowledged via the environment.
  auto env = cuda::std::execution::env{cuda::execution::require(
    cuda::execution::determinism::not_guaranteed,
    cuda::execution::tie_break::unspecified,
    cuda::execution::output_ordering::unsorted)};

  // Query temporary storage requirements
  size_t temp_storage_bytes = 0;
  auto error                = cub::DeviceBatchedTopK::MaxKeys(
    nullptr, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segs, env);

  // Allocate temporary storage and run
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
  // Each segment's k largest keys are written to keys_out in unspecified order. The result set is fixed,
  // shown here sorted per segment:
  auto expected_result_set = thrust::device_vector<int>{8, 7, 6, /* segment 0 */ 9, 8, 7 /* segment 1 */};
  // example-end batched-topk-max-keys

  REQUIRE(error == cudaSuccess);
  // keys_out is unordered, so sort each segment (descending) before comparing against the expected set.
  thrust::sort(keys_out.begin(), keys_out.begin() + k, cuda::std::greater<int>{});
  thrust::sort(keys_out.begin() + k, keys_out.begin() + 2 * k, cuda::std::greater<int>{});
  REQUIRE(keys_out == expected_result_set);
}

C2H_TEST("cub::DeviceBatchedTopK::MinKeys temp-storage API example", "[batched_topk][device]")
{
  // example-begin batched-topk-min-keys
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

  size_t temp_storage_bytes = 0;
  auto error                = cub::DeviceBatchedTopK::MinKeys(
    nullptr, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segs, env);
  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);
  error = cub::DeviceBatchedTopK::MinKeys(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    segment_sizes,
    k_arg,
    num_segs,
    env);
  // Each segment's k smallest keys are written to keys_out in unspecified order. The result set is fixed,
  // shown here sorted per segment:
  auto expected_result_set = thrust::device_vector<int>{-3, 1, 2, /* segment 0 */ 0, 1, 2 /* segment 1 */};
  // example-end batched-topk-min-keys

  REQUIRE(error == cudaSuccess);
  // keys_out is unordered, so sort each segment (ascending) before comparing against the expected set.
  thrust::sort(keys_out.begin(), keys_out.begin() + k);
  thrust::sort(keys_out.begin() + k, keys_out.begin() + 2 * k);
  REQUIRE(keys_out == expected_result_set);
}

C2H_TEST("cub::DeviceBatchedTopK::MaxPairs temp-storage API example", "[batched_topk][device]")
{
  // example-begin batched-topk-max-pairs
  constexpr int num_segments = 2;
  constexpr int segment_size = 8;
  constexpr int k            = 3;

  auto keys_in    = thrust::device_vector<int>{5, -3, 1, 7, 8, 2, 4, 6, /**/ 0, 9, 3, 2, 1, 8, 7, 4};
  auto keys_out   = thrust::device_vector<int>(num_segments * k, thrust::no_init);
  auto values_out = thrust::device_vector<int>(num_segments * k, thrust::no_init);

  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), segment_size);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);
  // Input values are the per-segment item indices [0, segment_size).
  auto d_values_in = cuda::make_constant_iterator(cuda::make_counting_iterator(0));
  auto d_values_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(values_out.data())), k);

  constexpr auto segment_sizes = cuda::args::constant<segment_size>{};
  constexpr auto k_arg         = cuda::args::constant<k>{};
  auto num_segs                = cuda::args::immediate{cuda::std::int64_t{num_segments}};
  auto env                     = cuda::std::execution::env{cuda::execution::require(
    cuda::execution::determinism::not_guaranteed,
    cuda::execution::tie_break::unspecified,
    cuda::execution::output_ordering::unsorted)};

  size_t temp_storage_bytes = 0;
  auto error                = cub::DeviceBatchedTopK::MaxPairs(
    nullptr, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, segment_sizes, k_arg, num_segs, env);
  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);
  error = cub::DeviceBatchedTopK::MaxPairs(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    segment_sizes,
    k_arg,
    num_segs,
    env);
  // keys_out holds each segment's k largest keys. The key set is fixed (shown here sorted per segment). For
  // keys that tie, which equal element's value is returned is unspecified.
  auto expected_result_set = thrust::device_vector<int>{8, 7, 6, /* segment 0 */ 9, 8, 7 /* segment 1 */};
  // example-end batched-topk-max-pairs

  REQUIRE(error == cudaSuccess);

  // Each returned value is the source index of its key within the segment. Check that every value indexes
  // back to the input element whose key was selected.
  thrust::host_vector<int> h_keys_in(keys_in);
  thrust::host_vector<int> h_keys_out(keys_out);
  thrust::host_vector<int> h_values_out(values_out);
  for (int s = 0; s < num_segments; ++s)
  {
    for (int j = 0; j < k; ++j)
    {
      const int idx = s * k + j;
      const int v   = h_values_out[idx];
      REQUIRE(v >= 0);
      REQUIRE(v < segment_size);
      REQUIRE(h_keys_in[s * segment_size + v] == h_keys_out[idx]);
    }
  }

  // keys_out is unordered, so sort each segment (descending) before comparing against the expected set.
  thrust::sort(keys_out.begin(), keys_out.begin() + k, cuda::std::greater<int>{});
  thrust::sort(keys_out.begin() + k, keys_out.begin() + 2 * k, cuda::std::greater<int>{});
  REQUIRE(keys_out == expected_result_set);
}

C2H_TEST("cub::DeviceBatchedTopK::MinPairs temp-storage API example", "[batched_topk][device]")
{
  // example-begin batched-topk-min-pairs
  constexpr int num_segments = 2;
  constexpr int segment_size = 8;
  constexpr int k            = 3;

  auto keys_in    = thrust::device_vector<int>{5, -3, 1, 7, 8, 2, 4, 6, /**/ 0, 9, 3, 2, 1, 8, 7, 4};
  auto keys_out   = thrust::device_vector<int>(num_segments * k, thrust::no_init);
  auto values_out = thrust::device_vector<int>(num_segments * k, thrust::no_init);

  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), segment_size);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);
  auto d_values_in = cuda::make_constant_iterator(cuda::make_counting_iterator(0));
  auto d_values_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(values_out.data())), k);

  constexpr auto segment_sizes = cuda::args::constant<segment_size>{};
  constexpr auto k_arg         = cuda::args::constant<k>{};
  auto num_segs                = cuda::args::immediate{cuda::std::int64_t{num_segments}};
  auto env                     = cuda::std::execution::env{cuda::execution::require(
    cuda::execution::determinism::not_guaranteed,
    cuda::execution::tie_break::unspecified,
    cuda::execution::output_ordering::unsorted)};

  size_t temp_storage_bytes = 0;
  auto error                = cub::DeviceBatchedTopK::MinPairs(
    nullptr, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, segment_sizes, k_arg, num_segs, env);
  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);
  error = cub::DeviceBatchedTopK::MinPairs(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    segment_sizes,
    k_arg,
    num_segs,
    env);
  // keys_out holds each segment's k smallest keys. The key set is fixed (shown here sorted per segment). For
  // keys that tie, which equal element's value is returned is unspecified.
  auto expected_result_set = thrust::device_vector<int>{-3, 1, 2, /* segment 0 */ 0, 1, 2 /* segment 1 */};
  // example-end batched-topk-min-pairs

  REQUIRE(error == cudaSuccess);

  // Each returned value is the source index of its key within the segment. Check that every value indexes
  // back to the input element whose key was selected.
  thrust::host_vector<int> h_keys_in(keys_in);
  thrust::host_vector<int> h_keys_out(keys_out);
  thrust::host_vector<int> h_values_out(values_out);
  for (int s = 0; s < num_segments; ++s)
  {
    for (int j = 0; j < k; ++j)
    {
      const int idx = s * k + j;
      const int v   = h_values_out[idx];
      REQUIRE(v >= 0);
      REQUIRE(v < segment_size);
      REQUIRE(h_keys_in[s * segment_size + v] == h_keys_out[idx]);
    }
  }

  // keys_out is unordered, so sort each segment (ascending) before comparing against the expected set.
  thrust::sort(keys_out.begin(), keys_out.begin() + k);
  thrust::sort(keys_out.begin() + k, keys_out.begin() + 2 * k);
  REQUIRE(keys_out == expected_result_set);
}
